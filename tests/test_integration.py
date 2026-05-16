"""End-to-end smoke tests: dataset → collator → model → loss."""

import pytest
import torch

from mech_uspto.data.dataset import MechUSPTODataset
from mech_uspto.data.loaders import collate_fn_with_spectators
from mech_uspto.losses.focal import MaskedFocalLossWithSpectators
from mech_uspto.models.transformer import ReactionTransformer


@pytest.mark.parametrize(
    "task_mode, num_classes, shift",
    [
        pytest.param(
            "stepwise",
            3,
            1,
            marks=pytest.mark.xfail(
                strict=True,
                reason="Parser does not yet decompose multi-step rxns into "
                "elementary steps; fixture has Δ=-2 which is out of stepwise range.",
            ),
        ),
        ("end_to_end", 7, 3),
    ],
)
def test_dataset_to_loss_pipeline(sample_reactions, task_mode, num_classes, shift):
    """Regression: this would have caught the per-atom vs per-pair spectator bug."""
    dataset = MechUSPTODataset(
        sample_reactions, task_mode=task_mode, compute_spectators=True
    )
    batch = collate_fn_with_spectators([dataset[i] for i in range(len(dataset))])
    assert batch is not None

    model = ReactionTransformer(
        hidden_dim=32, num_heads=4, num_layers=2, num_classes=num_classes
    )
    criterion = MaskedFocalLossWithSpectators(num_classes=num_classes, gamma=2.0)

    logits, mask = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)
    targets = batch.y_padded + shift
    loss = criterion(
        logits, targets, mask_2d, getattr(batch, "spectator_padded", None)
    )

    assert loss.dim() == 0
    assert torch.isfinite(loss)
    loss.backward()  # gradients must also flow
