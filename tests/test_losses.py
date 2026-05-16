"""Tests for ``MaskedFocalLossWithSpectators``."""

import torch

from mech_uspto.losses.focal import MaskedFocalLossWithSpectators


def _make_inputs(B=2, N=5, num_classes=3):
    logits = torch.randn(B, N, N, num_classes)
    targets = torch.randint(0, num_classes, (B, N, N))
    mask = torch.ones(B, N, dtype=torch.bool)
    mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)
    return logits, targets, mask_2d


def test_loss_is_finite_scalar():
    logits, targets, mask_2d = _make_inputs()
    criterion = MaskedFocalLossWithSpectators(num_classes=3, gamma=2.0)
    loss = criterion(logits, targets, mask_2d)

    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_spectator_downweighting_reduces_loss():
    logits, targets, mask_2d = _make_inputs()
    criterion = MaskedFocalLossWithSpectators(num_classes=3, gamma=2.0)

    no_spec_loss = criterion(logits, targets, mask_2d).item()
    all_spectator = torch.ones_like(targets, dtype=torch.bool)
    all_spec_loss = criterion(logits, targets, mask_2d, spectator_mask=all_spectator).item()

    # With every position downweighted by 0.1 the loss must shrink.
    assert all_spec_loss < no_spec_loss


def test_class_weights_alter_loss():
    logits, targets, mask_2d = _make_inputs()
    base = MaskedFocalLossWithSpectators(num_classes=3, gamma=2.0)
    weighted = MaskedFocalLossWithSpectators(
        num_classes=3, gamma=2.0, weights=torch.tensor([10.0, 0.1, 10.0])
    )
    assert base(logits, targets, mask_2d).item() != weighted(logits, targets, mask_2d).item()


def test_all_padding_returns_zero_loss():
    logits, targets, _ = _make_inputs()
    empty_mask = torch.zeros(logits.shape[0], logits.shape[1], logits.shape[1])
    criterion = MaskedFocalLossWithSpectators(num_classes=3, gamma=2.0)
    loss = criterion(logits, targets, empty_mask)
    assert loss.item() == 0.0


def test_spectator_mask_accepts_per_atom_shape():
    """Regression: collator emits per-atom (B, N) spectator masks; loss must
    broadcast them to per-pair (B, N, N) (pair is spectator iff both atoms are).
    """
    B, N, C = 2, 5, 3
    logits, targets, mask_2d = _make_inputs(B=B, N=N, num_classes=C)
    criterion = MaskedFocalLossWithSpectators(num_classes=C, gamma=2.0)

    per_atom = torch.tensor(
        [[True, True, False, False, False], [False, False, True, True, True]]
    )
    per_pair = per_atom.unsqueeze(2) & per_atom.unsqueeze(1)

    loss_atom = criterion(logits, targets, mask_2d, spectator_mask=per_atom).item()
    loss_pair = criterion(logits, targets, mask_2d, spectator_mask=per_pair).item()

    assert loss_atom == loss_pair
