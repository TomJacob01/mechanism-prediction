"""Tests for the ``DeltaMLP`` head."""

import pytest
import torch

from mech_uspto.models.heads import DeltaMLP


@pytest.mark.parametrize("num_classes", [3, 5])
def test_delta_mlp_output_shape_and_symmetry(num_classes):
    head = DeltaMLP(hidden_dim=64, num_classes=num_classes, dropout=0.0)
    head.eval()
    h = torch.randn(2, 7, 64)

    logits = head(h)
    assert logits.shape == (2, 7, 7, num_classes)
    # Predictions for (i, j) and (j, i) must match.
    assert torch.allclose(logits, logits.transpose(1, 2), atol=1e-6)


def test_delta_mlp_rejects_invalid_class_count():
    with pytest.raises(ValueError):
        DeltaMLP(hidden_dim=64, num_classes=4)
