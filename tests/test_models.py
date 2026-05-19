"""Tests for the ``DeltaMLP`` head."""

import pytest
import torch

from mech_uspto.models.heads import DeltaMLP


@pytest.mark.parametrize("num_classes", [3, 7])
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


def test_delta_mlp_bias_init_matches_class_prior():
    """RetinaNet bias-init: step-0 softmax should approximate ``class_prior``.

    With zero-input pair features the only non-noise contribution to the
    logits is the final-layer bias, so the softmax is exactly the (centred)
    prior. We test that path explicitly.
    """
    prior = torch.tensor([0.0005, 0.002, 0.010, 0.975, 0.010, 0.002, 0.0005])
    head = DeltaMLP(hidden_dim=32, num_classes=7, dropout=0.0, class_prior=prior)
    head.eval()

    # Bypass the encoder: feed zeros, so logits = bias of final linear layer.
    h = torch.zeros(1, 4, 32)
    logits = head(h)
    probs = torch.softmax(logits, dim=-1)
    # All pairs share the same bias, so any pair (i, j) has prior-ish softmax.
    # The first MLP layer's bias adds a small residual to the logits, so the
    # match is approximate (~few % relative) rather than exact.
    assert torch.allclose(probs[0, 0, 0], prior, atol=2e-3)
    # No-change class (idx 3) is the argmax everywhere.
    assert (logits.argmax(dim=-1) == 3).all()


def test_delta_mlp_bias_init_validates_prior_length():
    with pytest.raises(ValueError):
        DeltaMLP(hidden_dim=16, num_classes=3, class_prior=torch.tensor([0.5, 0.5]))
    with pytest.raises(ValueError):
        DeltaMLP(hidden_dim=16, num_classes=3, class_prior=torch.tensor([0.5, -0.1, 0.6]))


def test_delta_mlp_with_edge_features():
    """edge_dim>0 must accept (B, N, N, edge_dim) and produce same output shape."""
    edge_dim = 6
    head = DeltaMLP(hidden_dim=32, num_classes=3, dropout=0.0, edge_dim=edge_dim)
    head.eval()
    h = torch.randn(2, 5, 32)
    edge_dense = torch.randn(2, 5, 5, edge_dim)
    # Symmetrise edge features (mirrors the densified collator output).
    edge_dense = (edge_dense + edge_dense.transpose(1, 2)) / 2

    logits = head(h, edge_dense=edge_dense)
    assert logits.shape == (2, 5, 5, 3)
    # Output remains symmetric over (i, j).
    assert torch.allclose(logits, logits.transpose(1, 2), atol=1e-6)


def test_delta_mlp_edge_features_actually_used():
    """Changing edge features must change the head's output."""
    head = DeltaMLP(hidden_dim=16, num_classes=3, dropout=0.0, edge_dim=4)
    head.eval()
    h = torch.randn(1, 3, 16)
    e1 = torch.zeros(1, 3, 3, 4)
    e2 = torch.ones(1, 3, 3, 4)

    out1 = head(h, edge_dense=e1)
    out2 = head(h, edge_dense=e2)
    assert not torch.allclose(out1, out2)


def test_delta_mlp_edge_features_required_when_configured():
    head = DeltaMLP(hidden_dim=16, num_classes=3, edge_dim=4)
    head.eval()
    with pytest.raises(ValueError):
        head(torch.randn(1, 3, 16))  # missing edge_dense


def test_reaction_transformer_with_edge_features_in_head():
    """Densified edge features must flow end-to-end without shape errors."""
    from mech_uspto.models.transformer import ReactionTransformer

    prior = torch.tensor([0.01, 0.98, 0.01])
    model = ReactionTransformer(
        node_in=8,
        edge_in=6,
        hidden_dim=16,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
        num_classes=3,
        class_prior=prior,
        use_edge_features_in_head=True,
    )
    model.eval()

    # Two tiny graphs in one batch: graph 0 has 3 nodes, graph 1 has 2 nodes.
    x = torch.randn(5, 8)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]],
        dtype=torch.long,
    )
    edge_attr = torch.randn(6, 6)
    edge_attr = (edge_attr + edge_attr.flip(0)) / 2  # mimic featurize_edges symmetry
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)

    logits, mask = model(x, edge_index, edge_attr, batch)
    B, N = mask.shape
    assert logits.shape == (B, N, N, 3)
    assert torch.allclose(logits, logits.transpose(1, 2), atol=1e-5)
