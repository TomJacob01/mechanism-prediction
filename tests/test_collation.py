"""Tests for the collation function."""

from mech_uspto.data.loaders import collate_fn_with_spectators


def test_collation_single_item(stepwise_dataset):
    batch = collate_fn_with_spectators([stepwise_dataset[0]])
    assert batch is not None
    assert batch.y_padded.dim() == 3  # (B, N, N)
    assert batch.y_padded.shape[0] == 1
    assert hasattr(batch, "spectator_padded")
    # Spectator mask is per-atom (B, N), not per-bond (B, N, N).
    assert batch.spectator_padded.dim() == 2
    assert batch.spectator_padded.shape[0] == batch.y_padded.shape[0]
    assert batch.spectator_padded.shape[1] == batch.y_padded.shape[1]


def test_collation_filters_none_inputs(stepwise_dataset):
    batch = collate_fn_with_spectators([None, stepwise_dataset[0], None])
    assert batch is not None
    assert batch.y_padded.shape[0] == 1


def test_collation_all_none_returns_none():
    assert collate_fn_with_spectators([None, None]) is None
