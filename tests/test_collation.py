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


def test_create_dataloaders_smoke(sample_csv_path, tmp_path):
    """Regression: the Subset-based refactor must produce working dataloaders."""
    from mech_uspto.data.loaders import create_dataloaders

    loaders = create_dataloaders(
        csv_path=str(sample_csv_path),
        task_mode="end_to_end",
        batch_size=1,
        train_val_test_split=(0.5, 0.25, 0.25),
        cache_dir=str(tmp_path / "cache"),
    )
    assert set(loaders.keys()) == {"train", "val", "test"}
    total = (
        len(loaders["train"].dataset)
        + len(loaders["val"].dataset)
        + len(loaders["test"].dataset)
    )
    assert total == 2

    batch = next(iter(loaders["train"]))
    assert batch is not None
    assert batch.y_padded.dim() == 3


def test_collation_does_not_mutate_input_data(stepwise_dataset):
    """Regression: collate must NOT delete fields from the original Data objects.

    PyG ``Data`` items are shared by reference across epochs by the dataloader.
    If the collator does ``del d.y`` on the input, epoch 2 sees ``d.y is None``
    and crashes inside ``F.pad``. This bug killed slurm job 68205018 at the
    start of epoch 2 after a clean epoch 1.
    """
    item = stepwise_dataset[0]
    assert item.y is not None
    assert hasattr(item, "spectator_mask")

    # First "epoch": collate consumes the item.
    _ = collate_fn_with_spectators([item])
    assert item.y is not None, "collate_fn must not delete d.y from the source Data"
    assert hasattr(item, "spectator_mask"), "collate_fn must not delete d.spectator_mask"

    # Second "epoch": same item must still collate without errors.
    batch2 = collate_fn_with_spectators([item])
    assert batch2 is not None
    assert batch2.y_padded.dim() == 3
