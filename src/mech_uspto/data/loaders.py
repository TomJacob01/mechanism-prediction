"""Collation function and high-level dataloader factory."""

import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data

from mech_uspto.data.dataset import MechUSPTODataset
from mech_uspto.data.parser import MechUSPTOParser


def collate_fn_with_spectators(data_list: list[Optional[Data]]) -> Optional[Batch]:
    """Collate a list of ``Data`` items, padding Δ matrices and spectator masks.

    Returns ``None`` if all inputs are ``None`` (drop-empty-batch behaviour).

    NOTE: PyG ``Data`` objects are shared by reference across epochs (the
    dataloader returns the same Python objects each time). We must NOT mutate
    them (e.g. ``del d.y``) — doing so works on epoch 1 then crashes on epoch 2
    with ``F.pad(None, ...)``. Instead, temporarily pop the variable-shape
    fields, run PyG's batching on the stripped clones, then restore.
    """
    data_list = [item for item in data_list if item is not None]
    if not data_list:
        return None

    max_nodes = max(d.num_nodes for d in data_list)
    has_spectators = hasattr(data_list[0], "spectator_mask")

    y_padded: list[torch.Tensor] = []
    spectator_padded: list[torch.Tensor] = []
    # Build shallow clones with the variable-shape fields removed, so
    # ``Batch.from_data_list`` doesn't try to concatenate mismatched tensors.
    stripped: list[Data] = []

    for d in data_list:
        pad_size = max_nodes - d.num_nodes

        # Pad delta matrix to (max_nodes, max_nodes).
        y_padded.append(F.pad(d.y, (0, pad_size, 0, pad_size), value=0))

        if has_spectators:
            padded_spectator = F.pad(
                d.spectator_mask.unsqueeze(0), (0, pad_size), value=True
            ).squeeze(0)
            spectator_padded.append(padded_spectator)

        # Shallow copy + remove only the fields PyG can't batch (do NOT mutate d).
        d_clone = d.clone()
        del d_clone.y
        if has_spectators:
            del d_clone.spectator_mask
        stripped.append(d_clone)

    batch = Batch.from_data_list(stripped)
    batch.y_padded = torch.stack(y_padded)
    if has_spectators:
        batch.spectator_padded = torch.stack(spectator_padded)

    return batch


def create_dataloaders(
    csv_path: str,
    task_mode: str = "stepwise",
    batch_size: int = 16,
    train_val_test_split: tuple[float, float, float] = (0.7, 0.15, 0.15),
    num_workers: int = 0,
    seed: int = 42,
    compute_spectators: bool = True,
    cache_dir: Optional[str] = "./cache",
    use_cache: bool = True,
) -> dict[str, DataLoader]:
    """Parse ``csv_path``, build one cached dataset, return train/val/test DataLoaders."""
    from torch.utils.data import Subset

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"📂 Parsing mech-USPTO-31k from {csv_path}...")
    reactions = MechUSPTOParser.parse_csv_file(csv_path)
    print(f"   Loaded {len(reactions)} reactions")

    print(f"🔨 Building '{task_mode}' mode dataset (cache_dir={cache_dir})...")
    full_dataset = MechUSPTODataset(
        reactions,
        task_mode=task_mode,
        compute_spectators=compute_spectators,
        cache_dir=cache_dir,
        csv_path=csv_path,
        use_cache=use_cache,
    )

    # Reproducible shuffle of indices into a single featurized dataset.
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    n_train = int(len(indices) * train_val_test_split[0])
    n_val = int(len(indices) * train_val_test_split[1])
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    print(
        f"   Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}"
    )

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)

    print(f"📊 Creating dataloaders (batch_size={batch_size})...")
    return {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn_with_spectators,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn_with_spectators,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn_with_spectators,
        ),
    }
