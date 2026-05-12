"""Collation function and high-level dataloader factory."""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data

from mech_uspto.data.dataset import MechUSPTODataset
from mech_uspto.data.parser import MechUSPTOParser


def collate_fn_with_spectators(data_list: List[Optional[Data]]) -> Optional[Batch]:
    """Collate a list of ``Data`` items, padding Δ matrices and spectator masks.

    Returns ``None`` if all inputs are ``None`` (drop-empty-batch behaviour).
    """
    data_list = [item for item in data_list if item is not None]
    if not data_list:
        return None

    max_nodes = max(d.num_nodes for d in data_list)
    has_spectators = hasattr(data_list[0], "spectator_mask")

    y_padded: List[torch.Tensor] = []
    spectator_padded: List[torch.Tensor] = []

    for d in data_list:
        pad_size = max_nodes - d.num_nodes

        # Pad delta matrix to (max_nodes, max_nodes).
        y_padded.append(F.pad(d.y, (0, pad_size, 0, pad_size), value=0))

        if has_spectators:
            padded_spectator = F.pad(
                d.spectator_mask.unsqueeze(0), (0, pad_size), value=True
            ).squeeze(0)
            spectator_padded.append(padded_spectator)

        # Drop raw fields so PyG's default collation doesn't choke on shape mismatch.
        del d.y
        if has_spectators:
            del d.spectator_mask

    batch = Batch.from_data_list(data_list)
    batch.y_padded = torch.stack(y_padded)
    if has_spectators:
        batch.spectator_padded = torch.stack(spectator_padded)

    return batch


def create_dataloaders(
    json_dir: str,
    task_mode: str = "stepwise",
    batch_size: int = 16,
    train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    num_workers: int = 0,
    seed: int = 42,
    compute_spectators: bool = True,
) -> Dict[str, DataLoader]:
    """Parse ``json_dir``, build datasets, return train/val/test DataLoaders."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"📂 Parsing mech-USPTO-31k from {json_dir}...")
    reactions = MechUSPTOParser.parse_batch(json_dir)
    print(f"   Loaded {len(reactions)} reactions")

    random.shuffle(reactions)

    n_train = int(len(reactions) * train_val_test_split[0])
    n_val = int(len(reactions) * train_val_test_split[1])
    train_reactions = reactions[:n_train]
    val_reactions = reactions[n_train : n_train + n_val]
    test_reactions = reactions[n_train + n_val :]

    print(
        f"   Train: {len(train_reactions)}, "
        f"Val: {len(val_reactions)}, "
        f"Test: {len(test_reactions)}"
    )

    print(f"🔨 Building '{task_mode}' mode datasets...")
    train_dataset = MechUSPTODataset(
        train_reactions, task_mode=task_mode, compute_spectators=compute_spectators
    )
    val_dataset = MechUSPTODataset(
        val_reactions, task_mode=task_mode, compute_spectators=compute_spectators
    )
    test_dataset = MechUSPTODataset(
        test_reactions, task_mode=task_mode, compute_spectators=compute_spectators
    )

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
