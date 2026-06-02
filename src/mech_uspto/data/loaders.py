"""Collation function and high-level dataloader factory."""

import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data

from mech_uspto.data.dataset import MechUSPTODataset
from mech_uspto.data.parser import MechUSPTOParser


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG that affects training, for run-to-run reproducibility.

    Covers ``random``, ``numpy``, ``torch`` (CPU + all CUDA devices), and the
    ``PYTHONHASHSEED`` env var (affects dict / set ordering in child processes).
    When ``deterministic`` is True, also forces cuDNN into deterministic mode
    (disables autotuner) — costs ~5-10% throughput but gives bitwise-stable
    runs on the same hardware. Required for "did X actually help?" comparisons.

    Note: full bitwise reproducibility across runs *also* requires the
    DataLoader to use ``worker_init_fn=_seed_worker`` and an explicit
    ``generator`` — see ``create_dataloaders``.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` that re-seeds RNGs inside each worker process.

    Without this, multi-worker shuffling order depends on worker startup
    timing → non-reproducible epoch ordering with ``num_workers > 0``.
    PyTorch passes a base seed to each worker via ``initial_seed()``; we
    derive ``numpy`` / ``random`` seeds from it so all three RNGs agree.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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

    # Best-effort: seed everything before any RNG-touching work. Caller may
    # have already called ``seed_everything`` directly; this is idempotent.
    seed_everything(seed)

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

    print(f"   Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)

    print(f"📊 Creating dataloaders (batch_size={batch_size})...")
    # Dedicated CPU generator → reproducible shuffle order across runs even
    # with num_workers > 0 (pair with worker_init_fn=_seed_worker).
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    return {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn_with_spectators,
            worker_init_fn=_seed_worker,
            generator=train_generator,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn_with_spectators,
            worker_init_fn=_seed_worker,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn_with_spectators,
            worker_init_fn=_seed_worker,
        ),
    }

__all__ = [
    "collate_fn_with_spectators",
    "create_dataloaders",
    "seed_everything",
]
