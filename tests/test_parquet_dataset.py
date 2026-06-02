"""Smoke tests for ParquetMechDataset.

These tests require the canonical parquet cache to exist at
``data/cache/parquet``. They are skipped if it's missing so CI doesn't
fail on a fresh checkout (the cache is gitignored).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

CACHE_DIR = Path("data/cache/parquet")
HAS_CACHE = (CACHE_DIR / "reactions.parquet").exists() and (
    CACHE_DIR / "steps.parquet"
).exists()

pytestmark = pytest.mark.skipif(
    not HAS_CACHE, reason="parquet cache not present; run scripts/build_parquet_cache.py"
)


def test_stepwise_dataset_loads():
    from mech_uspto.data.parquet_dataset import ParquetMechDataset

    ds = ParquetMechDataset(CACHE_DIR, task_mode="stepwise", split="all", limit=64)
    assert len(ds) == 64

    item = ds[0]
    assert item.x.dim() == 2
    assert item.edge_index.shape[0] == 2
    assert item.y.dim() == 2
    assert item.y.shape[0] == item.y.shape[1] == item.x.shape[0]
    assert item.task_mode == "stepwise"
    assert isinstance(item.reaction_id, str)
    assert isinstance(item.step_id, int)
    # Δ values are integer-valued (stepwise: in {-1, 0, +1} typically)
    assert item.y.dtype == torch.long


def test_end_to_end_dataset_loads():
    from mech_uspto.data.parquet_dataset import ParquetMechDataset

    ds = ParquetMechDataset(CACHE_DIR, task_mode="end_to_end", split="all", limit=16)
    assert len(ds) == 16
    item = ds[0]
    assert item.task_mode == "end_to_end"
    assert item.step_id == -1
    # End-to-end Δ values clamped to {-3..3}.
    assert int(item.y.min()) >= -3 and int(item.y.max()) <= 3


def test_split_partitions_are_disjoint():
    from mech_uspto.data.parquet_dataset import ParquetMechDataset

    train = ParquetMechDataset(CACHE_DIR, task_mode="end_to_end", split="train")
    val = ParquetMechDataset(CACHE_DIR, task_mode="end_to_end", split="val")
    test = ParquetMechDataset(CACHE_DIR, task_mode="end_to_end", split="test")

    train_ids = {r["rxn_id"] for r in train._rows}
    val_ids = {r["rxn_id"] for r in val._rows}
    test_ids = {r["rxn_id"] for r in test._rows}

    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids & test_ids)
    total = len(train_ids) + len(val_ids) + len(test_ids)
    assert total == len(train) + len(val) + len(test)


def test_stepwise_steps_inherit_reaction_split():
    """A reaction's steps must all land in the same split as the reaction."""
    from mech_uspto.data.parquet_dataset import ParquetMechDataset, split_of

    ds = ParquetMechDataset(CACHE_DIR, task_mode="stepwise", split="train", limit=2000)
    for row in ds._rows[:200]:
        h = ds._rxn_split[row["rxn_id"]]
        assert split_of(h) == "train"
