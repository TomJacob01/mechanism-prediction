"""Regressions for #8 (deterministic seeding) and #12 (bf16 autocast)."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from mech_uspto.data.loaders import _seed_worker, seed_everything
from mech_uspto.training.config import Config
from mech_uspto.training.engine import TrainingEngine

# --------------------------------------------------------------------- seeding


def test_seed_everything_makes_random_numpy_torch_reproducible():
    """Two `seed_everything(42)` calls must yield identical RNG draws."""
    seed_everything(42)
    py = random.random()
    npy = np.random.rand(4)
    th = torch.rand(4)

    seed_everything(42)
    assert random.random() == py
    assert np.array_equal(np.random.rand(4), npy)
    assert torch.equal(torch.rand(4), th)


def test_seed_everything_sets_cudnn_deterministic_when_requested():
    seed_everything(0, deterministic=True)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_seed_everything_skips_cudnn_flags_when_deterministic_false():
    # Force flags to a known non-default state, then verify we don't touch them.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    seed_everything(0, deterministic=False)
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is True


def test_seed_worker_reseeds_numpy_and_random_from_torch_initial_seed():
    """Inside a worker, `_seed_worker` must derive numpy/random seeds from torch."""
    torch.manual_seed(123)
    _seed_worker(worker_id=0)
    expected_seed = torch.initial_seed() % 2**32

    # Compare against a fresh re-seed using the same derived value.
    np_draw = np.random.rand(3)
    py_draw = random.random()

    np.random.seed(expected_seed)
    random.seed(expected_seed)
    assert np.array_equal(np.random.rand(3), np_draw)
    assert random.random() == py_draw


# ------------------------------------------------------------------------ amp


def test_config_use_amp_defaults_true():
    cfg = Config(task_mode="stepwise")
    assert cfg.use_amp is True
    assert cfg.deterministic is True


def test_engine_amp_disabled_on_cpu_even_when_use_amp_true():
    """`use_amp=True` + CPU device must fall back to a no-op autocast context."""
    cfg = Config(task_mode="stepwise", device="cpu", use_amp=True, num_epochs=1)
    engine = TrainingEngine(cfg)
    assert engine._amp_enabled is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for AMP path")
def test_engine_amp_enabled_on_cuda():
    cfg = Config(task_mode="stepwise", device="cuda", use_amp=True, num_epochs=1)
    engine = TrainingEngine(cfg)
    assert engine._amp_enabled is True
    # _autocast() should yield a real autocast context, not nullcontext.
    ctx = engine._autocast()
    assert isinstance(ctx, torch.autocast)


def test_engine_amp_disabled_when_use_amp_false():
    cfg = Config(task_mode="stepwise", device="cpu", use_amp=False, num_epochs=1)
    engine = TrainingEngine(cfg)
    assert engine._amp_enabled is False
