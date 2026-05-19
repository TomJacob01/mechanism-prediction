"""Tests for ``TrainingEngine`` behaviors that don't require a full training run."""

import math

import torch

from mech_uspto.training.config import Config
from mech_uspto.training.engine import TrainingEngine


def _tiny_engine(warmup_steps: int = 0, lr: float = 1e-3) -> TrainingEngine:
    """Build a real engine on CPU with the smallest viable model.

    We don't need a dataset — only the optimizer + warmup helper are exercised.
    """
    cfg = Config(
        task_mode="end_to_end",
        device="cpu",
        learning_rate=lr,
        warmup_steps=warmup_steps,
        hidden_dim=8,
        num_heads=2,
        num_layers=1,
        output_dir="/tmp/_mech_test_warmup",
    )
    return TrainingEngine(cfg)


def test_warmup_disabled_keeps_base_lr():
    """``warmup_steps=0`` must not touch ``optimizer.param_groups[*]['lr']``."""
    engine = _tiny_engine(warmup_steps=0, lr=5e-4)
    before = [g["lr"] for g in engine.optimizer.param_groups]
    for _ in range(20):
        engine._apply_warmup_lr()
        engine.global_step += 1
    after = [g["lr"] for g in engine.optimizer.param_groups]
    assert before == after, "warmup disabled (steps=0) must be a no-op"


def test_warmup_ramps_linearly_then_holds():
    """Step k (1-indexed) gets lr = base * k/warmup; after warmup, lr is held at base."""
    base_lr = 1e-3
    warmup = 10
    engine = _tiny_engine(warmup_steps=warmup, lr=base_lr)

    observed: list[float] = []
    # Simulate 15 train steps (5 past the warmup boundary).
    for _ in range(15):
        engine._apply_warmup_lr()
        observed.append(engine.optimizer.param_groups[0]["lr"])
        engine.global_step += 1

    # Step 1 → 1/10 of base; step 10 → full base; step 11+ → held at base.
    assert math.isclose(observed[0], base_lr * 1 / warmup, rel_tol=1e-6), (
        f"step 1 should be base/warmup = {base_lr / warmup:.3e}, got {observed[0]:.3e}"
    )
    assert math.isclose(observed[9], base_lr, rel_tol=1e-6), (
        f"step 10 (warmup boundary) should reach base lr, got {observed[9]:.3e}"
    )
    # Once warmup is finished, the helper must not touch lr (plateau scheduler owns it).
    # Manually nudge lr to a sentinel and verify _apply_warmup_lr leaves it alone.
    sentinel = 4.2e-4
    for g in engine.optimizer.param_groups:
        g["lr"] = sentinel
    engine._apply_warmup_lr()
    assert engine.optimizer.param_groups[0]["lr"] == sentinel, (
        "post-warmup _apply_warmup_lr must not overwrite the plateau scheduler's lr"
    )


def test_warmup_state_round_trips_through_checkpoint(tmp_path):
    """``global_step`` survives a save → load cycle so warmup doesn't restart on resume."""
    engine = _tiny_engine(warmup_steps=50, lr=1e-3)
    engine.global_step = 17  # simulate mid-warmup interrupt

    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    engine.config.checkpoint_dir = str(ckpt_dir)
    engine._save_checkpoint(epoch=2, tag="latest")

    # Fresh engine loads the checkpoint and should resume at step 17, not 0.
    fresh = _tiny_engine(warmup_steps=50, lr=1e-3)
    fresh.config.checkpoint_dir = str(ckpt_dir)
    ckpt_path = next(ckpt_dir.glob("*.pt"))
    fresh.load_state(ckpt_path)

    assert fresh.global_step == 17, (
        f"resumed global_step should be 17, got {fresh.global_step} — "
        "warmup will restart from scratch on every resume"
    )


def test_warmup_legacy_checkpoint_defaults_to_zero(tmp_path):
    """Checkpoints saved before warmup existed must still load (no KeyError)."""
    engine = _tiny_engine(warmup_steps=10, lr=1e-3)
    ckpt_path = tmp_path / "legacy.pt"
    # Build a checkpoint dict WITHOUT a "global_step" key (legacy shape).
    torch.save(
        {
            "epoch": 3,
            "model_state_dict": engine.model.state_dict(),
            "optimizer_state_dict": engine.optimizer.state_dict(),
            "config": engine.config,
            "history": engine.history,
            "best_val_loss": 0.5,
        },
        ckpt_path,
    )
    fresh = _tiny_engine(warmup_steps=10, lr=1e-3)
    fresh.load_state(ckpt_path)
    assert fresh.global_step == 0, "legacy checkpoint must default global_step to 0"
