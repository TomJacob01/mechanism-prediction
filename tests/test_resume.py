"""Tests for resume-from-checkpoint behavior."""

import pytest

from mech_uspto.training.config import Config
from mech_uspto.training.engine import TrainingEngine


def _make_config(tmp_path, **overrides) -> Config:
    kwargs = dict(
        task_mode="end_to_end",
        output_dir=str(tmp_path / "results"),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        hidden_dim=16,
        num_layers=2,
        num_heads=2,
        num_epochs=10,
        device="cpu",
    )
    kwargs.update(overrides)
    return Config(**kwargs)


def test_load_state_restores_start_epoch_and_best(tmp_path):
    cfg = _make_config(tmp_path)
    engine = TrainingEngine(cfg)
    engine.best_val_loss = 0.42
    engine.best_epoch = 5
    engine.history["train_loss"].extend([1.0, 0.8, 0.6])
    engine._save_checkpoint(epoch=5, tag="best")

    ckpt_path = tmp_path / "checkpoints" / "end_to_end_best_ep5.pt"
    assert ckpt_path.exists()

    fresh = TrainingEngine(_make_config(tmp_path))
    assert fresh.start_epoch == 0
    fresh.load_state(ckpt_path)
    assert fresh.start_epoch == 6, "resume should continue from epoch + 1"
    assert fresh.best_epoch == 5
    assert fresh.best_val_loss == pytest.approx(0.42)
    assert fresh.history["train_loss"] == [1.0, 0.8, 0.6]


def test_load_state_rejects_incompatible_architecture(tmp_path):
    cfg = _make_config(tmp_path, hidden_dim=16)
    engine = TrainingEngine(cfg)
    engine._save_checkpoint(epoch=0, tag="best")
    ckpt_path = tmp_path / "checkpoints" / "end_to_end_best_ep0.pt"

    # Reload with a DIFFERENT hidden_dim — should refuse rather than silently
    # load a state-dict shape-mismatched into the model.
    bad_cfg = _make_config(tmp_path / "alt", hidden_dim=32)
    fresh = TrainingEngine(bad_cfg)
    with pytest.raises(ValueError, match="hidden_dim"):
        fresh.load_state(ckpt_path)


def test_load_state_warns_on_benign_change_and_loads(tmp_path, capsys):
    cfg = _make_config(tmp_path, learning_rate=1e-4)
    engine = TrainingEngine(cfg)
    engine._save_checkpoint(epoch=0, tag="best")
    ckpt_path = tmp_path / "checkpoints" / "end_to_end_best_ep0.pt"

    new_cfg = _make_config(tmp_path / "alt", learning_rate=5e-5)
    fresh = TrainingEngine(new_cfg)
    fresh.load_state(ckpt_path)
    out = capsys.readouterr().out
    assert "learning_rate" in out  # benign change reported
    assert fresh.start_epoch == 1
