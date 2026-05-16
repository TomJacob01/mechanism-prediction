"""Tests for scripts/plot_history.py."""

import json
import sys
from pathlib import Path

import pytest

# Make scripts/ importable.
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import plot_history  # noqa: E402


def _make_history(n_epochs: int = 3) -> dict:
    """Synthetic results dict matching TrainingEngine.save_results() shape."""

    def metric(epoch: int) -> dict:
        return {
            "precision": 0.1 * epoch,
            "recall": 0.05 * epoch,
            "f1": 0.07 * epoch,
            "pr_auc": 0.2 + 0.1 * epoch,
            "topk_acc": 0.3 + 0.1 * epoch,
            "tp": 10 * epoch,
            "fp": 100,
            "fn": 50,
            "n_rxn_preds": 110 * epoch,
            "n_rxn_targets": 60 * epoch,
        }

    return {
        "config": {"task_mode": "end_to_end"},
        "best_epoch": 1,
        "best_val_loss": 0.4,
        "history": {
            "train_loss": [0.9 - 0.2 * i for i in range(n_epochs)],
            "val_loss": [0.85 - 0.2 * i for i in range(n_epochs)],
            "train_metrics": [metric(i + 1) for i in range(n_epochs)],
            "val_metrics": [metric(i + 1) for i in range(n_epochs)],
        },
    }


def test_load_history_round_trip(tmp_path: Path) -> None:
    data = _make_history()
    p = tmp_path / "results.json"
    p.write_text(json.dumps(data))
    loaded = plot_history.load_history(p)
    assert loaded["history"]["train_loss"] == data["history"]["train_loss"]


def test_load_history_rejects_missing_history_key(tmp_path: Path) -> None:
    p = tmp_path / "no_history.json"
    p.write_text(json.dumps({"config": {}}))
    with pytest.raises(KeyError, match="no 'history' key"):
        plot_history.load_history(p)


def test_load_history_rejects_empty(tmp_path: Path) -> None:
    data = _make_history()
    data["history"]["train_loss"] = []
    p = tmp_path / "empty.json"
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="no epochs ran"):
        plot_history.load_history(p)


def test_plot_history_creates_png(tmp_path: Path) -> None:
    data = _make_history()
    out = tmp_path / "history.png"
    plot_history.plot_history(data, out)
    assert out.exists(), "plot file was not created"
    assert out.stat().st_size > 1024, "plot file is suspiciously small"


def test_series_handles_missing_keys() -> None:
    metrics = [{"f1": 0.5}, {}, {"f1": 0.7, "extra": 1}]
    assert plot_history._series(metrics, "f1") == [0.5, 0.0, 0.7]
    assert plot_history._series(metrics, "n_rxn_preds") == [0.0, 0.0, 0.0]
