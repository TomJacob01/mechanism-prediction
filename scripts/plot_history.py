"""Plot training history (loss + metrics) from a results JSON.

Usage::

    python scripts/plot_history.py results/end_to_end_results.json
    python scripts/plot_history.py results/end_to_end_results.json --output history.png
    python scripts/plot_history.py results/end_to_end_results.json --show

The input JSON is produced by ``TrainingEngine.save_results()`` and contains
``history`` with per-epoch ``train_loss``, ``val_loss``, ``train_metrics``,
and ``val_metrics`` (each metric dict has ``precision``, ``recall``, ``f1``,
``pr_auc``, ``topk_acc``, ``tp``, ``fp``, ``fn``, ``n_rxn_preds``,
``n_rxn_targets``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # no GUI; safe on cluster login nodes
import matplotlib.pyplot as plt


def load_history(results_path: Path) -> dict[str, Any]:
    """Load and validate the history block from a results JSON."""
    with results_path.open() as f:
        data = json.load(f)
    if "history" not in data:
        raise KeyError(
            f"{results_path} has no 'history' key. Expected the output of "
            "TrainingEngine.save_results()."
        )
    h = data["history"]
    for key in ("train_loss", "val_loss", "train_metrics", "val_metrics"):
        if key not in h:
            raise KeyError(f"history missing required key {key!r} in {results_path}")
    if not h["train_loss"]:
        raise ValueError(f"history is empty in {results_path} — no epochs ran?")
    return data


def _series(metrics_list: list[dict[str, Any]], key: str) -> list[float]:
    """Pull a single metric across epochs, returning 0.0 for missing keys."""
    return [float(m.get(key, 0.0)) for m in metrics_list]


def plot_history(results: dict[str, Any], output_path: Path) -> None:
    """Render a 4-panel figure: loss, F1/P/R, PR-AUC, rxn-prediction counts."""
    history = results["history"]
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    task_mode = results.get("config", {}).get("task_mode", "?")
    fig.suptitle(f"Training history — {task_mode}", fontsize=14)

    # --- Panel 1: loss curves --------------------------------------------
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], label="train", marker="o", linewidth=1.5)
    ax.plot(epochs, history["val_loss"], label="val", marker="s", linewidth=1.5)
    best_epoch = results.get("best_epoch")
    if isinstance(best_epoch, int):
        ax.axvline(best_epoch + 1, color="green", linestyle="--", alpha=0.5,
                   label=f"best (epoch {best_epoch + 1})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # --- Panel 2: F1 / Precision / Recall on val -------------------------
    ax = axes[0, 1]
    val_metrics = history["val_metrics"]
    ax.plot(epochs, _series(val_metrics, "f1"), label="F1", marker="o", linewidth=1.5)
    ax.plot(epochs, _series(val_metrics, "precision"), label="precision",
            marker="s", linewidth=1.5)
    ax.plot(epochs, _series(val_metrics, "recall"), label="recall",
            marker="^", linewidth=1.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("metric")
    ax.set_title("Validation F1 / Precision / Recall (rare classes only)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # --- Panel 3: PR-AUC + top-k acc -------------------------------------
    ax = axes[1, 0]
    ax.plot(epochs, _series(val_metrics, "pr_auc"), label="PR-AUC",
            marker="o", linewidth=1.5)
    ax.plot(epochs, _series(val_metrics, "topk_acc"), label="top-k acc",
            marker="s", linewidth=1.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("score")
    ax.set_title("Validation PR-AUC & top-k accuracy")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # --- Panel 4: rxn prediction counts (class-collapse diagnostic) ------
    ax = axes[1, 1]
    n_preds = _series(val_metrics, "n_rxn_preds")
    n_targets = _series(val_metrics, "n_rxn_targets")
    tp_vals = _series(val_metrics, "tp")
    ax.plot(epochs, n_targets, label="targets (constant)",
            linestyle=":", color="gray")
    ax.plot(epochs, n_preds, label="rxn predictions", marker="o", linewidth=1.5)
    ax.plot(epochs, tp_vals, label="true positives", marker="^", linewidth=1.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("count")
    ax.set_title("Reaction-pair predictions (class-collapse diagnostic)")
    ax.set_yscale("symlog")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=120)
    print(f"Saved plot: {output_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("results", type=Path,
                   help="Path to <task_mode>_results.json")
    p.add_argument("--output", type=Path, default=None,
                   help="Output PNG path (default: same dir as input, "
                        "<basename>_history.png)")
    p.add_argument("--show", action="store_true",
                   help="Open the plot interactively (requires a GUI backend; "
                        "ignored if matplotlib can't open a display).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = load_history(args.results)
    output = args.output or args.results.with_name(args.results.stem + "_history.png")
    plot_history(data, output)
    if args.show:
        try:
            matplotlib.use("TkAgg", force=True)
            plt.show()
        except Exception as e:  # pragma: no cover
            print(f"(--show failed: {e})")


if __name__ == "__main__":
    main()
