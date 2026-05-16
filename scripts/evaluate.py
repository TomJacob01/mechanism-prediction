"""Evaluate a trained checkpoint on the held-out test split.

Usage::

    python scripts/evaluate.py --checkpoint checkpoints/end_to_end_best_ep19.pt
    python scripts/evaluate.py --checkpoint checkpoints/end_to_end_best_ep19.pt \\
        --split val --output results/end_to_end_val_eval.json
    python scripts/evaluate.py --checkpoint checkpoints/end_to_end_best_ep19.pt \\
        --csv path/to/mech-USPTO-31k.csv \\
        --output results/eval.json --confusion-csv results/confusion.csv

Saves:
- A JSON with aggregate metrics (precision, recall, f1, pr_auc, topk_acc,
  per-class counts, n_rxn_preds, n_rxn_targets, confusion matrix).
- An optional CSV with the raw confusion matrix (rows=true, cols=pred,
  classes indexed 0..num_classes-1 corresponding to shifted Δ values).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from mech_uspto.data.loaders import create_dataloaders
from mech_uspto.models.transformer import ReactionTransformer
from mech_uspto.training.config import DEFAULT_DATA_PATH, Config
from mech_uspto.training.metrics import MetricsComputer


# ---------------------------------------------------------------- checkpoint IO


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """Load a training checkpoint; tolerate both new-style (with config) and old-style."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" not in ckpt:
        raise KeyError(f"{path} is not a valid training checkpoint (no model_state_dict)")
    if "config" not in ckpt:
        raise KeyError(
            f"{path} has no 'config' -- cannot reconstruct model architecture. "
            "Retrain with the current engine.py to embed the config."
        )
    return ckpt


def build_model_from_config(config: Config, device: torch.device) -> ReactionTransformer:
    """Mirror the architecture wiring done by TrainingEngine.__init__."""
    model = ReactionTransformer(
        node_in=config.node_in,
        edge_in=config.edge_in,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
        num_classes=config.num_classes,
    ).to(device)
    return model


# ------------------------------------------------------------------- evaluation


@torch.no_grad()
def evaluate_split(
    model: ReactionTransformer,
    loader,
    device: torch.device,
    task_mode: str,
    num_classes: int,
) -> dict[str, Any]:
    """Run the model on ``loader`` and return aggregate metrics + confusion matrix."""
    model.eval()

    # Use the same target shift logic as TrainingEngine._shift_targets.
    shift = 1 if task_mode == "stepwise" else 3

    # Confusion matrix over the upper-triangle pairs (true rows, pred cols).
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    tp = fp = fn = n_rxn_preds = n_rxn_targets = 0
    total_topk_hits = 0
    total_loss_weight = 0  # number of batches successfully processed
    pr_auc_running: list[float] = []  # per-batch PR-AUC, averaged at the end

    for batch in tqdm(loader, desc="Eval"):
        batch = batch.to(device)
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        batch_idx = batch.batch

        logits, mask = model(x, edge_index, edge_attr, batch_idx)
        targets = batch.y_padded + shift

        # Per-batch metrics (so we accumulate raw counts faithfully).
        bm = MetricsComputer.get_mechanism_metrics(logits, targets, mask)
        tp += int(bm["tp"])
        fp += int(bm["fp"])
        fn += int(bm["fn"])
        n_rxn_preds += int(bm["n_rxn_preds"])
        n_rxn_targets += int(bm["n_rxn_targets"])
        total_topk_hits += int(bm["topk_acc"] * bm["n_rxn_targets"])
        pr_auc_running.append(float(bm["pr_auc"]))
        total_loss_weight += 1

        # Build the confusion matrix over masked upper-triangle pairs.
        preds = logits.argmax(dim=-1)  # (B, N, N)
        B, N, _ = preds.shape
        mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)
        triu = torch.triu_indices(N, N, offset=1, device=preds.device)
        for b in range(B):
            m = mask_2d[b][triu[0], triu[1]]
            t = targets[b][triu[0], triu[1]][m].cpu().numpy()
            p = preds[b][triu[0], triu[1]][m].cpu().numpy()
            # np.add.at on a 2D index pair updates the confusion matrix in-place.
            np.add.at(confusion, (t, p), 1)

    # Final aggregated metrics from raw counts.
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    pr_auc = float(np.mean(pr_auc_running)) if pr_auc_running else 0.0
    topk_acc = total_topk_hits / n_rxn_targets if n_rxn_targets > 0 else 0.0

    # Per-class precision / recall / F1 derived from the confusion matrix.
    per_class = {}
    for c in range(num_classes):
        class_tp = int(confusion[c, c])
        class_fp = int(confusion[:, c].sum() - class_tp)
        class_fn = int(confusion[c, :].sum() - class_tp)
        cp = class_tp / (class_tp + class_fp) if (class_tp + class_fp) > 0 else 0.0
        cr = class_tp / (class_tp + class_fn) if (class_tp + class_fn) > 0 else 0.0
        cf = 2 * cp * cr / (cp + cr) if (cp + cr) > 0 else 0.0
        per_class[f"class_{c}"] = {
            "tp": class_tp,
            "fp": class_fp,
            "fn": class_fn,
            "support": int(confusion[c, :].sum()),
            "precision": cp,
            "recall": cr,
            "f1": cf,
        }

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "topk_acc": topk_acc,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_rxn_preds": n_rxn_preds,
        "n_rxn_targets": n_rxn_targets,
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }


def write_confusion_csv(confusion: list[list[int]], path: Path, shift: int) -> None:
    """Write the confusion matrix as CSV with Δ-labeled rows/cols."""
    n = len(confusion)
    deltas = [str(c - shift) for c in range(n)]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + [f"delta={d}" for d in deltas])
        for r, row in enumerate(confusion):
            w.writerow([f"delta={deltas[r]}"] + list(row))
    print(f"Saved confusion matrix CSV: {path}")


# -------------------------------------------------------------------- CLI glue


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to a training checkpoint .pt file.")
    p.add_argument("--csv", type=str, default=None,
                   help="Override the CSV path stored in the checkpoint's config. "
                        "Useful if data moved between training and eval.")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override checkpoint's batch_size (eval doesn't backprop, "
                        "you can usually use a larger batch).")
    p.add_argument("--output", type=Path, default=None,
                   help="Path to write the eval JSON (default: alongside checkpoint, "
                        "<checkpoint_stem>_<split>_eval.json).")
    p.add_argument("--confusion-csv", type=Path, default=None,
                   help="Optional CSV path for the confusion matrix.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading checkpoint {args.checkpoint}...")
    ckpt = load_checkpoint(args.checkpoint, device)
    config: Config = ckpt["config"]
    print(f"   task_mode={config.task_mode}, num_classes={config.num_classes}, "
          f"hidden_dim={config.hidden_dim}, epoch={ckpt.get('epoch', '?')}")

    if args.csv:
        config.csv_path = args.csv
    if args.batch_size:
        config.batch_size = args.batch_size

    # Reproduce the same train/val/test split that training used.
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    dataloaders = create_dataloaders(
        config.csv_path,
        task_mode=config.task_mode,
        batch_size=config.batch_size,
        train_val_test_split=config.train_val_test_split,
        seed=config.seed,
        compute_spectators=True,
    )

    model = build_model_from_config(config, device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"   Loaded weights ({sum(p.numel() for p in model.parameters()):,} params)")

    metrics = evaluate_split(
        model,
        dataloaders[args.split],
        device,
        task_mode=config.task_mode,
        num_classes=config.num_classes,
    )

    # --- pretty-print summary -----------------------------------------------
    print("\n" + "=" * 78)
    print(f"Eval on '{args.split}' split  ({metrics['n_rxn_targets']:,} reaction-pair targets)")
    print("=" * 78)
    print(f"  F1           : {metrics['f1']:.4f}")
    print(f"  Precision    : {metrics['precision']:.4f}")
    print(f"  Recall       : {metrics['recall']:.4f}")
    print(f"  PR-AUC (avg) : {metrics['pr_auc']:.4f}")
    print(f"  Top-k acc    : {metrics['topk_acc']:.4f}")
    print(f"  TP / FP / FN : {metrics['tp']:,} / {metrics['fp']:,} / {metrics['fn']:,}")
    print(f"  RxnPreds     : {metrics['n_rxn_preds']:,} / {metrics['n_rxn_targets']:,}")
    print("\nPer-class:")
    shift = 1 if config.task_mode == "stepwise" else 3
    for c in range(config.num_classes):
        pc = metrics["per_class"][f"class_{c}"]
        print(f"  delta={c - shift:+d}: support={pc['support']:>9,}  "
              f"P={pc['precision']:.3f}  R={pc['recall']:.3f}  F1={pc['f1']:.3f}")

    # --- save ----------------------------------------------------------------
    output = args.output or args.checkpoint.with_name(
        f"{args.checkpoint.stem}_{args.split}_eval.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "task_mode": config.task_mode,
        "num_classes": config.num_classes,
        "metrics": metrics,
    }
    with output.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved eval JSON: {output}")

    if args.confusion_csv:
        args.confusion_csv.parent.mkdir(parents=True, exist_ok=True)
        write_confusion_csv(metrics["confusion_matrix"], args.confusion_csv, shift)


if __name__ == "__main__":
    main()
