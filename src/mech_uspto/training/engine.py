"""Training / evaluation engine for the ablation study."""

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm

from mech_uspto.losses.focal import MaskedFocalLossWithSpectators
from mech_uspto.models.transformer import ReactionTransformer
from mech_uspto.training.config import Config
from mech_uspto.training.metrics import MetricsComputer


class TrainingEngine:
    """Train + validate + checkpoint a ``ReactionTransformer``."""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)

        self.model = ReactionTransformer(
            node_in=config.node_in,
            edge_in=config.edge_in,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
            num_classes=config.num_classes,
        ).to(self.device)

        self.criterion = MaskedFocalLossWithSpectators(
            num_classes=config.num_classes,
            weights=config.class_weights.to(self.device),
            gamma=config.gamma_focal,
            spectator_weight=config.spectator_weight,
            reduction="mean",
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
        )

        self.history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": [],
        }
        self.best_val_loss = float("inf")
        self.best_epoch = 0

    # ------------------------------------------------------------------ helpers

    def _shift_targets(self, y_padded: torch.Tensor) -> torch.Tensor:
        """Shift task-specific Δ range to non-negative class indices."""
        if self.config.task_mode == "stepwise":
            return y_padded + 1  # {-1, 0, 1} → {0, 1, 2}
        return y_padded + 2  # {-2..2} → {0..4}

    def _forward_and_loss(
        self, batch: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward + loss. Returns ``(loss, logits, targets, mask)``."""
        targets = self._shift_targets(batch.y_padded)

        logits, mask = self.model(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )
        mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)

        spectator_mask = getattr(batch, "spectator_padded", None)
        loss = self.criterion(logits, targets, mask_2d, spectator_mask)
        return loss, logits, targets, mask

    def _train_step(self, batch: Any) -> Tuple[float, Dict[str, float]]:
        """One optimization step."""
        loss, logits, targets, mask = self._forward_and_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )
        self.optimizer.step()

        metrics = MetricsComputer.get_mechanism_metrics(logits, targets, mask)
        return loss.item(), metrics

    def _val_step(self, batch: Any) -> Tuple[float, Dict[str, float]]:
        """One forward-only step."""
        with torch.no_grad():
            loss, logits, targets, mask = self._forward_and_loss(batch)
        metrics = MetricsComputer.get_mechanism_metrics(logits, targets, mask)
        return loss.item(), metrics

    # -------------------------------------------------------------------- core

    def run_epoch(self, loader, is_train: bool = True) -> Tuple[float, Dict]:
        """Run one full pass over ``loader``."""
        self.model.train() if is_train else self.model.eval()

        total_loss = 0.0
        all_metrics: Dict[str, float] = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "topk_acc": 0.0,
            "pr_auc": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }
        num_batches = 0

        pbar = tqdm(loader, desc="Train" if is_train else "Val", disable=not is_train)
        for batch in pbar:
            batch = batch.to(self.device)

            if is_train:
                loss_val, metrics = self._train_step(batch)
            else:
                loss_val, metrics = self._val_step(batch)

            total_loss += loss_val
            for key in all_metrics:
                if isinstance(all_metrics[key], (int, float)) and isinstance(
                    metrics[key], (int, float)
                ):
                    all_metrics[key] += metrics[key]

            num_batches += 1
            if is_train:
                pbar.set_postfix({"loss": f"{loss_val:.4f}"})

        avg_loss = total_loss / max(num_batches, 1)
        for key in all_metrics:
            if key not in ("tp", "fp", "fn"):
                all_metrics[key] /= max(num_batches, 1)

        # Recompute precision/recall/F1 from accumulated tp/fp/fn for accuracy.
        tp, fp, fn = all_metrics["tp"], all_metrics["fp"], all_metrics["fn"]
        if tp + fp > 0:
            all_metrics["precision"] = tp / (tp + fp)
        if tp + fn > 0:
            all_metrics["recall"] = tp / (tp + fn)
        p, r = all_metrics["precision"], all_metrics["recall"]
        if p + r > 0:
            all_metrics["f1"] = 2 * p * r / (p + r)

        return avg_loss, all_metrics

    def train(self, train_loader, val_loader) -> None:
        """Full training loop with checkpointing + early stopping."""
        print(f"\n{'=' * 80}")
        print(f"Starting training: {self.config.task_mode} mode")
        print(f"Classes: {self.config.num_classes}, Hidden: {self.config.hidden_dim}")
        print(f"{'=' * 80}\n")

        for epoch in range(self.config.num_epochs):
            train_loss, train_metrics = self.run_epoch(train_loader, is_train=True)
            self.history["train_loss"].append(train_loss)
            self.history["train_metrics"].append(train_metrics)

            val_loss, val_metrics = self.run_epoch(val_loader, is_train=False)
            self.history["val_loss"].append(val_loss)
            self.history["val_metrics"].append(val_metrics)

            self.scheduler.step(val_loss)

            print(
                f"\nEpoch {epoch + 1:3d}/{self.config.num_epochs} | "
                f"TrL: {train_loss:.4f} | "
                f"VL: {val_loss:.4f} | "
                f"F1: {val_metrics['f1']:.3f} | "
                f"Rec: {val_metrics['recall']:.3f} | "
                f"Prec: {val_metrics['precision']:.3f} | "
                f"PR-AUC: {val_metrics['pr_auc']:.3f}"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self._save_checkpoint(epoch, "best")
                print("   ✅ New best model saved!")

            if epoch - self.best_epoch >= self.config.patience:
                print(f"\n⏹️  Early stopping at epoch {epoch + 1}")
                break

            if epoch % 10 == 0:
                self._save_checkpoint(epoch, "latest")

    def _save_checkpoint(self, epoch: int, tag: str = "best") -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "history": self.history,
            "best_val_loss": self.best_val_loss,
        }
        path = (
            Path(self.config.checkpoint_dir)
            / f"{self.config.task_mode}_{tag}_ep{epoch}.pt"
        )
        torch.save(checkpoint, path)
        print(f"   Saved checkpoint: {path}")

    def save_results(self) -> None:
        """Persist training history + config to ``output_dir``."""
        results = {
            "config": vars(self.config),
            "history": self.history,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
        }
        results_path = (
            Path(self.config.output_dir) / f"{self.config.task_mode}_results.json"
        )
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Saved results: {results_path}")
