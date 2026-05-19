"""Training / evaluation engine for the ablation study."""

import contextlib
import json
from pathlib import Path
from typing import Any

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm

from mech_uspto.losses.focal import MaskedFocalLossWithSpectators
from mech_uspto.models.transformer import ReactionTransformer
from mech_uspto.training.config import Config
from mech_uspto.training.metrics import MetricsComputer, pooled_pr_auc
from mech_uspto.training.tracking import Tracker


class TrainingEngine:
    """Train + validate + checkpoint a ``ReactionTransformer``."""

    def __init__(self, config: Config, tracker: Tracker | None = None):
        self.config = config
        self.device = torch.device(config.device)
        self.tracker = tracker if tracker is not None else Tracker()

        self.model = ReactionTransformer(
            node_in=config.node_in,
            edge_in=config.edge_in,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
            num_classes=config.num_classes,
            class_prior=config.class_prior,
            use_edge_features_in_head=config.use_edge_features_in_head,
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

        self.history: dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": [],
        }
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        # Epoch to start training from. Bumped by ``load_state`` after a resume.
        self.start_epoch = 0
        # Global optimizer-step counter used by the linear LR warmup.
        # Restored from checkpoint on resume so warmup state doesn't repeat.
        self.global_step = 0

        # bf16 autocast on CUDA only. bf16 has fp32-equivalent dynamic range
        # so no GradScaler is required (unlike fp16). On CPU / mismatched
        # device, fall back to a no-op so behavior is unchanged.
        self._amp_enabled = bool(config.use_amp) and self.device.type == "cuda"

    def _autocast(self) -> Any:
        """Return the autocast context manager (or a no-op when AMP is off)."""
        if self._amp_enabled:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def load_state(self, checkpoint_path: str | Path) -> None:
        """Restore model + optimizer + history from a checkpoint to enable resume.

        Raises ``ValueError`` on architecture-incompatible config mismatch
        (``task_mode``, ``num_classes``, ``hidden_dim``, ``num_layers``).
        Warns on benign mismatches (e.g. ``learning_rate``, ``num_epochs``).
        """
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        ckpt_cfg = ckpt.get("config")
        if ckpt_cfg is not None:
            for k in ("task_mode", "num_classes", "hidden_dim", "num_layers", "num_heads"):
                old, new = getattr(ckpt_cfg, k, None), getattr(self.config, k)
                if old is not None and old != new:
                    raise ValueError(
                        f"Cannot resume: checkpoint {k}={old!r} but current config {k}={new!r}"
                    )
            for k in ("learning_rate", "weight_decay", "gamma_focal", "num_epochs"):
                old, new = getattr(ckpt_cfg, k, None), getattr(self.config, k)
                if old is not None and old != new:
                    print(f"   ℹ️  Resume: {k} changed {old!r} -> {new!r}")

        self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "history" in ckpt:
            self.history = ckpt["history"]
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        self.best_epoch = ckpt.get("epoch", 0)
        self.start_epoch = ckpt.get("epoch", -1) + 1
        # Restore warmup step counter (0 for legacy checkpoints saved before warmup existed).
        self.global_step = ckpt.get("global_step", 0)
        print(
            f"   ⏮️  Resumed from {checkpoint_path}: "
            f"start_epoch={self.start_epoch}, best_val_loss={self.best_val_loss:.3e}"
        )

    # ------------------------------------------------------------------ helpers

    def _shift_targets(self, y_padded: torch.Tensor) -> torch.Tensor:
        """Shift task-specific Δ range to non-negative class indices."""
        if self.config.task_mode == "stepwise":
            return y_padded + 1  # {-1, 0, 1} → {0, 1, 2}
        return y_padded + 3  # {-3..3} → {0..6}

    def _forward_and_loss(
        self, batch: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward + loss. Returns ``(loss, logits, targets, mask)``.

        Wrapped in ``torch.autocast(..., bfloat16)`` when ``use_amp`` is on and
        running on CUDA. The returned ``logits`` retain their autocast dtype
        (bf16) on the AMP path — downstream metric code casts to fp32 as needed.
        """
        targets = self._shift_targets(batch.y_padded)

        with self._autocast():
            logits, mask = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)

            spectator_mask = getattr(batch, "spectator_padded", None)
            loss = self.criterion(logits, targets, mask_2d, spectator_mask)
        return loss, logits, targets, mask

    def _train_step(self, batch: Any) -> tuple[float, dict[str, float]]:
        """One optimization step."""
        self._apply_warmup_lr()
        loss, logits, targets, mask = self._forward_and_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.global_step += 1

        metrics = MetricsComputer.get_mechanism_metrics(logits, targets, mask)
        return loss.item(), metrics

    def _apply_warmup_lr(self) -> None:
        """Linear LR ramp from 0 → ``learning_rate`` over ``warmup_steps`` steps.

        No-op once warmup is finished — the plateau scheduler then takes over
        on a per-epoch basis. The multiplier is applied to each param group's
        ``initial_lr`` (stamped by the optimizer at construction), so it
        composes correctly with any LR the plateau scheduler later sets.
        """
        warmup = self.config.warmup_steps
        if warmup <= 0 or self.global_step >= warmup:
            return
        scale = (self.global_step + 1) / warmup
        for group in self.optimizer.param_groups:
            base_lr = group.get("initial_lr", self.config.learning_rate)
            group["lr"] = base_lr * scale

    def _val_step(self, batch: Any) -> tuple[float, dict[str, float]]:
        """One forward-only step."""
        with torch.no_grad():
            loss, logits, targets, mask = self._forward_and_loss(batch)
        metrics = MetricsComputer.get_mechanism_metrics(logits, targets, mask)
        return loss.item(), metrics

    # -------------------------------------------------------------------- core

    def run_epoch(self, loader, is_train: bool = True) -> tuple[float, dict]:
        """Run one full pass over ``loader``."""
        self.model.train() if is_train else self.model.eval()

        total_loss = 0.0
        num_classes = self.config.num_classes
        all_metrics: dict[str, float] = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "topk_acc": 0.0,
            "pr_auc": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "n_rxn_preds": 0,
            "n_rxn_targets": 0,
            "pr_auc_per_class": [0.0] * num_classes,
            "n_preds_per_class": [0] * num_classes,
            "n_targets_per_class": [0] * num_classes,
        }
        # Raw buffers for epoch-level pooled PR-AUC (see metrics.pooled_pr_auc).
        # Averaging per-batch APs is mathematically wrong, so we pool the raw
        # scores/targets across the whole epoch and compute AP once at the end.
        pooled_buffers: dict[str, list[torch.Tensor]] = {
            "_pooled_scores": [],
            "_pooled_targets": [],
            "_pooled_probs": [],
            "_pooled_class_targets": [],
        }
        num_batches = 0
        total_batches = len(loader)

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
                elif isinstance(all_metrics[key], list) and isinstance(metrics[key], list):
                    # Element-wise sum (per-class counters and per-class AP).
                    for i in range(len(all_metrics[key])):
                        all_metrics[key][i] += metrics[key][i]
            for key in pooled_buffers:
                if key in metrics and metrics[key].numel() > 0:
                    pooled_buffers[key].append(metrics[key])

            num_batches += 1
            if is_train:
                pbar.set_postfix({"loss": f"{loss_val:.3e}"})
                # Periodic flushed line so `tail -f` on the slurm log shows progress
                # (tqdm only writes `\r`, which tail won't surface). Also fires on
                # batch 1 so the log confirms the model is actually training (not
                # stuck in data loading) — useful on slow cold-cache starts.
                # Reports running averages from accumulated tp/fp/fn — stable even
                # at small batch sizes, unlike per-batch P/R/F1.
                if num_batches == 1 or num_batches % 25 == 0 or num_batches == total_batches:
                    tp, fp, fn = all_metrics["tp"], all_metrics["fp"], all_metrics["fn"]
                    p_run = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    r_run = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1_run = 2 * p_run * r_run / (p_run + r_run) if (p_run + r_run) > 0 else 0.0
                    print(
                        f"  batch {num_batches}/{total_batches} "
                        f"loss={loss_val:.3e} "
                        f"P={p_run:.3e} R={r_run:.3e} F1={f1_run:.3e}",
                        flush=True,
                    )

        avg_loss = total_loss / max(num_batches, 1)
        for key in all_metrics:
            if key not in (
                "tp",
                "fp",
                "fn",
                "n_rxn_preds",
                "n_rxn_targets",
                "n_preds_per_class",
                "n_targets_per_class",
                # PR-AUC keys are overwritten below with the pooled epoch-level
                # value (averaging per-batch APs is mathematically invalid).
                "pr_auc",
                "pr_auc_per_class",
            ):
                if isinstance(all_metrics[key], list):
                    all_metrics[key] = [v / max(num_batches, 1) for v in all_metrics[key]]
                else:
                    all_metrics[key] /= max(num_batches, 1)

        # Epoch-level pooled PR-AUC (correct AP over the concatenated rankings).
        pooled_overall, pooled_per_class = pooled_pr_auc(
            scores_list=pooled_buffers["_pooled_scores"],
            targets_list=pooled_buffers["_pooled_targets"],
            probs_list=pooled_buffers["_pooled_probs"],
            class_targets_list=pooled_buffers["_pooled_class_targets"],
            num_classes=num_classes,
        )
        all_metrics["pr_auc"] = pooled_overall
        all_metrics["pr_auc_per_class"] = pooled_per_class

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
        print(
            f"AMP: {'bf16 autocast (CUDA)' if self._amp_enabled else 'disabled'}"
            f" | deterministic: {self.config.deterministic}"
        )
        if self.config.warmup_steps > 0:
            print(
                f"LR warmup: linear over {self.config.warmup_steps} steps "
                f"(starting from global_step={self.global_step})"
            )
        print(f"{'=' * 80}\n")

        self.tracker.init(vars(self.config), run_name=Path(self.config.output_dir).name)

        for epoch in range(self.start_epoch, self.config.num_epochs):
            train_loss, train_metrics = self.run_epoch(train_loader, is_train=True)
            self.history["train_loss"].append(train_loss)
            self.history["train_metrics"].append(train_metrics)

            val_loss, val_metrics = self.run_epoch(val_loader, is_train=False)
            self.history["val_loss"].append(val_loss)
            self.history["val_metrics"].append(val_metrics)

            self.scheduler.step(val_loss)

            print(
                f"\nEpoch {epoch + 1:3d}/{self.config.num_epochs} | "
                f"TrL: {train_loss:.3e} | "
                f"VL: {val_loss:.3e} | "
                f"F1: {val_metrics['f1']:.3e} | "
                f"Rec: {val_metrics['recall']:.3e} | "
                f"Prec: {val_metrics['precision']:.3e} | "
                f"PR-AUC: {val_metrics['pr_auc']:.3e} | "
                f"RxnPreds: {int(val_metrics['n_rxn_preds'])}/{int(val_metrics['n_rxn_targets'])} "
                f"(TP {int(val_metrics['tp'])} FP {int(val_metrics['fp'])} FN {int(val_metrics['fn'])})"
            )

            # Per-class diagnostic: shows which classes the model is even
            # *trying* to predict, and how well each is ranked. Distinguishes
            # "P=0 because model never predicted class c" (n_preds[c]==0,
            # PR-AUC tells the ranking story) from "P=0 because all attempts
            # were wrong" (n_preds[c]>0, but tp=0).
            n_preds = val_metrics["n_preds_per_class"]
            n_tgts = val_metrics["n_targets_per_class"]
            pr_aucs = val_metrics["pr_auc_per_class"]
            per_class_str = " ".join(
                f"c{c}(p={int(n_preds[c])}/t={int(n_tgts[c])},auc={pr_aucs[c]:.2e})"
                for c in range(self.config.num_classes)
            )
            print(f"   Per-class: {per_class_str}")

            # Push everything to the tracker (no-op if backend is "none").
            log_payload = {
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "val/loss": val_loss,
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **{f"val/{k}": v for k, v in val_metrics.items()},
            }
            self.tracker.log(log_payload, step=epoch + 1)

            # Empirical class-collapse warning: if the model has not predicted a
            # single rare class across the whole validation set, P/R are 0 by
            # construction (not by bug). Tell the user explicitly.
            if val_metrics["n_rxn_preds"] == 0 and val_metrics["n_rxn_targets"] > 0:
                print(
                    "   ⚠️  Model predicted no-change for ALL pairs this epoch — "
                    "class collapse. Consider higher focal gamma, stronger class "
                    "weights, or oversampling rare classes."
                )
            # Inverted collapse: predicting reaction everywhere. Signature is
            # n_rxn_preds >> n_rxn_targets AND precision close to zero.
            elif (
                val_metrics["n_rxn_targets"] > 0
                and val_metrics["n_rxn_preds"] > 10 * val_metrics["n_rxn_targets"]
                and val_metrics["precision"] < 0.01
            ):
                print(
                    "   ⚠️  Model predicted reaction for >10x more pairs than truth — "
                    "inverted collapse (loss likely under-weights the no-change class). "
                    "Consider weaker rare-class weights or lower focal gamma."
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

        self.tracker.finish()

    def _save_checkpoint(self, epoch: int, tag: str = "best") -> None:
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "history": self.history,
            "best_val_loss": self.best_val_loss,
        }
        path = Path(self.config.checkpoint_dir) / f"{self.config.task_mode}_{tag}_ep{epoch}.pt"
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
        results_path = Path(self.config.output_dir) / f"{self.config.task_mode}_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Saved results: {results_path}")
