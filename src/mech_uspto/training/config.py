"""Training configuration for the dual-mode ablation study."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

DEFAULT_DATA_PATH = os.environ.get("MECH_USPTO_DATA", "./data/mech-USPTO-31k.csv")


@dataclass
class Config:
    """Mode-aware training configuration.

    Mode-specific defaults (``num_classes``, ``class_weights``) are populated
    in ``__post_init__`` so callers only need to set ``task_mode``.
    """

    # Paths
    csv_path: str = DEFAULT_DATA_PATH
    output_dir: str = "./results"
    # If None, derived as f"{output_dir}/checkpoints" in __post_init__ so
    # concurrent runs don't share checkpoint files. Override to keep the
    # old single-shared-dir behavior.
    checkpoint_dir: Optional[str] = None

    # Dataset
    task_mode: str = "stepwise"  # or "end_to_end"
    train_val_test_split: tuple[float, float, float] = (0.7, 0.15, 0.15)
    batch_size: int = 32
    num_workers: int = 4

    # Model architecture
    node_in: int = 25
    edge_in: int = 6
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.15
    max_grad_norm: float = 1.0

    # Loss & optimization
    num_classes: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100
    patience: int = 20
    # Linear LR warmup over the first ``warmup_steps`` optimizer steps. LR
    # ramps from 0 → ``learning_rate`` then hands off to the plateau scheduler.
    # 0 disables warmup (backward compatible). Counted in optimizer steps,
    # NOT epochs — one warmup setting works across batch sizes.
    warmup_steps: int = 0

    # Loss shaping
    class_weights: Optional[torch.Tensor] = field(default=None, repr=False)
    gamma_focal: float = 3.5
    spectator_weight: float = 0.1

    # Head initialisation / wiring
    # Class prior for RetinaNet-style bias init on the head's final layer.
    # When None, a mode-aware default is filled in by __post_init__ matching
    # mech-USPTO-31k's observed ~97.5% no-change dominance. Pass an explicit
    # tensor to override (e.g. computed from your own train split).
    class_prior: Optional[torch.Tensor] = field(default=None, repr=False)
    # When True, the head concatenates the (B, N, N, edge_in) input edge
    # features into the pair representation so the classifier can condition
    # directly on the current bond order. Cheap F1 win; default on.
    use_edge_features_in_head: bool = True

    # Runtime
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    # Wrap forward+loss in ``torch.autocast(..., dtype=bfloat16)`` on CUDA.
    # bf16 has fp32-equivalent dynamic range so no GradScaler is needed.
    # Auto-disabled on non-CUDA devices regardless of this flag. ~2× speedup
    # on Ampere+ (A40 / L40S) with no observed accuracy loss; see #12.
    use_amp: bool = True
    # Deterministic-mode toggles. When True, sets
    # ``torch.backends.cudnn.deterministic = True`` and ``benchmark = False``
    # at seeding time. Trades ~5-10% throughput for run-to-run reproducibility.
    # Required for any "did intervention X actually help vs seed noise" comparison.
    deterministic: bool = True

    def __post_init__(self) -> None:
        if self.task_mode == "stepwise":
            self.num_classes = 3
            if self.class_weights is None:
                # Mildly upweight the rare ±1 classes vs the dominant 0 class.
                # Earlier 8x weights caused over-prediction collapse in end-to-end
                # runs (May 2026); kept proportional here.
                self.class_weights = torch.tensor([3.0, 1.0, 3.0])
        elif self.task_mode == "end_to_end":
            self.num_classes = 7
            if self.class_weights is None:
                # Δ ∈ {-3,-2,-1,0,1,2,3}; class 3 (idx 3) is the dominant "no change".
                # c4 (Δ=+1) bumped to 5.0 vs c2 (Δ=−1) at 3.0: test eval of run 68209851
                # showed asymmetric collapse — Δ=−1 F1=0.92, Δ=+1 F1=0.04 — even though
                # supports are similar (158k vs 99k). Bond *formation* loses to "keep
                # Δ=0" more than bond *breaking* does; extra weight on +1 to compensate.
                # Earlier [2,4,16,1,16,4,2] caused inverted collapse (RxnPreds > targets
                # by epoch 8); these values keep rare-class signal without dominating.
                self.class_weights = torch.tensor([1.0, 1.5, 3.0, 1.0, 5.0, 1.5, 1.0])
        else:
            raise ValueError(f"Unknown task_mode: {self.task_mode!r}")

        if self.class_weights is not None and len(self.class_weights) != self.num_classes:
            raise ValueError(
                f"class_weights length ({len(self.class_weights)}) does not match "
                f"num_classes for task_mode={self.task_mode!r} ({self.num_classes})"
            )

        # Class-prior defaults. Δ=0 dominates ~97.5% of pairs in mech-USPTO-31k;
        # rare classes split the remaining ~2.5% with a chemically-symmetric
        # tilt toward smaller |Δ|. Override by passing ``class_prior`` explicitly.
        if self.class_prior is None:
            if self.task_mode == "stepwise":
                # Δ ∈ {-1, 0, 1}; no-change at idx 1.
                self.class_prior = torch.tensor([0.0125, 0.975, 0.0125])
            else:
                # 7-class: bulk on idx 3, light tails on |Δ|=1, tiny on |Δ|≥2.
                self.class_prior = torch.tensor([0.0005, 0.002, 0.010, 0.975, 0.010, 0.002, 0.0005])
        if len(self.class_prior) != self.num_classes:
            raise ValueError(
                f"class_prior length ({len(self.class_prior)}) does not match "
                f"num_classes for task_mode={self.task_mode!r} ({self.num_classes})"
            )

        if self.checkpoint_dir is None:
            self.checkpoint_dir = str(Path(self.output_dir) / "checkpoints")

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

__all__ = [
    "Config",
    "DEFAULT_DATA_PATH",
]
