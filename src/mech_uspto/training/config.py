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
    checkpoint_dir: str = "./checkpoints"

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

    # Loss shaping
    class_weights: Optional[torch.Tensor] = field(default=None, repr=False)
    gamma_focal: float = 3.5
    spectator_weight: float = 0.1

    # Runtime
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.task_mode == "stepwise":
            self.num_classes = 3
            if self.class_weights is None:
                # Upweight the rare ±1 classes vs. the dominant 0 class.
                self.class_weights = torch.tensor([8.0, 1.0, 8.0])
        elif self.task_mode == "end_to_end":
            self.num_classes = 7
            if self.class_weights is None:
                # Δ ∈ {-3,-2,-1,0,1,2,3}; class 3 (idx 3) is the dominant "no change".
                # Strong weights to combat class collapse on imbalanced data
                # (Δ=0 is ~97.5% of all pairs in mech-USPTO-31k).
                self.class_weights = torch.tensor([2.0, 4.0, 16.0, 1.0, 16.0, 4.0, 2.0])
        else:
            raise ValueError(f"Unknown task_mode: {self.task_mode!r}")

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
