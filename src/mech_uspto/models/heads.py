"""Classification heads for delta-matrix prediction."""

import torch
import torch.nn as nn


class DeltaMLP(nn.Module):
    """Pairwise MLP head producing per-pair Δ logits.

    Supports either 3-class (stepwise: {-1, 0, 1}) or 7-class
    (end-to-end: {-3, -2, -1, 0, 1, 2, 3}) outputs.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_classes not in (3, 7):
            raise ValueError(f"num_classes must be 3 or 7, got {num_classes}")

        self.num_classes = num_classes
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, h_dense: torch.Tensor) -> torch.Tensor:
        """Args: ``h_dense`` of shape (B, N, D). Returns (B, N, N, num_classes)."""
        B, N, D = h_dense.size()
        h_src = h_dense.unsqueeze(2).expand(B, N, N, D)
        h_dst = h_dense.unsqueeze(1).expand(B, N, N, D)
        pair_features = torch.cat([h_src, h_dst], dim=-1)

        logits = self.mlp(pair_features)
        # Bond formation/breaking is symmetric (i,j) == (j,i).
        logits = (logits + logits.transpose(1, 2)) / 2
        return logits
