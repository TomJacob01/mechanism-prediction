"""Classification heads for delta-matrix prediction."""

from typing import Optional

import torch
import torch.nn as nn


class DeltaMLP(nn.Module):
    """Pairwise MLP head producing per-pair Δ logits.

    Supports either 3-class (stepwise: {-1, 0, 1}) or 7-class
    (end-to-end: {-3, -2, -1, 0, 1, 2, 3}) outputs.

    Optional features:
    - ``edge_dim``: when > 0, the head concatenates per-pair input edge
      features (a dense ``(B, N, N, edge_dim)`` tensor) into the pair
      representation so the classifier can condition directly on the *current*
      bond order between atoms (i, j) instead of relying on the encoder to
      preserve it through all message-passing layers.
    - ``class_prior``: when provided, the final linear layer's bias is set to
      ``log(π) − mean(log(π))`` so the step-0 softmax matches ``class_prior``
      (the RetinaNet trick). Eliminates the early-training oscillation between
      "predict no-change everywhere" and "predict reaction everywhere".
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_classes: int = 3,
        dropout: float = 0.1,
        edge_dim: int = 0,
        class_prior: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if num_classes not in (3, 7):
            raise ValueError(f"num_classes must be 3 or 7, got {num_classes}")

        self.num_classes = num_classes
        self.edge_dim = edge_dim

        pair_in = 2 * hidden_dim + edge_dim
        self.mlp = nn.Sequential(
            nn.Linear(pair_in, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        if class_prior is not None:
            self._init_bias_from_prior(class_prior)

    def _init_bias_from_prior(self, class_prior: torch.Tensor) -> None:
        """Set the final-layer bias so step-0 softmax ≈ ``class_prior``."""
        prior = torch.as_tensor(class_prior, dtype=torch.float32).flatten()
        if prior.numel() != self.num_classes:
            raise ValueError(
                f"class_prior length ({prior.numel()}) must equal num_classes ({self.num_classes})"
            )
        if (prior <= 0).any():
            raise ValueError("class_prior entries must all be > 0 (taking log)")
        log_prior = torch.log(prior / prior.sum())
        final_linear = self.mlp[-1]
        with torch.no_grad():
            final_linear.bias.data = log_prior - log_prior.mean()

    def forward(
        self,
        h_dense: torch.Tensor,
        edge_dense: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute symmetric pairwise Δ logits.

        Args:
            h_dense: ``(B, N, D)`` dense node embeddings.
            edge_dense: optional ``(B, N, N, edge_dim)`` dense edge features.
                Required iff ``edge_dim > 0`` was set at construction.

        Returns ``(B, N, N, num_classes)`` logits, symmetrised over (i, j).
        """
        B, N, D = h_dense.size()
        h_src = h_dense.unsqueeze(2).expand(B, N, N, D)
        h_dst = h_dense.unsqueeze(1).expand(B, N, N, D)
        parts = [h_src, h_dst]
        if self.edge_dim > 0:
            if edge_dense is None:
                raise ValueError(
                    "DeltaMLP was built with edge_dim > 0 but no edge_dense was passed"
                )
            if edge_dense.shape != (B, N, N, self.edge_dim):
                raise ValueError(
                    f"edge_dense shape {tuple(edge_dense.shape)} != expected "
                    f"{(B, N, N, self.edge_dim)}"
                )
            parts.append(edge_dense)
        pair_features = torch.cat(parts, dim=-1)

        logits = self.mlp(pair_features)
        # Bond formation/breaking is symmetric (i,j) == (j,i).
        logits = (logits + logits.transpose(1, 2)) / 2
        return logits

__all__ = [
    "DeltaMLP",
]
