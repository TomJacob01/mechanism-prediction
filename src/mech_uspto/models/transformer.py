"""Reaction transformer encoder + delta head (adapted from PMechDB POC)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import to_dense_batch

from mech_uspto.models.heads import DeltaMLP


class ReactionTransformer(nn.Module):
    """Graph-transformer encoder paired with a modular ``DeltaMLP`` head.

    The head is 3-class for stepwise mode and 5-class for end-to-end mode.
    """

    def __init__(
        self,
        node_in: int = 25,
        edge_in: int = 6,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        num_classes: int = 3,
    ):
        super().__init__()

        self.node_embedding = nn.Linear(node_in, hidden_dim)
        self.edge_embedding = nn.Linear(edge_in, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                TransformerConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    edge_dim=hidden_dim,
                    dropout=dropout,
                    concat=True,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.mlp = DeltaMLP(hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout)

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (dense node embeddings, valid-position mask)."""
        h = self.node_embedding(x)
        e = self.edge_embedding(edge_attr)

        for conv, norm in zip(self.convs, self.norms):
            h_res = h
            h = conv(h, edge_index, edge_attr=e)
            h = F.dropout(h, p=0.1, training=self.training)
            h = norm(h + h_res)

        h_dense, mask = to_dense_batch(h, batch)
        return h_dense, mask

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (logits of shape (B, N, N, num_classes), node mask)."""
        h_dense, mask = self.encode(x, edge_index, edge_attr, batch)
        logits = self.mlp(h_dense)
        return logits, mask
