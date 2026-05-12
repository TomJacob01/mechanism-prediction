"""Masked focal loss with spectator-atom downweighting."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedFocalLossWithSpectators(nn.Module):
    """Focal loss masked to valid positions, with optional spectator downweighting.

    Designed for USPTO-scale molecules where ~95% of atoms are spectators —
    those positions are downweighted (not zeroed) so the model still gets
    weak structural signal without being drowned in trivial bonds.
    """

    def __init__(
        self,
        num_classes: int = 3,
        weights: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        spectator_weight: float = 0.1,
        reduction: str = "mean",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.spectator_weight = spectator_weight
        self.reduction = reduction

        if weights is not None:
            self.register_buffer("weights", weights)
        else:
            self.weights = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask_2d: torch.Tensor,
        spectator_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the masked focal loss.

        Args:
            logits: (B, N, N, num_classes).
            targets: (B, N, N), values in [0, num_classes-1].
            mask_2d: (B, N, N) boolean / 0-1 mask of valid (non-padded) positions.
            spectator_mask: optional (B, N, N) where True marks spectator pairs.
        """
        logits_flat = logits.view(-1, self.num_classes)
        targets_flat = targets.view(-1)
        mask_flat = mask_2d.view(-1)

        ce_loss = F.cross_entropy(logits_flat, targets_flat, weight=self.weights, reduction="none")

        # Focal loss modulation.
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # Mask out padding.
        masked_loss = focal_loss * mask_flat

        # Downweight spectator positions.
        if spectator_mask is not None:
            spectator_flat = spectator_mask.view(-1)
            spectator_weight = torch.where(
                spectator_flat,
                torch.tensor(self.spectator_weight, device=spectator_flat.device),
                torch.tensor(1.0, device=spectator_flat.device),
            )
            masked_loss = masked_loss * spectator_weight

        if self.reduction == "mean":
            num_active = mask_flat.sum().clamp(min=1.0)
            return masked_loss.sum() / num_active

        return masked_loss.view_as(targets)
