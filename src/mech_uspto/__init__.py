"""mech_uspto: dual-mode reaction prediction on mech-USPTO-31k.

Public API re-exports so external code can do
``from mech_uspto import MechUSPTODataset, DeltaMLP, ...``.
"""

from mech_uspto.data.dataset import MechUSPTODataset
from mech_uspto.data.loaders import collate_fn_with_spectators, create_dataloaders
from mech_uspto.data.parser import MechUSPTOParser
from mech_uspto.data.schema import MultiStepReaction, ReactionStep
from mech_uspto.data.spectators import SpectatorDetector
from mech_uspto.data.transformations import DeltaMatrixGenerator
from mech_uspto.losses.focal import MaskedFocalLossWithSpectators
from mech_uspto.models.heads import DeltaMLP
from mech_uspto.models.transformer import ReactionTransformer

__all__ = [
    "DeltaMLP",
    "DeltaMatrixGenerator",
    "MaskedFocalLossWithSpectators",
    "MechUSPTODataset",
    "MechUSPTOParser",
    "MultiStepReaction",
    "ReactionStep",
    "ReactionTransformer",
    "SpectatorDetector",
    "collate_fn_with_spectators",
    "create_dataloaders",
]
