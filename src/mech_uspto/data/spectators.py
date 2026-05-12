"""Identify spectator atoms (those whose bonds do not change)."""

import torch
from rdkit import Chem  # noqa: F401  - kept for typing clarity in signatures


class SpectatorDetector:
    """Identify and summarise spectator atoms in USPTO reactions.

    A spectator atom is one whose connectivity is unchanged across the
    reaction. In USPTO ~95% of atoms are spectators, which would otherwise
    drown out gradient signal on the reactive atoms.
    """

    @staticmethod
    def identify_spectators(
        reactants_mol: "Chem.Mol",
        products_mol: "Chem.Mol",
        delta_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Return a boolean per-atom mask.

        ``True`` = spectator (downweight in loss).
        ``False`` = reactive (full loss weight).
        """
        num_atoms = delta_matrix.shape[0]
        spectator_mask = torch.ones(num_atoms, dtype=torch.bool)

        # Any atom touching a non-zero Δ entry is reactive.
        reactive_atoms: set[int] = set()
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                if delta_matrix[i, j] != 0:
                    reactive_atoms.add(i)
                    reactive_atoms.add(j)

        for idx in reactive_atoms:
            spectator_mask[idx] = False

        return spectator_mask

    @staticmethod
    def compute_spectator_ratio(spectator_mask: torch.Tensor) -> float:
        """Fraction of atoms that are spectators (for monitoring)."""
        return (spectator_mask.sum().float() / len(spectator_mask)).item()
