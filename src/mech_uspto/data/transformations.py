"""Compute Δ (delta) bond-order matrices for stepwise / end-to-end modes."""

import torch
from rdkit import Chem


class DeltaMatrixGenerator:
    """Build delta matrices from molecule pairs or mechanism arrow strings."""

    @staticmethod
    def compute_adjacency_matrix(mol: Chem.Mol) -> torch.Tensor:
        """Return the bond-order adjacency matrix for ``mol``."""
        num_atoms = mol.GetNumAtoms()
        adjacency = torch.zeros((num_atoms, num_atoms), dtype=torch.float)

        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_order = bond.GetBondTypeAsDouble()
            adjacency[i, j] = bond_order
            adjacency[j, i] = bond_order

        return adjacency

    @staticmethod
    def delta_from_reactants_products(
        reactants_mol: Chem.Mol,
        products_mol: Chem.Mol,
    ) -> torch.Tensor:
        """Compute Δ = A_products − A_reactants.

        Stepwise mode: result is constrained to {-1, 0, 1}.
        End-to-end mode: result may include {-2, -1, 0, 1, 2} for double-bond changes.
        """
        adjacency_reactants = DeltaMatrixGenerator.compute_adjacency_matrix(reactants_mol)
        adjacency_products = DeltaMatrixGenerator.compute_adjacency_matrix(products_mol)

        n_react = adjacency_reactants.shape[0]
        n_prod = adjacency_products.shape[0]

        # Pad smaller adjacency to the larger size when atom counts differ.
        if n_react != n_prod:
            max_n = max(n_react, n_prod)
            padded_react = torch.zeros((max_n, max_n), dtype=torch.float)
            padded_prod = torch.zeros((max_n, max_n), dtype=torch.float)
            padded_react[:n_react, :n_react] = adjacency_reactants
            padded_prod[:n_prod, :n_prod] = adjacency_products
            adjacency_reactants = padded_react
            adjacency_products = padded_prod

        delta = adjacency_products - adjacency_reactants
        # Bond changes are symmetric (A→B == B→A).
        delta = (delta + delta.T) / 2
        return delta

    @staticmethod
    def delta_from_mechanism_arrows(
        arrow_string: str,
        mol: Chem.Mol,
    ) -> torch.Tensor:
        """Parse a mechanism arrow string (e.g. ``"1*2=2*3"``) into a Δ matrix.

        Format: ``"1*2=; 3*4=5*6"`` means *break bond 1-2*, then *shift from
        3-4 to 3-6*. Atom indices reference RDKit atom map numbers.
        """
        num_nodes = mol.GetNumAtoms()
        delta = torch.zeros((num_nodes, num_nodes), dtype=torch.long)
        map_to_idx = {
            atom.GetAtomMapNum(): atom.GetIdx()
            for atom in mol.GetAtoms()
            if atom.GetAtomMapNum() > 0
        }

        if not isinstance(arrow_string, str) or not arrow_string.strip():
            return delta

        for arrow in arrow_string.split(";"):
            arrow = arrow.strip().replace("*", "").replace(" ", "")
            if "=" not in arrow:
                continue

            src_s, dst_s = arrow.split("=")
            src_maps = [int(m) for m in src_s.split(",") if m.isdigit()]
            dst_maps = [int(m) for m in dst_s.split(",") if m.isdigit()]

            # Bond breaking: "1,2=" → remove bond 1-2.
            if len(src_maps) == 2 and len(dst_maps) == 0:
                u, v = src_maps[0], src_maps[1]
                if u in map_to_idx and v in map_to_idx:
                    delta[map_to_idx[u], map_to_idx[v]] = -1
                    delta[map_to_idx[v], map_to_idx[u]] = -1

            # Bond formation: "=1,2" → form bond 1-2.
            elif len(src_maps) == 0 and len(dst_maps) == 2:
                u, v = dst_maps[0], dst_maps[1]
                if u in map_to_idx and v in map_to_idx:
                    delta[map_to_idx[u], map_to_idx[v]] = 1
                    delta[map_to_idx[v], map_to_idx[u]] = 1

            # Shift: "1,2=1,3" → remove 1-2 / form 1-3 (only "form" recorded here).
            elif len(src_maps) == 2 and len(dst_maps) == 2:
                if src_maps[0] == dst_maps[0]:  # shared pivot
                    remove_atom = src_maps[1]
                    add_atom = dst_maps[1]
                    if remove_atom in map_to_idx and add_atom in map_to_idx:
                        delta[map_to_idx[remove_atom], map_to_idx[add_atom]] = 1
                        delta[map_to_idx[add_atom], map_to_idx[remove_atom]] = 1

        return delta
