"""Graph featurization for mapped SMILES.

Produces 25-dim node features and 6-dim edge features (matching the PMechDB POC).
Featurization helpers operate on RDKit ``Mol`` objects with mapped atoms.
"""

from typing import Any

import torch
from rdkit import Chem
from torch_geometric.data import Data

from mech_uspto.constants import (
    ALLOWED_BOND_TYPES,
    ALLOWED_CHIRAL,
    ALLOWED_ELEMENTS,
    ALLOWED_HYBRIDIZATIONS,
    EDGE_FEATURE_DIM,
)


def one_hot_encode(value: Any, allowed_choices: list[Any]) -> list[int]:
    """One-hot encode ``value`` against ``allowed_choices``.

    Adds a trailing "unknown" bin so the encoding length is always
    ``len(allowed_choices) + 1``.
    """
    encoding = [0] * (len(allowed_choices) + 1)
    if value in allowed_choices:
        encoding[allowed_choices.index(value)] = 1
    else:
        encoding[-1] = 1
    return encoding


def align_atoms(mol: Chem.Mol) -> Chem.Mol:
    """Reorder atoms so mapped atoms come first (sorted by map number)."""
    sorted_atoms = sorted(
        mol.GetAtoms(),
        key=lambda a: a.GetAtomMapNum() if a.GetAtomMapNum() > 0 else float("inf"),
    )
    ordered_indices = [a.GetIdx() for a in sorted_atoms]
    return Chem.RenumberAtoms(mol, ordered_indices)


def featurize_nodes(aligned_mol: Chem.Mol) -> torch.Tensor:
    """Build the (N, 25) node-feature tensor."""
    node_features: list[list[int]] = []
    for atom in aligned_mol.GetAtoms():
        element_one_hot = one_hot_encode(atom.GetAtomicNum(), ALLOWED_ELEMENTS)
        num_hs = atom.GetTotalNumHs() if atom.GetNumRadicalElectrons() == 0 else 0
        hyb_one_hot = one_hot_encode(atom.GetHybridization(), ALLOWED_HYBRIDIZATIONS)
        chiral_one_hot = one_hot_encode(atom.GetChiralTag(), ALLOWED_CHIRAL)

        features = (
            element_one_hot
            + [
                atom.GetDegree(),
                atom.GetFormalCharge(),
                int(atom.GetIsAromatic()),
                num_hs,
                int(atom.IsInRing()),
            ]
            + hyb_one_hot
            + chiral_one_hot
        )
        node_features.append(features)

    return torch.tensor(node_features, dtype=torch.float)


def featurize_edges(aligned_mol: Chem.Mol) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (edge_index, edge_attr) for the molecule."""
    edge_indices: list[list[int]] = []
    edge_features: list[list[int]] = []

    for bond in aligned_mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = 4.0 if bond.GetIsAromatic() else bond.GetBondTypeAsDouble()
        bond_one_hot = one_hot_encode(bond_type, ALLOWED_BOND_TYPES)
        bond_feats = bond_one_hot + [int(bond.IsInRing())]

        edge_indices.extend([[i, j], [j, i]])
        edge_features.extend([bond_feats, bond_feats])

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, EDGE_FEATURE_DIM), dtype=torch.float)

    return edge_index, edge_attr


def process_mapped_smiles(
    mapped_smiles: str,
    add_hs: bool = True,
) -> tuple[Chem.Mol, Data]:
    """Parse a mapped SMILES into an RDKit ``Mol`` and a PyG ``Data`` object.

    Raises:
        ValueError: if ``mapped_smiles`` is not a valid molecule.
    """
    mol = Chem.MolFromSmiles(mapped_smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {mapped_smiles}")

    if add_hs:
        mol = Chem.AddHs(mol)

    aligned_mol = align_atoms(mol)
    x = featurize_nodes(aligned_mol)
    edge_index, edge_attr = featurize_edges(aligned_mol)

    return aligned_mol, Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
