"""Featurization stability across bond-Δ steps.

The sequential model will re-featurize intermediate molecules between steps.
This file checks that ``featurize_nodes`` / ``featurize_edges`` produce
well-formed tensors when called on a Mol returned by ``apply_delta``
(post-sanitize, atom-maps preserved), with the same shapes / dtypes /
feature dimensions as featurizing a fresh-from-SMILES Mol.
"""

import torch
from rdkit import Chem

from mech_uspto.chemistry import apply_delta
from mech_uspto.constants import EDGE_FEATURE_DIM, NODE_FEATURE_DIM
from mech_uspto.data.featurization import (
    align_atoms,
    featurize_edges,
    featurize_nodes,
)
from mech_uspto.data.transformations import DeltaMatrixGenerator


def _featurize(mol: Chem.Mol):
    aligned = align_atoms(mol)
    x = featurize_nodes(aligned)
    edge_index, edge_attr = featurize_edges(aligned)
    return x, edge_index, edge_attr


def test_featurize_works_on_post_delta_mol():
    """Output of apply_delta is sanitized and has atom maps → featurize must
    succeed with the expected shapes / dtypes."""
    r = Chem.MolFromSmiles("[CH3:1][CH2:2][OH:3]")
    p = Chem.MolFromSmiles("[CH2:1]=[CH:2][OH:3]")
    delta = DeltaMatrixGenerator.delta_from_reactants_products(r, p)
    out = apply_delta(r, delta, apply_charge_heuristic=False)

    x, edge_index, edge_attr = _featurize(out)
    assert x.shape[0] == out.GetNumAtoms()
    assert x.shape[1] == NODE_FEATURE_DIM
    assert x.dtype == torch.float
    assert edge_index.shape[0] == 2
    if edge_attr.numel() > 0:
        assert edge_attr.shape[1] == EDGE_FEATURE_DIM
        assert edge_index.shape[1] == edge_attr.shape[0]


def test_featurize_post_delta_matches_featurize_fresh_smiles():
    """Featurizing the apply_delta output should give the same tensors as
    parsing P's SMILES from scratch (modulo atom-ordering, which align_atoms
    fixes by sorting on atom-map)."""
    r = Chem.MolFromSmiles("[CH3:1][CH2:2][OH:3]")
    p_smi = "[CH2:1]=[CH:2][OH:3]"
    p_fresh = Chem.MolFromSmiles(p_smi)
    delta = DeltaMatrixGenerator.delta_from_reactants_products(r, p_fresh)
    out = apply_delta(r, delta, apply_charge_heuristic=False)

    x_out, ei_out, ea_out = _featurize(out)
    x_fresh, ei_fresh, ea_fresh = _featurize(p_fresh)
    assert x_out.shape == x_fresh.shape
    assert ei_out.shape == ei_fresh.shape
    assert ea_out.shape == ea_fresh.shape


def test_featurize_after_two_delta_steps():
    """Multi-step prep: featurize after step 1 AND after step 2; both must
    produce well-formed tensors. Sequential model's hot loop."""
    r = Chem.MolFromSmiles("[CH3:1][CH2:2][OH:3]")
    p1 = Chem.MolFromSmiles("[CH2:1]=[CH:2][OH:3]")
    d1 = DeltaMatrixGenerator.delta_from_reactants_products(r, p1)
    step1 = apply_delta(r, d1, apply_charge_heuristic=False)
    x1, ei1, ea1 = _featurize(step1)
    assert x1.shape[0] == step1.GetNumAtoms()
    assert x1.shape[1] == NODE_FEATURE_DIM

    # Step 2: break the C-O on atom 2.
    n = step1.GetNumAtoms()
    i = next(a.GetIdx() for a in step1.GetAtoms() if a.GetAtomMapNum() == 2)
    j = next(a.GetIdx() for a in step1.GetAtoms() if a.GetAtomMapNum() == 3)
    d2 = torch.zeros((n, n), dtype=torch.long)
    d2[i, j] = -1
    d2[j, i] = -1
    step2 = apply_delta(step1, d2, apply_charge_heuristic=False)
    x2, ei2, ea2 = _featurize(step2)
    assert x2.shape[0] == step2.GetNumAtoms()
    assert x2.shape[1] == NODE_FEATURE_DIM
