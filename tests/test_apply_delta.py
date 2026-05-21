"""Unit tests for :func:`mech_uspto.chemistry.apply_delta.apply_delta`."""

import pytest
import torch
from rdkit import Chem

from mech_uspto.chemistry import ApplyDeltaError, apply_delta
from mech_uspto.data.transformations import DeltaMatrixGenerator


def _canon(mol: Chem.Mol, with_maps: bool = False) -> str:
    """Canonical SMILES; optionally strip atom-map numbers first."""
    m = Chem.Mol(mol)
    if not with_maps:
        for a in m.GetAtoms():
            a.SetAtomMapNum(0)
    return Chem.MolToSmiles(m)


def _delta(n: int, edits: dict[tuple[int, int], int]) -> torch.Tensor:
    d = torch.zeros((n, n), dtype=torch.long)
    for (i, j), v in edits.items():
        d[i, j] = v
        d[j, i] = v
    return d


# ---------------------------------------------------------------------- ops


def test_break_bond_removes_bond():
    # Methanol + methane → break the methanol C-O, get methyl fragment + OH fragment.
    # (Not realistic chemistry; just exercises the bond-removal codepath.)
    mol = Chem.MolFromSmiles("[CH3:1][OH:2]")
    i = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum() == 1)
    j = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum() == 2)
    d = _delta(mol.GetNumAtoms(), {(i, j): -1})

    out = apply_delta(mol, d, apply_charge_heuristic=False)
    assert out.GetBondBetweenAtoms(i, j) is None
    # CH3 (4H, closed shell after H bookkeeping reset) + H2O.
    assert _canon(out) == _canon(Chem.MolFromSmiles("C.O"))


def test_form_bond_adds_bond():
    # Two methanols → form a C-C bond between their carbons.
    mol = Chem.MolFromSmiles("[CH3:1][OH:2].[CH3:3][OH:4]")
    i = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum() == 1)
    j = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum() == 3)
    d = _delta(mol.GetNumAtoms(), {(i, j): +1})

    out = apply_delta(mol, d, apply_charge_heuristic=False)
    assert out.GetBondBetweenAtoms(i, j) is not None
    # Ethylene glycol-like (2-hydroxyethanol): HOCH2CH2OH.
    assert _canon(out) == _canon(Chem.MolFromSmiles("OCCO"))


def test_change_order_single_to_double():
    mol = Chem.MolFromSmiles("[CH3:1][CH2:2][OH:3]")
    i = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum() == 1)
    j = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum() == 2)
    d = _delta(mol.GetNumAtoms(), {(i, j): +1})

    out = apply_delta(mol, d, apply_charge_heuristic=False)
    bond = out.GetBondBetweenAtoms(i, j)
    assert bond.GetBondTypeAsDouble() == 2.0


def test_invalid_negative_order_raises():
    mol = Chem.MolFromSmiles("[CH3:1][CH3:2]")
    i, j = 0, 1
    d = _delta(mol.GetNumAtoms(), {(i, j): -2})  # 1 + (-2) = -1
    with pytest.raises(ApplyDeltaError) as exc:
        apply_delta(mol, d, apply_charge_heuristic=False)
    assert exc.value.reason == "invalid_order"


def test_invalid_quadruple_order_raises():
    mol = Chem.MolFromSmiles("[C:1]#[C:2]")
    i, j = 0, 1
    d = _delta(mol.GetNumAtoms(), {(i, j): +1})  # 3 + 1 = 4
    with pytest.raises(ApplyDeltaError) as exc:
        apply_delta(mol, d, apply_charge_heuristic=False)
    assert exc.value.reason == "invalid_order"


def test_shape_mismatch_raises():
    mol = Chem.MolFromSmiles("[CH3:1][CH3:2]")
    with pytest.raises(ApplyDeltaError) as exc:
        apply_delta(mol, torch.zeros((1, 1)))
    assert exc.value.reason == "shape_mismatch"


def test_diagonal_is_ignored():
    mol = Chem.MolFromSmiles("[CH3:1][OH:2]")
    n = mol.GetNumAtoms()
    d = torch.zeros((n, n), dtype=torch.long)
    d[0, 0] = 5  # nonsense diagonal — must be ignored
    out = apply_delta(mol, d, apply_charge_heuristic=False)
    assert _canon(out) == _canon(mol)


# ----------------------------------------------------------- atom maps


def test_preserves_atom_maps():
    mol = Chem.MolFromSmiles("[CH3:1][CH2:2][OH:3]")
    d = torch.zeros((mol.GetNumAtoms(), mol.GetNumAtoms()), dtype=torch.long)
    out = apply_delta(mol, d, apply_charge_heuristic=False)
    in_maps = sorted(a.GetAtomMapNum() for a in mol.GetAtoms())
    out_maps = sorted(a.GetAtomMapNum() for a in out.GetAtoms())
    assert in_maps == out_maps == [1, 2, 3]


# ----------------------------------------------------- round-trip via Δ


def test_roundtrip_via_delta_from_reactants_products():
    """Δ from R→P, applied to R, should yield a Mol matching P."""
    r = Chem.MolFromSmiles("[CH3:1][Br:2].[OH:3][CH3:4]")
    # Methyl swap: form CH3-O, break CH3-Br, break O-CH3, form Br-something?
    # Use the simpler case: condensation losing nothing — just bond change in place.
    # Pick: ethanol → acetaldehyde (loses 2 H's; harder).
    # Easiest: rotate a single bond change.
    r = Chem.MolFromSmiles("[CH3:1][CH2:2][OH:3]")
    p = Chem.MolFromSmiles("[CH2:1]=[CH:2][OH:3]")  # dehydrogenation, vinyl alcohol
    # Build Δ
    delta = DeltaMatrixGenerator.delta_from_reactants_products(r, p)
    out = apply_delta(r, delta, apply_charge_heuristic=False)
    assert _canon(out) == _canon(p)
