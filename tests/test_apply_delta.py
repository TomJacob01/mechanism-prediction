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


# ---------------------------------------------------------------- stereo


def _atom_by_map(mol: Chem.Mol, mapnum: int) -> Chem.Atom:
    return next(a for a in mol.GetAtoms() if a.GetAtomMapNum() == mapnum)


def test_chirality_preserved_on_untouched_atom():
    """An atom whose bonds are not in the Δ keeps its R/S tag."""
    # (R)-2-bromobutane plus an unrelated methanol; break the methanol O-H.
    # The chiral C (map 2) has no Δ touching it → @ must survive.
    mol = Chem.MolFromSmiles("[CH3:1][C@H:2]([Br:3])[CH2:4][CH3:5].[CH3:6][OH:7]")
    in_tag = _atom_by_map(mol, 2).GetChiralTag()
    i = _atom_by_map(mol, 6).GetIdx()
    j = _atom_by_map(mol, 7).GetIdx()
    d = _delta(mol.GetNumAtoms(), {(i, j): -1})
    out = apply_delta(mol, d, apply_charge_heuristic=False)
    chiral_atom = _atom_by_map(out, 2)
    assert chiral_atom.GetChiralTag() == in_tag


def test_chirality_preserved_when_neighbor_elements_unchanged():
    """Mitsunobu-like: touched atom keeps @ tag when neighbour element multiset
    is preserved (OH → OR' is still {O, C, C, H})."""
    # [C@@H](OH)(CH3)(CH3) → swap OH for OMe by breaking C-O and forming C-O
    # to a different oxygen. Both oxygens are O, so element multiset preserved.
    smi = "[CH3:1][C@@H:2]([OH:3])[CH3:4].[OH:5][CH3:6]"
    mol = Chem.MolFromSmiles(smi)
    in_tag = _atom_by_map(mol, 2).GetChiralTag()
    c2 = _atom_by_map(mol, 2).GetIdx()
    o3 = _atom_by_map(mol, 3).GetIdx()
    o5 = _atom_by_map(mol, 5).GetIdx()
    d = _delta(mol.GetNumAtoms(), {(c2, o3): -1, (c2, o5): +1})
    out = apply_delta(mol, d, apply_charge_heuristic=False)
    chiral_atom = _atom_by_map(out, 2)
    assert chiral_atom.GetChiralTag() == in_tag, (
        f"Expected tag {in_tag} preserved, got {chiral_atom.GetChiralTag()}"
    )


def test_chirality_blanked_when_neighbor_elements_change():
    """Touched atom with neighbour element multiset CHANGED → UNSPECIFIED.
    Replace Br with N → element set goes from {C, C, Br, H} to {C, C, N, H}."""
    smi = "[CH3:1][C@@H:2]([Br:3])[CH3:4].[NH3:5]"
    mol = Chem.MolFromSmiles(smi)
    c2 = _atom_by_map(mol, 2).GetIdx()
    br3 = _atom_by_map(mol, 3).GetIdx()
    n5 = _atom_by_map(mol, 5).GetIdx()
    d = _delta(mol.GetNumAtoms(), {(c2, br3): -1, (c2, n5): +1})
    out = apply_delta(mol, d, apply_charge_heuristic=False)
    chiral_atom = _atom_by_map(out, 2)
    assert chiral_atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED


def test_bond_stereo_blanked_when_double_bond_touched():
    """E/Z on a double bond touched by Δ must be blanked."""
    # (E)-but-2-ene + Δ that flips the C=C to single
    mol = Chem.MolFromSmiles("[CH3:1]/[CH:2]=[CH:3]/[CH3:4]")
    i = _atom_by_map(mol, 2).GetIdx()
    j = _atom_by_map(mol, 3).GetIdx()
    d = _delta(mol.GetNumAtoms(), {(i, j): -1})  # 2 → 1
    out = apply_delta(mol, d, apply_charge_heuristic=False)
    bond = out.GetBondBetweenAtoms(i, j)
    assert bond.GetStereo() == Chem.BondStereo.STEREONONE
    # And the canonical SMILES must not contain / or \
    canon = _canon(out)
    assert "/" not in canon and "\\" not in canon


# --------------------------------------------------------- charge_delta


def test_explicit_charge_delta_overrides_heuristic():
    """When charge_delta is provided, it sets the charges regardless of
    apply_charge_heuristic."""
    # Ammonia + proton → ammonium. Δ on bonds = 0 (no bond change here;
    # represents a pure charge transfer step that the diagonal-Δ head will
    # eventually emit). charge_delta says: +1 on N.
    mol = Chem.MolFromSmiles("[NH3:1]")
    n = mol.GetNumAtoms()
    d = torch.zeros((n, n), dtype=torch.long)
    cd = torch.tensor([1], dtype=torch.int64)
    out = apply_delta(mol, d, charge_delta=cd, apply_charge_heuristic=True)
    atom = _atom_by_map(out, 1)
    assert atom.GetFormalCharge() == 1


def test_charge_heuristic_applies_when_no_explicit_delta():
    """With apply_charge_heuristic=True and no charge_delta, the row-sum
    heuristic sets fc[i] = fc_in[i] + (-Σⱼ Δ[i,j])."""
    # Form a bond between two neutral atoms → row sum = +1 each → fc = -1 each.
    # Use methide + methide (artificial but unambiguous): just demonstrate the
    # heuristic fires.
    mol = Chem.MolFromSmiles("[CH3:1][CH3:2]")  # ethane
    n = mol.GetNumAtoms()
    # Break the C-C: Δ = -1. Row sum for atom 0 = -1, so heuristic says fc = +1.
    d = _delta(n, {(0, 1): -1})
    out = apply_delta(mol, d, apply_charge_heuristic=True)
    # After breaking, both atoms should carry the heuristic charge.
    assert _atom_by_map(out, 1).GetFormalCharge() == 1
    assert _atom_by_map(out, 2).GetFormalCharge() == 1


# --------------------------------------------------- multi-step chaining


def test_two_step_delta_preserves_atom_maps_and_indices():
    """The sequential model relies on applying multiple Δ steps in sequence,
    with atom-map numbers (and their index ordering) stable across steps."""
    # Step 1: ethanol → vinyl alcohol (dehydrogenation, +1 between C1 and C2)
    # Step 2: vinyl alcohol → acetaldehyde-like (tautomerise: not a clean
    # bond-Δ; instead use a synthetic step that just removes the OH).
    r = Chem.MolFromSmiles("[CH3:1][CH2:2][OH:3]")
    p1 = Chem.MolFromSmiles("[CH2:1]=[CH:2][OH:3]")
    d1 = DeltaMatrixGenerator.delta_from_reactants_products(r, p1)
    after_step1 = apply_delta(r, d1, apply_charge_heuristic=False)

    # Atom maps must survive step 1.
    maps_after_step1 = sorted(a.GetAtomMapNum() for a in after_step1.GetAtoms())
    assert maps_after_step1 == [1, 2, 3]

    # Step 2: break C2-O3 to get vinyl + OH fragments.
    i = _atom_by_map(after_step1, 2).GetIdx()
    j = _atom_by_map(after_step1, 3).GetIdx()
    n = after_step1.GetNumAtoms()
    d2 = _delta(n, {(i, j): -1})
    after_step2 = apply_delta(after_step1, d2, apply_charge_heuristic=False)

    # Atom maps must survive step 2 too.
    maps_after_step2 = sorted(a.GetAtomMapNum() for a in after_step2.GetAtoms())
    assert maps_after_step2 == [1, 2, 3]
    # No bond between maps 2 and 3 anymore.
    a2 = _atom_by_map(after_step2, 2)
    a3 = _atom_by_map(after_step2, 3)
    assert after_step2.GetBondBetweenAtoms(a2.GetIdx(), a3.GetIdx()) is None


def test_two_step_delta_chirality_propagation():
    """Chirality on an atom that's untouched across both steps must survive."""
    # Chiral atom at map 2 (a Br carbon). Step 1: form an unrelated C-O on
    # different atoms. Step 2: break that O-H. Atom 2 is never touched.
    smi = "[CH3:1][C@H:2]([Br:3])[CH2:4][CH3:5].[CH3:6][OH:7].[CH3:8][CH3:9]"
    r = Chem.MolFromSmiles(smi)
    i6 = _atom_by_map(r, 6).GetIdx()
    i8 = _atom_by_map(r, 8).GetIdx()
    d1 = _delta(r.GetNumAtoms(), {(i6, i8): +1})  # form C-C (6-8)
    step1 = apply_delta(r, d1, apply_charge_heuristic=False)
    assert _atom_by_map(step1, 2).GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED

    i6b = _atom_by_map(step1, 6).GetIdx()
    i7b = _atom_by_map(step1, 7).GetIdx()
    d2 = _delta(step1.GetNumAtoms(), {(i6b, i7b): -1})  # break methanol O-H bond
    step2 = apply_delta(step1, d2, apply_charge_heuristic=False)
    assert _atom_by_map(step2, 2).GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED


# ----------------------------------------------- additional edge cases


def test_new_stereocentre_emerges_as_unspecified():
    """sp²→sp³ where R had no chirality must NOT invent a stereo tag.
    Example: protonate a ketone carbon (C=O reduced to C-OH) — the carbon
    becomes a new stereocentre but the patent wouldn't fix R/S from a Δ alone.
    """
    # acetaldehyde-ish: [CH3:1][CH:2]=[O:3] + [H:4] → secondary alcohol carbon
    # Use [CH3:1][C:2](=[O:3])[CH3:4] (acetone) and add CH3 via map 5.
    mol = Chem.MolFromSmiles("[CH3:1][C:2](=[O:3])[CH3:4].[CH4:5]")
    c2 = _atom_by_map(mol, 2).GetIdx()
    o3 = _atom_by_map(mol, 3).GetIdx()
    c5 = _atom_by_map(mol, 5).GetIdx()
    # Reduce C=O to C-O, form new C-C → atom 2 has 4 different substituents
    d = _delta(mol.GetNumAtoms(), {(c2, o3): -1, (c2, c5): +1})
    out = apply_delta(mol, d, apply_charge_heuristic=False)
    chiral_atom = _atom_by_map(out, 2)
    # Should be UNSPECIFIED — bond-Δ alone cannot decide R/S on a newly-formed
    # stereocentre. (The patent dataset's annotation gap policy: emit nothing.)
    assert chiral_atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED


def test_charge_delta_with_heuristic_disabled():
    """Explicit charge_delta works even when apply_charge_heuristic=False."""
    mol = Chem.MolFromSmiles("[NH3:1]")
    n = mol.GetNumAtoms()
    d = torch.zeros((n, n), dtype=torch.long)
    cd = torch.tensor([1], dtype=torch.int64)
    out = apply_delta(mol, d, charge_delta=cd, apply_charge_heuristic=False)
    assert _atom_by_map(out, 1).GetFormalCharge() == 1


def test_zero_delta_is_noop():
    """A Δ of all zeros should return an identical molecule (atom maps,
    chirality, charges all preserved)."""
    smi = "[CH3:1][C@H:2]([Br:3])[CH2:4][CH3:5]"
    mol = Chem.MolFromSmiles(smi)
    in_tag = _atom_by_map(mol, 2).GetChiralTag()
    n = mol.GetNumAtoms()
    d = torch.zeros((n, n), dtype=torch.long)
    out = apply_delta(mol, d, apply_charge_heuristic=False)
    assert _atom_by_map(out, 2).GetChiralTag() == in_tag
    assert _canon(out) == _canon(mol)
