"""Smoke tests for the chain-rule arrow grouper.

The validity-check grouper was removed in cleanup batch B per ADR-0001.
"""

from __future__ import annotations

from rdkit import Chem

from mech_uspto.data.arrow_parser import (
    group_arrows_into_steps,
    parse_arrows,
)
from mech_uspto.data.featurization import align_atoms


def _mol(smi: str) -> tuple[Chem.Mol, dict[int, int]]:
    mol = align_atoms(Chem.MolFromSmiles(smi))
    m2i = {a.GetAtomMapNum(): a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum() > 0}
    return mol, m2i


def test_chain_rule_returns_all_arrow_indices():
    """Every arrow must land in exactly one group, in order."""
    arrows = parse_arrows("[(29, 1), ([1, 2], 2), (2, [2, 1]), ([1, 101], 101)]")
    groups = group_arrows_into_steps(arrows)
    flat = [i for g in groups for i in g]
    assert flat == list(range(len(arrows)))


def test_chain_rule_groups_acyl_substitution():
    """Acyl-Cl + NH3: chain rule fuses the 4 arrows into 1 concerted step."""
    arrows = parse_arrows("[(29, 1), ([1, 2], 2), (2, [2, 1]), ([1, 101], 101)]")
    groups = group_arrows_into_steps(arrows)
    assert groups == [[0, 1, 2, 3]]


def test_chain_rule_runs_on_sn2():
    """Smoke: chain rule completes on a 2-arrow SN2 without crashing."""
    arrows = parse_arrows("[(11, 1), ([1, 101], 101)]")
    groups = group_arrows_into_steps(arrows)
    assert [i for g in groups for i in g] == [0, 1]
