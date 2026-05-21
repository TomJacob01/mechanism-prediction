"""Regression tests for ``DeltaMatrixGenerator``.

Focus: ``delta_from_mechanism_arrows`` correctness across the three arrow
shapes (break / form / shift). The shift case had a latent bug
(`transformations.py` shift branch) where ``"u,v=u,w"`` wrote ``delta[v, w] = +1``
(a phantom non-event pair) and never recorded the actual break (u, v) or
form (u, w). This file covers all four cases including the previously-bugged
phantom-cell assertion so the regression cannot recur silently.
"""

import torch
from rdkit import Chem

from mech_uspto.data.transformations import DeltaMatrixGenerator


def _linear_chain(n: int) -> Chem.Mol:
    """Return a single-bond linear chain of ``n`` carbons with atom maps 1..n."""
    smiles = "-".join(f"[CH3:{i + 1}]" if i in (0, n - 1) else f"[CH2:{i + 1}]" for i in range(n))
    smiles = smiles.replace("-", "")  # SMILES separator is implicit
    # Build via RWMol so atom-map ordering is unambiguous.
    rw = Chem.RWMol()
    for i in range(n):
        atom = Chem.Atom(6)
        atom.SetAtomMapNum(i + 1)
        rw.AddAtom(atom)
    for i in range(n - 1):
        rw.AddBond(i, i + 1, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def test_break_arrow_records_negative_delta() -> None:
    """``"1,2="`` breaks bond between atom-map 1 and 2."""
    mol = _linear_chain(4)
    delta = DeltaMatrixGenerator.delta_from_mechanism_arrows("1,2=", mol)

    expected = torch.zeros((4, 4), dtype=torch.long)
    expected[0, 1] = -1
    expected[1, 0] = -1
    assert torch.equal(delta, expected)


def test_form_arrow_records_positive_delta() -> None:
    """``"=1,3"`` forms bond between atom-map 1 and 3."""
    mol = _linear_chain(4)
    delta = DeltaMatrixGenerator.delta_from_mechanism_arrows("=1,3", mol)

    expected = torch.zeros((4, 4), dtype=torch.long)
    expected[0, 2] = 1
    expected[2, 0] = 1
    assert torch.equal(delta, expected)


def test_shift_arrow_records_break_and_form() -> None:
    """``"1,2=1,3"`` must record BOTH break(1,2)=-1 AND form(1,3)=+1.

    Pre-fix bug: this branch instead wrote ``delta[2, 3] = +1`` (the
    leaving group <-> incoming nucleophile pair, a non-event), and left
    the actual broken bond (1,2) and formed bond (1,3) unrecorded.
    """
    mol = _linear_chain(4)
    delta = DeltaMatrixGenerator.delta_from_mechanism_arrows("1,2=1,3", mol)

    # Indices: atom-map 1 -> idx 0, map 2 -> idx 1, map 3 -> idx 2.
    expected = torch.zeros((4, 4), dtype=torch.long)
    # Break pivot-leaving (1-2):
    expected[0, 1] = -1
    expected[1, 0] = -1
    # Form pivot-incoming (1-3):
    expected[0, 2] = 1
    expected[2, 0] = 1
    assert torch.equal(delta, expected)


def test_shift_arrow_does_not_write_phantom_leaving_incoming_pair() -> None:
    """Explicit guard: the (leaving, incoming) cell must remain zero.

    This is the exact cell that the pre-fix bug touched. Asserting it
    independently means a regression to the old behaviour fails this
    test even if the break/form assertions are loosened later.
    """
    mol = _linear_chain(4)
    delta = DeltaMatrixGenerator.delta_from_mechanism_arrows("1,2=1,3", mol)

    # leaving = atom-map 2 -> idx 1; incoming = atom-map 3 -> idx 2.
    assert delta[1, 2].item() == 0
    assert delta[2, 1].item() == 0


def test_shift_arrow_with_non_first_pivot_atom_is_skipped() -> None:
    """Current parser only handles shifts where ``src_maps[0] == dst_maps[0]``.

    Document the behaviour: if the pivot is encoded in a different
    position (e.g. ``"2,1=3,1"``), the branch is silently skipped \u2014
    delta stays all-zero. This is a known limitation, not the fix
    target, but pinning it makes a future generalisation visible.
    """
    mol = _linear_chain(4)
    delta = DeltaMatrixGenerator.delta_from_mechanism_arrows("2,1=3,1", mol)
    assert torch.equal(delta, torch.zeros((4, 4), dtype=torch.long))


def test_multiple_arrows_in_one_string_are_all_recorded() -> None:
    """Semicolon-separated arrows compose into one delta matrix."""
    mol = _linear_chain(5)
    # Break 1-2, form 4-5 (already bonded so this is a contrived but parseable case).
    delta = DeltaMatrixGenerator.delta_from_mechanism_arrows("1,2=; =4,5", mol)

    expected = torch.zeros((5, 5), dtype=torch.long)
    expected[0, 1] = -1
    expected[1, 0] = -1
    expected[3, 4] = 1
    expected[4, 3] = 1
    assert torch.equal(delta, expected)


def test_empty_arrow_string_returns_zero_delta() -> None:
    mol = _linear_chain(3)
    delta = DeltaMatrixGenerator.delta_from_mechanism_arrows("", mol)
    assert torch.equal(delta, torch.zeros((3, 3), dtype=torch.long))
