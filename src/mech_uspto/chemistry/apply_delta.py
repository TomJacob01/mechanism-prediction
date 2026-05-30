"""Apply a Δ bond-order matrix to an RDKit ``Mol`` to obtain a new molecule.

Design contract (see ``docs/rollout_design.md`` §6.1-6.7):

- Input: ``mol`` with atom-map numbers and a square ``delta`` tensor whose
  rows/cols are indexed by ``mol.GetAtoms()`` order.
- Diagonal of ``delta`` is **ignored** (the dataset's R→P diff is charge-neutral
  on shared atoms — see ``scripts/charge_diagnostic.py`` results: 0.05% of
  atoms change charge; deferred to a future "diagonal-Δ" extension).
- For each upper-triangular off-diagonal non-zero entry ``delta[i,j]``:
  ``b_new = b_current + delta[i,j]``. Remove the bond if ``b_new == 0``,
  add if ``b_current == 0``, change order otherwise. Reject ``b_new < 0`` or
  ``b_new > 3`` (no quadruple bonds, no negative orders).
- Formal charges follow the row-sum heuristic ``Δq[i] = −Σⱼ delta[i,j]``
  applied to the **off-diagonal** sum only. For overall-neutral R→P this
  is ≈ 0 for shared atoms.
- Sanitize the result; raise ``ApplyDeltaError`` with a structured ``reason``
  field on failure so callers can implement the §6.7 sanitize-fail policy
  (``skip_step`` / ``terminate`` / ``force_allow``).
- Atom-map preservation: ``GetAtomMapNum()`` is preserved across the edit;
  the post-sanitize result is asserted to have unique non-zero maps on
  atoms that had them in the input.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from rdkit import Chem


_BOND_ORDER_TO_TYPE = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
}


@dataclass
class ApplyDeltaError(Exception):
    """Structured failure from :func:`apply_delta`.

    ``reason`` is one of: ``"invalid_order"``, ``"sanitize_failed"``,
    ``"atom_map_collision"``, ``"shape_mismatch"``.
    """

    reason: str
    message: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"[{self.reason}] {self.message}"


def apply_delta(
    mol: Chem.Mol,
    delta: torch.Tensor,
    *,
    charge_delta: torch.Tensor | None = None,
    apply_charge_heuristic: bool = True,
) -> Chem.Mol:
    """Return a new ``Mol`` with ``delta`` applied to ``mol``'s bond orders.

    Args:
        mol: Input molecule. Atom-map numbers are preserved on output.
        delta: Square tensor (N, N) where N >= mol.GetNumAtoms(). Off-diagonal
            entries are bond-order deltas; diagonal entries are ignored.
        charge_delta: Optional 1-D tensor of per-atom formal-charge deltas
            (length >= mol.GetNumAtoms()). When provided, ``charge_delta[i]``
            is *added* to atom ``i``'s formal charge before sanitize, and the
            row-sum heuristic is suppressed regardless of
            ``apply_charge_heuristic``. Used by the e2e verifier with
            ground-truth Δq computed from R vs P, and by the future
            diagonal-Δ head at inference time.
        apply_charge_heuristic: If True and ``charge_delta is None``, set each
            atom's formal charge to ``fc_in[i] + (−Σⱼ delta_off_diag[i,j])``.
            Ignored when ``charge_delta`` is supplied.

    Raises:
        ApplyDeltaError: on shape mismatch, invalid resulting bond order,
            atom-map collision post-sanitize, or RDKit sanitize failure.
    """
    if delta.ndim != 2 or delta.shape[0] != delta.shape[1]:
        raise ApplyDeltaError(
            "shape_mismatch",
            f"delta must be square 2-D, got shape {tuple(delta.shape)}",
        )
    n_mol = mol.GetNumAtoms()
    if delta.shape[0] < n_mol:
        raise ApplyDeltaError(
            "shape_mismatch",
            f"delta size {delta.shape[0]} < mol.GetNumAtoms() {n_mol}",
        )

    rw = Chem.RWMol(mol)

    # Snapshot the input atom-map set (we assert preservation post-sanitize).
    input_maps = {a.GetIdx(): a.GetAtomMapNum() for a in rw.GetAtoms()}

    # Snapshot chiral tags so we can restore them on stereocenters whose
    # neighbour set is unchanged by the Δ (i.e. atoms NOT in ``touched_atoms``).
    # RDKit's default sanitize (SANITIZE_CLEANUPCHIRALITY) and the bond-list
    # mutations during RWMol surgery can otherwise flip or drop a tag whose
    # local environment didn't actually change. Featurization encodes R/S
    # chirality (constants.ALLOWED_CHIRAL), so without this restore step every
    # prediction would silently strip input chirality.
    input_chiral_tags = {a.GetIdx(): a.GetChiralTag() for a in rw.GetAtoms()}

    # Also snapshot the multiset of neighbour atomic numbers per atom. After
    # surgery we use this to decide whether a touched atom's chirality tag
    # should be PRESERVED (element multiset unchanged → CIP priorities at
    # that centre likely still resolve the same way; SMILES ``@``/``@@``
    # parity is invariant under bond-order changes that don't add/remove a
    # substituent element) or BLANKED (element multiset changed → CIP
    # priorities may have reshuffled and the original tag is unreliable).
    # This recovers chirality across reactions like Mitsunobu, where the
    # ``[C@@H]`` tag persists but the CIP code flips because the new
    # substituent (OAr) outranks the old (OH) — sanitize handles the flip
    # automatically when we preserve the parity tag.
    input_neighbor_elems = {
        a.GetIdx(): sorted(n.GetAtomicNum() for n in a.GetNeighbors())
        for a in rw.GetAtoms()
    }

    # Walk strictly upper-triangular off-diagonal entries.
    # Round to nearest int — accept float tensors from model outputs.
    delta_int = delta.round().to(torch.int64)

    # Atoms whose bond environment changes need their implicit-H bookkeeping
    # recomputed during sanitize. SMILES like ``[CH3:1]`` set NoImplicit=True
    # with a fixed H count, which makes sanitize fail after a bond-order edit
    # unless we hand control of H count back to RDKit.
    touched_atoms: set[int] = set()

    for i in range(n_mol):
        for j in range(i + 1, n_mol):
            d = int(delta_int[i, j].item())
            if d == 0:
                continue

            touched_atoms.add(i)
            touched_atoms.add(j)

            bond = rw.GetBondBetweenAtoms(i, j)
            cur_order = int(bond.GetBondTypeAsDouble()) if bond is not None else 0
            new_order = cur_order + d

            if new_order < 0 or new_order > 3:
                raise ApplyDeltaError(
                    "invalid_order",
                    f"bond ({i},{j}): {cur_order} + {d} = {new_order} ∉ [0,3]",
                )

            if new_order == 0:
                # bond exists (cur_order > 0) by construction here.
                rw.RemoveBond(i, j)
            elif cur_order == 0:
                rw.AddBond(i, j, _BOND_ORDER_TO_TYPE[new_order])
            else:
                bond.SetBondType(_BOND_ORDER_TO_TYPE[new_order])

    if apply_charge_heuristic and charge_delta is None:
        # Off-diagonal row sum per atom (lower + upper since delta is
        # expected symmetric; use the full matrix and zero the diagonal).
        off_diag = delta_int[:n_mol, :n_mol].clone()
        off_diag.fill_diagonal_(0)
        dq = (-off_diag.sum(dim=1)).tolist()
        for i, atom in enumerate(rw.GetAtoms()):
            shift = int(dq[i])
            if shift != 0:
                atom.SetFormalCharge(atom.GetFormalCharge() + shift)

    if charge_delta is not None:
        # Explicit per-atom Δq (overrides the heuristic). The dataset is
        # 99.95% Δq=0 on shared atoms (charge_diagnostic.py) but the residual
        # 0.05% are real and break sanitize otherwise — e.g. azide
        # [N⁻]=[N⁺]=[N⁻] adding to an electrophile neutralises to [N⁻]−N=N
        # (Δq on the central N goes -1, on the terminal N goes +1).
        if charge_delta.ndim != 1 or charge_delta.shape[0] < n_mol:
            raise ApplyDeltaError(
                "shape_mismatch",
                f"charge_delta must be 1-D length >= {n_mol}, "
                f"got shape {tuple(charge_delta.shape)}",
            )
        dq_int = charge_delta.round().to(torch.int64).tolist()
        for i, atom in enumerate(rw.GetAtoms()):
            shift = int(dq_int[i])
            if shift != 0:
                atom.SetFormalCharge(atom.GetFormalCharge() + shift)
                touched_atoms.add(i)

    # Hand H bookkeeping back to RDKit for any atom we touched, so sanitize
    # can re-derive implicit H counts from the new valence + formal charge.
    # Also clear radical-electron count: the dataset is closed-shell (the
    # charge_diagnostic confirms 0 radical-changing reactions), so the
    # post-edit valence should be filled with Hs rather than unpaired electrons.
    for idx in touched_atoms:
        atom = rw.GetAtomWithIdx(idx)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(False)
        atom.SetNumRadicalElectrons(0)

    try:
        Chem.SanitizeMol(rw)
    except Exception as first_err:
        # Retry path for the "aromatic ring composition changed" case:
        # touched atoms (and their incident bonds) may carry stale aromatic
        # flags from R's ring system that no longer exists in P. Clearing
        # them lets sanitize re-perceive aromaticity from the new graph.
        # We do this only on failure to avoid breaking untouched aromatic
        # rings that merely share a bond with a touched atom.
        for idx in touched_atoms:
            atom = rw.GetAtomWithIdx(idx)
            atom.SetIsAromatic(False)
        for bond in rw.GetBonds():
            if (
                bond.GetBeginAtomIdx() in touched_atoms
                and bond.GetEndAtomIdx() in touched_atoms
            ):
                bond.SetIsAromatic(False)
        try:
            Chem.SanitizeMol(rw)
        except Exception as e:
            raise ApplyDeltaError("sanitize_failed", str(e)) from e

    # Stereo handling, post-sanitize. Three cases per atom:
    #   1. Untouched stereocentre → restore the snapshot tag verbatim
    #      (sanitize / RWMol bookkeeping may have flipped or dropped it).
    #   2. Touched stereocentre with neighbour ELEMENT multiset unchanged
    #      → restore the snapshot tag. The CIP descriptor (R/S) is recomputed
    #      by RDKit from the new neighbour priorities, so the tag transfers
    #      cleanly: Mitsunobu-style inversion is captured automatically when
    #      a new substituent of the same element outranks the old one.
    #   3. Touched stereocentre with neighbour element multiset CHANGED
    #      → blank to UNSPECIFIED. The bond-order Δ alone cannot determine
    #      stereo at a centre whose substituent set has fundamentally
    #      changed (e.g. sp²→sp³ where a new substituent is added).
    for idx, original_tag in input_chiral_tags.items():
        if original_tag == Chem.ChiralType.CHI_UNSPECIFIED:
            continue
        atom = rw.GetAtomWithIdx(idx)
        if idx not in touched_atoms:
            atom.SetChiralTag(original_tag)
            continue
        post_elems = sorted(n.GetAtomicNum() for n in atom.GetNeighbors())
        if post_elems == input_neighbor_elems[idx]:
            atom.SetChiralTag(original_tag)
        else:
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)

    # Bond stereo on bonds incident to a touched atom: blank.
    # E/Z geometry on a newly-formed (or order-changed) double bond cannot be
    # determined from bond-order Δ alone — that lives in 3D transition-state
    # geometry. Leaving sanitize's best guess produces a confidently-wrong
    # label; UNSPECIFIED is the honest answer.
    for bond in rw.GetBonds():
        if (
            bond.GetBeginAtomIdx() in touched_atoms
            or bond.GetEndAtomIdx() in touched_atoms
        ):
            if bond.GetStereo() != Chem.BondStereo.STEREONONE:
                bond.SetStereo(Chem.BondStereo.STEREONONE)

    # Atom-map preservation check (per design §6.1).
    out_maps = [a.GetAtomMapNum() for a in rw.GetAtoms()]
    nonzero = [m for m in out_maps if m > 0]
    if len(nonzero) != len(set(nonzero)):
        raise ApplyDeltaError(
            "atom_map_collision",
            f"duplicate atom-map numbers after sanitize: {sorted(nonzero)}",
        )
    expected_nonzero = {m for m in input_maps.values() if m > 0}
    actual_nonzero = set(nonzero)
    if not expected_nonzero.issubset(actual_nonzero):
        missing = expected_nonzero - actual_nonzero
        raise ApplyDeltaError(
            "atom_map_collision",
            f"atom-map numbers dropped during sanitize: {sorted(missing)}",
        )

    return rw.GetMol()
