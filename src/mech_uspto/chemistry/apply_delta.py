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

# Standard neutral valences for heteroatoms we can rescue by raising charge.
# Excess valence by exactly +1 ⇒ add +1 formal charge (oxonium / iminium /
# sulfonium / phosphonium). S and P have legitimate hypervalent states
# (sulfone S=6, phosphate P=5); we only rescue the one-electron-shy case.
_HETERO_STD_VALENCE = {"O": 2, "N": 3, "S": 2, "P": 3, "C": 4}


def _h_overvalence_rescue(rw: Chem.RWMol, original: Chem.Mol, err_msg: str,
                          max_passes: int = 4) -> bool:
    """Neutralise an over-valent H atom and retry sanitize.

    Hydride transfer arrows like ``([1, 101], 301)`` form a new H-X bond
    from a starting ``[H-]`` (hydride, fc=-1, valence 0). RDKit's permitted
    valence for H- is 0, so the post-surgery H with 1 bond raises
    "Explicit valence for atom # N H greater than permitted". The arrow's
    intent is hydride attack: the H becomes neutral after binding. Set
    fc=0 and retry. Also handles H+ (fc=+1) acquiring a bond on
    deprotonation reversals.
    """
    import re
    pat = re.compile(r"atom # (\d+) H greater than permitted")
    msg = err_msg
    for _ in range(max_passes):
        m = pat.search(msg)
        if not m:
            return False
        idx = int(m.group(1))
        if idx >= rw.GetNumAtoms():
            return False
        atom = rw.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() != 1:
            return False
        fc = atom.GetFormalCharge()
        deg = atom.GetDegree()
        # H with `deg` bonds is only permitted at fc = 1 - deg.
        # (deg=1 -> fc=0, deg=0 -> fc=+/-1, deg=2 -> fc=-1 which is rare).
        target_fc = 1 - deg
        if fc == target_fc:
            return False
        atom.SetFormalCharge(target_fc)
        try:
            Chem.SanitizeMol(rw)
            return True
        except Exception as e:
            msg = str(e)
    return False


def _hypervalent_rescue(rw: Chem.RWMol, err_msg: str, max_passes: int = 8) -> bool:
    """Parse a SpecificValenceException, set the offending O/N/S/P's formal
    charge to absorb the excess valence, retry sanitize. Loops to handle
    multiple offenders.

    For element X with neutral valence ``std`` and observed valence ``actual``,
    the permitted valence at charge ``q`` is ``std + q`` (oxonium / ammonium
    pattern). We set ``q = actual - std`` so the resulting valence is exactly
    permitted. This covers both "neutral atom needs +1" (oxonium, iminium)
    and "negative atom needs to neutralize" (e.g. an N⁻ that just gained a
    bond and should now read as neutral N).

    Returns True if sanitize eventually succeeds, False otherwise.
    """
    import re
    pat = re.compile(r"atom # (\d+) (\w+), (\d+)")
    msg = err_msg
    for _ in range(max_passes):
        m = pat.search(msg)
        if not m:
            return False
        idx = int(m.group(1))
        elem = m.group(2)
        actual = int(m.group(3))
        std = _HETERO_STD_VALENCE.get(elem)
        if std is None:
            return False
        needed_fc = actual - std
        # Sanity bounds: only rescue charges in [-1, +1]. Anything larger
        # implies a fundamentally bad arrow, not a chargeable hypervalence.
        if needed_fc < -1 or needed_fc > 1:
            return False
        atom = rw.GetAtomWithIdx(idx)
        if atom.GetFormalCharge() == needed_fc:
            # Sanitize asked for this exact charge; can't make progress.
            return False
        atom.SetFormalCharge(needed_fc)
        try:
            Chem.SanitizeMol(rw)
            return True
        except Exception as e:
            msg = str(e)
    return False


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

    # Kekulize before any bond surgery so aromatic systems become explicit
    # single/double bonds with aromatic flags cleared. Without this, mutating
    # one bond inside an aromatic ring leaves the *other* ring atoms still
    # flagged aromatic; sanitize then either fails outright or silently
    # produces a half-aromatic Frankenstein whose SMILES contains `:[CH]:`
    # placeholders. After surgery, sanitize re-perceives aromaticity cleanly
    # from the new bond graph. Failure is non-fatal — fall through to the
    # existing retry ladder if kekulize can't resolve the input.
    try:
        Chem.Kekulize(rw, clearAromaticFlags=True)
    except Exception:
        pass

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
        # breaking even one bond of an aromatic ring leaves the *other*
        # ring atoms (which were never touched by Δ) still flagged
        # aromatic, so RDKit complains "non-ring atom marked aromatic".
        # Empirically (intermediate verifier on USPTO-31k) clearing only
        # touched atoms misses these stranded ring-mates, so on first
        # failure we wipe every aromatic flag and let RDKit re-perceive
        # from scratch. This is safe — aromaticity is fully derivable
        # from the bond graph + atom valences, which sanitize already
        # touches anyway.
        for atom in rw.GetAtoms():
            atom.SetIsAromatic(False)
        for bond in rw.GetBonds():
            bond.SetIsAromatic(False)
        try:
            Chem.SanitizeMol(rw)
        except Exception as e:
            # Third retry: hypervalent-heteroatom rescue. The arrow-derived
            # charge rule misses ~5% of reactions where a lone pair on
            # O/N/P/S donates into a new bond (oxonium / iminium / sulfonium
            # / phosphonium). The row-sum heuristic doesn't help either —
            # for donation the atom *gains* a bond but *loses* electrons, so
            # Δfc = +1, opposite to what -Σrow predicts. Detect the offending
            # atom from the SpecificValenceException, set its formal charge
            # to absorb the excess valence, and retry.
            if _hypervalent_rescue(rw, str(e)):
                pass
            elif _h_overvalence_rescue(rw, mol, str(e)):
                pass
            else:
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
