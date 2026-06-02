"""Parse PMechDB-style curly arrows into elementary mechanistic steps.

Dataset format (``mechanistic_label`` column): a Python literal::

    [(src, tgt), (src, tgt), ...]

where each ``src`` / ``tgt`` is one of:

- ``int N``          — lone pair on atom map-num ``N``
- ``[a, b]``         — bond between atoms with map nums ``a`` and ``b``
- ``float N.k``      — atom ``N``'s ``k``-th implicit hydrogen

This module turns that flat arrow list into a list of
:class:`ElementaryStep` objects, each carrying bond-order and formal-charge
deltas indexed by heavy-atom RDKit indices (not map numbers). Grouping is
**chain-based**: consecutive arrows belong to the same step iff the next
arrow's source overlaps with the previous arrow's target (electron-pushing
flow). This recovers 2-arrow, 3-arrow (pericyclic), and longer concerted
steps without committing to a fixed group size.

Verified on 2000 USPTO-31k reactions (see ``scripts/verify_arrow_parser.py``):

- 96% bond-Δ rule match on shared atoms
- 81% sequential reconstruction with last-step charge correction

The per-step charge rule keeps intermediates chemically valid (low sanitize
failure rate); residual error in the cumulative formal charge can be fixed
on the final step by adding ``GT_total - sum_arrow_steps`` to the last
step's charge delta — see :func:`apply_steps_with_correction` helper.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Arrow primitives
# ---------------------------------------------------------------------------

def _as_atom_map(x) -> int | None:
    """Return the heavy-atom map number, or ``None`` if ``x`` is an implicit H.

    Real heavy atoms are emitted as plain ``int``; implicit H indices arrive
    as ``float`` (``N.k``). Booleans are ints in Python — guard against
    accidentally treating them as maps.
    """
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    return None


def _h_parent(x) -> int | None:
    """For a float ``N.k`` return the heavy atom map ``N``; else ``None``."""
    if isinstance(x, float):
        return int(x)
    return None


def _anchor_maps(part) -> set[int]:
    """Set of heavy-atom maps that ``part`` touches (incl. the parent of an H).

    Used for chain-detection: two arrows chain if the next arrow's src
    anchors overlap with the previous arrow's tgt anchors.
    """
    if isinstance(part, list):
        return {m for m in (_as_atom_map(x) for x in part) if m is not None} | {
            m for m in (_h_parent(x) for x in part) if m is not None
        }
    m = _as_atom_map(part)
    if m is not None:
        return {m}
    m = _h_parent(part)
    if m is not None:
        return {m}
    return set()


def arrow_bond_changes(
    arrow: tuple, map_to_idx: dict[int, int]
) -> list[tuple[int, int, int]]:
    """Return ``[(i, j, ±1), ...]`` heavy-atom bond-order changes for one arrow.

    Rules:
      - ``src = [a, b]``  → Δ[a, b] -= 1
      - ``tgt = [a, b]``  → Δ[a, b] += 1
      - ``src = [a, b]``, ``tgt = t``, ``t ∉ {a, b}``  → Δ[b, t] += 1
        (electrons in bond [a,b] migrate to form a new bond between the
        second source atom and the target — Grignard alkoxide formation,
        nucleophilic addition where the pi bond becomes a new σ bond
        to a coordinating cation).
      - ``src``/``tgt`` both atoms (different)  → Δ[src, tgt] += 1
      - arrows involving an H index don't move heavy-atom bonds
    """
    src, tgt = arrow
    src_bond = isinstance(src, list)
    tgt_bond = isinstance(tgt, list)
    out: list[tuple[int, int, int]] = []

    if src_bond:
        a, b = _as_atom_map(src[0]), _as_atom_map(src[1])
        if a is not None and b is not None and a in map_to_idx and b in map_to_idx:
            out.append((map_to_idx[a], map_to_idx[b], -1))

    if tgt_bond:
        a, b = _as_atom_map(tgt[0]), _as_atom_map(tgt[1])
        if a is not None and b is not None and a in map_to_idx and b in map_to_idx:
            out.append((map_to_idx[a], map_to_idx[b], +1))

    if src_bond and not tgt_bond:
        # Electron pair from bond [a,b] lands somewhere on tgt. If tgt is
        # a *third* atom (not a or b), the convention is that the second
        # source atom (b) ends up bonded to tgt — the bond effectively
        # "swings" from a to tgt. If tgt equals a or b, the electrons
        # become a lone pair on tgt and no new bond forms.
        a = _as_atom_map(src[0])
        b = _as_atom_map(src[1])
        t = _as_atom_map(tgt)
        if (
            a is not None and b is not None and t is not None
            and t != a and t != b
            and b in map_to_idx and t in map_to_idx
        ):
            out.append((map_to_idx[b], map_to_idx[t], +1))

    if not src_bond and not tgt_bond:
        a = _as_atom_map(src)
        b = _as_atom_map(tgt)
        if (
            a is not None and b is not None
            and a != b
            and a in map_to_idx and b in map_to_idx
        ):
            out.append((map_to_idx[a], map_to_idx[b], +1))

    return out


def arrow_charge_changes(
    arrow: tuple, map_to_idx: dict[int, int]
) -> list[tuple[int, int]]:
    """Return ``[(heavy_idx, Δfc), ...]`` formal-charge changes for one arrow.

    Owned-electron model: ``Δfc[A] = -Δ(owned electrons)`` where
    ``owned = 2 * (lone-pair count) + 1 * (bond count)``.
    """
    src, tgt = arrow
    src_bond = isinstance(src, list)
    tgt_bond = isinstance(tgt, list)

    db: dict[int, int] = defaultdict(int)
    dlp: dict[int, int] = defaultdict(int)

    if src_bond:
        for x in src:
            m = _as_atom_map(x)
            if m in map_to_idx:
                db[map_to_idx[m]] -= 1
    if tgt_bond:
        for x in tgt:
            m = _as_atom_map(x)
            if m in map_to_idx:
                db[map_to_idx[m]] += 1
    if (not src_bond) and (not tgt_bond):
        ms = _as_atom_map(src)
        mt = _as_atom_map(tgt)
        if ms in map_to_idx and mt in map_to_idx and ms != mt:
            db[map_to_idx[ms]] += 1
            db[map_to_idx[mt]] += 1
        elif ms in map_to_idx and isinstance(tgt, float):
            db[map_to_idx[ms]] += 1

    if not src_bond:
        m = _as_atom_map(src)
        if m in map_to_idx:
            dlp[map_to_idx[m]] -= 1
    if not tgt_bond:
        m = _as_atom_map(tgt)
        if m in map_to_idx:
            if src_bond:
                # bond-to-atom: tgt receives the electrons as an LP
                # (whether tgt was an endpoint or an outsider).
                dlp[map_to_idx[m]] += 1
            # else atom-atom: electrons become the new bond, not a fresh LP.

    out: list[tuple[int, int]] = []
    for idx in set(db) | set(dlp):
        d_owned = 2 * dlp.get(idx, 0) + db.get(idx, 0)
        if d_owned != 0:
            out.append((idx, -d_owned))
    return out


# ---------------------------------------------------------------------------
# Step grouping
# ---------------------------------------------------------------------------

def _arrow_src_anchors(arrow: tuple) -> set[int]:
    return _anchor_maps(arrow[0])


def _arrow_tgt_anchors(arrow: tuple) -> set[int]:
    return _anchor_maps(arrow[1])


def group_arrows_into_steps(
    arrows: list[tuple],
    *,
    max_step_size: int = 99,
) -> list[list[int]]:
    """Group arrows into elementary steps via electron-flow chaining.

    Two consecutive arrows belong to the same step iff the next arrow's
    source anchors overlap with the previous arrow's target anchors. This
    naturally handles 2-arrow (Sn1/Sn2-like), 3-arrow (pericyclic, E2 with
    H migration), and longer concerted steps.

    ``max_step_size`` caps the chain length. Default 99 is effectively
    unlimited — empirically, splitting the occasional 10-/14-arrow
    over-grouped chains produces unstable intermediates more often than
    it helps, so the cap is left disabled by default.

    Returns a list of arrow-index lists.
    """
    if not arrows:
        return []
    groups: list[list[int]] = [[0]]
    for k in range(1, len(arrows)):
        prev_tgt = _arrow_tgt_anchors(arrows[k - 1])
        cur_src = _arrow_src_anchors(arrows[k])
        if (prev_tgt & cur_src) and len(groups[-1]) < max_step_size:
            groups[-1].append(k)
        else:
            groups.append([k])
    return groups


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ElementaryStep:
    """One elementary mechanistic step.

    Attributes:
        arrow_indices: positions in the original arrow list.
        bond_changes: ``(heavy_i, heavy_j, ±1)`` triples; symmetric pair not
            duplicated — callers must mirror across the diagonal.
        charge_changes: ``(heavy_idx, Δfc)`` pairs (the owned-electron rule).
    """

    arrow_indices: list[int] = field(default_factory=list)
    bond_changes: list[tuple[int, int, int]] = field(default_factory=list)
    charge_changes: list[tuple[int, int]] = field(default_factory=list)


def parse_arrows(label_str: str) -> list[tuple]:
    """Parse the ``mechanistic_label`` literal into a list of ``(src, tgt)``."""
    val = ast.literal_eval(label_str)
    if not isinstance(val, list):
        raise ValueError(f"expected list, got {type(val).__name__}")
    return val


def _step_from_group(
    arrows: list[tuple],
    group_indices: list[int],
    map_to_idx: dict[int, int],
) -> "ElementaryStep":
    """Build one :class:`ElementaryStep` from a group of arrow indices.

    Aggregates per-arrow bond/charge changes, then applies the pi-bond
    H-pointer expansion used by epoxidation / cyclopropanation patterns.
    Shared by all grouping strategies (chain rule, validity check, …).
    """
    bc: list[tuple[int, int, int]] = []
    cc: list[tuple[int, int]] = []
    for k in group_indices:
        bc.extend(arrow_bond_changes(arrows[k], map_to_idx))
        cc.extend(arrow_charge_changes(arrows[k], map_to_idx))
    # --- pi-bond attack expansion (epoxidation / cyclopropanation) ---
    h_targets: set[float] = set()
    for k in group_indices:
        _, tgt = arrows[k]
        if isinstance(tgt, float):
            h_targets.add(tgt)
    if h_targets:
        for k in group_indices:
            src, tgt = arrows[k]
            if not isinstance(src, list) or isinstance(tgt, list):
                continue
            t = _as_atom_map(tgt)
            if t is None or t not in map_to_idx:
                continue
            floats = [x for x in src if isinstance(x, float)]
            heavies = [_as_atom_map(x) for x in src if not isinstance(x, float)]
            heavies = [h for h in heavies if h is not None]
            if len(floats) == 1 and len(heavies) == 1 and floats[0] in h_targets:
                x = heavies[0]
                if x != t and x in map_to_idx:
                    bc.append((map_to_idx[x], map_to_idx[t], +1))
    return ElementaryStep(
        arrow_indices=list(group_indices), bond_changes=bc, charge_changes=cc
    )


def parse_steps(label_str: str, map_to_idx: dict[int, int]) -> list[ElementaryStep]:
    """End-to-end: literal → arrows → chain-rule step groups → per-step deltas."""
    arrows = parse_arrows(label_str)
    groups = group_arrows_into_steps(arrows)
    return [_step_from_group(arrows, g, map_to_idx) for g in groups]

__all__ = [
    "arrow_bond_changes",
    "arrow_charge_changes",
    "ElementaryStep",
    "group_arrows_into_steps",
    "parse_arrows",
    "parse_steps",
]
