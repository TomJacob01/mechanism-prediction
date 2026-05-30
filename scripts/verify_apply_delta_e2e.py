"""End-to-end verification of :func:`apply_delta`.

For every reaction in mech-USPTO-31k:

1. Parse the atom-mapped R and P SMILES (no Hs, matching ``delta_from_reactants_products``).
2. Build the ground-truth Δ from R→P with :class:`DeltaMatrixGenerator`.
3. Apply Δ to R via :func:`apply_delta` → predicted molecule.
4. The reactant side contains *leaving groups* (atom-map ≥ 101 in MechFinder
   convention) that are absent from P. After Δ is applied, the bonds joining
   them to productive atoms break and they fall out as disconnected fragments.
   Keep only the fragments containing at least one atom whose map number
   appears in P.
5. Strip atom-map numbers and compare canonical SMILES against P canonical.

Decision gate (matches ``docs/rollout_design.md`` §4 / phase 1.2):
- ≥ 0.90 pass rate → bond-Δ formulation is sound; proceed to model rollout.
- 0.60 – 0.90 → investigate failures, possibly need diagonal-Δ for problem classes.
- < 0.60 → bond-Δ alone insufficient; reconsider formulation.

CLI::

    python scripts/verify_apply_delta_e2e.py [--limit N] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem, RDLogger
from tqdm.auto import tqdm
import torch

from mech_uspto.chemistry import ApplyDeltaError, apply_delta
from mech_uspto.data.featurization import align_atoms
from mech_uspto.data.parser import MechUSPTOParser
from mech_uspto.data.transformations import DeltaMatrixGenerator

RDLogger.DisableLog("rdApp.*")


def _canon_no_maps(mol: Chem.Mol) -> str:
    m = Chem.Mol(mol)
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    return Chem.MolToSmiles(m)


def _canon_no_maps_no_stereo(mol: Chem.Mol) -> str:
    """Canonical SMILES with atom maps AND all stereo descriptors stripped.

    Used for the "stereo-blind" pass rate: bond-order Δ alone cannot recover
    stereo on newly-formed centres / bonds, so comparing the heavy-atom
    skeleton ignoring stereo isolates whether the *chemistry* (connectivity,
    aromaticity, charges) is correct independent of stereo prediction.

    Uses ``isomericSmiles=False`` to strip *every* stereo descriptor at the
    output level — ChiralTag, BondStereo, AND BondDir (the ``/``/``\\`` on
    single bonds adjacent to a double bond, which ``SetStereo(STEREONONE)``
    alone does NOT clear).
    """
    m = Chem.Mol(mol)
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    return Chem.MolToSmiles(m, isomericSmiles=False)


def _canon_strip_unrecoverable(mol: Chem.Mol, r_mol: Chem.Mol) -> str:
    """Canonical SMILES with stereo descriptors stripped at every position
    where ``r_mol`` has no stereo annotation either.

    Rationale: the dataset is internally inconsistent — for ~5% of reactions
    the patent's product SMILES records chirality on atoms / bonds whose
    *reactant* SMILES has no stereo marker at all. apply_delta cannot
    conjure stereo that isn't in the input; counting these as failures
    inflates the strict-error rate with dataset annotation gaps rather than
    model / algorithm errors.

    This metric strips chirality on any atom whose R-side counterpart (matched
    by atom-map number) has ``CHI_UNSPECIFIED``, and bond-stereo on any bond
    whose R-side counterpart has ``STEREONONE`` or doesn't exist in R.
    Comparing ``_canon_strip_unrecoverable(pred, r)`` to
    ``_canon_strip_unrecoverable(true, r)`` yields ``pass_rate_recoverable``:
    "did pred match true on everything that was knowable from R?"
    """
    r_chir = {
        a.GetAtomMapNum(): a.GetChiralTag()
        for a in r_mol.GetAtoms()
        if a.GetAtomMapNum() > 0
    }
    r_bonds: dict[tuple[int, int], Chem.BondStereo] = {}
    for b in r_mol.GetBonds():
        m1 = b.GetBeginAtom().GetAtomMapNum()
        m2 = b.GetEndAtom().GetAtomMapNum()
        if m1 > 0 and m2 > 0:
            r_bonds[tuple(sorted((m1, m2)))] = b.GetStereo()

    m = Chem.Mol(mol)
    for a in m.GetAtoms():
        mn = a.GetAtomMapNum()
        if mn > 0 and r_chir.get(mn, Chem.ChiralType.CHI_UNSPECIFIED) == Chem.ChiralType.CHI_UNSPECIFIED:
            a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for b in m.GetBonds():
        m1 = b.GetBeginAtom().GetAtomMapNum()
        m2 = b.GetEndAtom().GetAtomMapNum()
        if m1 > 0 and m2 > 0:
            key = tuple(sorted((m1, m2)))
            if r_bonds.get(key, Chem.BondStereo.STEREONONE) == Chem.BondStereo.STEREONONE:
                b.SetStereo(Chem.BondStereo.STEREONONE)
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    # Round-trip via SMILES to clear stale BondDir flags on single bonds
    # adjacent to a double bond whose stereo we just blanked.
    smi = Chem.MolToSmiles(m)
    rt = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(rt) if rt is not None else smi


def _extract_productive_fragments(
    out_mol: Chem.Mol, productive_maps: set[int]
) -> Chem.Mol | None:
    """Return a single ``Mol`` containing only fragments that touch ``productive_maps``.

    Returns ``None`` if no fragment contains any productive atom (shouldn't
    happen in practice for well-formed reactions).
    """
    frag_indices = Chem.GetMolFrags(out_mol, asMols=False)
    keep_idx: list[int] = []
    for frag in frag_indices:
        if any(out_mol.GetAtomWithIdx(i).GetAtomMapNum() in productive_maps for i in frag):
            keep_idx.extend(frag)
    if not keep_idx:
        return None

    rw = Chem.RWMol(out_mol)
    # Remove atoms NOT in keep_idx, in descending order so indices stay valid.
    drop = sorted(set(range(out_mol.GetNumAtoms())) - set(keep_idx), reverse=True)
    for idx in drop:
        rw.RemoveAtom(idx)
    return rw.GetMol()


def _analyse_reaction(rxn) -> dict:
    r_smi = rxn.overall_reactants_smi
    p_smi = rxn.overall_products_smi

    r_mol = Chem.MolFromSmiles(r_smi)
    p_mol = Chem.MolFromSmiles(p_smi)
    if r_mol is None or p_mol is None:
        return {"status": "parse_failed", "reason": "MolFromSmiles returned None"}

    # CRITICAL: ``delta_from_reactants_products`` diffs adjacency matrices by
    # atom **index**, not atom-map. R has reagent atoms (map ≥101) that P
    # doesn't have; without alignment the diff compares unrelated atom pairs.
    # ``align_atoms`` reorders so mapped atoms come first sorted by map number,
    # giving R and P a common prefix indexing on the productive atoms.
    r_mol = align_atoms(r_mol)
    p_mol = align_atoms(p_mol)

    try:
        delta = DeltaMatrixGenerator.delta_from_reactants_products(r_mol, p_mol)
    except Exception as e:
        return {"status": "delta_failed", "reason": repr(e)}

    # Ground-truth diagonal Δ (formal-charge change per shared atom). The
    # verifier knows P, so we feed the exact Δq into apply_delta instead of
    # relying on the row-sum heuristic. This handles the 0.05% of atoms
    # (e.g. azide central N) where Δq ≠ 0 — necessary to reach a true
    # ceiling on bond-Δ pass rate. The future inference-time path will
    # consume a *predicted* Δq vector from a diagonal head.
    p_map_to_charge = {
        a.GetAtomMapNum(): a.GetFormalCharge()
        for a in p_mol.GetAtoms()
        if a.GetAtomMapNum() > 0
    }
    charge_delta = torch.zeros(r_mol.GetNumAtoms(), dtype=torch.int64)
    for idx, a in enumerate(r_mol.GetAtoms()):
        m = a.GetAtomMapNum()
        if m in p_map_to_charge:
            charge_delta[idx] = p_map_to_charge[m] - a.GetFormalCharge()

    try:
        out_mol = apply_delta(
            r_mol,
            delta,
            charge_delta=charge_delta,
            apply_charge_heuristic=False,
        )
    except ApplyDeltaError as e:
        return {"status": "apply_failed", "reason": e.reason, "detail": e.message}
    except Exception as e:
        return {"status": "apply_failed", "reason": "unexpected", "detail": repr(e)}

    productive_maps = {a.GetAtomMapNum() for a in p_mol.GetAtoms() if a.GetAtomMapNum() > 0}
    pred_productive = _extract_productive_fragments(out_mol, productive_maps)
    if pred_productive is None:
        return {"status": "no_productive_frag"}

    try:
        Chem.SanitizeMol(pred_productive)
    except Exception as e:
        return {"status": "post_extract_sanitize_failed", "reason": repr(e)}

    pred_canon = _canon_no_maps(pred_productive)
    true_canon = _canon_no_maps(p_mol)
    if pred_canon == true_canon:
        return {"status": "ok"}
    # Try the "recoverable" comparison: ignore stereo at positions where
    # the dataset's R-side has no annotation either (~75% of strict misses
    # in the 200-rxn investigation were dataset gaps, not algorithm errors).
    pred_recov = _canon_strip_unrecoverable(pred_productive, r_mol)
    true_recov = _canon_strip_unrecoverable(p_mol, r_mol)
    if pred_recov == true_recov:
        return {
            "status": "ok_recoverable",
            "pred": pred_canon,
            "true": true_canon,
        }
    # Re-compare ignoring all stereo descriptors. apply_delta deliberately
    # leaves new stereocentres / new double-bond geometry as UNSPECIFIED
    # because bond-order Δ cannot determine them. A "stereo_only" mismatch
    # means the heavy-atom skeleton is correct — chemistry is right but the
    # patent's specific stereo label can't be recovered from R alone.
    pred_blind = _canon_no_maps_no_stereo(pred_productive)
    true_blind = _canon_no_maps_no_stereo(p_mol)
    if pred_blind == true_blind:
        return {
            "status": "stereo_only_mismatch",
            "pred": pred_canon,
            "true": true_canon,
        }
    return {
        "status": "mismatch",
        "pred": pred_canon,
        "true": true_canon,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="data/mech-USPTO-31k.csv")
    ap.add_argument("--out", default="results/apply_delta_e2e.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip the first OFFSET reactions (useful for held-out smoke tests).",
    )
    ap.add_argument(
        "--max-mismatch-examples",
        type=int,
        default=20,
        help="How many mismatch examples to keep per class.",
    )
    args = ap.parse_args()

    print(f"📖 Parsing {args.csv}")
    reactions = MechUSPTOParser.parse_csv_file(args.csv)
    if args.offset:
        reactions = reactions[args.offset:]
    if args.limit is not None:
        reactions = reactions[: args.limit]
    print(f"   {len(reactions)} reactions loaded")

    status_counts: Counter[str] = Counter()
    failure_reasons: Counter[str] = Counter()
    class_status: dict[str, Counter] = defaultdict(Counter)
    mismatch_examples: list[dict] = []
    failure_examples: list[dict] = []  # captures apply_failed too

    for rxn in tqdm(reactions, desc="Verifying"):
        result = _analyse_reaction(rxn)
        status = result["status"]
        status_counts[status] += 1

        cls = rxn.metadata.get("mechanistic_class", "<unknown>") or "<unknown>"
        class_status[cls][status] += 1

        if status != "ok":
            tag = status
            if "reason" in result:
                tag = f"{status}:{result['reason']}"
            failure_reasons[tag] += 1

            if len(failure_examples) < 30:
                failure_examples.append(
                    {
                        "reaction_id": rxn.reaction_id,
                        "mechanistic_class": cls,
                        "status": status,
                        "reason": result.get("reason", ""),
                        "detail": result.get("detail", ""),
                        "r_smi": rxn.overall_reactants_smi,
                        "p_smi": rxn.overall_products_smi,
                    }
                )

        if status == "mismatch" and len(mismatch_examples) < args.max_mismatch_examples:
            mismatch_examples.append(
                {
                    "reaction_id": rxn.reaction_id,
                    "mechanistic_class": cls,
                    "true": result["true"],
                    "pred": result["pred"],
                }
            )

    total = sum(status_counts.values())
    n_ok = status_counts["ok"]
    n_recoverable = status_counts.get("ok_recoverable", 0)
    n_stereo_only = status_counts.get("stereo_only_mismatch", 0)
    pass_rate = n_ok / total if total else 0.0
    pass_rate_recoverable = (n_ok + n_recoverable) / total if total else 0.0
    pass_rate_stereo_blind = (
        (n_ok + n_recoverable + n_stereo_only) / total if total else 0.0
    )

    # Per-class pass rate (sorted by count, then accuracy).
    class_summary = {}
    for cls, counts in class_status.items():
        c_total = sum(counts.values())
        class_summary[cls] = {
            "total": c_total,
            "ok": counts.get("ok", 0),
            "pass_rate": counts.get("ok", 0) / c_total if c_total else 0.0,
            "by_status": dict(counts),
        }

    summary = {
        "n_total": total,
        "n_ok": n_ok,
        "pass_rate": pass_rate,
        "pass_rate_recoverable": pass_rate_recoverable,
        "pass_rate_stereo_blind": pass_rate_stereo_blind,
        "status_counts": dict(status_counts),
        "top_failure_reasons": dict(failure_reasons.most_common(20)),
        "per_class": dict(
            sorted(class_summary.items(), key=lambda kv: -kv[1]["total"])
        ),
        "mismatch_examples": mismatch_examples,
        "failure_examples": failure_examples,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("apply_delta E2E VERIFICATION")
    print("=" * 60)
    print(f"Total reactions:  {total:,}")
    print(f"Passed (ok):      {n_ok:,}  ({pass_rate:6.2%})")
    print(
        f"Recoverable ok:   {n_ok + n_recoverable:,}  "
        f"({pass_rate_recoverable:6.2%})  "
        f"(ignoring stereo at positions R-side doesn't annotate)"
    )
    print(
        f"Stereo-blind ok:  {n_ok + n_recoverable + n_stereo_only:,}  "
        f"({pass_rate_stereo_blind:6.2%})  (ignoring ALL stereo)"
    )
    print()
    print("Status breakdown:")
    for status, count in status_counts.most_common():
        print(f"  {status:30s}  {count:6,}  ({count/total:6.2%})")
    print()
    if failure_reasons:
        print("Top failure reasons:")
        for reason, count in failure_reasons.most_common(10):
            print(f"  {count:6,}  {reason}")
    print()
    print("Lowest pass-rate classes (min 50 reactions):")
    sortable = [
        (cls, c["pass_rate"], c["total"])
        for cls, c in class_summary.items()
        if c["total"] >= 50
    ]
    sortable.sort(key=lambda x: x[1])
    for cls, rate, n in sortable[:10]:
        print(f"  {rate:6.2%}  n={n:5,}  {cls}")
    print()
    print(f"💾 Wrote {out_path}")
    print()
    print("Decision gate:")
    print(
        "  (recoverable is the chemistry-honest metric — it excludes "
        "stereo positions the dataset doesn't annotate in R)"
    )
    if pass_rate_recoverable >= 0.95:
        print(
            f"  ✅ recoverable pass rate {pass_rate_recoverable:.2%} ≥ 95% — "
            "bond-Δ formulation is sound."
        )
    elif pass_rate_recoverable >= 0.80:
        print(
            f"  ⚠️  recoverable pass rate {pass_rate_recoverable:.2%} in "
            "[80%, 95%) — investigate failure modes."
        )
    else:
        print(
            f"  ❌ recoverable pass rate {pass_rate_recoverable:.2%} < 80% — "
            "reconsider formulation."
        )


if __name__ == "__main__":
    main()
