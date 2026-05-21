"""Phase-0 diagnostic for the diagonal-Δ / charge encoding plan.

Iterates the full mech-USPTO-31k CSV and measures, per reaction:

1. Whether any heavy atom changes formal charge (R → P).
2. Whether any heavy atom changes radical-electron count.
3. Per-atom accuracy of the row-sum heuristic ``q_hat[i] = −Σⱼ Δbond[i,j]``
   at predicting the true ``Δq[i] = fc_P[i] − fc_R[i]``.

Output:
- Prints a summary table.
- Writes ``results/charge_diagnostic.json`` with full aggregates.

Decision gate:
- ``heuristic_atom_accuracy ≥ 0.95`` → skip diagonal-Δ encoding; keep
  the row-sum heuristic in ``apply_delta`` and proceed to Phase D3.
- Otherwise → proceed with the full diagonal-Δ plan (Phases D1–D5).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Optional

import torch
from rdkit import Chem, RDLogger
from tqdm.auto import tqdm

from mech_uspto.data.parser import MechUSPTOParser
from mech_uspto.data.transformations import DeltaMatrixGenerator
from mech_uspto.data.schema import MultiStepReaction

# Silence RDKit's parse warnings — the dataset has occasional valence quirks
# and we surface our own per-reaction skip messages.
RDLogger.DisableLog("rdApp.*")


def _atom_map_table(mol: Chem.Mol) -> dict[int, int]:
    """Return ``{atom_map_num: atom_idx}`` for atoms with a non-zero map."""
    return {
        atom.GetAtomMapNum(): atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomMapNum() > 0
    }


def _per_atom_charge_and_radical(mol: Chem.Mol) -> dict[int, tuple[int, int]]:
    """Return ``{atom_map_num: (formal_charge, num_radical_electrons)}``."""
    return {
        atom.GetAtomMapNum(): (atom.GetFormalCharge(), atom.GetNumRadicalElectrons())
        for atom in mol.GetAtoms()
        if atom.GetAtomMapNum() > 0
    }


def _analyse_reaction(rxn: MultiStepReaction) -> Optional[dict]:
    """Compute per-atom charge / radical diffs and heuristic accuracy.

    Returns ``None`` if either side of the reaction fails to parse.
    """
    r_mol = Chem.MolFromSmiles(rxn.overall_reactants_smi)
    p_mol = Chem.MolFromSmiles(rxn.overall_products_smi)
    if r_mol is None or p_mol is None:
        return None

    r_state = _per_atom_charge_and_radical(r_mol)
    p_state = _per_atom_charge_and_radical(p_mol)
    shared_maps = set(r_state) & set(p_state)
    if not shared_maps:
        return None

    # Heuristic q_hat: requires a bond-Δ matrix indexed by reactant atom idx.
    # Compute it using the no-Hs mols so atom indices match _atom_map_table.
    try:
        delta = DeltaMatrixGenerator.delta_from_reactants_products(r_mol, p_mol)
    except Exception:
        return None
    r_idx = _atom_map_table(r_mol)

    n_atoms_shared = 0
    n_atoms_charge_changed = 0
    n_atoms_radical_changed = 0
    n_heuristic_correct = 0
    n_heuristic_correct_on_changed = 0
    charge_change_hist: Counter[int] = Counter()
    radical_change_hist: Counter[int] = Counter()

    for m in shared_maps:
        fc_r, rad_r = r_state[m]
        fc_p, rad_p = p_state[m]
        dq = fc_p - fc_r
        drad = rad_p - rad_r

        n_atoms_shared += 1
        charge_change_hist[dq] += 1
        radical_change_hist[drad] += 1

        if dq != 0:
            n_atoms_charge_changed += 1
        if drad != 0:
            n_atoms_radical_changed += 1

        # Heuristic: q_hat = −Σⱼ Δ[i,j]. Indexed by reactant atom idx.
        i = r_idx.get(m)
        if i is None or i >= delta.shape[0]:
            continue
        q_hat = int(-delta[i].sum().item())
        if q_hat == dq:
            n_heuristic_correct += 1
            if dq != 0:
                n_heuristic_correct_on_changed += 1

    return {
        "reaction_id": rxn.reaction_id,
        "mechanistic_class": rxn.metadata.get("mechanistic_class", ""),
        "n_atoms_shared": n_atoms_shared,
        "n_atoms_charge_changed": n_atoms_charge_changed,
        "n_atoms_radical_changed": n_atoms_radical_changed,
        "n_heuristic_correct": n_heuristic_correct,
        "n_heuristic_correct_on_changed": n_heuristic_correct_on_changed,
        "charge_change_hist": dict(charge_change_hist),
        "radical_change_hist": dict(radical_change_hist),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--csv",
        default="data/mech-USPTO-31k.csv",
        help="Path to the mech-USPTO-31k CSV.",
    )
    ap.add_argument(
        "--out",
        default="results/charge_diagnostic.json",
        help="Where to write the JSON summary.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of reactions (for quick smoke tests).",
    )
    args = ap.parse_args()

    print(f"📖 Parsing {args.csv}")
    reactions = MechUSPTOParser.parse_csv_file(args.csv)
    if args.limit is not None:
        reactions = reactions[: args.limit]
    print(f"   {len(reactions)} reactions loaded")

    # Aggregates.
    n_reactions = 0
    n_reactions_parsed = 0
    n_reactions_with_charge_change = 0
    n_reactions_with_radical = 0
    n_atoms_total = 0
    n_atoms_charge_changed = 0
    n_atoms_radical_changed = 0
    n_heuristic_correct = 0
    n_heuristic_correct_on_changed = 0
    charge_change_hist: Counter[int] = Counter()
    radical_change_hist: Counter[int] = Counter()
    class_charge_change: Counter[str] = Counter()
    class_totals: Counter[str] = Counter()

    for rxn in tqdm(reactions, desc="Analysing"):
        n_reactions += 1
        result = _analyse_reaction(rxn)
        if result is None:
            continue
        n_reactions_parsed += 1

        rxn_class = result["mechanistic_class"] or "<unknown>"
        class_totals[rxn_class] += 1
        if result["n_atoms_charge_changed"] > 0:
            n_reactions_with_charge_change += 1
            class_charge_change[rxn_class] += 1
        if result["n_atoms_radical_changed"] > 0:
            n_reactions_with_radical += 1

        n_atoms_total += result["n_atoms_shared"]
        n_atoms_charge_changed += result["n_atoms_charge_changed"]
        n_atoms_radical_changed += result["n_atoms_radical_changed"]
        n_heuristic_correct += result["n_heuristic_correct"]
        n_heuristic_correct_on_changed += result["n_heuristic_correct_on_changed"]

        for k, v in result["charge_change_hist"].items():
            charge_change_hist[int(k)] += v
        for k, v in result["radical_change_hist"].items():
            radical_change_hist[int(k)] += v

    def _pct(num: int, den: int) -> float:
        return float(num) / den if den > 0 else 0.0

    summary = {
        "n_reactions_total": n_reactions,
        "n_reactions_parsed": n_reactions_parsed,
        "n_reactions_skipped": n_reactions - n_reactions_parsed,
        "n_atoms_total": n_atoms_total,
        "pct_reactions_with_charge_change": _pct(
            n_reactions_with_charge_change, n_reactions_parsed
        ),
        "pct_reactions_with_radicals": _pct(n_reactions_with_radical, n_reactions_parsed),
        "pct_atoms_with_charge_change": _pct(n_atoms_charge_changed, n_atoms_total),
        "pct_atoms_with_radical_change": _pct(n_atoms_radical_changed, n_atoms_total),
        "heuristic_atom_accuracy": _pct(n_heuristic_correct, n_atoms_total),
        "heuristic_atom_accuracy_on_changed": _pct(
            n_heuristic_correct_on_changed, n_atoms_charge_changed
        ),
        "charge_change_histogram": dict(sorted(charge_change_hist.items())),
        "radical_change_histogram": dict(sorted(radical_change_hist.items())),
        "top_classes_by_charge_change": dict(class_charge_change.most_common(20)),
        "top_classes_total": dict(class_totals.most_common(20)),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Pretty summary.
    print("\n" + "=" * 60)
    print("CHARGE DIAGNOSTIC — SUMMARY")
    print("=" * 60)
    print(f"Reactions:                   {n_reactions_parsed:,} parsed / {n_reactions:,} total")
    print(f"Atoms (atom-mapped, shared): {n_atoms_total:,}")
    print()
    print(f"Reactions with Δq ≠ 0:       {summary['pct_reactions_with_charge_change']:6.1%}")
    print(f"Reactions with radicals:     {summary['pct_reactions_with_radicals']:6.1%}")
    print(f"Atoms with Δq ≠ 0:           {summary['pct_atoms_with_charge_change']:6.1%}")
    print(f"Atoms with Δradical ≠ 0:     {summary['pct_atoms_with_radical_change']:6.1%}")
    print()
    print(f"Heuristic accuracy (all atoms):     {summary['heuristic_atom_accuracy']:6.1%}")
    print(
        f"Heuristic accuracy (changed atoms): "
        f"{summary['heuristic_atom_accuracy_on_changed']:6.1%}"
    )
    print()
    print(f"Δq histogram:       {summary['charge_change_histogram']}")
    print(f"Δradical histogram: {summary['radical_change_histogram']}")
    print()
    print("Decision gate:")
    acc = summary["heuristic_atom_accuracy"]
    if acc >= 0.95:
        print(f"  ✅ heuristic accuracy {acc:.1%} ≥ 95% — skip diagonal-Δ, use heuristic.")
    else:
        print(f"  ⚠️  heuristic accuracy {acc:.1%} < 95% — proceed with diagonal-Δ plan.")
    print()
    print(f"💾 Wrote {out_path}")


if __name__ == "__main__":
    main()
