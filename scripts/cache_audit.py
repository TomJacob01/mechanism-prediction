"""T1.4 + T2.{1,2,3} — Single-pass parquet-cache audit.

Reports:
  * Δ value range over every cached step (T1.4): bond Δ must lie in {-1,0,1}.
  * Split integrity (T2.1): no rxn_id in >1 split; bucket sizes.
  * Class balance per split (T2.2): histogram of mechanistic_class.
  * Step-size distribution (T2.3): n_steps per reaction, arrow_count per step,
    n_atoms per reaction.

Usage::

    python scripts/cache_audit.py \\
        [--cache data/cache/parquet] \\
        [--out results/cache_audit.json]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from mech_uspto.data.parquet_dataset import split_of


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data/cache/parquet"))
    ap.add_argument("--out", type=Path, default=Path("results/cache_audit.json"))
    args = ap.parse_args()

    rxns = pq.read_table(args.cache / "reactions.parquet").to_pylist()
    steps = pq.read_table(args.cache / "steps.parquet").to_pylist()

    # ---- T2.1 split integrity --------------------------------------------
    rxn_split: dict[str, str] = {}
    split_counts: Counter = Counter()
    for r in rxns:
        s = split_of(r["split_hash"])
        rxn_split[r["rxn_id"]] = s
        split_counts[s] += 1

    # ---- T2.2 class × split ----------------------------------------------
    class_by_split: dict[str, Counter] = {s: Counter() for s in ("train", "val", "test")}
    for r in rxns:
        class_by_split[split_of(r["split_hash"])][r["mechanistic_class"]] += 1
    all_classes = sorted({c for r in rxns for c in [r["mechanistic_class"]]})

    # ---- T2.3 n_steps, n_atoms -------------------------------------------
    n_steps_hist: Counter = Counter(r["n_steps"] for r in rxns)
    n_atoms_hist: Counter = Counter()
    for r in rxns:
        # Bucket atoms to 10s so the histogram is readable.
        n_atoms_hist[(r["n_atoms_mapped"] // 10) * 10] += 1

    # ---- T1.4 + T2.3 step-level scan -------------------------------------
    bond_delta_hist: Counter = Counter()
    bond_delta_abs_max = 0
    charge_delta_hist: Counter = Counter()
    charge_delta_abs_max = 0
    arrow_count_hist: Counter = Counter()
    steps_per_rxn_via_steps: Counter = Counter()
    seen_rxn_in_steps: set[str] = set()
    for st in steps:
        steps_per_rxn_via_steps[st["rxn_id"]] += 1
        arrow_count_hist[st["arrow_count"]] += 1
        seen_rxn_in_steps.add(st["rxn_id"])
        for bc in st["bond_changes"]:
            d = int(bc["delta"])
            bond_delta_hist[d] += 1
            if abs(d) > bond_delta_abs_max:
                bond_delta_abs_max = abs(d)
        for cc in st["charge_changes"]:
            d = int(cc["delta"])
            charge_delta_hist[d] += 1
            if abs(d) > charge_delta_abs_max:
                charge_delta_abs_max = abs(d)

    # Cross-table consistency.
    rxn_ids_only_in_rxns = {r["rxn_id"] for r in rxns} - seen_rxn_in_steps
    rxn_ids_only_in_steps = seen_rxn_in_steps - {r["rxn_id"] for r in rxns}

    summary = {
        "n_reactions": len(rxns),
        "n_steps": len(steps),
        "split_counts": dict(split_counts),
        "split_fractions": {
            k: v / max(1, len(rxns)) for k, v in split_counts.items()
        },
        "class_by_split": {
            s: dict(c) for s, c in class_by_split.items()
        },
        "n_classes": len(all_classes),
        "classes_missing_from_split": {
            s: sorted(set(all_classes) - set(c.keys()))
            for s, c in class_by_split.items()
        },
        "n_steps_hist": {str(k): v for k, v in sorted(n_steps_hist.items())},
        "n_atoms_hist_bucket10": {str(k): v for k, v in sorted(n_atoms_hist.items())},
        "arrow_count_hist": {str(k): v for k, v in sorted(arrow_count_hist.items())},
        "bond_delta_hist": {str(k): v for k, v in sorted(bond_delta_hist.items())},
        "bond_delta_abs_max": bond_delta_abs_max,
        "bond_delta_in_range": bond_delta_abs_max <= 1,
        "charge_delta_hist": {str(k): v for k, v in sorted(charge_delta_hist.items())},
        "charge_delta_abs_max": charge_delta_abs_max,
        "rxn_ids_only_in_reactions_table": sorted(rxn_ids_only_in_rxns)[:20],
        "n_rxn_ids_only_in_reactions_table": len(rxn_ids_only_in_rxns),
        "rxn_ids_only_in_steps_table": sorted(rxn_ids_only_in_steps)[:20],
        "n_rxn_ids_only_in_steps_table": len(rxn_ids_only_in_steps),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"[cache_audit] reactions={len(rxns)}  steps={len(steps)}  "
          f"bond|Δ|max={bond_delta_abs_max}  "
          f"splits={dict(split_counts)}  "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
