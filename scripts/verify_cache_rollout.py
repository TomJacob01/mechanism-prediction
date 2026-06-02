"""T1.2 — Sequential apply_delta over cached steps.

For every row in ``reactions.parquet``, take ``reactant_mol`` and apply each
step's ``bond_changes`` (+ ``charge_changes``) in order, then compare the
final molecule's canonical SMILES (stereo-blind) to ``product_mol``.

This is the property the stepwise training assumes: that applying the K
cached Δ-matrices in sequence reproduces the recorded product. The cache
builder already verified this at build time, but apply_delta and the
parser may have drifted since.

Output: JSON with overall pass-rate, per-class pass-rate, and the top
failure reasons.

Usage::

    python scripts/verify_cache_rollout.py \\
        --cache data/cache/parquet \\
        [--limit N] \\
        [--out results/cache_rollout.json]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import torch
from rdkit import Chem, RDLogger
from tqdm import tqdm

from mech_uspto.chemistry import ApplyDeltaError, apply_delta

RDLogger.DisableLog("rdApp.*")


def _canon(mol: Chem.Mol) -> str:
    m = Chem.Mol(mol)
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    return Chem.MolToSmiles(m, isomericSmiles=False)


def _build_deltas(mol: Chem.Mol, bond_changes, charge_changes):
    n = mol.GetNumAtoms()
    map_to_idx = {
        a.GetAtomMapNum(): a.GetIdx()
        for a in mol.GetAtoms() if a.GetAtomMapNum() > 0
    }
    bd = torch.zeros((n, n), dtype=torch.long)
    for bc in bond_changes:
        i = map_to_idx.get(bc["map_i"])
        j = map_to_idx.get(bc["map_j"])
        if i is None or j is None:
            return None, None
        bd[i, j] += int(bc["delta"])
        bd[j, i] += int(bc["delta"])
    cd = torch.zeros(n, dtype=torch.long)
    for cc in charge_changes:
        i = map_to_idx.get(cc["map_i"])
        if i is None:
            return None, None
        cd[i] += int(cc["delta"])
    return bd, cd


def _replay(rxn_row, step_rows):
    r_mol = Chem.Mol(rxn_row["reactant_mol"])
    p_mol = Chem.Mol(rxn_row["product_mol"])
    cur = r_mol
    for step in step_rows:
        bd, cd = _build_deltas(cur, step["bond_changes"], step["charge_changes"])
        if bd is None:
            return "missing_map", None, None
        try:
            cur = apply_delta(cur, bd, charge_delta=cd)
        except ApplyDeltaError as e:
            return f"apply_delta:{e.reason}", None, None
        except Exception as e:  # noqa: BLE001
            return f"exception:{type(e).__name__}", None, None
    return "ok", _canon(cur), _canon(p_mol)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data/cache/parquet"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/cache_rollout.json"))
    args = ap.parse_args()

    rxns_tbl = pq.read_table(args.cache / "reactions.parquet")
    steps_tbl = pq.read_table(args.cache / "steps.parquet")

    # Group steps by rxn_id, preserving step_idx order.
    steps_by_rxn: dict[str, list[dict]] = defaultdict(list)
    for row in steps_tbl.to_pylist():
        steps_by_rxn[row["rxn_id"]].append(row)
    for v in steps_by_rxn.values():
        v.sort(key=lambda r: r["step_idx"])

    rxn_rows = rxns_tbl.to_pylist()
    if args.limit > 0:
        rxn_rows = rxn_rows[: args.limit]

    status_counts: Counter = Counter()
    per_class: dict[str, Counter] = defaultdict(Counter)
    failure_examples: list[dict] = []
    n_match = 0
    n_mismatch = 0

    for rxn in tqdm(rxn_rows, desc="replay"):
        steps = steps_by_rxn.get(rxn["rxn_id"], [])
        if not steps:
            status_counts["no_steps"] += 1
            per_class[rxn["mechanistic_class"]]["no_steps"] += 1
            continue
        status, pred, true = _replay(rxn, steps)
        if status != "ok":
            status_counts[status] += 1
            per_class[rxn["mechanistic_class"]][status] += 1
            if len(failure_examples) < 20:
                failure_examples.append({"rxn_id": rxn["rxn_id"], "status": status})
            continue
        if pred == true:
            status_counts["ok"] += 1
            per_class[rxn["mechanistic_class"]]["ok"] += 1
            n_match += 1
        else:
            status_counts["mismatch"] += 1
            per_class[rxn["mechanistic_class"]]["mismatch"] += 1
            n_mismatch += 1
            if len(failure_examples) < 20:
                failure_examples.append({
                    "rxn_id": rxn["rxn_id"],
                    "status": "mismatch",
                    "pred": pred,
                    "true": true,
                })

    total = len(rxn_rows)
    summary = {
        "n_total": total,
        "n_ok": status_counts["ok"],
        "pass_rate": status_counts["ok"] / max(1, total),
        "status_counts": dict(status_counts),
        "per_class": {
            cls: {
                "total": sum(cnt.values()),
                "ok": cnt["ok"],
                "pass_rate": cnt["ok"] / max(1, sum(cnt.values())),
                "by_status": dict(cnt),
            }
            for cls, cnt in per_class.items()
        },
        "failure_examples": failure_examples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"[verify_cache_rollout] pass_rate={summary['pass_rate']:.4f}  "
          f"ok={summary['n_ok']}/{total}  "
          f"mismatches={n_mismatch}  "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
