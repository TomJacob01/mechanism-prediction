"""Sequential apply_delta verifier.

Tests whether applying a Δ matrix in K chunks (sequentially) produces the
same molecule as applying it in one shot. This is the foundational property
the sequential model relies on — the model will emit a sequence of partial
Δs, and we need to know apply_delta can consume them safely.

Method (per reaction):
    1. Compute full Δ from R → P (same as the single-step verifier).
    2. Apply Δ in one shot → ``mol_one_shot``.
    3. Extract nonzero off-diagonal upper-tri entries from Δ.
       Shuffle deterministically and partition into K chunks.
    4. Build a sparse Δ tensor per chunk; apply each chunk sequentially
       to the running mol, using the charge heuristic each step (so
       intermediates are self-consistent — no need to know ground-truth
       charges mid-reaction).
    5. Apply the same one-shot heuristic call for the baseline.
    6. Compare canonical SMILES of sequential-final vs one-shot.

Statuses:
    - ``ok``                      : sequential final == one-shot
    - ``diverged``                : sequential final != one-shot (path-dependent)
    - ``intermediate_failed``     : a chunk raised ApplyDeltaError mid-chain
    - ``one_shot_failed``         : one-shot itself errored (filter out;
                                    irrelevant for this experiment)
    - ``no_arrows``               : Δ is all zeros (no work to do)
    - ``parse_failed`` /
      ``delta_failed``            : upstream failures (also filter out)

Output: JSON summary with counts per status and a ``decomposability_rate``
= ok / (ok + diverged + intermediate_failed). This tells us what fraction
of reactions are safely sequential-decomposable.

Usage::

    python scripts/verify_apply_delta_sequential.py \\
        [--limit N] [--offset N] [--num-chunks K] [--seed S] \\
        [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
from rdkit import Chem
from tqdm import tqdm

from mech_uspto.chemistry import ApplyDeltaError, apply_delta
from mech_uspto.data.featurization import align_atoms
from mech_uspto.data.parser import MechUSPTOParser
from mech_uspto.data.transformations import DeltaMatrixGenerator


def _canon_no_maps(mol: Chem.Mol) -> str:
    m = Chem.Mol(mol)
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    return Chem.MolToSmiles(m, isomericSmiles=False)  # stereo-blind for fairness


def _chunk_delta(delta: torch.Tensor, num_chunks: int, rng: random.Random) -> list[torch.Tensor]:
    """Partition Δ's nonzero off-diagonal upper-tri entries into ``num_chunks``
    sparse Δ tensors that sum to ``delta`` on the upper triangle (and to
    ``delta + delta.T`` overall by symmetry)."""
    n = delta.shape[0]
    entries: list[tuple[int, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            v = int(delta[i, j].item())
            if v != 0:
                entries.append((i, j, v))
    if not entries:
        return []
    rng.shuffle(entries)
    chunks: list[list[tuple[int, int, int]]] = [[] for _ in range(num_chunks)]
    for k, entry in enumerate(entries):
        chunks[k % num_chunks].append(entry)
    out: list[torch.Tensor] = []
    for chunk in chunks:
        if not chunk:
            continue
        d = torch.zeros((n, n), dtype=delta.dtype)
        for i, j, v in chunk:
            d[i, j] = v
            d[j, i] = v
        out.append(d)
    return out


def _analyse_reaction(rxn, num_chunks: int, rng: random.Random, use_gt_charges: bool) -> dict:
    r_smi = rxn.overall_reactants_smi
    p_smi = rxn.overall_products_smi
    r_mol = Chem.MolFromSmiles(r_smi)
    p_mol = Chem.MolFromSmiles(p_smi)
    if r_mol is None or p_mol is None:
        return {"status": "parse_failed"}
    r_mol = align_atoms(r_mol)
    p_mol = align_atoms(p_mol)

    try:
        delta = DeltaMatrixGenerator.delta_from_reactants_products(r_mol, p_mol)
    except Exception as e:  # noqa: BLE001
        return {"status": "delta_failed", "reason": repr(e)}

    delta_long = delta.long()
    if int(delta_long.abs().sum().item()) == 0:
        return {"status": "no_arrows"}

    # Ground-truth Δq from R vs P for shared atoms (when use_gt_charges).
    charge_delta = None
    if use_gt_charges:
        p_map_to_charge = {
            a.GetAtomMapNum(): a.GetFormalCharge()
            for a in p_mol.GetAtoms()
            if a.GetAtomMapNum() > 0
        }
        cd = torch.zeros(r_mol.GetNumAtoms(), dtype=torch.int64)
        for idx, a in enumerate(r_mol.GetAtoms()):
            m = a.GetAtomMapNum()
            if m in p_map_to_charge:
                cd[idx] = p_map_to_charge[m] - a.GetFormalCharge()
        charge_delta = cd

    # One-shot baseline.
    try:
        mol_oneshot = apply_delta(
            r_mol,
            delta_long,
            charge_delta=charge_delta,
            apply_charge_heuristic=(not use_gt_charges),
        )
    except ApplyDeltaError as e:
        return {"status": "one_shot_failed", "reason": e.reason}
    except Exception as e:  # noqa: BLE001
        return {"status": "one_shot_failed", "reason": repr(e)}

    chunks = _chunk_delta(delta_long, num_chunks, rng)
    if not chunks:
        return {"status": "no_arrows"}

    running = r_mol
    for step_idx, chunk_delta in enumerate(chunks):
        is_last = (step_idx == len(chunks) - 1)
        # Intermediate chunks use the heuristic so charges stay self-consistent
        # with the bond changes that happened. Last chunk additionally applies
        # the ground-truth Δq (when use_gt_charges) so the final state matches
        # the one-shot baseline.
        step_charge_delta = charge_delta if (is_last and use_gt_charges) else None
        step_heuristic = (not use_gt_charges) or (not is_last)
        try:
            running = apply_delta(
                running,
                chunk_delta,
                charge_delta=step_charge_delta,
                apply_charge_heuristic=step_heuristic,
            )
        except ApplyDeltaError as e:
            return {
                "status": "intermediate_failed",
                "failed_step": step_idx,
                "n_chunks": len(chunks),
                "reason": e.reason,
            }
        except Exception as e:  # noqa: BLE001
            return {
                "status": "intermediate_failed",
                "failed_step": step_idx,
                "n_chunks": len(chunks),
                "reason": repr(e),
            }

    seq_canon = _canon_no_maps(running)
    one_canon = _canon_no_maps(mol_oneshot)
    if seq_canon == one_canon:
        return {"status": "ok", "n_chunks": len(chunks)}
    return {
        "status": "diverged",
        "n_chunks": len(chunks),
        "sequential": seq_canon,
        "one_shot": one_canon,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="data/mech-USPTO-31k.csv")
    ap.add_argument("--out", default="results/apply_delta_sequential.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--num-chunks", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--use-gt-charges",
        action="store_true",
        help="Use ground-truth Δq (R vs P) on the last chunk + one-shot, instead "
        "of the heuristic. Matches the real e2e verifier's charge handling.",
    )
    ap.add_argument("--max-examples", type=int, default=10)
    args = ap.parse_args()

    print(f"Parsing {args.csv}")
    reactions = MechUSPTOParser.parse_csv_file(args.csv)
    if args.offset:
        reactions = reactions[args.offset:]
    if args.limit is not None:
        reactions = reactions[: args.limit]
    print(
        f"  {len(reactions)} reactions loaded; num_chunks={args.num_chunks}; "
        f"seed={args.seed}; use_gt_charges={args.use_gt_charges}"
    )

    rng = random.Random(args.seed)
    counts: Counter[str] = Counter()
    examples: dict[str, list[dict]] = defaultdict(list)
    failure_reasons: Counter[str] = Counter()

    for rxn in tqdm(reactions, desc="Sequential"):
        result = _analyse_reaction(rxn, args.num_chunks, rng, args.use_gt_charges)
        counts[result["status"]] += 1
        if result["status"] in {"diverged", "intermediate_failed"}:
            if len(examples[result["status"]]) < args.max_examples:
                examples[result["status"]].append({"rxn_id": rxn.reaction_id, **result})
            if "reason" in result:
                failure_reasons[result["reason"]] += 1

    total = sum(counts.values())
    n_ok = counts["ok"]
    n_div = counts.get("diverged", 0)
    n_int = counts.get("intermediate_failed", 0)
    eligible = n_ok + n_div + n_int  # exclude no_arrows / parse_failed / one_shot_failed
    decomposability = n_ok / eligible if eligible else 0.0

    print()
    print(f"Total reactions:          {total:,}")
    print(f"Eligible (had nonzero Δ): {eligible:,}")
    print(f"Sequential ok:            {n_ok:,}  ({decomposability:6.2%})")
    print(f"Diverged from one-shot:   {n_div:,}")
    print(f"Intermediate sanitize:    {n_int:,}")
    print()
    for st, c in counts.most_common():
        print(f"  {st:30s} {c:6,}  ({c / total:6.2%})")
    if failure_reasons:
        print("\nTop failure reasons:")
        for reason, c in failure_reasons.most_common(10):
            print(f"  {c:5,}  {reason}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(
            {
                "n_total": total,
                "n_eligible": eligible,
                "decomposability_rate": decomposability,
                "num_chunks": args.num_chunks,
                "seed": args.seed,
                "status_counts": dict(counts),
                "failure_reasons": dict(failure_reasons.most_common(20)),
                "examples": dict(examples),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
