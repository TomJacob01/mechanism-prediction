"""Verify the curly-arrow parser on real USPTO-31k data.

Reports three numbers per run:

(A) **Per-arrow bond-Δ rule match**: sum of per-arrow bond changes vs
    R→P bond Δ on shared atoms. Mismatch ⇒ parser rule is wrong.

(B) **Per-arrow charge-Δ rule match**: sum of per-arrow Δfc vs R→P Δfc
    on shared atoms (no last-step correction).

(C) **Sequential reconstruction**: group arrows into elementary steps
    (chain-based), apply each step's Δ to the running intermediate via
    :func:`apply_delta`, force cumulative charges to match GT on the
    last step, and check ``canonical(final) == canonical(P)``.

The parser logic lives in :mod:`mech_uspto.data.arrow_parser`; this
script only does I/O, the rule-check comparisons, and the sequential
rollout.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import torch
from rdkit import Chem, RDLogger
from tqdm.auto import tqdm

from mech_uspto.chemistry import ApplyDeltaError, apply_delta
from mech_uspto.data.arrow_parser import (
    arrow_bond_changes,
    arrow_charge_changes,
    parse_arrows,
    parse_steps,
)
from mech_uspto.data.featurization import align_atoms
from mech_uspto.data.transformations import DeltaMatrixGenerator

RDLogger.DisableLog("rdApp.*")


def _accumulate(delta: torch.Tensor, changes: list[tuple[int, int, int]]) -> None:
    for i, j, s in changes:
        delta[i, j] += s
        if i != j:
            delta[j, i] += s


# ---------------------------------------------------------------------------
# Per-reaction analysis
# ---------------------------------------------------------------------------

def _canon_no_maps_no_stereo(mol: Chem.Mol) -> str:
    m = Chem.Mol(mol)
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    smi = Chem.MolToSmiles(m, isomericSmiles=False)
    # Round-trip: apply_delta's RWMol surgery leaves partial-aromatic states
    # where SMILES emits placeholders (`C2:N:C(C):...:S:`) rather than clean
    # lowercase aromatic. A simple parse-then-stringify usually fixes it.
    rt = Chem.MolFromSmiles(smi)
    if rt is not None:
        return Chem.MolToSmiles(rt, isomericSmiles=False)
    # Round-trip parse failed (un-parseable partial-aromatic SMILES like
    # `:C:[C]2F`). Force-clear aromaticity and re-sanitize as a last resort.
    m2 = Chem.Mol(mol)
    for a in m2.GetAtoms():
        a.SetAtomMapNum(0)
        a.SetIsAromatic(False)
    for b in m2.GetBonds():
        b.SetIsAromatic(False)
        if b.GetBondType() == Chem.BondType.AROMATIC:
            b.SetBondType(Chem.BondType.SINGLE)
    try:
        Chem.SanitizeMol(m2)
    except Exception:
        pass
    return Chem.MolToSmiles(m2, isomericSmiles=False)


def _diagnose_roundtrip(orig: Chem.Mol, rt: Chem.Mol, s1: str) -> str:
    """Best-effort categorisation of why ``orig``'s SMILES isn't idempotent.

    Compares orig vs the round-trip Mol on a few cheap features. The
    first matching bucket wins; multiple may apply but we want a single
    label per intermediate for histogramming.
    """
    if orig.GetNumAtoms() != rt.GetNumAtoms():
        return "atom_count_diff"
    # Aromaticity flip on any atom
    o_arom = sum(1 for a in orig.GetAtoms() if a.GetIsAromatic())
    r_arom = sum(1 for a in rt.GetAtoms() if a.GetIsAromatic())
    if o_arom != r_arom:
        return "aromaticity_flip"
    # H redistribution across heteroatoms (tautomer drift)
    o_h_by_elem: dict[str, int] = {}
    r_h_by_elem: dict[str, int] = {}
    for a in orig.GetAtoms():
        o_h_by_elem[a.GetSymbol()] = o_h_by_elem.get(a.GetSymbol(), 0) + a.GetTotalNumHs()
    for a in rt.GetAtoms():
        r_h_by_elem[a.GetSymbol()] = r_h_by_elem.get(a.GetSymbol(), 0) + a.GetTotalNumHs()
    if o_h_by_elem != r_h_by_elem:
        return "h_redistribution(tautomer)"
    # Formal charge redistribution (same net, different atoms)
    o_fc = sorted(a.GetFormalCharge() for a in orig.GetAtoms() if a.GetFormalCharge() != 0)
    r_fc = sorted(a.GetFormalCharge() for a in rt.GetAtoms() if a.GetFormalCharge() != 0)
    if o_fc != r_fc:
        return "charge_redistribution"
    # Bond-type histogram (Kekulé flip; aromatic counts already matched)
    o_bt = sorted(b.GetBondTypeAsDouble() for b in orig.GetBonds())
    r_bt = sorted(b.GetBondTypeAsDouble() for b in rt.GetBonds())
    if o_bt != r_bt:
        return "bond_type_shuffle(kekule)"
    # Same Mol features but different canonical SMILES -> stereo / atom-order
    return "canonical_smiles_only"


def _extract_productive_fragments(
    out_mol: Chem.Mol, productive_maps: set[int]
) -> Chem.Mol | None:
    """Return the sub-mol containing product atoms.

    Among fragments that contain at least one productive atom-map, keep
    only those whose *productive coverage* (count of distinct productive
    maps) is maximal-tied with the best fragment. POCl3-style activators
    sometimes end up bonded to a single productive atom by the end of a
    rollout, producing an extra fragment that shares one productive map
    but carries the catalyst Ps/Cls. The real product fragment carries
    nearly all productive maps, so coverage-based ranking drops the
    catalyst fragment without losing legitimate split products (which
    have comparable coverage).
    """
    frag_indices = Chem.GetMolFrags(out_mol, asMols=False)
    scored: list[tuple[int, tuple[int, ...]]] = []
    for frag in frag_indices:
        prod_count = sum(
            1 for i in frag
            if out_mol.GetAtomWithIdx(i).GetAtomMapNum() in productive_maps
        )
        if prod_count == 0:
            continue
        scored.append((prod_count, frag))
    if not scored:
        return None
    best = max(s[0] for s in scored)
    # Keep fragments whose productive coverage is within a factor of the
    # best; this preserves legitimate co-products (which usually have
    # ~equal coverage to the main product) while dropping catalyst-leak
    # fragments (which carry only 1-2 productive maps vs the main
    # product's 10+).
    threshold = max(1, best // 4)
    keep_idx: list[int] = []
    for prod_count, frag in scored:
        if prod_count >= threshold:
            keep_idx.extend(frag)
    if not keep_idx:
        return None
    rw = Chem.RWMol(out_mol)
    drop = sorted(set(range(out_mol.GetNumAtoms())) - set(keep_idx), reverse=True)
    for idx in drop:
        rw.RemoveAtom(idx)
    return rw.GetMol()


def _build_charge_delta(r_mol: Chem.Mol, p_mol: Chem.Mol) -> torch.Tensor:
    """Ground-truth Δq from R vs P, aligned to ``r_mol`` atom indices."""
    p_map_to_charge = {
        a.GetAtomMapNum(): a.GetFormalCharge()
        for a in p_mol.GetAtoms() if a.GetAtomMapNum() > 0
    }
    cd = torch.zeros(r_mol.GetNumAtoms(), dtype=torch.int64)
    for idx, a in enumerate(r_mol.GetAtoms()):
        m = a.GetAtomMapNum()
        if m in p_map_to_charge:
            cd[idx] = p_map_to_charge[m] - a.GetFormalCharge()
    return cd


def analyse_reaction(
    rxn_id: str, r_smi: str, p_smi: str, label_str: str
) -> dict:
    """Return per-reaction result dict with rule-check + sequential-test outcomes."""
    # --- parse mols + arrows ---
    try:
        arrows = parse_arrows(label_str)
    except Exception as e:
        return {"rxn_id": rxn_id, "status": "arrow_parse_failed", "reason": repr(e)}
    if not isinstance(arrows, list) or not arrows:
        return {"rxn_id": rxn_id, "status": "no_arrows"}

    r_mol = Chem.MolFromSmiles(r_smi)
    p_mol = Chem.MolFromSmiles(p_smi)
    if r_mol is None or p_mol is None:
        return {"rxn_id": rxn_id, "status": "mol_parse_failed"}
    r_mol = align_atoms(r_mol)
    p_mol = align_atoms(p_mol)

    map_to_idx = {
        a.GetAtomMapNum(): a.GetIdx()
        for a in r_mol.GetAtoms() if a.GetAtomMapNum() > 0
    }

    # --- (A) rule check: arrow-sum Δ vs R→P Δ (on shared atoms) ---
    n = r_mol.GetNumAtoms()
    arrow_delta = torch.zeros((n, n), dtype=torch.int64)
    arrow_charge = torch.zeros(n, dtype=torch.int64)
    for ar in arrows:
        _accumulate(arrow_delta, arrow_bond_changes(ar, map_to_idx))
        for idx, dfc in arrow_charge_changes(ar, map_to_idx):
            arrow_charge[idx] += dfc

    try:
        rp_delta_f = DeltaMatrixGenerator.delta_from_reactants_products(r_mol, p_mol)
    except Exception as e:
        return {"rxn_id": rxn_id, "status": "rp_delta_failed", "reason": repr(e)}
    # Round to int; R→P Δ uses float adjacency (aromatic 1.5 etc).
    rp_delta = rp_delta_f.round().to(torch.int64)
    # The rule check must restrict to the *shared* atoms (atoms whose
    # atom-map appears in both R and P). Arrows correctly track bonds to
    # catalytic atoms (maps 301+) and leaving spectators, but R→P Δ zeros
    # them out (they disappear from the product). Comparing on shared atoms
    # only isolates the rule's correctness from this scope mismatch.
    p_maps = {a.GetAtomMapNum() for a in p_mol.GetAtoms() if a.GetAtomMapNum() > 0}
    shared_idxs = sorted(idx for m, idx in map_to_idx.items() if m in p_maps)
    if shared_idxs:
        shared_t = torch.tensor(shared_idxs, dtype=torch.long)
        ad_sub = arrow_delta.index_select(0, shared_t).index_select(1, shared_t)
        rp_sub = rp_delta[:n, :n].index_select(0, shared_t).index_select(1, shared_t)
        rule_match = bool(torch.equal(ad_sub, rp_sub))
        rule_diff_nnz = int((ad_sub != rp_sub).sum().item())
        # Charge rule check: summed arrow Δfc vs (P.fc - R.fc) on shared atoms.
        gt_cd = _build_charge_delta(r_mol, p_mol)
        ac_sub = arrow_charge.index_select(0, shared_t)
        gt_sub = gt_cd.index_select(0, shared_t)
        charge_rule_match = bool(torch.equal(ac_sub, gt_sub))
        charge_diff_nnz = int((ac_sub != gt_sub).sum().item())
    else:
        rule_match = True
        rule_diff_nnz = 0
        charge_rule_match = True
        charge_diff_nnz = 0

    # --- (B) sequential test: chain-grouped elementary steps ---
    steps = parse_steps(label_str, map_to_idx)
    steps_bond_changes = [s.bond_changes for s in steps]
    steps_charge_changes = [s.charge_changes for s in steps]
    step_sizes = [len(s.arrow_indices) for s in steps]

    cur_mol = r_mol
    cur_n = n
    n_fallbacks = 0  # per-step heuristic-charge fallbacks fired in this rxn
    suspect_intermediates = 0  # intermediates that don't round-trip cleanly
    fallback_errors: list[str] = []  # error message that triggered each fallback
    suspect_reasons: list[str] = []  # diagnostic bucket per suspect intermediate
    # Cumulative arrow Δfc across applied steps (for last-step correction).
    arrow_cum_cd = torch.zeros(cur_n, dtype=torch.int64)
    gt_cd_total = _build_charge_delta(r_mol, p_mol)
    for step_i, (bc, cc) in enumerate(zip(steps_bond_changes, steps_charge_changes)):
        is_last = step_i == len(steps_bond_changes) - 1
        step_delta = torch.zeros((cur_n, cur_n), dtype=torch.int64)
        for i, j, s in bc:
            if i < cur_n and j < cur_n:
                step_delta[i, j] += s
                step_delta[j, i] += s
        step_cd = torch.zeros(cur_n, dtype=torch.int64)
        for idx, dfc in cc:
            if idx < cur_n:
                step_cd[idx] += dfc

        if is_last:
            # Force cumulative arrow charges to land on GT for shared atoms.
            # correction = GT_total - (arrow_cum_so_far + step_cd_arrow_last)
            # Apply only on shared atoms; leave non-shared atoms with the
            # arrow-derived charges (they leave with the catalyst fragment).
            correction = torch.zeros(cur_n, dtype=torch.int64)
            for m, idx in map_to_idx.items():
                if m in p_maps and idx < cur_n:
                    target = int(gt_cd_total[idx].item())
                    have = int(arrow_cum_cd[idx].item() + step_cd[idx].item())
                    correction[idx] = target - have
            step_cd = step_cd + correction

        # Snapshot pre-step formal charges so we can recover the *actual*
        # per-atom Δfc applied (whether arrow-derived or heuristic). The
        # last-step cumulative-charge correction depends on this being
        # right even when we fall back to the heuristic mid-rollout.
        pre_fc = [a.GetFormalCharge() for a in cur_mol.GetAtoms()]
        n_pre = cur_mol.GetNumAtoms()
        used_fallback = False
        first_err_msg = ""
        try:
            new_mol = apply_delta(
                cur_mol, step_delta,
                charge_delta=step_cd,
                apply_charge_heuristic=False,
            )
        except (ApplyDeltaError, Exception) as e:
            first_err_msg = getattr(e, "message", repr(e))
            # Fallback: retry the SAME step with the row-sum charge heuristic.
            # The arrow-derived charge rule misses ~7% of patterns (mostly
            # oxonium/iminium formation); the heuristic catches most of them
            # because Δfc = -Σ row delta correctly handles the net bond gain.
            try:
                new_mol = apply_delta(
                    cur_mol, step_delta,
                    apply_charge_heuristic=True,
                )
                used_fallback = True
            except Exception as e2:
                return {
                    "rxn_id": rxn_id,
                    "status": "intermediate_failed",
                    "failed_step": step_i,
                    "n_steps": len(steps_bond_changes),
                    "reason": getattr(e, "reason", "unexpected"),
                    "detail": getattr(e, "message", repr(e)),
                    "fallback_reason": getattr(e2, "reason", "unexpected"),
                    "fallback_detail": getattr(e2, "message", repr(e2)),
                    "rule_match": rule_match,
                    "rule_diff_nnz": rule_diff_nnz,
                    "charge_rule_match": charge_rule_match,
                    "charge_diff_nnz": charge_diff_nnz,
                }

        # Update cumulative tracker with what was *actually* applied.
        actual_step_cd = torch.zeros(cur_n, dtype=torch.int64)
        for i, atom in enumerate(new_mol.GetAtoms()):
            if i < n_pre:
                actual_step_cd[i] = atom.GetFormalCharge() - pre_fc[i]
        arrow_cum_cd = arrow_cum_cd + actual_step_cd
        if used_fallback:
            n_fallbacks += 1
            fallback_errors.append(first_err_msg)
        # Round-trip sanity on the intermediate: the first MolFromSmiles
        # normalises explicit ``[H]`` atoms into implicit-H counts (an
        # expected representation shift, not a chemistry issue). We
        # therefore check idempotency of the *second* round-trip:
        # SMILES -> Mol -> SMILES should be a fixed point. If it isn't,
        # the intermediate carries a real perception instability
        # (aromaticity / tautomer / Kekulé drift).
        try:
            s1 = Chem.MolToSmiles(new_mol)
            rt = Chem.MolFromSmiles(s1)
            if rt is None:
                suspect_intermediates += 1
                suspect_reasons.append("rt_parse_fail")
            else:
                s2 = Chem.MolToSmiles(rt)
                rt2 = Chem.MolFromSmiles(s2)
                if rt2 is None:
                    suspect_intermediates += 1
                    suspect_reasons.append("rt2_parse_fail")
                elif Chem.MolToSmiles(rt2) != s2:
                    suspect_intermediates += 1
                    suspect_reasons.append(_diagnose_roundtrip(rt, rt2, s2))
        except Exception as ex:
            suspect_intermediates += 1
            suspect_reasons.append(f"exception:{type(ex).__name__}")
        cur_mol = new_mol

    productive_maps = {a.GetAtomMapNum() for a in p_mol.GetAtoms() if a.GetAtomMapNum() > 0}
    pred = _extract_productive_fragments(cur_mol, productive_maps)
    base = {
        "rxn_id": rxn_id,
        "n_steps": len(steps_bond_changes),
        "step_sizes": step_sizes,
        "n_fallbacks": n_fallbacks,
        "suspect_intermediates": suspect_intermediates,
        "fallback_errors": fallback_errors,
        "suspect_reasons": suspect_reasons,
        "rule_match": rule_match,
        "rule_diff_nnz": rule_diff_nnz,
        "charge_rule_match": charge_rule_match,
        "charge_diff_nnz": charge_diff_nnz,
    }
    if pred is None:
        return {**base, "status": "no_productive_frag"}
    try:
        Chem.SanitizeMol(pred)
    except Exception as e:
        return {**base, "status": "post_extract_sanitize_failed", "reason": repr(e)}

    pred_canon = _canon_no_maps_no_stereo(pred)
    true_canon = _canon_no_maps_no_stereo(p_mol)
    if pred_canon == true_canon:
        return {**base, "status": "ok"}
    return {**base, "status": "diverged", "pred": pred_canon, "true": true_canon}


def _analyse_row(row: dict) -> dict:
    # Module-level wrapper so multiprocessing.Pool can pickle it.
    return analyse_reaction(row["rxn_id"], row["r_smi"], row["p_smi"], row["label"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/mech-USPTO-31k.csv")
    ap.add_argument("--out", default="results/arrow_parser_verify.json")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--max-examples", type=int, default=8)
    ap.add_argument("--workers", type=int, default=1,
                    help="Process-pool workers. 1 = serial (debuggable). 0 = os.cpu_count().")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= args.limit:
                break
            r, p = row["updated_reaction"].split(">>")
            rows.append({
                "rxn_id": f"rxn_{i:06d}",
                "r_smi": r,
                "p_smi": p,
                "label": row["mechanistic_label"],
            })

    workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)
    if workers == 1:
        results = [
            analyse_reaction(r["rxn_id"], r["r_smi"], r["p_smi"], r["label"])
            for r in tqdm(rows, desc="verify-arrows")
        ]
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        # chunksize tuned for ~30k rows / many workers: keep dispatch cheap
        # while still surfacing tqdm progress smoothly.
        chunksize = max(1, len(rows) // (workers * 32))
        with ctx.Pool(processes=workers) as pool:
            results = list(tqdm(
                pool.imap(_analyse_row, rows, chunksize=chunksize),
                total=len(rows), desc=f"verify-arrows[{workers}w]",
            ))

    n_total = len(results)
    status_counts = Counter(r["status"] for r in results)
    rule_checked = [r for r in results if "rule_match" in r]
    rule_ok = sum(1 for r in rule_checked if r["rule_match"])
    charge_ok = sum(1 for r in rule_checked if r.get("charge_rule_match", False))
    seq_eligible = [r for r in results if r["status"] in ("ok", "diverged", "intermediate_failed", "post_extract_sanitize_failed", "no_productive_frag")]
    seq_ok = status_counts.get("ok", 0)
    ok_results = [r for r in results if r["status"] == "ok"]
    ok_clean = sum(1 for r in ok_results if r.get("n_fallbacks", 0) == 0 and r.get("suspect_intermediates", 0) == 0)
    ok_with_fallback = sum(1 for r in ok_results if r.get("n_fallbacks", 0) > 0)
    ok_with_suspect = sum(1 for r in ok_results if r.get("suspect_intermediates", 0) > 0)

    # Step-size distribution from chain grouping (e.g. 2-arrow Sn1/Sn2,
    # 3-arrow pericyclic). Useful diagnostic for the grouping heuristic.
    step_size_counts: Counter[int] = Counter()
    for r in results:
        for sz in r.get("step_sizes", []) or []:
            step_size_counts[sz] += 1

    examples: dict[str, list] = defaultdict(list)
    all_failures: list[dict] = []
    for r in results:
        if r["status"] in ("diverged", "intermediate_failed"):
            all_failures.append(r)
            if len(examples[r["status"]]) < args.max_examples:
                examples[r["status"]].append(r)
        if r["status"] == "ok" and not r.get("rule_match", True) and len(examples["ok_but_rule_mismatch"]) < args.max_examples:
            examples["ok_but_rule_mismatch"].append(r)

    # Categorize failure detail strings into coarse buckets.
    import re
    def _classify(detail: str) -> tuple[str, str]:
        d = detail or ""
        m = re.search(r"Explicit valence for atom # \d+ ([A-Za-z]+), (\d+)", d)
        if m:
            return ("hypervalent", f"{m.group(1)}={m.group(2)}")
        if "kekulize" in d.lower() or "Can't kekulize" in d:
            return ("kekulize", "")
        if "non-ring atom" in d.lower():
            return ("non_ring_aromatic", "")
        if "Aromatic" in d or "aromatic" in d:
            return ("aromatic_other", "")
        if "valence" in d.lower():
            return ("valence_other", "")
        if "sanitize" in d.lower():
            return ("sanitize_other", "")
        return ("other", d[:60])

    def _categorize_fallback_errors(ok: list[dict]) -> Counter:
        c: Counter = Counter()
        for r in ok:
            for msg in r.get("fallback_errors", []) or []:
                cat, sub = _classify(msg)
                c[f"{cat}:{sub}" if sub else cat] += 1
        return c

    failure_breakdown: dict[str, Counter] = {
        "intermediate_failed": Counter(),
        "diverged": Counter(),
    }
    failed_step_dist: dict[str, Counter] = {
        "intermediate_failed": Counter(),
    }
    for r in all_failures:
        status = r["status"]
        detail = r.get("fallback_detail") or r.get("detail") or ""
        cat, sub = _classify(detail)
        failure_breakdown[status][f"{cat}:{sub}" if sub else cat] += 1
        if status == "intermediate_failed":
            fs = r.get("failed_step")
            ns = r.get("n_steps")
            if fs is not None and ns:
                pos = "first" if fs == 0 else ("last" if fs == ns - 1 else "middle")
                failed_step_dist["intermediate_failed"][pos] += 1

    summary = {
        "n_total": n_total,
        "csv": str(csv_path),
        "rule_check": {
            "checked": len(rule_checked),
            "bond_rule_match": rule_ok,
            "bond_rule_match_rate": rule_ok / max(1, len(rule_checked)),
            "charge_rule_match": charge_ok,
            "charge_rule_match_rate": charge_ok / max(1, len(rule_checked)),
        },
        "sequential": {
            "eligible": len(seq_eligible),
            "ok": seq_ok,
            "ok_rate": seq_ok / max(1, len(seq_eligible)),
            "ok_clean_rollout": ok_clean,
            "ok_with_fallback": ok_with_fallback,
            "ok_with_suspect_intermediates": ok_with_suspect,
        },
        "status_counts": dict(status_counts),
        "step_size_counts": dict(step_size_counts),
        "examples": {k: v for k, v in examples.items()},
        "failure_breakdown": {k: dict(v) for k, v in failure_breakdown.items()},
        "failed_step_position": {k: dict(v) for k, v in failed_step_dist.items()},
        "fallback_error_breakdown": dict(_categorize_fallback_errors(ok_results)),
        "suspect_reason_breakdown": dict(Counter(
            reason
            for r in ok_results
            for reason in (r.get("suspect_reasons") or [])
        )),
        "all_failures": all_failures,
    }

    out_path.write_text(json.dumps(summary, indent=2))

    print()
    print("=" * 60)
    print(f"Total reactions:               {n_total}")
    print(f"Bond-Δ rule match:             {rule_ok}/{len(rule_checked)}  ({100*rule_ok/max(1,len(rule_checked)):.1f}%)")
    print(f"Charge-Δ rule match:           {charge_ok}/{len(rule_checked)}  ({100*charge_ok/max(1,len(rule_checked)):.1f}%)")
    print(f"Sequential eligible:           {len(seq_eligible)}")
    print(f"Sequential ok (final==P):      {seq_ok}  ({100*seq_ok/max(1,len(seq_eligible)):.1f}%)")
    print(f"  of which clean rollout:      {ok_clean}  ({100*ok_clean/max(1,len(ok_results)):.1f}% of ok)")
    print(f"  of which used fallback:      {ok_with_fallback}  ({100*ok_with_fallback/max(1,len(ok_results)):.1f}% of ok)")
    print(f"  of which suspect intermed.:  {ok_with_suspect}  ({100*ok_with_suspect/max(1,len(ok_results)):.1f}% of ok)")
    print(f"Status counts:")
    for k, v in sorted(status_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:35s}  {v:5d}  ({100*v/n_total:.1f}%)")
    print(f"Step-size distribution (arrows per step):")
    for sz, c in sorted(step_size_counts.items()):
        print(f"  {sz}-arrow steps: {c}")
    print(f"\nFull report: {out_path}")


if __name__ == "__main__":
    main()
