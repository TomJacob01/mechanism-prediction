"""Build the canonical Parquet cache from the verified rollout.

For each reaction in the CSV:
  1. Parse mapped reactant/product SMILES + curly-arrow label.
  2. Roll out the elementary steps via :func:`apply_delta`, using the same
     logic as ``scripts/verify_arrow_parser.py``.
  3. If the final intermediate matches the ground-truth product
     (canonical SMILES, no maps/stereo) **AND** the rollout used no
     heuristic-charge fallbacks **AND** no intermediate is suspect, emit:
       - one row in ``reactions.parquet``
       - one row per elementary step in ``steps.parquet``
  4. Otherwise skip the reaction. The whole point of "safe training pool"
     is to exclude these.

Also writes ``cache_manifest.json`` with build metadata: source CSV
fingerprint, git HEAD, counts, totals, schema version.

Usage::

    python scripts/build_parquet_cache.py \
        --csv data/mech-USPTO-31k.csv \
        --out data/cache/parquet \
        --workers 0

(Mirrors the verifier's CLI; ``workers=0`` ⇒ ``os.cpu_count()``.)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
import torch
from rdkit import Chem, RDLogger
from tqdm.auto import tqdm

from mech_uspto.chemistry import ApplyDeltaError, apply_delta
from mech_uspto.data.arrow_parser import parse_arrows, parse_steps
from mech_uspto.data.cache_schema import (
    REACTIONS_SCHEMA,
    STEPS_SCHEMA,
    open_writer,
    split_hash,
    write_batch,
)
from mech_uspto.data.featurization import align_atoms
from mech_uspto.data.transformations import DeltaMatrixGenerator

RDLogger.DisableLog("rdApp.*")

SCHEMA_VERSION = 1
BATCH_FLUSH_EVERY = 256  # reactions per parquet batch flush


# ---------------------------------------------------------------------------
# Per-reaction rollout (clean-only; returns parquet-ready dicts or None)
# ---------------------------------------------------------------------------

@dataclass
class _RolloutResult:
    """Output of :func:`rollout_clean`: parquet rows or skip reason."""

    rxn_row: dict | None = None
    step_rows: list[dict] | None = None
    skip_reason: str | None = None
    n_steps: int = 0


def _build_charge_delta(r_mol: Chem.Mol, p_mol: Chem.Mol) -> torch.Tensor:
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


def _canon_no_maps_no_stereo(mol: Chem.Mol) -> str:
    m = Chem.Mol(mol)
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    smi = Chem.MolToSmiles(m, isomericSmiles=False)
    rt = Chem.MolFromSmiles(smi)
    if rt is not None:
        return Chem.MolToSmiles(rt, isomericSmiles=False)
    return smi


def _extract_productive_fragments(
    out_mol: Chem.Mol, productive_maps: set[int]
) -> Chem.Mol | None:
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


def _suspect_intermediate(mol: Chem.Mol) -> bool:
    """Idempotency check: canonical(canonical(SMILES)) must be a fixed point.

    The first round-trip normalises explicit ``[H]`` -> implicit H counts
    (a representation shift, not a chemistry issue). We therefore evaluate
    stability of the *second* round-trip — if that's not idempotent, the
    intermediate carries real perception drift (aromaticity / tautomer /
    Kekulé instability) and we shouldn't put it in the training cache.
    """
    try:
        s1 = Chem.MolToSmiles(mol)
        rt = Chem.MolFromSmiles(s1)
        if rt is None:
            return True
        s2 = Chem.MolToSmiles(rt)
        rt2 = Chem.MolFromSmiles(s2)
        if rt2 is None:
            return True
        return Chem.MolToSmiles(rt2) != s2
    except Exception:
        return True


def _index_changes_to_map_keyed(
    bc: list[tuple[int, int, int]],
    cc: list[tuple[int, int]],
    idx_to_map: dict[int, int],
) -> tuple[list[dict], list[dict]]:
    """Convert (idx, idx, delta) -> (map, map, delta), dropping unmapped atoms."""
    bond_rows: list[dict] = []
    for i, j, d in bc:
        mi = idx_to_map.get(i)
        mj = idx_to_map.get(j)
        if mi is None or mj is None:
            continue
        bond_rows.append({"map_i": int(mi), "map_j": int(mj), "delta": int(d)})
    charge_rows: list[dict] = []
    for idx, d in cc:
        mi = idx_to_map.get(idx)
        if mi is None:
            continue
        charge_rows.append({"map_i": int(mi), "delta": int(d)})
    return bond_rows, charge_rows


def rollout_clean(
    rxn_id: str,
    r_smi: str,
    p_smi: str,
    label_str: str,
    mechanistic_class: str,
    data_source: str,
) -> _RolloutResult:
    """Run a single reaction's rollout. Return parquet rows iff fully clean."""
    # --- parse ---
    try:
        arrows = parse_arrows(label_str)
    except Exception as e:
        return _RolloutResult(skip_reason=f"arrow_parse_failed:{type(e).__name__}")
    if not isinstance(arrows, list) or not arrows:
        return _RolloutResult(skip_reason="no_arrows")

    r_mol = Chem.MolFromSmiles(r_smi)
    p_mol = Chem.MolFromSmiles(p_smi)
    if r_mol is None or p_mol is None:
        return _RolloutResult(skip_reason="mol_parse_failed")
    r_mol = align_atoms(r_mol)
    p_mol = align_atoms(p_mol)

    map_to_idx = {
        a.GetAtomMapNum(): a.GetIdx()
        for a in r_mol.GetAtoms() if a.GetAtomMapNum() > 0
    }
    p_maps = {a.GetAtomMapNum() for a in p_mol.GetAtoms() if a.GetAtomMapNum() > 0}

    steps = parse_steps(label_str, map_to_idx)
    if not steps:
        return _RolloutResult(skip_reason="no_steps")

    # --- rollout ---
    cur_mol = r_mol
    cur_n = cur_mol.GetNumAtoms()
    arrow_cum_cd = torch.zeros(cur_n, dtype=torch.int64)
    gt_cd_total = _build_charge_delta(r_mol, p_mol)

    step_payloads: list[dict] = []

    for step_i, step in enumerate(steps):
        bc = step.bond_changes
        cc = step.charge_changes
        is_last = step_i == len(steps) - 1
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
            correction = torch.zeros(cur_n, dtype=torch.int64)
            for m, idx in map_to_idx.items():
                if m in p_maps and idx < cur_n:
                    target = int(gt_cd_total[idx].item())
                    have = int(arrow_cum_cd[idx].item() + step_cd[idx].item())
                    correction[idx] = target - have
            step_cd = step_cd + correction

        pre_fc = [a.GetFormalCharge() for a in cur_mol.GetAtoms()]
        n_pre = cur_mol.GetNumAtoms()
        pre_idx_to_map = {
            a.GetIdx(): a.GetAtomMapNum()
            for a in cur_mol.GetAtoms() if a.GetAtomMapNum() > 0
        }
        mol_pre_bytes = cur_mol.ToBinary()

        used_fallback = False
        try:
            new_mol = apply_delta(
                cur_mol, step_delta,
                charge_delta=step_cd,
                apply_charge_heuristic=False,
            )
        except (ApplyDeltaError, Exception):
            try:
                new_mol = apply_delta(
                    cur_mol, step_delta,
                    apply_charge_heuristic=True,
                )
                used_fallback = True
            except Exception:
                return _RolloutResult(
                    skip_reason=f"intermediate_failed:step={step_i}",
                    n_steps=len(steps),
                )

        # Reject the rxn if this step used a fallback (clean-only).
        if used_fallback:
            return _RolloutResult(
                skip_reason=f"used_fallback:step={step_i}",
                n_steps=len(steps),
            )

        # Reject the rxn if the resulting intermediate is suspect.
        if _suspect_intermediate(new_mol):
            return _RolloutResult(
                skip_reason=f"suspect_intermediate:step={step_i}",
                n_steps=len(steps),
            )

        # Convert this step's index-keyed changes to map-keyed.
        bond_map_rows, charge_map_rows = _index_changes_to_map_keyed(
            bc, cc, pre_idx_to_map
        )

        step_payloads.append({
            "rxn_id": rxn_id,
            "step_idx": int(step_i),
            "mol_pre": mol_pre_bytes,
            "mol_post": new_mol.ToBinary(),
            "bond_changes": bond_map_rows,
            "charge_changes": charge_map_rows,
            "arrow_count": int(len(step.arrow_indices)),
        })

        # Update cumulative charge tracker with what was actually applied.
        actual_step_cd = torch.zeros(cur_n, dtype=torch.int64)
        for i, atom in enumerate(new_mol.GetAtoms()):
            if i < n_pre:
                actual_step_cd[i] = atom.GetFormalCharge() - pre_fc[i]
        arrow_cum_cd = arrow_cum_cd + actual_step_cd
        cur_mol = new_mol

    # --- final-product check ---
    pred = _extract_productive_fragments(cur_mol, p_maps)
    if pred is None:
        return _RolloutResult(skip_reason="no_productive_frag", n_steps=len(steps))
    try:
        Chem.SanitizeMol(pred)
    except Exception:
        return _RolloutResult(
            skip_reason="post_extract_sanitize_failed", n_steps=len(steps)
        )

    if _canon_no_maps_no_stereo(pred) != _canon_no_maps_no_stereo(p_mol):
        return _RolloutResult(skip_reason="diverged", n_steps=len(steps))

    # Clean! Build the reaction row.
    rxn_row = {
        "rxn_id": rxn_id,
        "n_steps": int(len(steps)),
        "n_atoms_mapped": int(len(map_to_idx)),
        "mechanistic_class": mechanistic_class,
        "mechanistic_label": label_str,
        "data_source": data_source,
        "reactant_mol": r_mol.ToBinary(),
        "product_mol": p_mol.ToBinary(),
        "split_hash": split_hash(rxn_id),
    }
    return _RolloutResult(
        rxn_row=rxn_row, step_rows=step_payloads, n_steps=len(steps)
    )


# ---------------------------------------------------------------------------
# Multiprocessing wrapper
# ---------------------------------------------------------------------------

def _rollout_row(row: dict) -> _RolloutResult:
    return rollout_clean(
        rxn_id=row["rxn_id"],
        r_smi=row["r_smi"],
        p_smi=row["p_smi"],
        label_str=row["label"],
        mechanistic_class=row["mechanistic_class"],
        data_source=row["data_source"],
    )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _git_head() -> str:
    """Return ``<sha>`` for a clean tree or ``<sha>-dirty`` if there are uncommitted changes.

    Returns ``"unknown"`` if git is unavailable or this is not a git repo. The
    dirty marker uses ``git status --porcelain`` (any modified, added, or
    untracked file flips it). Recorded in the cache manifest so future readers
    can answer "which code built this cache?" without guessing.
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"
    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return sha
    return f"{sha}-dirty" if dirty else sha


def _is_tree_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return False
    return bool(out)


def _csv_fingerprint(csv_path: Path) -> dict:
    st = csv_path.stat()
    return {
        "path": str(csv_path),
        "size_bytes": st.st_size,
        "mtime": int(st.st_mtime),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/mech-USPTO-31k.csv")
    ap.add_argument("--out", default="data/cache/parquet")
    ap.add_argument("--limit", type=int, default=999_999)
    ap.add_argument("--workers", type=int, default=1,
                    help="Process-pool workers. 1 = serial (debug). 0 = os.cpu_count().")
    ap.add_argument("--strict", action="store_true",
                    help="Refuse to build when the git working tree is dirty (for CI / reproducible runs).")
    args = ap.parse_args()

    if args.strict and _is_tree_dirty():
        print(
            "[build_parquet_cache] ERROR: --strict was set but git working tree is dirty. "
            "Commit or stash changes before rebuilding the cache.",
            file=sys.stderr,
        )
        sys.exit(2)

    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    reactions_path = out_dir / "reactions.parquet"
    steps_path = out_dir / "steps.parquet"
    manifest_path = out_dir / "cache_manifest.json"

    # --- read CSV into row dicts ---
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
                "mechanistic_class": row.get("mechanistic_class", "") or "",
                "data_source": row.get("data_source", "") or "",
            })

    workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)

    # --- run rollouts ---
    rxn_writer = open_writer(reactions_path, REACTIONS_SCHEMA)
    step_writer = open_writer(steps_path, STEPS_SCHEMA)

    rxn_batch: list[dict] = []
    step_batch: list[dict] = []

    n_kept = 0
    n_steps_kept = 0
    skip_counts: Counter[str] = Counter()
    t0 = time.time()

    def _flush() -> None:
        nonlocal rxn_batch, step_batch
        if rxn_batch:
            write_batch(rxn_writer, rxn_batch, REACTIONS_SCHEMA)
            rxn_batch = []
        if step_batch:
            write_batch(step_writer, step_batch, STEPS_SCHEMA)
            step_batch = []

    try:
        if workers == 1:
            iterator = (
                _rollout_row(r)
                for r in tqdm(rows, desc="build-cache")
            )
        else:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            chunksize = max(1, len(rows) // (workers * 32))
            pool = ctx.Pool(processes=workers)
            iterator = tqdm(
                pool.imap(_rollout_row, rows, chunksize=chunksize),
                total=len(rows), desc=f"build-cache[{workers}w]",
            )

        for result in iterator:
            if result.rxn_row is not None:
                rxn_batch.append(result.rxn_row)
                step_batch.extend(result.step_rows or [])
                n_kept += 1
                n_steps_kept += len(result.step_rows or [])
                if len(rxn_batch) >= BATCH_FLUSH_EVERY:
                    _flush()
            else:
                skip_counts[result.skip_reason or "unknown"] += 1
        _flush()
    finally:
        rxn_writer.close()
        step_writer.close()
        if workers != 1:
            pool.close()
            pool.join()

    elapsed = time.time() - t0

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "built_at": int(time.time()),
        "git_head": _git_head(),
        "source_csv": _csv_fingerprint(csv_path),
        "n_rows_input": len(rows),
        "n_reactions_kept": n_kept,
        "n_steps_kept": n_steps_kept,
        "keep_rate": n_kept / max(1, len(rows)),
        "skip_reason_breakdown": dict(skip_counts),
        "workers": workers,
        "elapsed_sec": round(elapsed, 2),
        "outputs": {
            "reactions": str(reactions_path),
            "steps": str(steps_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # --- summary ---
    print()
    print("=" * 60)
    print(f"Input rows:           {len(rows)}")
    print(f"Reactions kept:       {n_kept}  ({100 * n_kept / max(1, len(rows)):.2f}%)")
    print(f"Steps kept:           {n_steps_kept}")
    print(f"Elapsed:              {elapsed:.1f}s ({workers}w)")
    print(f"Reactions parquet:    {reactions_path} "
          f"({reactions_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Steps parquet:        {steps_path} "
          f"({steps_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Manifest:             {manifest_path}")
    if skip_counts:
        print("Skip reasons:")
        for k, v in sorted(skip_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {k:40s}  {v:5d}")

    # Verify the written files parse back.
    rxn_meta = pq.read_metadata(reactions_path)
    step_meta = pq.read_metadata(steps_path)
    assert rxn_meta.num_rows == n_kept, (rxn_meta.num_rows, n_kept)
    assert step_meta.num_rows == n_steps_kept, (step_meta.num_rows, n_steps_kept)
    print(f"\nVerified: reactions={rxn_meta.num_rows} rows, "
          f"steps={step_meta.num_rows} rows")


if __name__ == "__main__":
    main()
