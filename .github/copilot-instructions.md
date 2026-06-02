# Copilot instructions — mech_uspto

Pre-loaded context for any AI agent working in this repo. Read [docs/glossary.md](../docs/glossary.md) for domain vocabulary, [docs/architecture.md](../docs/architecture.md) for data flow, and [docs/adr/](../docs/adr/) for "why we chose X" decisions.

## What this repo is
Dual-mode (stepwise vs end-to-end) graph-transformer ablation on the **mech-USPTO-31k** multi-step reaction dataset. Source SMILES + atom-mapped arrows → bond-order Δ matrices → training targets.

## Environment
- Python 3.10, venv at `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python` (Linux).
- Cluster: Newton-L40S at Technion (see [CLUSTER_SETUP.md](../CLUSTER_SETUP.md) and user memory `cluster.md`). Conda env name: `mechuspto`. SLURM partition `newton`, account `cslab`.
- Sync: `sync push` / `sync pull` (PowerShell function; per-project config in `sync.config.ps1`).

## Conventions

| Topic | Rule |
|---|---|
| Arrow grouping | **Chain-rule (`parse_steps`) is canonical.** Validity grouper (`parse_steps_by_validity`) is being deprecated; do not extend it. See [ADR-0001](../docs/adr/0001-canonical-chain-rule-grouper.md). |
| Training pool | **Use the parquet cache** (`data/cache/parquet/`), not raw CSV. The cache contains only the clean-rollout subset (~99.75%). See [ADR-0002](../docs/adr/0002-clean-training-pool-filter.md) and [ADR-0004](../docs/adr/0004-parquet-canonical-cache.md). |
| Charges | `apply_delta` uses a row-sum heuristic by default; pass `charge_delta=...` when ground truth is known. See [ADR-0003](../docs/adr/0003-heuristic-charges-in-apply-delta.md). |
| Δ value range | Stepwise mode: `{-1, 0, 1}`. End-to-end mode: `{-2, -1, 0, 1, 2}` (sometimes ±3 after clamp). |
| Atom indexing | Cache stores changes keyed by **atom-map number** (`map_i`, `map_j`). Convert to RDKit indices at load time via `_delta_from_map_changes`. |
| Splits | `split_hash = sha1(rxn_id)[:4]` mod buckets, 80/10/10 train/val/test. Deterministic across machines. |
| Test command | `pytest -q` from repo root. No GPU required (uses synthetic fixtures). |

## Don't
- **Don't commit `data/cache/parquet/`** — it's a build artifact, regenerated from the CSV.
- **Don't commit `checkpoints/`, `results/`, `slurm-logs/`** — output artifacts.
- **Don't add `--no-verify` to git commands** or otherwise bypass hooks.
- **Don't add new arrow-grouping algorithms** without an ADR explaining why chain-rule is insufficient.
- **Don't change `apply_delta`'s charge semantics** without an ADR.

## Do
- **Run `pytest -q` after any change to `src/mech_uspto/` or `scripts/build_parquet_cache.py`.**
- **Rebuild the cache** (via `scripts/build_parquet_cache.py` or `scripts/slurm/verify_all.sbatch`) after touching arrow_parser, apply_delta, or transformations.
- **Use absolute paths** in tool calls (the agent tools expect them on Windows).
- **Update [docs/repo-map.md](../docs/repo-map.md)** by running `python scripts/gen_repo_map.py` after adding or removing modules.

## Workflows (skills)
For common multi-step procedures, follow these step-by-step playbooks:
- [docs/skills/rebuild-cache.md](../docs/skills/rebuild-cache.md)
- [docs/skills/launch-training.md](../docs/skills/launch-training.md)
- [docs/skills/debug-diverged-reaction.md](../docs/skills/debug-diverged-reaction.md)

## Where things live
See [docs/repo-map.md](../docs/repo-map.md) for the auto-generated symbol index. High-level layout in [docs/architecture.md](../docs/architecture.md). Cluster ops in [CLUSTER_SETUP.md](../CLUSTER_SETUP.md). Data format in [docs/DATA.md](../docs/DATA.md). Model architecture in [docs/MODEL.md](../docs/MODEL.md).
