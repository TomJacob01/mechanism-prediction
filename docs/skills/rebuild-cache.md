# Skill: Rebuild the parquet cache

Use this procedure after any change to: `arrow_parser.py`, `chemistry/apply_delta.py`, `chemistry/valence.py`, `data/transformations.py`, `data/parser.py`, `data/cache_schema.py`, or `scripts/build_parquet_cache.py`.

## When to run
- After a code change that affects rollout (see list above).
- When `data/cache/parquet/cache_manifest.json`'s `built_at` is older than the latest commit touching those files.
- When `git_head` in the manifest is `"unknown"` (built from a dirty tree) and you want a clean rebuild.

## Pre-check
```powershell
# What does the current manifest say?
Get-Content data/cache/parquet/cache_manifest.json | ConvertFrom-Json | Select-Object built_at, keep_rate, skip_reason_breakdown
```

## Local rebuild (Windows, dev machine)
```powershell
.\.venv\Scripts\python.exe scripts/build_parquet_cache.py `
    --csv data/mech-USPTO-31k.csv `
    --out data/cache/parquet `
    --workers 0    # 0 = os.cpu_count()
```
Wall time: ~10 min on 16 cores, ~1.5 min on 64 cores.

## Cluster rebuild (preferred for big rebuilds)
```powershell
sync push                                              # ship local code
wsl bash -c 'ssh "STAFF\tom-jacob@132.68.39.200" "cd ~/projects/mechanism-prediction && sbatch --parsable scripts/slurm/verify_all.sbatch"'
```
The `verify_all.sbatch` job rebuilds the cache as stage 0, then runs the verifiers. Job tag prefix: `mech-verify-<jobid>`. Output: `slurm-logs/mech-verify-<jobid>.out`.

## Post-check
```powershell
# Local
.\.venv\Scripts\python.exe scripts/cache_audit.py --cache data/cache/parquet --out results/cache_audit.json

# Confirm:
# - keep_rate ≥ 0.995 (any drop means the parser/grouper/apply_delta regressed)
# - bond_delta_abs_max == 1 (stepwise Δ must stay in {-1, 0, 1})
# - split_counts roughly 80/10/10 (deterministic from rxn_id hash)
# - no class missing from any split
```

If anything looks wrong, check `results/cache_rollout.json` for which reactions failed and why.

## Pitfalls
- **Don't commit `data/cache/parquet/`.** It's a build artifact (gitignored). The manifest is the only persisted record.
- **`workers=0` means cpu_count.** On the cluster you'll get 32–96 workers depending on the partition; locally it depends on your machine.
- **A rebuild invalidates cached checkpoints' assumptions about step boundaries.** If the grouper changed, the model's step-by-step training meaning shifted, so old checkpoints aren't directly comparable to new training runs.
