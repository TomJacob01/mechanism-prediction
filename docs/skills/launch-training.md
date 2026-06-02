# Skill: Launch a training run

## Prerequisites
- Parquet cache exists at `data/cache/parquet/` and is recent (see [rebuild-cache.md](rebuild-cache.md)).
- `pytest -q` passes locally.
- For cluster runs: code is in sync (`sync push`) and you've decided on a budget.

## Local smoke (CPU, <5 min)
```powershell
.\.venv\Scripts\python.exe scripts/train.py `
    --task-mode stepwise `
    --batch-size 16 `
    --num-epochs 1 `
    --limit 200 `
    --hidden-dim 64 `
    --output-dir results/smoke_local
```
Use this to validate that a code change doesn't crash the training loop end-to-end before submitting to the cluster. Loss should decrease over the 1 epoch; if not, abort and debug locally.

## Cluster training (GPU)
```powershell
# Submit a real training run
wsl bash -c 'ssh "STAFF\tom-jacob@132.68.39.200" "cd ~/projects/mechanism-prediction && sbatch --parsable scripts/slurm/train.sbatch --task-mode end_to_end --num-epochs 50 --batch-size 128 --hidden-dim 128"'
```
Returns the jobid. Defaults baked into `scripts/slurm/train.sbatch`:
- partition `newton`, account `cslab`, 1 GPU, 8 CPUs, 32 GB RAM, 24h walltime
- `--class-weights "1,1.5,3,1,5,1.5,1"` (end-to-end only, see comment in sbatch)
- `--warmup-steps 130`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Output dir: `results/mech-train-<jobid>/`

Override any of the above by passing the flag explicitly — your CLI args take precedence over the sbatch defaults.

## Monitor
```powershell
# Job state
wsl bash -c 'ssh "STAFF\tom-jacob@132.68.39.200" "squeue -j <jobid> -o %.10i\ %.20j\ %.2t\ %.10M\ %R"'

# Live tail (Ctrl-C to detach; job keeps running)
wsl bash -c 'ssh "STAFF\tom-jacob@132.68.39.200" "tail -F ~/projects/mechanism-prediction/slurm-logs/mech-train-<jobid>.out"'
```

## Pull results
```powershell
sync pull                          # pulls results/ + checkpoints/
sync pull -Only results            # results only (faster, no big .pt files)
```

## Pitfalls
- **Conda activation in sbatch.** Use `source ~/miniconda3/etc/profile.d/conda.sh && conda activate mechuspto`. Never `source ~/miniconda3/bin/activate` — it re-processes `$@` and breaks. Already correct in `train.sbatch`.
- **`set -u` before conda activation breaks.** The mechuspto env's activate hooks reference unset vars. Set `set -eo pipefail` first; `set -u` only *after* `conda activate`.
- **`squeue --me` not supported.** Use `squeue -u "$(id -u)"` or `squeue | grep tom-jacob`.
- **The default `USER` column in squeue is 8 chars.** Widen with `-o "%.10i %.20j %.20u %.2t %.10M %.6D %R"`.
- **CRLF line endings in sbatch files cause SLURM to reject them** with "Batch script contains DOS line breaks". Fix: ` $c = [IO.File]::ReadAllText($f); [IO.File]::WriteAllText($f, ($c -replace "`r`n","`n"))` before pushing, or run `sed -i 's/\r$//'` on the cluster.

## After training
- Pull results.
- Plot loss curves: `python scripts/plot_history.py results/mech-train-<jobid>/`
- Evaluate the best checkpoint: `python scripts/evaluate.py --checkpoint checkpoints/<...>.pt --split test`
