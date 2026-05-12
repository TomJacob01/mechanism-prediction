# Newton cluster deployment

Quick-start for deploying `mech_uspto` to the 8× A100 Newton partition
(`cslab` account). For research design and API, see [README.md](README.md).

---

## 1. Layout on the cluster

```
/cluster/data/mech-USPTO-31k/        # input JSON files (provided)
/project/mech_uspto/                  # this repository (sync from local)
/project/results/{stepwise,e2e}/      # training outputs (created automatically)
/project/logs/                        # SLURM stdout/stderr
```

---

## 2. Sync code

From your workstation:

```powershell
# Copy the whole repo (excluding .venv, results, etc. — see .gitignore)
rsync -av --exclude-from=.gitignore . cslab@newton:/project/mech_uspto/
```

---

## 3. Environment setup

On the cluster, once per environment:

```bash
ssh cslab@newton
cd /project/mech_uspto

module load python/3.10 cuda/12.1
python -m venv .venv
source .venv/bin/activate

# Install torch + torch_geometric matching the cluster's CUDA build first,
# then the rest of the package.
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric
pip install -e .

# Smoke test (no GPU required)
pytest -q
```

Set the data path once:

```bash
echo 'export MECH_USPTO_DATA=/cluster/data/mech-USPTO-31k' >> ~/.bashrc
```

---

## 4. SLURM job templates

### Stepwise — `submit_stepwise.sh`

```bash
#!/bin/bash
#SBATCH --job-name=mech-uspto-stepwise
#SBATCH --partition=newton
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --account=cslab
#SBATCH --output=/project/logs/stepwise-%j.log
#SBATCH --error=/project/logs/stepwise-%j.err

source /project/mech_uspto/.venv/bin/activate
cd /project/mech_uspto

python scripts/train.py \
  --task-mode stepwise \
  --batch-size 32 \
  --hidden-dim 256 \
  --num-epochs 100 \
  --output-dir /project/results/stepwise \
  --seed 42
```

### End-to-end — `submit_e2e.sh`

Same as above with `--task-mode end_to_end` and `--output-dir /project/results/e2e`.

Submit:

```bash
sbatch submit_stepwise.sh
sbatch submit_e2e.sh
```

---

## 5. Monitoring

```bash
squeue -u cslab                                # job status
tail -f /project/logs/stepwise-<JOBID>.log     # live log
ls -lh /project/results/stepwise/              # checkpoints + JSON results
```

What to watch:

- **Training loss** decreases monotonically.
- **Validation loss** improves then plateaus — early-stop kicks in after
  20 epochs without improvement.
- **F1** should reach > 0.7 in stepwise; ~0.5–0.7 in end-to-end.
- **Spectator ratio** logged at dataset build time should be 0.85 ± 0.05.

Best checkpoints are saved as `{task_mode}_best_ep<N>.pt`; periodic snapshots
as `{task_mode}_latest_ep<N>.pt`.

---

## 6. Troubleshooting

| Symptom                  | Try                                                                |
| ------------------------ | ------------------------------------------------------------------ |
| CUDA out of memory       | `--batch-size 16` or `--hidden-dim 128`                            |
| Loss is NaN              | Lower class weights in `Config` (e.g. `[2.0, 1.0, 2.0]`)           |
| Slow data loading        | Bump `num_workers` in `create_dataloaders` and add `pin_memory=True` |
| Model not improving      | Check spectator ratio (~0.85), try `gamma_focal=1.5` or `3.0`, lower LR |

---

## 7. Next steps after training

1. Implement an autoregressive inference loop for stepwise mode
   (`predict → apply Δ → repeat` until a stop condition).
2. Implement the FPR (Final Product Recovery) evaluation metric and compare
   the two modes head-to-head.
3. Compare per-mechanism breakdowns to see *where* end-to-end fails.
