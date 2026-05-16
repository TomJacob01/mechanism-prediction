#!/bin/bash
#SBATCH --job-name=mech-train
#SBATCH --partition=newton
#SBATCH --account=cslab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --error=slurm-logs/%x-%j.err

# Usage:
#   sbatch scripts/slurm/train.sbatch [extra train.py args...]
#
# Example:
#   sbatch scripts/slurm/train.sbatch \
#       --task-mode end_to_end --num-epochs 50 --batch-size 128 --hidden-dim 128
#
# All args after the script name are forwarded verbatim to scripts/train.py.

set -eo pipefail   # NOTE: no `-u` — conda activation hooks reference unset vars (e.g. MKL_INTERFACE_LAYER)

mkdir -p slurm-logs

# --- Environment -----------------------------------------------------------
# Use the conda init script (NOT `source ~/miniconda3/bin/activate`), which
# would re-process the sbatch script's $@ and forward them to `conda activate`.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mechuspto

# Re-enable strict unset checking AFTER conda is activated.
set -u

cd ~/projects/mechanism-prediction

# --- Default args (overridable from CLI) -----------------------------------
CSV_DEFAULT=/home/tom-jacob/data/mech-USPTO-31k/mech-USPTO-31k.csv

# If --csv not passed, prepend default.
if [[ "$*" != *"--csv"* ]]; then
    set -- --csv "$CSV_DEFAULT" "$@"
fi

# Tag the output dir with the job id so concurrent runs don't collide.
if [[ "$*" != *"--output-dir"* ]]; then
    set -- "$@" --output-dir "results/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
fi

echo "================================================================"
echo "Job:        ${SLURM_JOB_ID} on $(hostname)"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started:    $(date -Iseconds)"
echo "Command:    python -u scripts/train.py $*"
echo "================================================================"

# -u: unbuffered stdout/stderr so the slurm log tails live.
python -u scripts/train.py "$@"

echo "================================================================"
echo "Finished:   $(date -Iseconds)"
echo "================================================================"
