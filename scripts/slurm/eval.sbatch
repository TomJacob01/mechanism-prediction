#!/bin/bash
#SBATCH --job-name=mech-eval
#SBATCH --partition=newton
#SBATCH --account=cslab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --error=slurm-logs/%x-%j.err

# Usage:
#   sbatch scripts/slurm/eval.sbatch --checkpoint checkpoints/end_to_end_best_ep28.pt
#   sbatch scripts/slurm/eval.sbatch \
#       --checkpoint checkpoints/end_to_end_best_ep28.pt \
#       --split test \
#       --output results/mech-train-68205034/test_eval.json \
#       --confusion-csv results/mech-train-68205034/test_confusion.csv
#
# All args after the script name are forwarded verbatim to scripts/evaluate.py.

set -eo pipefail

mkdir -p slurm-logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mechuspto
set -u

cd ~/projects/mechanism-prediction

CSV_DEFAULT=/home/tom-jacob/data/mech-USPTO-31k/mech-USPTO-31k.csv

if [[ "$*" != *"--csv"* ]]; then
    set -- --csv "$CSV_DEFAULT" "$@"
fi

echo "================================================================"
echo "Job:        ${SLURM_JOB_ID} on $(hostname)"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started:    $(date -Iseconds)"
echo "Command:    python -u scripts/evaluate.py $*"
echo "================================================================"

python -u scripts/evaluate.py "$@"

echo "================================================================"
echo "Finished:   $(date -Iseconds)"
echo "================================================================"
