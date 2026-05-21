#!/bin/bash
#SBATCH --job-name=mech-train
#SBATCH --partition=newton
#SBATCH --account=cslab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --error=slurm-logs/%x-%j.out

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

# Expandable-segments allocator: lets PyTorch grow CUDA memory segments
# contiguously instead of pinning fixed pools. Required for graph batches
# with high N variance — fixed allocator fragments 30+ GB of cached memory
# into unusable chunks and OOMs on the next big tensor request (observed
# epoch 2 of job 68210046, May 2026: 9.8 GB allocated + 28 GB reserved but
# fragmented; needed an 18 GB contiguous block). Cost: trivial.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Default args (overridable from CLI) -----------------------------------
CSV_DEFAULT=/home/tom-jacob/data/mech-USPTO-31k/mech-USPTO-31k.csv

# If --csv not passed, prepend default.
if [[ "$*" != *"--csv"* ]]; then
    set -- --csv "$CSV_DEFAULT" "$@"
fi

# Tag the output dir with the job id so concurrent runs don't collide.
OUTPUT_DIR="results/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
if [[ "$*" != *"--output-dir"* ]]; then
    set -- "$@" --output-dir "$OUTPUT_DIR"
fi

# Task mode controls which JSON file the engine writes
# (engine.save_results -> <output_dir>/<task_mode>_results.json).
TASK_MODE="end_to_end"
if [[ "$*" == *"--task-mode"*"stepwise"* ]]; then
    TASK_MODE="stepwise"
fi

# --- Mode-specific loss-shaping defaults (overridable from CLI) ------------
# These mirror the Config defaults adopted after the May 2026 runs showed
# strong rare-class weights + gamma=2 focal pushed the model into inverted
# collapse (RxnPreds > targets by epoch 8). Setting them here explicitly
# documents the run config in the slurm log AND guards against a future
# Config edit silently changing what this sbatch ships.
if [[ "$TASK_MODE" == "end_to_end" && "$*" != *"--class-weights"* ]]; then
    set -- "$@" --class-weights "1,1.5,3,1,5,1.5,1"
fi
if [[ "$*" != *"--warmup-steps"* ]]; then
    set -- "$@" --warmup-steps "130"
fi

# Always fail fast on accidental CPU runs on the cluster. If the conda env
# ever loses its CUDA-build torch, this aborts before training instead of
# burning the wall-clock at ~1/50 GPU throughput. Override by removing the
# flag explicitly for CPU-only debugging runs.
if [[ "$*" != *"--require-cuda"* ]]; then
    set -- "$@" --require-cuda
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
echo "Training finished: $(date -Iseconds)"
echo "================================================================"

# --- Auto-eval + auto-plot -------------------------------------------------
# Pick the best checkpoint (highest epoch with the "best" tag) from this run's
# per-run checkpoint dir. Per-run dir is set automatically by Config when
# checkpoint_dir is not overridden.
RESULTS_JSON="${OUTPUT_DIR}/${TASK_MODE}_results.json"
CKPT_DIR="${OUTPUT_DIR}/checkpoints"
BEST_CKPT="$(ls -1 ${CKPT_DIR}/${TASK_MODE}_best_ep*.pt 2>/dev/null \
    | sort -V | tail -n 1)"

# Pick up --csv that was either user-supplied or defaulted above.
CSV_ARG=""
prev=""
for tok in "$@"; do
    if [[ "$prev" == "--csv" ]]; then
        CSV_ARG="$tok"
        break
    fi
    prev="$tok"
done

if [[ -n "$BEST_CKPT" ]]; then
    echo "Auto-eval: $BEST_CKPT on test split"
    python -u scripts/evaluate.py \
        --checkpoint "$BEST_CKPT" \
        --csv "$CSV_ARG" \
        --split test \
        --output "${OUTPUT_DIR}/test_eval.json" \
        --confusion-csv "${OUTPUT_DIR}/test_confusion.csv" \
        || echo "WARN: auto-eval failed (non-fatal)"
else
    echo "WARN: no best checkpoint found in ${CKPT_DIR}, skipping auto-eval"
fi

if [[ -f "$RESULTS_JSON" ]]; then
    echo "Auto-plot: $RESULTS_JSON"
    python -u scripts/plot_history.py "$RESULTS_JSON" \
        || echo "WARN: auto-plot failed (non-fatal)"
else
    echo "WARN: $RESULTS_JSON not found, skipping auto-plot"
fi

echo "================================================================"
echo "Finished:   $(date -Iseconds)"
echo "================================================================"
