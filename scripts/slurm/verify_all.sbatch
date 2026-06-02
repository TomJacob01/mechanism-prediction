#!/bin/bash
#SBATCH --job-name=mech-verify
#SBATCH --partition=newton
#SBATCH --account=cslab
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --error=slurm-logs/%x-%j.out

# Full pre-training verification sweep (CPU-only).
#
# Stages:
#   0. Rebuild parquet cache from current arrow_parser + apply_delta.
#   1. T1.1 verify_apply_delta_e2e on full corpus.
#   2. T1.2 verify_cache_rollout on every cached reaction.
#   3. T1.4 + T2.* cache_audit (single parquet scan).
#   4. pytest -q (unit tests).
#
# Each stage writes to results/verify_<jobid>/ ; the script keeps going
# past per-stage failures so we get a complete picture in one shot, but
# exits non-zero at the end if anything failed.

set -eo pipefail
mkdir -p slurm-logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mechuspto
set -u

cd ~/projects/mechanism-prediction

OUT_DIR="results/verify-${SLURM_JOB_ID}"
mkdir -p "${OUT_DIR}"
CSV=/home/tom-jacob/data/mech-USPTO-31k/mech-USPTO-31k.csv

# Allow CSV to live in the repo too (fallback).
if [[ ! -f "${CSV}" ]]; then
    CSV=data/mech-USPTO-31k.csv
fi

declare -A STATUS

run_stage() {
    local name="$1"; shift
    echo ""
    echo "================ STAGE: ${name} ================"
    if "$@"; then
        STATUS[$name]="OK"
    else
        STATUS[$name]="FAIL($?)"
    fi
}

# ---- Stage 0: rebuild cache --------------------------------------------
run_stage "rebuild_cache" \
    python scripts/build_parquet_cache.py \
        --csv "${CSV}" \
        --out data/cache/parquet \
        --workers 0

# ---- Stage 1: T1.1 e2e on full corpus ---------------------------------
run_stage "verify_apply_delta_e2e" \
    python scripts/verify_apply_delta_e2e.py \
        --limit 0 \
        --out "${OUT_DIR}/apply_delta_e2e_full.json"

# ---- Stage 2: T1.2 cache rollout --------------------------------------
run_stage "verify_cache_rollout" \
    python scripts/verify_cache_rollout.py \
        --cache data/cache/parquet \
        --out "${OUT_DIR}/cache_rollout.json"

# ---- Stage 3: T1.4 + T2.* cache audit ---------------------------------
run_stage "cache_audit" \
    python scripts/cache_audit.py \
        --cache data/cache/parquet \
        --out "${OUT_DIR}/cache_audit.json"

# ---- Stage 4: unit tests ----------------------------------------------
run_stage "pytest" \
    python -m pytest -q --maxfail=10

# ---- Summary -----------------------------------------------------------
echo ""
echo "================ SUMMARY ================"
fail=0
for k in "${!STATUS[@]}"; do
    printf "  %-30s %s\n" "$k" "${STATUS[$k]}"
    [[ "${STATUS[$k]}" == "OK" ]] || fail=1
done
echo "OUT_DIR=${OUT_DIR}"
exit "${fail}"
