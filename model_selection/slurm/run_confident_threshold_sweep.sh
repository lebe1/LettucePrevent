#!/bin/bash
#SBATCH -J conf-thr-sweep
#SBATCH --partition=GPU-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --output=confident_threshold_experiment.txt

set -euo pipefail

echo "${SLURM_JOB_NODELIST}"
nvidia-smi -L

# Hardcode repo path — SLURM copies scripts to /var/spool so dynamic
# resolution via BASH_SOURCE does not work on this cluster.
REPO_DIR="/home/e12133103/LettucePrevent"
BENCHMARK_DIR="${REPO_DIR}/analysis/benchmark"
RESULTS_DIR="${RESULTS_DIR:-/share/${USER}/confident-threshold-sweep-results}"
VENV_PATH="${VENV_PATH:-$HOME/Python312/bin/activate}"
PYTHON_BIN="${PYTHON_BIN:-python}"

WANDB_ENTITY="${WANDB_ENTITY:-lebeccard-technical-university-wien}"
WANDB_PROJECT="${WANDB_PROJECT:-hdm-benchmark-rq3-threshold-sweep}"
SWEEP_COUNT="${SWEEP_COUNT:-8}"
SWEEP_SEED="${SWEEP_SEED:-42}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-rq3_confident_threshold_sweep}"

# Use /share as scratch replacement (A100s VMs have no /scratch)
JOB_TAG="${SLURM_JOB_ID:-manual_$$}"
WORK_DIR="/share/${USER}/scratch_jobs/${JOB_TAG}"
mkdir -p "${WORK_DIR}"
chmod 700 "${WORK_DIR}"
echo "Using WORK_DIR=${WORK_DIR}"
echo "Using BENCHMARK_DIR=${BENCHMARK_DIR}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "WARNING: SLURM_JOB_ID is not set. Script appears to run outside sbatch."
fi

mkdir -p "${RESULTS_DIR}"

# Redirect cache/temp writes to work dir (avoid heavy /home writes)
export SCRATCH_DIR="${WORK_DIR}"
export WANDB_DIR="${WORK_DIR}/wandb"
export WANDB_CACHE_DIR="${WORK_DIR}/.wandb_cache"
export WANDB_DATA_DIR="${WORK_DIR}/.wandb_data"
export WEAVE_DISABLED="true"
export HF_HOME="/share/${USER}/.cache/huggingface"
unset TRANSFORMERS_CACHE || true

mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}" "${HF_HOME}"

# Cleanup on exit — copy results, then remove work dir
cleanup() {
    echo "Copying outputs back to ${RESULTS_DIR}"
    cp -r "${WORK_DIR}"/${OUTPUT_PREFIX}_*.json "${RESULTS_DIR}/" 2>/dev/null || true
    cp -r "${WORK_DIR}"/wandb "${RESULTS_DIR}/wandb_${JOB_TAG}" 2>/dev/null || true
    rm -rf "${WORK_DIR}"
    echo "Job finished. Results in ${RESULTS_DIR}"
}
trap cleanup EXIT

# Activate Python environment
if [[ -f "${VENV_PATH}" ]]; then
    source "${VENV_PATH}"
else
    echo "WARNING: VENV_PATH not found (${VENV_PATH}). Continuing without activation."
fi

# Verify source files exist before copying
if [[ ! -f "${BENCHMARK_DIR}/confident_threshold_comparison.py" ]]; then
    echo "ERROR: ${BENCHMARK_DIR}/confident_threshold_comparison.py not found"
    exit 1
fi
if [[ ! -f "${BENCHMARK_DIR}/hallucination_benchmark.py" ]]; then
    echo "ERROR: ${BENCHMARK_DIR}/hallucination_benchmark.py not found"
    exit 1
fi

# Copy scripts from repo to work dir
cp "${BENCHMARK_DIR}/confident_threshold_comparison.py" "${WORK_DIR}/"
cp "${BENCHMARK_DIR}/hallucination_benchmark.py" "${WORK_DIR}/"
cd "${WORK_DIR}"

export PYTHONPATH="${WORK_DIR}:${PYTHONPATH:-}"

if [[ -n "${SWEEP_ID:-}" ]]; then
    echo "Using existing sweep id: ${SWEEP_ID}"
    "${PYTHON_BIN}" "${WORK_DIR}/confident_threshold_comparison.py" \
        --entity "${WANDB_ENTITY}" \
        --project "${WANDB_PROJECT}" \
        --sweep-id "${SWEEP_ID}" \
        --count "${SWEEP_COUNT}" \
        --seed "${SWEEP_SEED}" \
        --output-prefix "${OUTPUT_PREFIX}"
else
    echo "Creating new sweep and running agent."
    "${PYTHON_BIN}" "${WORK_DIR}/confident_threshold_comparison.py" \
        --entity "${WANDB_ENTITY}" \
        --project "${WANDB_PROJECT}" \
        --count "${SWEEP_COUNT}" \
        --seed "${SWEEP_SEED}" \
        --output-prefix "${OUTPUT_PREFIX}"
fi
