#!/bin/bash
#SBATCH -J lp-thr-sweep-uncal
#SBATCH --partition=GPU-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a40:1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --output=lettuceprevent_threshold_sweep_uncal_%j.txt

set -euo pipefail

echo "${SLURM_JOB_NODELIST}"
nvidia-smi -L

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
REPO_DIR="/home/e12133103/LettucePrevent"
BENCHMARK_DIR="${REPO_DIR}/model_selection"
RESULTS_DIR="${RESULTS_DIR:-/share/${USER}/lettuceprevent-rq3-results}"
VENV_PATH="${VENV_PATH:-${REPO_DIR}/Python312/bin/activate}"
PYTHON_BIN="${PYTHON_BIN:-python}"

WANDB_ENTITY="${WANDB_ENTITY:-lebeccard-technical-university-wien}"
WANDB_PROJECT="${WANDB_PROJECT:-hdm-rq3-threshold-sweep-lettuceprevent}"
SWEEP_COUNT="${SWEEP_COUNT:-4}"
SWEEP_SEED="${SWEEP_SEED:-42}"
SWEEP_PROMPTS_PER_TASK="${SWEEP_PROMPTS_PER_TASK:-50}"
SWEEP_OUTPUT_PREFIX="${SWEEP_OUTPUT_PREFIX:-rq3_lettuceprevent_sweep}"

# Standalone job — fresh WORK_DIR keyed on SLURM_JOB_ID
JOB_TAG="${SLURM_JOB_ID:-manual_$$}"
WORK_DIR="/share/${USER}/scratch_jobs/lp_uncal_${JOB_TAG}"
mkdir -p "${WORK_DIR}"
chmod 700 "${WORK_DIR}"

echo "Using WORK_DIR=${WORK_DIR}"
echo "Using BENCHMARK_DIR=${BENCHMARK_DIR}"

mkdir -p "${RESULTS_DIR}"

# ----------------------------------------------------------------------------
# Cache / temp redirection
# ----------------------------------------------------------------------------
export SCRATCH_DIR="${WORK_DIR}"
export WANDB_DIR="${WORK_DIR}/wandb"
export WANDB_CACHE_DIR="${WORK_DIR}/.wandb_cache"
export WANDB_DATA_DIR="${WORK_DIR}/.wandb_data"
export WEAVE_DISABLED="true"
export HF_HOME="/share/${USER}/.cache/huggingface"
unset TRANSFORMERS_CACHE || true
mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}" "${HF_HOME}"

# ----------------------------------------------------------------------------
# HF auth
# ----------------------------------------------------------------------------
HF_TOKEN_FILE_DEFAULT="${HOME}/.cache/huggingface/token"
HF_TOKEN_FILE_LEGACY="${HOME}/.huggingface/token"
if [[ -z "${HF_TOKEN:-}" ]]; then
    if [[ -f "${HF_TOKEN_FILE_DEFAULT}" ]]; then
        export HF_TOKEN="$(cat "${HF_TOKEN_FILE_DEFAULT}")"
        echo "[INFO] HF_TOKEN loaded from ${HF_TOKEN_FILE_DEFAULT} (length: ${#HF_TOKEN})"
    elif [[ -f "${HF_TOKEN_FILE_LEGACY}" ]]; then
        export HF_TOKEN="$(cat "${HF_TOKEN_FILE_LEGACY}")"
        echo "[INFO] HF_TOKEN loaded from ${HF_TOKEN_FILE_LEGACY} (length: ${#HF_TOKEN})"
    else
        echo "[ERROR] No HF token cache found."
        exit 1
    fi
fi
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"

# ----------------------------------------------------------------------------
# Cleanup on exit — copy results, then remove work dir
# ----------------------------------------------------------------------------
cleanup() {
    echo "Copying outputs back to ${RESULTS_DIR}"
    cp -r "${WORK_DIR}"/${SWEEP_OUTPUT_PREFIX}_*.json "${RESULTS_DIR}/" 2>/dev/null || true
    cp -r "${WORK_DIR}"/wandb                          "${RESULTS_DIR}/wandb_uncal_${JOB_TAG}" 2>/dev/null || true
    rm -rf "${WORK_DIR}"
    echo "Job complete. Results in ${RESULTS_DIR}"
}
trap cleanup EXIT

# ----------------------------------------------------------------------------
# Activate venv
# ----------------------------------------------------------------------------
if [[ -f "${VENV_PATH}" ]]; then
    source "${VENV_PATH}"
else
    echo "WARNING: VENV_PATH not found (${VENV_PATH}). Continuing without activation."
fi

# ----------------------------------------------------------------------------
# Auth sanity check
# ----------------------------------------------------------------------------
echo "[INFO] Verifying HF auth on $(hostname)..."
"${PYTHON_BIN}" -c "
from huggingface_hub import whoami
import os
info = whoami(token=os.environ.get('HF_TOKEN'))
print(f'[INFO] Authenticated as: {info[\"name\"]}')
" || { echo "[ERROR] HF auth check failed"; exit 1; }

# ----------------------------------------------------------------------------
# Copy sweep script
# ----------------------------------------------------------------------------
SCRIPT_NAME="lettuceprevent_threshold_experiment.py"
if [[ ! -f "${BENCHMARK_DIR}/${SCRIPT_NAME}" ]]; then
    echo "ERROR: ${BENCHMARK_DIR}/${SCRIPT_NAME} not found"
    exit 1
fi
cp "${BENCHMARK_DIR}/${SCRIPT_NAME}" "${WORK_DIR}/"
cd "${WORK_DIR}"
export PYTHONPATH="${WORK_DIR}:${PYTHONPATH:-}"

# ----------------------------------------------------------------------------
# Run threshold sweep
# ----------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "Threshold sweep "
echo "================================================================"

if [[ -n "${SWEEP_ID:-}" ]]; then
    echo "Using existing sweep id: ${SWEEP_ID}"
    "${PYTHON_BIN}" "${WORK_DIR}/${SCRIPT_NAME}" \
        --entity "${WANDB_ENTITY}" \
        --project "${WANDB_PROJECT}" \
        --sweep-id "${SWEEP_ID}" \
        --count "${SWEEP_COUNT}" \
        --seed "${SWEEP_SEED}" \
        --output-prefix "${SWEEP_OUTPUT_PREFIX}" \
        --prompts-per-task "${SWEEP_PROMPTS_PER_TASK}"
else
    echo "Creating new sweep and running agent."
    "${PYTHON_BIN}" "${WORK_DIR}/${SCRIPT_NAME}" \
        --entity "${WANDB_ENTITY}" \
        --project "${WANDB_PROJECT}" \
        --count "${SWEEP_COUNT}" \
        --seed "${SWEEP_SEED}" \
        --output-prefix "${SWEEP_OUTPUT_PREFIX}" \
        --prompts-per-task "${SWEEP_PROMPTS_PER_TASK}"
fi