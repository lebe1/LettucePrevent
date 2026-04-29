#!/bin/bash
#SBATCH -J lp-calibrate
#SBATCH --partition=GPU-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH --mem=32GB
#SBATCH --output=lettuceprevent_calibration_%j.txt



set -euo pipefail

echo "${SLURM_JOB_NODELIST}"
nvidia-smi -L

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
REPO_DIR="/home/e12133103/LettucePrevent"
BENCHMARK_DIR="${REPO_DIR}/model_selection"
RESULTS_DIR="${RESULTS_DIR:-/share/${USER}/lettuceprevent-rq3-results}"
VENV_PATH="${VENV_PATH:-/home/e12133103/Python312/bin/activate}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CAL_PROMPTS_PER_TASK="${CAL_PROMPTS_PER_TASK:-20}"
CAL_OUTPUT_PREFIX="${CAL_OUTPUT_PREFIX:-lettuceprevent_calibration}"
CAL_SEED="${CAL_SEED:-42}"

# WORK_DIR is intentionally NOT job-scoped — it's shared with the sweep job
# via PIPELINE_TAG so both jobs see the same calibration JSON.
PIPELINE_TAG="${PIPELINE_TAG:-${SLURM_JOB_ID:-manual_$$}}"
WORK_DIR="/share/${USER}/scratch_jobs/lp_pipeline_${PIPELINE_TAG}"
mkdir -p "${WORK_DIR}"
chmod 700 "${WORK_DIR}"

echo "Using WORK_DIR=${WORK_DIR}"
echo "Using BENCHMARK_DIR=${BENCHMARK_DIR}"
echo "PIPELINE_TAG=${PIPELINE_TAG}"

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
# Hugging Face authentication
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
        echo "[ERROR] No HF token cache found. Run 'huggingface-cli login' first."
        exit 1
    fi
fi
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"

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
# Copy script to work dir
# ----------------------------------------------------------------------------
SCRIPT_NAME="calibrate_lettuceprevent.py"
if [[ ! -f "${BENCHMARK_DIR}/${SCRIPT_NAME}" ]]; then
    echo "ERROR: ${BENCHMARK_DIR}/${SCRIPT_NAME} not found"
    exit 1
fi
cp "${BENCHMARK_DIR}/${SCRIPT_NAME}" "${WORK_DIR}/"
cd "${WORK_DIR}"
export PYTHONPATH="${WORK_DIR}:${PYTHONPATH:-}"

# ----------------------------------------------------------------------------
# Run calibration
# ----------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "STEP 1/2 — Calibration"
echo "================================================================"

"${PYTHON_BIN}" "${WORK_DIR}/${SCRIPT_NAME}" \
    --prompts-per-task "${CAL_PROMPTS_PER_TASK}" \
    --output-prefix "${CAL_OUTPUT_PREFIX}" \
    --seed "${CAL_SEED}"

# Verify the calibration JSON was actually produced
CAL_JSON="${WORK_DIR}/${CAL_OUTPUT_PREFIX}.json"
if [[ ! -f "${CAL_JSON}" ]]; then
    echo "[ERROR] Calibration JSON not found at ${CAL_JSON}"
    exit 1
fi

# Copy calibration JSON to results dir so it persists even if scratch is cleaned
cp "${CAL_JSON}" "${RESULTS_DIR}/"
echo "[INFO] Calibration JSON copied to ${RESULTS_DIR}/${CAL_OUTPUT_PREFIX}.json"

echo "[INFO] Calibration complete. WORK_DIR retained for sweep job: ${WORK_DIR}"
