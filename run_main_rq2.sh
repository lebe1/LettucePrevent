#!/bin/bash
#SBATCH -J rq2-skip-threshold-sweep
#SBATCH --partition=GPU-a100s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100s:1
#SBATCH --time=72:00:00
#SBATCH --mem=128GB
#SBATCH --output=rq2_skip_threshold_sweep.txt

set -euo pipefail

echo "${SLURM_JOB_NODELIST}"
nvidia-smi -L

REPO_DIR="/home/e12133103/LettucePrevent"
RQ1_DIR="${REPO_DIR}"
RESULTS_DIR="${RESULTS_DIR:-/share/${USER}/rq2-results-llama3-1}"
VENV_PATH="${VENV_PATH:-/home/e12133103/Python312/bin/activate}"
PYTHON_BIN="${PYTHON_BIN:-python}"

WANDB_ENTITY="${WANDB_ENTITY:-lebeccard-technical-university-wien}"
WANDB_PROJECT="${WANDB_PROJECT:-hdm-rq2-llama3.1}"
# 3 generators x 2 detectors x 4 skip thresholds = 24 cells
SWEEP_COUNT="${SWEEP_COUNT:-24}"
N_PER_TASK="${N_PER_TASK:-20}"
POST_EVAL_FLOOR="${POST_EVAL_FLOOR:-0.70}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-rq2_llama_3_1}"

JOB_TAG="${SLURM_JOB_ID:-manual_$$}"
WORK_DIR="/share/${USER}/scratch_jobs/${JOB_TAG}"
mkdir -p "${WORK_DIR}"
chmod 700 "${WORK_DIR}"

echo "Using WORK_DIR=${WORK_DIR}"
echo "Using RQ1_DIR=${RQ1_DIR}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "WARNING: SLURM_JOB_ID is not set. Script appears to run outside sbatch."
fi

mkdir -p "${RESULTS_DIR}"

export SCRATCH_DIR="${WORK_DIR}"
export WANDB_DIR="${WORK_DIR}/wandb"
export WANDB_CACHE_DIR="${WORK_DIR}/.wandb_cache"
export WANDB_DATA_DIR="${WORK_DIR}/.wandb_data"
export WEAVE_DISABLED="true"

# HF cache config. Modern transformers reads HF_HOME and derives subpaths.
# Do NOT set TRANSFORMERS_CACHE (deprecated, causes warnings + broke last run).
export HF_HOME="/share/${USER}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
unset TRANSFORMERS_CACHE || true
unset HF_DATASETS_OFFLINE || true

mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}" \
         "${HF_HOME}" "${HF_DATASETS_CACHE}"

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export TOKENIZERS_PARALLELISM="false"

# ---------------------------------------------------------------------------
# Hugging Face authentication
# ---------------------------------------------------------------------------
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

cleanup() {
    echo "Copying outputs back to ${RESULTS_DIR}"
    if [[ -d "${WORK_DIR}/data" ]]; then
        mkdir -p "${RESULTS_DIR}/data_${JOB_TAG}"
        cp -r "${WORK_DIR}/data/." "${RESULTS_DIR}/data_${JOB_TAG}/" 2>/dev/null || true
    fi
    cp -r "${WORK_DIR}/wandb" "${RESULTS_DIR}/wandb_${JOB_TAG}" 2>/dev/null || true
    rm -rf "${WORK_DIR}"
    echo "Job finished. Results in ${RESULTS_DIR}"
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Activate venv 
# ---------------------------------------------------------------------------
if [[ -f "${VENV_PATH}" ]]; then
    source "${VENV_PATH}"
    echo "[INFO] Activated venv: ${VENV_PATH}"
else
    echo "[ERROR] VENV_PATH not found (${VENV_PATH})."
    exit 1
fi

# ---------------------------------------------------------------------------
# HF auth sanity check
# ---------------------------------------------------------------------------
echo "[INFO] Verifying HF auth on $(hostname)..."
"${PYTHON_BIN}" -c "
from huggingface_hub import whoami
import os
info = whoami(token=os.environ.get('HF_TOKEN'))
print(f'[INFO] Authenticated as: {info[\"name\"]}')
" || { echo "[ERROR] HF auth check failed"; exit 1; }

# ---------------------------------------------------------------------------
# Stage source files
# ---------------------------------------------------------------------------
REQUIRED_FILES=(
    "main.py"
    "analysis/post_eval.py"
    "detectors/dataset_loader.py"
    "detectors/factory.py"
    "detectors/base_detector.py"
    "detectors/lettucedetect_detector.py"
    "detectors/lettuceprevent_detector.py"
    "detectors/number_detector.py"
    "logits_processors/hallucination_logits_processor.py"
)
for rel in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "${RQ1_DIR}/${rel}" ]]; then
        echo "ERROR: ${RQ1_DIR}/${rel} not found"
        exit 1
    fi
done

mkdir -p "${WORK_DIR}/analysis" \
         "${WORK_DIR}/detectors" \
         "${WORK_DIR}/logits_processors" \
         "${WORK_DIR}/data"

for rel in "${REQUIRED_FILES[@]}"; do
    cp "${RQ1_DIR}/${rel}" "${WORK_DIR}/${rel}"
done

for d in analysis detectors logits_processors; do
    if [[ -f "${RQ1_DIR}/${d}/__init__.py" ]]; then
        cp "${RQ1_DIR}/${d}/__init__.py" "${WORK_DIR}/${d}/__init__.py"
    else
        : > "${WORK_DIR}/${d}/__init__.py"
    fi
done

if [[ -f "${RQ1_DIR}/data/ragtruth_unique_summary_prompts.json" ]]; then
    cp "${RQ1_DIR}/data/ragtruth_unique_summary_prompts.json" \
       "${WORK_DIR}/data/ragtruth_unique_summary_prompts.json"
fi

cd "${WORK_DIR}"
export PYTHONPATH="${WORK_DIR}:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# Sweep execution
# ---------------------------------------------------------------------------
if [[ -n "${SWEEP_ID:-}" ]]; then
    echo "Using existing sweep id: ${SWEEP_ID}"
    "${PYTHON_BIN}" "${WORK_DIR}/main.py" \
        --mode sweep-agent \
        --sweep-id "${SWEEP_ID}" \
        --entity "${WANDB_ENTITY}" \
        --project "${WANDB_PROJECT}" \
        --count "${SWEEP_COUNT}" \
        --n-per-task "${N_PER_TASK}" \
        --post-eval-floor "${POST_EVAL_FLOOR}" \
        --output-prefix "${OUTPUT_PREFIX}"
else
    echo "Creating new sweep and running agent."
    "${PYTHON_BIN}" "${WORK_DIR}/main.py" \
        --mode sweep-agent \
        --entity "${WANDB_ENTITY}" \
        --project "${WANDB_PROJECT}" \
        --count "${SWEEP_COUNT}" \
        --n-per-task "${N_PER_TASK}" \
        --post-eval-floor "${POST_EVAL_FLOOR}" \
        --output-prefix "${OUTPUT_PREFIX}"
fi