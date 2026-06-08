#!/bin/bash
#SBATCH -J rq1-hallucination-reduction
#SBATCH --partition=GPU-a100s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100s:1
#SBATCH --time=72:00:00
#SBATCH --mem=128GB
#SBATCH --output=rq1_hallucination_reduction_%j.txt

set -euo pipefail

echo "${SLURM_JOB_NODELIST}"
nvidia-smi -L

REPO_DIR="/home/e12133103/LettucePrevent"
RESULTS_DIR="${RESULTS_DIR:-/share/${USER}/rq1-results}"
VENV_PATH="${VENV_PATH:-/home/e12133103/Python312/bin/activate}"
PYTHON_BIN="${PYTHON_BIN:-python}"

WANDB_ENTITY="${WANDB_ENTITY:-lebeccard-technical-university-wien}"
POST_EVAL_FLOOR="${POST_EVAL_FLOOR:-0.70}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-rq1}"

JOB_TAG="${SLURM_JOB_ID:-manual_$$}"
WORK_DIR="/share/${USER}/scratch_jobs/${JOB_TAG}"
mkdir -p "${WORK_DIR}"
chmod 700 "${WORK_DIR}"

echo "Using WORK_DIR=${WORK_DIR}"
echo "Using REPO_DIR=${REPO_DIR}"

mkdir -p "${RESULTS_DIR}"

export SCRATCH_DIR="${WORK_DIR}"
export WANDB_DIR="${WORK_DIR}/wandb"
export WANDB_CACHE_DIR="${WORK_DIR}/.wandb_cache"
export WANDB_DATA_DIR="${WORK_DIR}/.wandb_data"
export WEAVE_DISABLED="true"
export HF_HOME="/share/${USER}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
unset HF_DATASETS_OFFLINE || true

mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}" "${HF_HOME}"

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

# ---------------------------------------------------------------------------
# Cleanup: copy results back and remove scratch
# ---------------------------------------------------------------------------
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
# Activate virtual environment
# ---------------------------------------------------------------------------
if [[ -f "${VENV_PATH}" ]]; then
    source "${VENV_PATH}"
else
    echo "WARNING: VENV_PATH not found (${VENV_PATH}). Continuing without activation."
fi

# ---------------------------------------------------------------------------
# Verify HF authentication
# ---------------------------------------------------------------------------
echo "[INFO] Verifying HF auth on $(hostname)..."
"${PYTHON_BIN}" -c "
from huggingface_hub import whoami
import os
info = whoami(token=os.environ.get('HF_TOKEN'))
print(f'[INFO] Authenticated as: {info[\"name\"]}')
" || { echo "[ERROR] HF auth check failed"; exit 1; }

# ---------------------------------------------------------------------------
# Copy required source files to WORK_DIR
# ---------------------------------------------------------------------------
REQUIRED_FILES=(
    "main.py"
    "analysis/factual_detection/post_eval.py"
    "detectors/dataset_loader.py"
    "detectors/factory.py"
    "detectors/base_detector.py"
    "detectors/lettucedetect_detector.py"
    "detectors/lettuceprevent_detector.py"
    "detectors/number_detector.py"
    "logits_processors/hallucination_logits_processor.py"
)

for rel in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "${REPO_DIR}/${rel}" ]]; then
        echo "ERROR: ${REPO_DIR}/${rel} not found"
        exit 1
    fi
done

mkdir -p "${WORK_DIR}/analysis" \
         "${WORK_DIR}/detectors" \
         "${WORK_DIR}/logits_processors" \
         "${WORK_DIR}/data"

for rel in "${REQUIRED_FILES[@]}"; do
    cp "${REPO_DIR}/${rel}" "${WORK_DIR}/${rel}"
done

for d in analysis detectors logits_processors; do
    if [[ -f "${REPO_DIR}/${d}/__init__.py" ]]; then
        cp "${REPO_DIR}/${d}/__init__.py" "${WORK_DIR}/${d}/__init__.py"
    else
        : > "${WORK_DIR}/${d}/__init__.py"
    fi
done

if [[ -f "${REPO_DIR}/data/ragtruth_unique_summary_prompts.json" ]]; then
    cp "${REPO_DIR}/data/ragtruth_unique_summary_prompts.json" \
       "${WORK_DIR}/data/ragtruth_unique_summary_prompts.json"
fi

cd "${WORK_DIR}"
export PYTHONPATH="${WORK_DIR}:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# RQ1 sweep execution
# Launches a W&B grid sweep over GENERATOR_MODELS × DETECTOR_TYPES_SWEEPED.
# Skip threshold is resolved per model inside sweep_fn_rq1 via
# MODELS_BEST_SKIP_THRESHOLDS — it is NOT a sweep dimension here.
# ---------------------------------------------------------------------------
SWEEP_COUNT="${SWEEP_COUNT:-6}"  # = len(GENERATOR_MODELS) × len(DETECTOR_TYPES_SWEEPED)

if [[ -n "${SWEEP_ID:-}" ]]; then
    echo "[INFO] Attaching agent to existing RQ1 sweep: ${SWEEP_ID}"
    "${PYTHON_BIN}" "${WORK_DIR}/main.py" \
        --rq1 \
        --sweep-id "${SWEEP_ID}" \
        --entity   "${WANDB_ENTITY}" \
        --count    "${SWEEP_COUNT}" \
        --post-eval-floor "${POST_EVAL_FLOOR}" \
        --output-prefix   "${OUTPUT_PREFIX}"
else
    echo "[INFO] Creating new RQ1 sweep and running agent."
    "${PYTHON_BIN}" "${WORK_DIR}/main.py" \
        --rq1 \
        --entity  "${WANDB_ENTITY}" \
        --count   "${SWEEP_COUNT}" \
        --post-eval-floor "${POST_EVAL_FLOOR}" \
        --output-prefix   "${OUTPUT_PREFIX}"
fi