#!/bin/bash
#SBATCH -J rq2-smoke-test
#SBATCH --partition=GPU-a100s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100s:1
#SBATCH --time=02:00:00
#SBATCH --mem=64GB
#SBATCH --output=rq2_smoke_test.txt

set -euo pipefail

echo "${SLURM_JOB_NODELIST}"
nvidia-smi -L

REPO_DIR="/home/e12133103/LettucePrevent"
RQ1_DIR="${REPO_DIR}"
RESULTS_DIR="${RESULTS_DIR:-/share/${USER}/rq2-smoke-results}"
VENV_PATH="${VENV_PATH:-/home/e12133103/Python312/bin/activate}"
PYTHON_BIN="${PYTHON_BIN:-python}"

OUTPUT_PREFIX="${OUTPUT_PREFIX:-smoke}"

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
export WEAVE_DISABLED="true"
export HF_HOME="/share/${USER}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
unset HF_DATASETS_OFFLINE || true

mkdir -p "${HF_HOME}"

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
    rm -rf "${WORK_DIR}"
    echo "Smoke test finished. Results in ${RESULTS_DIR}"
}
trap cleanup EXIT

if [[ -f "${VENV_PATH}" ]]; then
    source "${VENV_PATH}"
else
    echo "WARNING: VENV_PATH not found (${VENV_PATH}). Continuing without activation."
fi

echo "[INFO] Verifying HF auth on $(hostname)..."
"${PYTHON_BIN}" -c "
from huggingface_hub import whoami
import os
info = whoami(token=os.environ.get('HF_TOKEN'))
print(f'[INFO] Authenticated as: {info[\"name\"]}')
" || { echo "[ERROR] HF auth check failed"; exit 1; }

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

cd "${WORK_DIR}"
export PYTHONPATH="${WORK_DIR}:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# Smoke test: single cell, Mistral + lettuceprevent + skip=1.0, 2 prompts/task
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "Running smoke test:"
echo "  Generator:   mistralai/Mistral-7B-Instruct-v0.2"
echo "  Detector:    lettuceprevent"
echo "  Skip thr:    1.0"
echo "  N per task:  2  (=> 6 prompts total across 3 tasks)"
echo "================================================================"
echo ""

"${PYTHON_BIN}" "${WORK_DIR}/main.py" \
    --mode single \
    --generator-model mistralai/Mistral-7B-Instruct-v0.2 \
    --detector-type lettuceprevent \
    --skip-threshold 1.0 \
    --n-per-task 2 \
    --output-prefix "${OUTPUT_PREFIX}" \
    --no-wandb

# ---------------------------------------------------------------------------
# Smoke test: single cell, Qwen + lettuceprevent + skip=1.0, 2 prompts/task
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "Running smoke test:"
echo "  Generator:   Qwen"
echo "  Detector:    lettuceprevent"
echo "  Skip thr:    1.0"
echo "  N per task:  2  (=> 6 prompts total across 3 tasks)"
echo "================================================================"
echo ""

"${PYTHON_BIN}" "${WORK_DIR}/main.py" \
    --mode single \
    --generator-model Qwen/Qwen2.5-14B-Instruct \
    --detector-type lettuceprevent \
    --skip-threshold 1.0 \
    --n-per-task 2 \
    --output-prefix "${OUTPUT_PREFIX}" \
    --no-wandb

# ---------------------------------------------------------------------------
# Smoke test: single cell, Llama + lettuceprevent + skip=1.0, 2 prompts/task
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "Running smoke test:"
echo "  Generator:   Llama"
echo "  Detector:    lettuceprevent"
echo "  Skip thr:    1.0"
echo "  N per task:  2  (=> 6 prompts total across 3 tasks)"
echo "================================================================"
echo ""

"${PYTHON_BIN}" "${WORK_DIR}/main.py" \
    --mode single \
    --generator-model meta-llama/Llama-2-7b-chat-hf \
    --detector-type lettuceprevent \
    --skip-threshold 1.0 \
    --n-per-task 2 \
    --output-prefix "${OUTPUT_PREFIX}" \
    --no-wandb

# ---------------------------------------------------------------------------
# Smoke test: single cell, Mistral + baseline + skip=1.0, 2 prompts/task
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "Running smoke test:"
echo "  Generator:   mistralai/Mistral-7B-Instruct-v0.2"
echo "  Detector:    lettuceprevent"
echo "  Skip thr:    1.0"
echo "  N per task:  2  (=> 6 prompts total across 3 tasks)"
echo "================================================================"
echo ""

"${PYTHON_BIN}" "${WORK_DIR}/main.py" \
    --mode single \
    --generator-model mistralai/Mistral-7B-Instruct-v0.2 \
    --detector-type baseline-run-facts \
    --skip-threshold 1.0 \
    --n-per-task 2 \
    --output-prefix "${OUTPUT_PREFIX}" \
    --no-wandb