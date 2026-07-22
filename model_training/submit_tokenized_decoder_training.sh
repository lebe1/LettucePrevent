#!/bin/bash
#SBATCH -J train-tokenized-decoder-model
#SBATCH --partition=GPU-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB

echo $SLURM_JOB_NODELIST
nvidia-smi -L

# Create job-specific scratch directory
scratch_dir="/scratch/${USER}_${SLURM_JOB_ID}"
mkdir -p "${scratch_dir}"
chmod 700 "${scratch_dir}"
cd "${scratch_dir}"

# Persistent results on /share
export results_dir="/share/${USER}/train-tokenized-decoder-results"
mkdir -p "${results_dir}"

# Redirect all cache/temp writes away from /home to scratch
export SCRATCH_DIR="${scratch_dir}"
export WANDB_DIR="${scratch_dir}"
export WANDB_CACHE_DIR="${scratch_dir}/.wandb_cache"
export WANDB_DATA_DIR="${scratch_dir}/.wandb_data"
export HF_HOME="/share/${USER}/.cache/huggingface"
export TRANSFORMERS_CACHE="/share/${USER}/.cache/huggingface/transformers"

source $HOME/Python312/bin/activate


cp /home/e12133103/LettucePrevent/model_training/ettin_tokenized_decoder_training.py "${scratch_dir}/"
export HF_TOKEN="YOUR_HF_TOKEN"
python "${scratch_dir}/ettin_tokenized_decoder_training.py"

# Copy results back to persistent storage
cp -r "${scratch_dir}"/sweep_run_* "${results_dir}/"

# Clean up scratch
rm -rf "${scratch_dir}"
echo "Job finished. Results in ${results_dir}"
