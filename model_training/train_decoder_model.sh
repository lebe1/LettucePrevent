#!/bin/bash

#SBATCH -J ettin-sweep
#SBATCH --partition=GPU-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a40:1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB

# Log node and GPU info
echo $SLURM_JOB_NODELIST
nvidia-smi -L

# Create job-specific scratch directory
scratch_dir="/scratch/${USER}_${SLURM_JOB_ID}"
if [ ! -d "${scratch_dir}" ]; then
    mkdir "${scratch_dir}"
    chmod 700 "${scratch_dir}"
fi

cd "${scratch_dir}"

# Where to persist results after the job
export results_dir="/home/${USER}/ettin-sweep-results"
mkdir -p "${results_dir}"

# Activate Python environment
source $HOME/Python312/bin/activate

# Copy training script to scratch
cp /home/e12133103/LettucePrevent/playground/ettin_decoder_training.py "${scratch_dir}/"

# Run the training
python ettin_decoder_training.py

# Copy results back to persistent storage
cp -r "${scratch_dir}"/sweep_run_* "${results_dir}/"

# Clean up scratch
rm -rf "${scratch_dir}"

echo "Job finished. Results in ${results_dir}"