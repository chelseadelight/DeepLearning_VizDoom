#!/bin/bash

# Make sure to CHMOD scripts!

#SBATCH --job-name=example-train
#SBATCH --gpus-per-node=a100:2
#SBATCH --partition=gpulong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/example_train_%j.out
#SBATCH --error=logs/example_train_%j.err

# Leave the script if an error is encountered
set -e

# Create log and temp directories
mkdir -p logs
export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"

# Load system modules for GPU support
module purge
module load CUDA/12.4.0

# Load Conda
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"

# Activate Conda environment
conda activate some_environment

# Navigate to code
cd "$HOME/path/to/your/code"

# Print environment info for logging
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
nvidia-smi
echo "Python path: $(which python)"
echo "Working directory: $(pwd)"
echo "Conda version: $(conda --version)"
echo "Python version: $(python --version)"
echo ""

# Launch training
python3 path/to/your/actual/train/script --gpu-flags

echo "Training finished at: $(date)"
