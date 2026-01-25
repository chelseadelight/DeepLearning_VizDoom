#!/bin/bash

#SBATCH --job-name=train-vizdoom-dm
#SBATCH --gpus-per-node=a100:1
#SBATCH --partition=gpulong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/train-vizdoom-dm-%j.out
#SBATCH --error=logs/train-vizdoom-dm-%j.err

# Leave the script if an error is encountered
set -e

# Create log and temp directories
mkdir -p logs
export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"

# Purge modules
module purge

# Load CUDA
module load CUDA/12.4.0
nvidia-smi

# Load Conda
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"

# Activate the project environment
conda activate vizdoom-dm

# Navigate to project root
cd ..

# Print environment info for logging
echo "Conda version: $(conda --version)"
echo "Python version: $(python --version)"
echo ""

# Launch training
python -m project.main "$@"

echo ""
echo "Training finished at: $(date)"
