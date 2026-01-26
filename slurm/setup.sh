#!/bin/bash
# Leave script if an error is encountered
set -e

# Purge modules
module purge

# Setup the Conda environment on-machine
module load Anaconda3/2024.02-1
echo "Successfully loaded Anaconda with Conda $(conda --version)"

# Remove the project environment if it already exists
conda env remove -n vizdoom-dm -y

# Set up the project environment
cd ..
conda env create -f environment.yml
