#!/bin/bash
# ============================================================
# NF-TSF Environment Setup Script
# ============================================================
# This script helps set up the conda environment and directory structure
# Run this once after uploading files to your supercomputer

set -e  # Exit on error

echo "============================================"
echo "NF-TSF Environment Setup"
echo "============================================"
echo ""

# ============================================================
# Configuration - Edit these as needed
# ============================================================
ENV_NAME="nf_tsf"
PYTHON_VERSION="3.10"
CUDA_VERSION="11.8"  # Adjust to match your system

# ============================================================
# Create directory structure
# ============================================================
echo "Creating directory structure..."
mkdir -p logs
mkdir -p results
mkdir -p data
mkdir -p Models

echo "  Created: logs/"
echo "  Created: results/"
echo "  Created: data/"
echo "  Created: Models/"
echo ""

# ============================================================
# Check if conda is available
# ============================================================
module load mamba/latest

if command -v conda &> /dev/null; then
    echo "Conda found: $(conda --version)"
else
    echo "ERROR: Conda not found after 'module load mamba/latest'!"
    echo "Check available modules with: module spider mamba"
    exit 1
fi

# ============================================================
# Create conda environment
# ============================================================
echo ""
echo "Checking for existing environment '$ENV_NAME'..."

if conda env list | grep -q "^$ENV_NAME "; then
    echo "Environment '$ENV_NAME' already exists."
    read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n $ENV_NAME -y
    else
        echo "Keeping existing environment."
        echo "Activate it with: conda activate $ENV_NAME"
        exit 0
    fi
fi

echo ""
echo "Creating new conda environment: $ENV_NAME (Python $PYTHON_VERSION)..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# ============================================================
# Activate and install packages
# ============================================================
echo ""
echo "Activating environment and installing packages..."
source activate $ENV_NAME

# Verify the environment is actually active before installing
if [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then
    echo "ERROR: Failed to activate environment '$ENV_NAME'!"
    echo "CONDA_DEFAULT_ENV is: $CONDA_DEFAULT_ENV"
    exit 1
fi
echo "Confirmed active environment: $CONDA_DEFAULT_ENV"

# Install PyTorch with CUDA
echo "Installing PyTorch with CUDA $CUDA_VERSION..."
conda install pytorch torchvision torchaudio pytorch-cuda=$CUDA_VERSION -c pytorch -c nvidia -y

# Install other dependencies (environment is activated above)
echo "Installing other dependencies via pip inside '$CONDA_DEFAULT_ENV'..."
pip install normflows numpy matplotlib seaborn tqdm scipy

# ============================================================
# Verify installation
# ============================================================
echo ""
echo "Verifying installation..."
python -c "
import torch
import numpy as np
import matplotlib
import normflows
print('='*50)
print('Installation Verification')
print('='*50)
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'NumPy version: {np.__version__}')
print(f'normflows version: {normflows.__version__}')
print('='*50)
print('All packages installed successfully!')
print('='*50)
"

# ============================================================
# Make bash scripts executable
# ============================================================
echo ""
echo "Making bash scripts executable..."
chmod +x run_training.sh
chmod +x run_testing.sh
chmod +x run_sweep.sh

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================"
echo "Setup Complete!"
echo "============================================"
echo ""
echo "To use this environment:"
echo "  conda activate $ENV_NAME"
echo ""
echo "Next steps:"
echo "  1. Replace architecture.py with your actual architecture file"
echo "  2. Put your data files in the data/ directory"
echo "  3. Edit run_training.sh with your data paths"
echo "  4. Submit with: sbatch run_training.sh"
echo ""
echo "============================================"
