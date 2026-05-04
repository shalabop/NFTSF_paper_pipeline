#!/usr/bin/env bash
#SBATCH --job-name=lgCTSFdiff
#SBATCH --mail-user=meahmed@asu.edu
#SBATCH --mail-type=ALL
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --output=logs/tsdiff_cond/linear_gaussian.%j.out
#SBATCH --error=logs/tsdiff_cond/linear_gaussian.%j.err

source venv310/bin/activate
which python

mkdir -p checkpoints/tsdiff_cond/linear_gaussian
mkdir -p results/tsdiff_cond/
#mkdir -p logs/tsdiff_cond/double_well

PROJECT_ROOT=$(pwd) 
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
# Load CUDA
module purge
module load cuda-12.8.1-gcc-12.1.0

# CUDA paths
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$CUDA_PATH/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$CUDA_PATH/targets/x86_64-linux/lib:$LIBRARY_PATH"
export CPATH="$CUDA_PATH/targets/x86_64-linux/include:$CPATH"

rm -rf ~/.cache/keops* 2>/dev/null
rm -rf ~/.cache/pykeops* 2>/dev/null
rm -rf /tmp/keops* 2>/dev/null
rm -rf /tmp/pykeops* 2>/dev/null

# Use node-local temp directory (fastest, auto-cleaned)
export PYKEOPS_BUILD_DIR="$SLURM_TMPDIR/pykeops_build"
export KEOPS_CACHE_FOLDER="$SLURM_TMPDIR/keops_cache"

# Fallback to home directory if SLURM_TMPDIR not available
if [ -z "$SLURM_TMPDIR" ]; then
    export PYKEOPS_BUILD_DIR="$HOME/.cache/pykeops_build"
    export KEOPS_CACHE_FOLDER="$HOME/.cache/keops_cache"
fi

mkdir -p "$PYKEOPS_BUILD_DIR"
mkdir -p "$KEOPS_CACHE_FOLDER"

echo "PyKeOps build dir: $PYKEOPS_BUILD_DIR"
echo "KeOps cache dir: $KEOPS_CACHE_FOLDER"

echo "Checking CUDA installation..."
ls $CUDA_PATH/include/cuda.h || echo "WARNING: cuda.h not found"
ls $CUDA_PATH/include/nvrtc.h || echo "WARNING: nvrtc.h not found"
ls $CUDA_PATH/lib64/libnvrtc.so* || ls $CUDA_PATH/targets/x86_64-linux/lib/libnvrtc.so* || echo "WARNING: libnvrtc.so not found"

python train/TSDiff/train_cond_tsdiff.py \
        --dataset_path gluonts_datasets/linear_gaussian \
        --config configs/tsdiff_cond_train/linear_gaussian.yaml \
        --out_dir checkpoints/tsdiff_cond/linear_gaussian

python eval/TSDiff/forecast_tsdiff_cond.py   \
        --config configs/tsdiff_forecast/linear_gaussian_cond.yaml  \
        --dataset_path gluonts_datasets/linear_gaussian \
        --out results/tsdiff_cond/linear_gaussian.npz