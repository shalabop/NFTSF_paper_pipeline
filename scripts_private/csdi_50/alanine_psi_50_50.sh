#!/usr/bin/env bash
#SBATCH --job-name=CSDI
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-user=meahmed@asu.edu
#SBATCH --mail-type=ALL
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=logs/csdi/alanine_shi_50_50.%j.out
#SBATCH --error=logs/csdi/alanine_psi_50_50.%j.err

mkdir -p results/csdi
mkdir -p results/csdi/alanine_psi_50_50
#mkdir -p logs/csdi/alanine_phi
mkdir -p checkpoints_25_25/csdi_with_early_stopping_50/alanine_psi
mkdir -p checkpoints_50_50/csdi_with_early_stopping_50/alanine_psi
mkdir -p checkpoints_25_25/csdi_with_early_stopping_50/alanine_phi
mkdir -p checkpoints_50_50/csdi_with_early_stopping_50/alanine_phi
mkdir -p checkpoints_25_25/csdi_with_early_stopping_50/double_well
mkdir -p checkpoints_50_50/csdi_with_early_stopping_50/double_well
mkdir -p checkpoints_25_25/csdi_with_early_stopping_50/single_well
mkdir -p checkpoints_50_50/csdi_with_early_stopping_50/single_well

source venv310/bin/activate
module purge
module load cuda-12.8.1-gcc-12.1.0

export NO_AI_TRACKING=false
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$CUDA_PATH/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$CUDA_PATH/targets/x86_64-linux/lib:$LIBRARY_PATH"
export CPATH="$CUDA_PATH/targets/x86_64-linux/include:$CPATH"
export KEOPS_CACHE_DIR=$HOME/.cache/keops
mkdir -p $KEOPS_CACHE_DIR

ls $CUDA_PATH/include/cuda.h      || echo "cuda.h not found"
ls $CUDA_PATH/include/nvrtc.h     || echo "nvrtc.h not found"
ls $CUDA_PATH/lib64/libnvrtc.so*  || echo "libnvrtc.so not found in lib64"
ls $CUDA_PATH/targets/x86_64-linux/lib/libnvrtc.so* || echo "libnvrtc.so not found in targets/x86_64-linux/lib"

echo "=== Training CSDI: double_well ==="
python train/CSDI/train_csdi_with_early_stopping.py \
    --config configs/csdi_train_50/alanine_psi_50_50.yaml \
    --input  DATA/alanine_psi_train.npz \
    --out    checkpoints_50_50/csdi_with_early_stopping_50/alanine_psi \
    --device cuda:0