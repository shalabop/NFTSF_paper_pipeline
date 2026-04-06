#!/usr/bin/env bash
#SBATCH --job-name=ApsiCSDI
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-user=meahmed@asu.edu
#SBATCH --mail-type=ALL
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/csdi/alanine_psi.%j.out
#SBATCH --error=logs/csdi/alanine_psi.%j.err

mkdir -p results/csdi/alanine_psi
#mkdir -p logs/csdi/alanine_psi
mkdir -p checkpoints/csdi/alanine_psi

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

echo "=== Training CSDI: alanine_psi ==="
python train/CSDI/train_csdi.py \
    --config configs/csdi_train/alanine_psi.yaml \
    --input  DATA/alanine_psi_train.npz \
    --out    checkpoints/csdi/alanine_psi \
    --device cuda:0

echo "=== Forecasting CSDI: alanine_psi ==="
python eval/CSDI/forecast_csdi.py \
    --config configs/csdi_train/double_well.yaml \
    --input  DATA/alanine_psi_test.npz \
    --ckpt   checkpoints/csdi/alanine_psi/model.pth \
    --out    results/csdi/alanine_psi.npz \
    --device cuda:0