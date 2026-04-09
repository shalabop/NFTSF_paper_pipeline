#!/usr/bin/env bash
#SBATCH --job-name=linCSDI
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-user=meahmed@asu.edu
#SBATCH --mail-type=ALL
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/csdi/linear_gaussian.%j.out
#SBATCH --error=logs/csdi/linear_gaussian.%j.err

mkdir -p results/csdi/linear_gaussian
#mkdir -p logs/csdi/linear_gaussian
mkdir -p checkpoints/csdi/linear_gaussian

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


echo "=== Forecasting CSDI: linear_gaussian ==="
python eval/CSDI/forecast_csdi.py \
    --config configs/csdi_train/linear_gaussian.yaml \
    --input  DATA/linear_gaussian_test.npz \
    --ckpt   checkpoints/csdi/linear_gaussian/model.pth \
    --out    results/csdi/linear_gaussian.npz \
    --device cuda:0