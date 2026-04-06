#!/usr/bin/env bash
#SBATCH --job-name=DWCSDI
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-user=meahmed@asu.edu
#SBATCH --mail-type=ALL
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/csdi/double_well.%j.out
#SBATCH --error=logs/csdi/double_well.%j.err

mkdir -p results/csdi/double_well
#mkdir -p logs/csdi/double_well
mkdir -p checkpoints/csdi/double_well

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
python train/CSDI/train_csdi.py \
    --config configs/csdi_train/double_well.yaml \
    --input  DATA/double_well_train.npz \
    --out    checkpoints/csdi/double_well \
    --device cuda:0

echo "=== Forecasting CSDI: double_well ==="
python eval/CSDI/forecast_csdi.py \
    --config configs/csdi_train/double_well.yaml \
    --input  DATA/double_well_test.npz \
    --ckpt   checkpoints/csdi/double_well/model.pth \
    --out    results/csdi/double_well.npz \
    --device cuda:0
