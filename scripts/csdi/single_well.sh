#!/usr/bin/env bash
#SBATCH --job-name=SWCSDI
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-user=meahmed@asu.edu
#SBATCH --mail-type=ALL
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/csdi/single_well.%j.out
#SBATCH --error=logs/csdi/single_well.%j.err

mkdir -p results/csdi/single_well
mkdir -p logs/csdi/single_well
mkdir -p checkpoints/csdi/single_well

source /home/meahmed/venv310/bin/activate
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

echo "=== Training CSDI: single_well ==="
python train/CSDI/train_csdi.py \
    --config configs/csdi_train/single_well.yaml \
    --input  DATA/single_well_train.npz \
    --out    checkpoints/csdi/single_well \
    --device cuda:0

echo "=== Forecasting CSDI: single_well ==="
python eval/CSDI/forecast_csdi.py \
    --config configs/csdi_train/single_well.yaml \
    --input  DATA/single_well_test.npz \
    --ckpt   checkpoints/csdi/single_well/model.pth \
    --out    results/csdi/single_well.npz \
    --device cuda:0

python plot_heatmap.py  \
    --results results/csdi/single_well.npz   \
    --name   csdi  \
    --dataset single_well    \
    --out   plots/csdi/single_well.pdf    \
    --indices 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 \
