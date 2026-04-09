#!/usr/bin/env bash
#SBATCH --job-name=SWarima
#SBATCH --mail-user=meahmed@asu.edu
#SBATCH --mail-type=ALL
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --output=logs/arima/single_well.%j.out
#SBATCH --error=logs/arima/single_well.%j.err


source venv310/bin/activate
which python

mkdir -p results/arima
mkdir -p logs/arima

PROJECT_ROOT=$(pwd) 
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

module purge
module load cuda-12.8.1-gcc-12.1.0

export NO_AI_TRACKING=false
PPexport CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_PATH/bin:$PATH"

export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$CUDA_PATH/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$CUDA_PATH/targets/x86_64-linux/lib:$LIBRARY_PATH"
export CPATH="$CUDA_PATH/targets/x86_64-linux/include:$CPATH"

export KEOPS_CACHE_DIR=$HOME/.cache/keops
mkdir -p $KEOPS_CACHE_DIR


ls $CUDA_PATH/include/cuda.h || echo "cuda.h not found"
ls $CUDA_PATH/include/nvrtc.h || echo "nvrtc.h not found"

ls $CUDA_PATH/lib64/libnvrtc.so* || echo "libnvrtc.so not found in lib64"
ls $CUDA_PATH/targets/x86_64-linux/lib/libnvrtc.so* || echo "libnvrtc.so not found in targets/x86_64-linux/lib"

python eval/ARIMA/run_auto_arima.py \
    --input DATA/single_well_test.npz \
    --out results/arima/single_well.npz \
    --model 3000 \
    --num-samples 500 \
    -c configs/arima/single_well.yaml