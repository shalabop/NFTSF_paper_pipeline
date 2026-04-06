#!/usr/bin/env bash
#SBATCH --job-name=AphiArima
#SBATCH --mail-user=meahmed@asu.edu
#SBATCH --mail-type=ALL
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=lightwork
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/arima/alanine_phi.%j.out
#SBATCH --error=logs/arima/alanine_phi.%j.err


source venv310/bin/activate
which python

mkdir -p results/arima/alanine_phi
#mkdir -p logs/arima/alanine_phi

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
    --input DATA/alanine_phi_test.npz \
    --out results/arima/alanine_phi.npz \
    --num-samples 500 \
    -c configs/arima/alanine_phi.yaml


python eval/ARIMA/run_auto_arima.py \
    --input DATA/alanine_psi_test.npz \
    --out results/arima/alanine_psi_fh.npz \
    --num-samples 500 \
    -c configs/arima/alanine_psi.yaml


python eval/ARIMA/run_auto_arima.py \
    --input DATA/linear_guassian_test.npz \
    --out results/arima/linear_guassian.npz \
    --num-samples 500 \
    -c configs/arima/linear_guassian.yaml