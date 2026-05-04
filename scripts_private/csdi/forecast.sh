#!/usr/bin/env bash
#SBATCH --job-name=evaluation
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=meahmed@asu.edu
#SBATCH --mail-type=ALL
#SBATCH --partition=general
#SBATCH --qos=grp_spresse
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=32G
#SBATCH --output=logs/csdi/evaluation.%j.out
#SBATCH --error=logs/csdi/evaluation.%j.err

mkdir -p results/csdi
mkdir -p results/csdi/alanine_phi
#mkdir -p logs/csdi/alanine_phi
mkdir -p checkpoints/csdi/
mkdir -p checkpoints/csdi/alanine_phi

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


python eval/CSDI/forecast_csdi.py \
    --config configs/csdi_train/alanine_phi_25_25.yaml \
    --input  DATA/alanine_phi_test.npz \
    --ckpt   checkpoints_25_25/csdi/alanine_phi/model.pth \
    --out    results/csdi/alanine_phi_25_25.npz \
    --device cuda:0

python eval/CSDI/forecast_csdi.py \
    --config configs/csdi_train/alanine_psi_25_25.yaml \
    --input  DATA/alanine_psi_test.npz \
    --ckpt   checkpoints_25_25/csdi/alanine_psi/model.pth \
    --out    results/csdi/alanine_psi_25_25.npz \
    --device cuda:0
