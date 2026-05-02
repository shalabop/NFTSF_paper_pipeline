#!/usr/bin/env bash
#SBATCH --job-name=forecast
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=meahmed@asu.edu
#SBATCH --mail-type=ALL
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=32G
#SBATCH --output=logs/forecaaaaaaaaaaaaaaaaaaaaast.%j.out
#SBATCH --error=logs/forecaaaaaaaaaaaaaaaaaaaaast.%j.err


mkdir -p results/csdi
mkdir -p results//alanine_phi_50_50
#mkdir -p logs/csdi/alanine_phi
mkdir -p checkpoints_50_50/csdi/
mkdir -p checkpoints_50_50/csdi/alanine_phi

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
    --config configs/csdi_train_50/single_well_25_25.yaml \
    --input  DATA/single_well_test.npz \
    --ckpt   checkpoints_25_25/csdi_with_early_stopping_50/single_well/model.pth \
    --out    results/csdi_with_early_stopping_50/single_well_25_25.npz \
    --device cuda:0

python eval/CSDI/forecast_csdi.py \
    --config configs/csdi_train_50/single_well_50_50.yaml \
    --input  DATA/single_well_test.npz \
    --ckpt   checkpoints_50_50/csdi_with_early_stopping_50/single_well/model.pth \
    --out    results/csdi_with_early_stopping_50/single_well_50_50.npz \
    --device cuda:0

python eval/CSDI/forecast_csdi.py \
    --config configs/csdi_train_50/double_well_25_25.yaml \
    --input  DATA/double_well_test.npz \
    --ckpt   checkpoints_25_25/csdi_with_early_stopping_50/double_well/model.pth \
    --out    results/csdi_with_early_stopping_50/double_well_25_25.npz \
    --device cuda:0

python eval/CSDI/forecast_csdi.py \
    --config configs/csdi_train_50/double_well_50_50.yaml \
    --input  DATA/double_well_test.npz \
    --ckpt   checkpoints_50_50/csdi_with_early_stopping_50/double_well/model.pth \
    --out    results/csdi_with_early_stopping_50/double_well_50_50.npz \
    --device cuda:0


python eval/CSDI/forecast_csdi.py \
    --config configs/csdi_train_50/alanine_phi_25_25.yaml \
    --input  DATA/alanine_phi_test.npz \
    --ckpt   checkpoints_25_25/csdi_with_early_stopping_50/alanine_phi/model.pth \
    --out    results/csdi_with_early_stopping_50/alanine_phi_25_25.npz \
    --device cuda:0

python eval/CSDI/forecast_csdi.py \
    --config configs/csdi_train_50/alanine_phi_50_50.yaml \
    --input  DATA/alanine_phi_test.npz \
    --ckpt   checkpoints_50_50/csdi_with_early_stopping_50/alanine_phi/model.pth \
    --out    results/csdi_with_early_stopping_50/alanine_phi_50_50.npz \
    --device cuda:0

python eval/CSDI/forecast_csdi.py \
    --config configs/csdi_train_50/alanine_psi_25_25.yaml \
    --input  DATA/alanine_psi_test.npz \
    --ckpt   checkpoints_25_25/csdi_with_early_stopping_50/alanine_psi/model.pth \
    --out    results/csdi_with_early_stopping_50/alanine_psi_25_25.npz \
    --device cuda:0

python eval/CSDI/forecast_csdi.py \
    --config configs/csdi_train_50/alanine_psi_50_50.yaml \
    --input  DATA/alanine_psi_test.npz \
    --ckpt   checkpoints_50_50/csdi_with_early_stopping_50/alanine_psi/model.pth \
    --out    results/csdi_with_early_stopping_50/alanine_psi_50_50.npz \
    --device cuda:0