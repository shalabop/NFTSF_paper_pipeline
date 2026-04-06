#!/usr/bin/env bash
#SBATCH --job-name=ApsiTSFdiffcond
#SBATCH --mail-user=meahmed@asu.edu
#SBATCH --mail-type=ALL
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --output=logs/tsdiff_cond/alanine_psi.%j.out
#SBATCH --error=logs/tsdiff_cond/alanine_psi.%j.err


source venv310/bin/activate
which python

mkdir -p checkpoints/tsdiff_cond/alanine_psi
mkdir -p results/tsdiff_cond/alanine_psi
#mkdir -p logs/tsdiff_cond/double_well

PROJECT_ROOT=$(pwd) 
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

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


ls $CUDA_PATH/include/cuda.h || echo "cuda.h not found"
ls $CUDA_PATH/include/nvrtc.h || echo "nvrtc.h not found"

ls $CUDA_PATH/lib64/libnvrtc.so* || echo "libnvrtc.so not found in lib64"
ls $CUDA_PATH/targets/x86_64-linux/lib/libnvrtc.so* || echo "libnvrtc.so not found in targets/x86_64-linux/lib"

python train/TSDiff/train_cond_tsdiff.py \
        --dataset_path gluonts_datasets/alanine_psi \
        --config configs/tsdiff_cond_train/alanine_psi.yaml \
        --out_dir checkpoints/tsdiff_cond/alanine_psi

python eval/TSDiff/forecast_tsdiff_cond.py   \
        --config configs/tsdiff_forecast/alanine_psi_cond.yaml  \
        --dataset_path gluonts_datasets/alanine_psi \
        --out results/tsdiff_cond/alanine_psi