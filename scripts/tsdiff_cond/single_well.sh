#!/usr/bin/env bash
#SBATCH --job-name=SWTSFdiffcond
#SBATCH --mail-user=meahmed@asu.edu
#SBATCH --mail-type=ALL
#SBATCH --time=2-15:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=logs/tsdiff_cond/SW_TSFdiff_cond.%j.out
#SBATCH --error=logs/tsdiff_cond/SW_TSFdiff_cond.%j.err


source venv310/bin/activate
which python

mkdir -p checkpoints/tsdiff_cond/single_well
mkdir -p results/tsdiff_cond/single_well
mkdir -p logs/tsdiff_cond/single_well

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
        --dataset_path gluonts_datasets/single_well \
        --config configs/tsdiff_cond_train/single_well_25_25_cond.yaml \
        --out_dir checkpoints_25_25/tsdiff_cond/single_well

python train/TSDiff/train_cond_tsdiff.py \
        --dataset_path gluonts_datasets/single_well \
        --config configs/tsdiff_cond_train/single_well_50_50_cond.yaml \
        --out_dir checkpoints_50_50/tsdiff_cond/single_well