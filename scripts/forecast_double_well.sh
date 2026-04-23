#!/usr/bin/env bash
#SBATCH --job-name=evaluation
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=meahmed@asu.edu
#SBATCH --mail-type=ALL
#SBATCH --partition=general
#SBATCH --qos=grp_spresse
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=32G
#SBATCH --output=logs/nf/evaluation.%j.out
#SBATCH --error=logs/nf/evaluation.%j.err


source /packages/apps/mamba/2.0.8/etc/profile.d/conda.sh
conda activate /home/meahmed/.conda/envs/venv310
which python

mkdir -p results/nf
mkdir -p logs/nf

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


python eval/NF/forecast_nf.py \
    --model_path   checkpoints_25_25/nf/double_well/model.pth     \
    --config configs/nf/forecast_25_25.yaml \
    --data_path    DATA/double_well_test.npz    \
    --out          results/nf/double_well_25_25.npz \
    --context_length     25    \
    --prediction_length     25   \
    --n_samples    1000 \
    --train_test_split 900 \
    --test_size 3000

python eval/NF/forecast_nf.py \
    --model_path   checkpoints_25_25/nf/double_well_with_memory/model.pth     \
    --config configs/nf/forecast_25_25.yaml \
    --data_path    DATA/double_well_with_memory_test.npz    \
    --out          results/nf/double_well_with_memory_25_25.npz \
    --context_length     25    \
    --prediction_length     25   \
    --n_samples    1000 \
    --train_test_split 900 \
    --test_size 3000

python eval/CSDI/forecast_csdi.py \
    --config configs/csdi_train/double_well_25_25.yaml \
    --input  DATA/double_well_test.npz \
    --ckpt   checkpoints_25_25/csdi/double_well/model.pth \
    --out    results/csdi/double_well_25_25.npz \
    --device cuda:0

python eval/CSDI/forecast_csdi.py \
    --config configs/csdi_train/double_well_with_memory_25_25.yaml \
    --input  DATA/double_well_with_memory_test.npz \
    --ckpt   checkpoints_25_25/csdi/double_well_with_memory_test/model.pth \
    --out    results/csdi/double_well_with_memory_25_25.npz \
    --device cuda:0

python eval/TSDiff/forecast_tsdiff_cond.py   \
        --config configs/tsdiff_forecast/double_well_cond_25_25.yaml  \
        --dataset_path gluonts_datasets/double_well \
        --out results/tsdiff_cond/double_well_25_25.npz


python eval/TSDiff/forecast_tsdiff_cond.py   \
        --config configs/tsdiff_forecast/double_well_with_memory_cond_25_25.yaml  \
        --dataset_path gluonts_datasets/double_well_with_memory \
        --out results/tsdiff_cond/double_well_with_memory_25_25.npz
