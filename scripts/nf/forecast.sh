#!/usr/bin/env bash
#SBATCH --job-name=evaluation
#SBATCH --time=22:00:00
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

source venv310/bin/activate
which python

mkdir -p results/nf
mkdir -p logs/nf

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


python eval/NF/forecast_nf.py \
    --model_path   checkpoints_100_100/nf/alanine_phi.pth     \
    --config configs/nf/alanine_phi_100_100_forecast.yaml \
    --data_path    DATA/alanine_phi_test.npz    \
    --out          results/nf/alanine_phi_100_100.npz \
    --context_length     100    \
    --prediction_length     100   \
    --n_samples    1000 \
    --train_test_split 900 \
    --test_size 3000

python eval/NF/forecast_nf.py \
    --model_path   checkpoints_50_50/nf/alanine_phi/model.pth     \
    --config configs/nf/alanine_phi_50_50_forecast.yaml \
    --data_path    DATA/alanine_phi_test.npz    \
    --out          results/nf/alanine_phi_50_50.npz \
    --context_length     50    \
    --prediction_length     50   \
    --n_samples    1000 \
    --train_test_split 900 \
    --test_size 3000

python eval/NF/forecast_nf.py \
    --model_path   checkpoints_25_25/nf/alanine_phi/model.pth     \
    --config configs/nf/alanine_phi_25_25_forecast.yaml \
    --data_path    DATA/alanine_phi_test.npz    \
    --out          results/nf/alanine_phi_25_25.npz \
    --context_length     25    \
    --prediction_length     25   \
    --n_samples    1000 \
    --train_test_split 900 \
    --test_size 3000


python eval/NF/forecast_nf.py \
    --model_path   checkpoints_25_25/nf/alanine_psi/model.pth     \
    --config configs/nf/alanine_psi_25_25_forecast.yaml \
    --data_path    DATA/alanine_psi_test.npz    \
    --out          results/nf/alanine_psi_25_25.npz \
    --context_length     25    \
    --prediction_length     25   \
    --n_samples    1000 \
    --train_test_split 900 \
    --test_size 3000

python eval/NF/forecast_nf.py \
    --model_path   checkpoints_50_50/nf/alanine_psi/model.pth     \
    --config configs/nf/alanine_psi_50_50_forecast.yaml \
    --data_path    DATA/alanine_psi_test.npz    \
    --out          results/nf/alanine_psi_50_50.npz \
    --context_length     50  \
    --prediction_length     50   \
    --n_samples    1000 \
    --train_test_split 900 \
    --test_size 3000

python eval/NF/forecast_nf.py \
    --model_path   checkpoints_100_100/nf/alanine_psi.pth     \
    --config configs/nf/alanine_psi_100_100_forecast.yaml \
    --data_path    DATA/alanine_psi_test.npz    \
    --out          results/nf/alanine_psi_100_100.npz \
    --context_length     100  \
    --prediction_length     100   \
    --n_samples    1000 \
    --train_test_split 900 \
    --test_size 3000