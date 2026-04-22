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
#SBATCH --output=logs/nf/evaluation_test.%j.out
#SBATCH --error=logs/nf/evaluation_test.%j.err


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
Pexport CUDA_HOME=$(dirname $(dirname $(which nvcc)))
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


python eval/NF/forecast_nf_ex.py \
  --config configs/nf/alanine_phi_100_100_forecast.yaml \
  --input DATA/alanine_phi_test.npz \
  --ckpt checkpoints_100_100/nf/alanine_phi.pth \
  --out results/nf/alanine_phi_100_100_test.npz \
  --device cuda:0 --test_size 3000 \
  --prediction_length 100 --context_length 100 \
  --num_of_samples 1000 --train_test_split 900
  
python eval/NF/forecast_nf_ex.py \
  --config configs/nf/alanine_psi_100_100_forecast.yaml \
  --input DATA/alanine_psi_test.npz \
  --ckpt checkpoints_100_100/nf/alanine_psi.pth \
  --out results/nf/alanine_psi_100_100_test.npz \
  --device cuda:0 --test_size 3000 \
  --prediction_length 100 --context_length 100 \
  --num_of_samples 1000 --train_test_split 900
  