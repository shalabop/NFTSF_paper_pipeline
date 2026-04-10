#!/usr/bin/env bash
#SBATCH --job-name=phiRATD
#SBATCH --mail-user=meahmed@asu.edu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=meahmed@asu.com
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --output=logs/ratd/alanine_phi.%j.out
#SBATCH --error=logs/ratd/alanine_phi.%j.err

mkdir -p logs/ratd
mkdir -p results/ratd
mkdir -p checkpoints/ratd
mkdir -p checkpoints/tcn
mkdir -p references/alanine_phi

source venvRATD/bin/activate

module purge
module load cuda-12.8.1-gcc-12.1.0

export NO_AI_TRACKING=false

#Verify the path before running (helps with debugging)
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


python train/RATD/train_tcn.py --config configs/tcn/alanine_phi.yaml

python train/RATD/retrieval_builder.py --config configs/tcn/alanine_phi.yaml --type encode

python train/RATD/retrieval_builder.py --config configs/tcn/alanine_phi.yaml --type retrieval

python train/RATD/train_ratd.py --config configs/ratd/alanine_phi.yaml

python eval/RATD/forecast_ratd.py --config configs/ratd/alanine_phi.yaml
