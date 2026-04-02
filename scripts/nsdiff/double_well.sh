#!/usr/bin/env bash
#SBATCH --job-name=NSDIFFdw
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --mail-user=meahmed@asu.edu
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/nsdiff/double_well.%j.out
#SBATCH --error=logs/nsdiff/double_well.%j.err

mkdir -p checkpoints/nsdiff/double_well
mkdir -p results/nsdiff/double_well
mkdir -p logs/nsdiff/double_well


source /venv310/bin/activate
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

export PYTHONPATH=~/architectures/NsDiff:$PYTHONPATH

python train/NsDiff/train_nsdiff.py \
    --config configs/nsdiff_train/double_well.yaml \
    --input  DATA/double_well.npz \
    --out    checkpoints/nsdiff/double_well \
    --device cuda:0

python eval/NsDiff/forecast_nsdiff.py    \
     --config configs/nsdiff_train/double_well.yaml   \
    --input  DATA/double_well.npz    \
    --ckpt   checkpoints/nsdiff/double_well \
    --out    results/nsdiff/double_well.npz \
    --device cuda:0
