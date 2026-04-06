#!/usr/bin/env bash
#SBATCH --job-name=uncondTSFdiff
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --output=electricity_train.%j.out
#SBATCH --error=electricity_train.%j.err

module purge
module load cuda-12.8.1-gcc-12.1.0

export CUDA_HOME="$CUDA_HOME"    
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$CUDA_PATH/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"

echo "=== Checking Cuda Headers ==="
ls $CUDA_PATH/include/cuda.h
ls $CUDA_PATH/include/nvrtc.h

echo "=== Checking NVRTC Libraries ==="
ls $CUDA_PATH/lib64/libnvrtc.so*
ls $CUDA_PATH/targets/x86_64-linux/lib/libnvrtc.so*

rm -rf ~/.cache/keops*

python bin/train_model.py -c configs/train_tsdiff/train_electricity.yaml

