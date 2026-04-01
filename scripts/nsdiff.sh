#!/usr/bin/env bash
#SBATCH --job-name=NSDIFFdw
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/NSDIFFdw.%j.out
#SBATCH --error=logs/NSDIFFdw.%j.err

source /home/meahmed/venv310/bin/activate
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

export PYTHONPATH=~/tsf_models/models/NsDiff:$PYTHONPATH

cd ~/tsf_models
mkdir -p logs checkpoints/nsdiff results/nsdiff

echo "=== Training NsDiff: double_well ==="
python train/NsDiff/train_nsdiff.py \
    --config configs/nsdiff_train/double_well.yaml \
    --input  DATA/double_well_std.npz \
    --out    checkpoints/nsdiff/double_well \
    --device cuda:0

echo "=== NsDiff training complete ==="

echo "=== Forecasting NsDiff: double_well ==="

python eval/NsDiff/forecast_nsdiff.py    \
     --config configs/nsdiff_train/double_well.yaml   \
    --input  DATA/double_well_std.npz    \
    --ckpt   checkpoints/nsdiff/double_well \
    --out    results/nsdiff/double_well.npz \
    --device cuda:0

echo "=== NsDiff forecasting complete ==="

python plot_heatmap.py \
    --results results/nsdiff/test.npz   \
    --name   nsdiff  \
    --dataset double_well    \
    --out   nsdifftest.pdf    \
    --indices 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 \
    --denorm data/double_well_std.npz