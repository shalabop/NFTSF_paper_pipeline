#!/usr/bin/env bash
#SBATCH --job-name=sw50
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=meahmed@asu.edu
#SBATCH --mail-type=ALL
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=32G
#SBATCH --output=logs/fsctsw50.%j.out
#SBATCH --error=logs/fsctsw50.%j.err


source /packages/apps/mamba/2.0.8/etc/profile.d/conda.sh
conda activate /home/meahmed/.conda/envs/venv310
which python

mkdir -p results/nf
mkdir -p logs/forecast

which python

PROJECT_ROOT=$(pwd) 
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

module purge
module load cuda-12.8.1-gcc-12.1.0

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$CUDA_PATH/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$CUDA_PATH/targets/x86_64-linux/lib:$LIBRARY_PATH"
export CPATH="$CUDA_PATH/targets/x86_64-linux/include:$CPATH"

rm -rf ~/.cache/keops* 2>/dev/null
rm -rf ~/.cache/pykeops* 2>/dev/null
rm -rf /tmp/keops* 2>/dev/null
rm -rf /tmp/pykeops* 2>/dev/null

export PYKEOPS_BUILD_DIR="$SLURM_TMPDIR/pykeops_build"
export KEOPS_CACHE_FOLDER="$SLURM_TMPDIR/keops_cache"

if [ -z "$SLURM_TMPDIR" ]; then
    export PYKEOPS_BUILD_DIR="$HOME/.cache/pykeops_build"
    export KEOPS_CACHE_FOLDER="$HOME/.cache/keops_cache"
fi

mkdir -p "$PYKEOPS_BUILD_DIR"
mkdir -p "$KEOPS_CACHE_FOLDER"

echo "PyKeOps build dir: $PYKEOPS_BUILD_DIR"
echo "KeOps cache dir: $KEOPS_CACHE_FOLDER"

echo "Checking CUDA installation..."
ls $CUDA_PATH/include/cuda.h || echo "WARNING: cuda.h not found"
ls $CUDA_PATH/include/nvrtc.h || echo "WARNING: nvrtc.h not found"
ls $CUDA_PATH/lib64/libnvrtc.so* || ls $CUDA_PATH/targets/x86_64-linux/lib/libnvrtc.so* || echo "WARNING: libnvrtc.so not found"

python eval/NF/forecast_nf.py \
    --model_path   checkpoints_50_50/nf/single_well/model.pth     \
    --config configs/nf/single_well_50_50.yaml \
    --data_path DATA/single_well_test.npz    \
    --out results/nf/single_well_50_50.npz \
    --context_length  50 \
    --prediction_length 50  \
    --n_samples 1000 \
    --train_test_split 900 \
    --test_size 3000


python eval/TSDiff/forecast_tsdiff_cond.py   \
        --config configs/tsdiff_forecast/single_well_cond_50_50.yaml  \
        --dataset_path gluonts_datasets/single_well \
        --out results/tsdiff_cond/single_well_50_50.npz

python eval/CSDI/forecast_csdi.py \
    --config configs/csdi_train/single_well_50_50.yaml \
    --input  DATA/single_well_test.npz \
    --ckpt   checkpoints_25_25/csdi/single_well/model.pth \
    --out    results/csdi/single_well_50_50.npz \
    --device cuda:0