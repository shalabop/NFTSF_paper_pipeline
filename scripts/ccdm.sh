#!/usr/bin/env bash
#SBATCH --job-name=ccdm
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/ccdm.%j.out
#SBATCH --error=logs/ccdm.%j.err

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

export PYTHONPATH=~/NFTSF_paper_pipeline/architectures/CCDM:$PYTHONPATH

cd ~/NFTSF_paper_pipeline
mkdir -p logs results/ccdm checkpoints/ccdm

echo "=== Stage 1: Training CCDM double_well (end-to-end) ==="
python train/CCDM/train_ccdm.py \
    --config configs/ccdm_train/double_well.yaml \
    --input  DATA/double_well_std.npz \
    --out    checkpoints/ccdm/double_well \
    --device cuda:0

echo "=== Stage 2: Fine-tuning CCDM double_well (contrast) ==="
python train/CCDM/train_ccdm.py \
    --config configs/ccdm_train/double_well.yaml \
    --input  data/double_well_std.npz \
    --out    checkpoints/ccdm/double_well \
    --device cuda:0 \
    --two_stage \
    --refine_epochs 30

echo "=== CCDM training complete ==="

echo "=== Forecasting CCDM: double_well (best model) ==="
python eval/CCDM/forecast_ccdm.py \
    --config configs/ccdm_train/double_well.yaml \
    --input  DATA/double_well_std.npz \
    --ckpt   checkpoints/ccdm/double_well/model_best.pt \
    --out    results/ccdm/double_well.npz \
    --device cuda:0

python plot_heatmap.py    \
    --results results/ccdm/double_well.npz  \
    --name    CCDM  \
    --dataset double_well    \
    --out   ccdm_dw.pdf    \
    --indices 1 2 3 4 5 6 7 8 9 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 \
    --denorm data/double_well_std.npz

echo "=== CCDM forecasting complete ==="


python train_ccdm.py     \
  --config configs/ccdm_train/test.yaml    \
   --input  data/double_well_std.npz     \
   --out    checkpoints/ccdm/test     \
  --device cuda:0

 

  python forecast_ccdm.py \
    --config configs/ccdm_train/test.yaml \
    --input  data/double_well_std.npz \
    --ckpt   checkpoints/ccdm/test/model_best.pt \
    --out    results/ccdm/double_well.npz \
    --device cuda:0


python plot_heatmap.py    \
    --results results/ccdm/double_well.npz  \
    --name    CCDM  \
    --dataset double_well    \
    --out   ccdm_dw.pdf    \
    --indices 1 2 3 4 5 6 7 8 9 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 \
    --denorm data/double_well_std.npz

echo "=== CCDM forecasting complete ==="


 python train_ccdm.py \
    --config configs/ccdm_train/test_stage_2.yaml \
    --input  data/double_well_std.npz \
    --out    checkpoints/ccdm/test_stage_2 \
    --device cuda:0 \
    --two_stage \
    --refine_epochs 30

  python forecast_ccdm.py \
    --config configs/ccdm_train/test_stage_2.yaml \
    --input  data/double_well_std.npz \
    --ckpt   checkpoints/ccdm/test_stage_2/model_best.pt \
    --out    results/ccdm/double_well_stage_2.npz \
    --device cuda:0


python plot_heatmap.py    \
    --results results/ccdm/double_well_stage_2.npz  \
    --name    CCDM  \
    --dataset double_well    \
    --out   ccdm_dw_stage_2.pdf    \
    --indices 1 2 3 4 5 6 7 8 9 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 \
    --denorm data/double_well_std.npz
