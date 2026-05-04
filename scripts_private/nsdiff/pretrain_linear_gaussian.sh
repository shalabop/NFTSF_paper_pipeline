#!/usr/bin/env bash
#SBATCH --job-name=LGNsdiff
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --mail-user=meahmed@asu.edu
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --partition=general
#SBATCH --qos=grp_spresse
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/nsdiff/linear_gaussian.%j.out
#SBATCH --error=logs/nsdiff/linear_gaussian.%j.err

#
source venv310/bin/activate

mkdir -p checkpoints/nsdiff/linear_gaussian/pretrain
mkdir -p checkpoints/nsdiff/linear_gaussian/diffusion
mkdir -p results/nsdiff
mkdir -p logs/nsdiff

python train/NsDiff/pretrain_mu.py \
    --config  configs/nsdiff_train/linear_gaussian.yaml \
    --input   DATA/linear_gaussian.npz \
    --out     checkpoints/nsdiff/linear_gaussian/pretrain \
    --device  cuda:0

python train/NsDiff/pretrain_g.py \
    --config  configs/nsdiff_train/linear_gaussian.yaml \
    --input   DATA/linear_gaussian.npz \
    --out     checkpoints/nsdiff/linear_gaussian/pretrain \
    --device  cuda:0

python train/NsDiff/train_nsdiff_pretrained.py \
    --config   configs/nsdiff_train/linear_gaussian.yaml \
    --input    DATA/linear_gaussian.npz \
    --pretrain checkpoints/nsdiff/linear_gaussian/pretrain \
    --out      checkpoints/nsdiff/linear_gaussian/diffusion \
    --device   cuda:0

python eval/NsDiff/forecast_nsdiff.py \
      --config  configs/nsdiff_train/linear_gaussian.yaml \
      --input   DATA/linear_gaussian.npz \
      --ckpt    checkpoints/nsdiff/linear_gaussian/diffusion \
      --out     results/nsdiff/linear_gaussian.npz
