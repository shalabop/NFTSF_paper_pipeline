#!/usr/bin/env bash
#SBATCH --job-name=phiNsdiff
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --mail-user=meahmed@asu.edu
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --partition=general
#SBATCH --qos=grp_spresse
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/nsdiff/alanine_phi.%j.out
#SBATCH --error=logs/nsdiff/alanine_phi.%j.err

#
source venv310/bin/activate

mkdir -p checkpoints/nsdiff/alanine_phi/pretrain
mkdir -p checkpoints/nsdiff/alanine_phi/diffusion
mkdir -p results/nsdiff
mkdir -p logs/nsdiff

python train/NsDiff/pretrain_mu.py \
    --config  configs/nsdiff_train/alanine_phi.yaml \
    --input   DATA/alanine_phi.npz \
    --out     checkpoints/nsdiff/alanine_phi/pretrain \
    --device  cuda:0

python train/NsDiff/pretrain_g.py \
    --config  configs/nsdiff_train/alanine_phi.yaml \
    --input   DATA/alanine_phi.npz \
    --out     checkpoints/nsdiff/alanine_phi/pretrain \
    --device  cuda:0

python train/NsDiff/train_nsdiff_pretrained.py \
    --config   configs/nsdiff_train/alanine_phi.yaml \
    --input    DATA/alanine_phi.npz \
    --pretrain checkpoints/nsdiff/alanine_phi/pretrain \
    --out      checkpoints/nsdiff/alanine_phi/diffusion \
    --device   cuda:0

python eval/NsDiff/forecast_nsdiff.py \
      --config  configs/nsdiff_train/alanine_phi.yaml \
      --input   DATA/alanine_phi.npz \
      --ckpt    checkpoints/nsdiff/alanine_phi/diffusion \
      --out     results/nsdiff/alanine_phi.npz
