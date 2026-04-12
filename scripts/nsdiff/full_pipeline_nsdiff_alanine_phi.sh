#!/usr/bin/env bash
#SBATCH --job-name=NSDIFFdw
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --mail-user=meahmed@asu.edu
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --partition=general
#SBATCH --qos=grp_spresse
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/nsdiff/double_well.%j.out
#SBATCH --error=logs/nsdiff/double_well.%j.err

# Stage 1: Pretrain f_phi (mu_backbone)
echo ""
echo "=== Stage 1: Pretrain mu_backbone (f_phi) ==="
python train/NsDiff/pretrain_mu.py \
    --config  configs/nsdiff_train/alanine_phi.yaml \
    --input   DATA/alanine_phi.npz \
    --out     checkpoints/nsdiff/alanine_phi/pretrain \
    --device  cuda:0

if [ $? -ne 0 ]; then
    echo "ERROR: Stage 1 (pretrain_mu.py) failed"
    exit 1
fi
echo "Stage 1 complete. Saved: ${PRETRAIN_DIR}/cond_pred_model.pth"

# Stage 2: Pretrain g_psi (g_backbone)
# Independent of Stage 1 — could be parallelised
echo ""
echo "=== Stage 2: Pretrain g_backbone (g_psi) ==="
python train/NsDiff/pretrain_g.py \
    --config  configs/nsdiff_train/alanine_phi.yaml \
    --input   DATA/alanine_phi.npz \
    --out     checkpoints/nsdiff/alanine_phi/pretrain \
    --device  cuda:0

if [ $? -ne 0 ]; then
    echo "ERROR: Stage 2 (pretrain_g.py) failed"
    exit 1
fi
echo "Stage 2 complete. Saved: ${PRETRAIN_DIR}/cond_pred_model_g.pth"

# Stage 3: Train diffusion model with frozen f_phi and g_psi
echo ""
echo "=== Stage 3: Train diffusion model xi_theta ==="
python train/NsDiff/train_nsdiff_pretrained.py \
    --config   configs/nsdiff_train/alanine_phi.yaml \
    --input    DATA/alanine_phi.npz \
    --pretrain checkpoints/nsdiff/alanine_phi/pretrain \
    --out      checkpoints/nsdiff/alanine_phi/diffusion \
    --device   cuda:0

if [ $? -ne 0 ]; then
    echo "ERROR: Stage 3 (train_nsdiff_pretrained.py) failed"
    exit 1
fi
echo "Stage 3 complete."

echo "Run forecasting:"
python eval/NsDiff/forecast_nsdiff.py \
      --config  configs/nsdiff_train/alanine_phi.yaml \
      --input   DATA/alanine_phi.npz \
      --ckpt    checkpoints/nsdiff/alanine_phi/diffusion \
      --out     results/nsdiff/test.npz
echo "========================================"