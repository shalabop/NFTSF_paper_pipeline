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

DATASET=${1:-double_well}
DEVICE=${2:-cuda:0}

CONFIG="configs/nsdiff/${DATASET}.yaml"
INPUT="DATA/${DATASET}.npz"
PRETRAIN_DIR="checkpoints/nsdiff/${DATASET}/pretrain"
DIFFUSION_DIR="checkpoints/nsdiff/${DATASET}/diffusion"

echo "========================================"
echo "NsDiff Pipeline: ${DATASET}"
echo "Device         : ${DEVICE}"
echo "Config         : ${CONFIG}"
echo "Input data     : ${INPUT}"
echo "Pretrain dir   : ${PRETRAIN_DIR}"
echo "Diffusion dir  : ${DIFFUSION_DIR}"
echo "========================================"

# Stage 1: Pretrain f_phi (mu_backbone)
echo ""
echo "=== Stage 1: Pretrain mu_backbone (f_phi) ==="
python pretrain_mu.py \
    --config  "${CONFIG}" \
    --input   "${INPUT}" \
    --out     "${PRETRAIN_DIR}" \
    --device  "${DEVICE}"

if [ $? -ne 0 ]; then
    echo "ERROR: Stage 1 (pretrain_mu.py) failed"
    exit 1
fi
echo "Stage 1 complete. Saved: ${PRETRAIN_DIR}/cond_pred_model.pth"

# Stage 2: Pretrain g_psi (g_backbone)
# Independent of Stage 1 — could be parallelised
echo ""
echo "=== Stage 2: Pretrain g_backbone (g_psi) ==="
python pretrain_g.py \
    --config  "${CONFIG}" \
    --input   "${INPUT}" \
    --out     "${PRETRAIN_DIR}" \
    --device  "${DEVICE}"

if [ $? -ne 0 ]; then
    echo "ERROR: Stage 2 (pretrain_g.py) failed"
    exit 1
fi
echo "Stage 2 complete. Saved: ${PRETRAIN_DIR}/cond_pred_model_g.pth"

# Stage 3: Train diffusion model with frozen f_phi and g_psi
echo ""
echo "=== Stage 3: Train diffusion model xi_theta ==="
python train_nsdiff_pretrained.py \
    --config   "${CONFIG}" \
    --input    "${INPUT}" \
    --pretrain "${PRETRAIN_DIR}" \
    --out      "${DIFFUSION_DIR}" \
    --device   "${DEVICE}"

if [ $? -ne 0 ]; then
    echo "ERROR: Stage 3 (train_nsdiff_pretrained.py) failed"
    exit 1
fi
echo "Stage 3 complete."

echo ""
echo "========================================"
echo "Pipeline complete for ${DATASET}"
echo "Forecast checkpoint: ${DIFFUSION_DIR}/"
echo "  model.pth              (xi_theta)"
echo "  cond_pred_model.pth    (f_phi — copy of pretrained)"
echo "  cond_pred_model_g.pth  (g_psi — copy of pretrained)"
echo ""
echo "Run forecasting:"
echo "  python forecast_nsdiff.py \\"
echo "      --config  ${CONFIG} \\"
echo "      --input   DATA/${DATASET}_test.npz \\"
echo "      --ckpt    ${DIFFUSION_DIR} \\"
echo "      --out     results/nsdiff/${DATASET}.npz"
echo "========================================"