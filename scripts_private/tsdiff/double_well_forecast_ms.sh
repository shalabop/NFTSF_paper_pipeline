#!/bin/bash
#SBATCH --job-name=forecastms
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --output=logs/tsdiff/double_well_ms.%j.out
#SBATCH --error=logs/tsdiff/double_well_ms.%j.err

echo "========================================"
echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "========================================"

# ----------------------------------------
# Activate environment
# ----------------------------------------
source ~/NFTSF_paper_pipeline/venv310/bin/activate

# ----------------------------------------
# 🔥 CRITICAL FIX: Clean ALL KeOps caches
# ----------------------------------------
echo "[1] Cleaning KeOps cache..."
rm -rf ~/.cache/keops
rm -rf ~/.cache/pykeops
rm -rf /tmp/keops*
rm -rf /tmp/pykeops*

# ----------------------------------------
# 🔥 CRITICAL FIX: correct env variable name
# (you used KEOPS_CACHE_DIR ❌)
# ----------------------------------------
export KEOPS_CACHE_FOLDER="/tmp/keops_cache_${SLURM_JOB_ID}"
mkdir -p "$KEOPS_CACHE_FOLDER"

# Optional but VERY useful for debugging
export PYKEOPS_VERBOSE=1

# ----------------------------------------
# Debug environment
# ----------------------------------------
echo "[2] Environment check"
python - <<EOF
import torch
print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

import os
print("KEOPS CACHE:", os.environ.get("KEOPS_CACHE_FOLDER"))
EOF

# ----------------------------------------
# Test PyKeOps BEFORE running model
# ----------------------------------------
echo "[3] Testing PyKeOps"
python - <<EOF
from pykeops.torch import Genred
print("✅ KeOps import OK")
EOF

# ----------------------------------------
# Run forecasting
# ----------------------------------------
echo "[4] Running forecast"

python eval/TSDiff/forecast_tsdiff.py \
    --config configs/tsdiff_forecast/double_well_mse_05.yaml \
    --dataset_path gluonts_datasets/double_well \
    --out results/tsdiff/double_well_mse_05

echo "========================================"
echo "DONE"
echo "========================================"