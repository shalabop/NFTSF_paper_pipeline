#!/usr/bin/env bash
# =============================================================================
# train_sol.sh — SLURM sbatch script for Sol (ASU HPC)
# =============================================================================
# Trains ONE model through NFTSF_paper_pipeline on a single A100 GPU.
# After training, automatically runs evaluation and writes results.npz.
#
# Usage — direct sbatch:
#   sbatch train_sol.sh
#
# Usage — override MODEL at submission time:
#   sbatch --export=ALL,MODEL=csdi,RUN_ID=run_paper train_sol.sh
#
# Usage — local smoke test (2 epochs, CPU):
#   MODEL=nftsf RUN_ID=smoke bash train_sol.sh
#
# Supported MODEL values:
#   nftsf | tsdiff_q | tsdiff_ms | tsdiff_cond | csdi | ratd | nsdiff | arima
# =============================================================================

#SBATCH --job-name=tsf_train
#SBATCH --output=logs/tsf_%x_%j.out
#SBATCH --error=logs/tsf_%x_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --export=NONE
#
# Notes for Sol:
#   • Adjust --time if diffusion models need longer (e.g. 3-00:00:00 for csdi).
#   • Adjust --mem / --gres if you switch to a V100 partition.
#   • ARIMA needs no GPU and runs in seconds — you can drop --gres for it.

set -euo pipefail

# =============================================================================
# CONFIGURATION — edit these or pass via SLURM --export
# =============================================================================

# Model to train. One of:
#   nftsf | tsdiff_q | tsdiff_ms | tsdiff_cond | csdi | ratd | nsdiff | arima
MODEL="${MODEL:-tsdiff_q}"

# Conda environment name for NFTSF_paper_pipeline.
# !! CHANGE THIS to your actual environment name !!
CONDA_ENV="${CONDA_ENV:-unified_tsf}"

# Unique run identifier (used to name the output sub-directory).
RUN_ID="${RUN_ID:-run_$(date +%Y%m%d_%H%M%S)}"

# Absolute path to the NFTSF_paper_pipeline repo root.
PROJECT_DIR="${PROJECT_DIR:-/home/user/NFTSF_paper_pipeline}"

# Canonical .npz dataset.  Built by convert_nftsf_to_canonical.sh if absent.
DATA_NPZ="${DATA_NPZ:-${PROJECT_DIR}/outputs/canonical/alanine_phi.npz}"

# Source .npy files (only needed if the canonical .npz must be built here).
TRAIN_NPY="${TRAIN_NPY:-/home/user/NFTSF_ssh/alanine_phi_train.npy}"
TEST_NPY="${TEST_NPY:-/home/user/NFTSF_ssh/alanine_phi_test.npy}"

# Root output directory (sub-dirs created automatically by train.py).
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs}"

# Optional: number of forecast samples for evaluation (default 500).
N_EVAL_SAMPLES="${N_EVAL_SAMPLES:-500}"

# Optional: override total training epochs (leave empty to use config default).
EPOCHS_OVERRIDE="${EPOCHS_OVERRIDE:-}"

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================
START_TIME=$(date +%s)
echo "============================================================"
echo " TSF Training — Sol SLURM Job"
echo "============================================================"
echo " Job started : $(date)"
echo " Node        : $(hostname)"
echo " Job ID      : ${SLURM_JOB_ID:-local}"
echo " Model       : $MODEL"
echo " Run ID      : $RUN_ID"
echo " Dataset     : $DATA_NPZ"
echo " Output root : $OUTPUT_DIR"
echo "============================================================"

mkdir -p "${PROJECT_DIR}/logs"

# Load Sol's mamba module and activate the conda environment.
module load mamba/latest
source activate "$CONDA_ENV"
echo "[env] Conda env  : $CONDA_DEFAULT_ENV"
echo "[env] Python     : $(which python) — $(python --version 2>&1)"

# Help PyTorch manage GPU memory more efficiently.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Print GPU info (non-fatal if no GPU available, e.g. local test).
python -c "
import torch
print(f'[env] PyTorch    : {torch.__version__}')
print(f'[env] CUDA avail : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'[env] GPU        : {torch.cuda.get_device_name(0)}')
" || true
echo ""

# =============================================================================
# STEP 0 — Build canonical .npz (skipped if already present)
# =============================================================================
if [[ -f "$DATA_NPZ" ]]; then
    echo "[step 0] Canonical dataset already exists — skipping conversion."
    echo "         $DATA_NPZ"
else
    echo "[step 0] Building canonical dataset from NFTSF .npy files …"
    for F in "$TRAIN_NPY" "$TEST_NPY"; do
        if [[ ! -f "$F" ]]; then
            echo "[ERROR] Required file not found: $F"
            echo "        Run convert_nftsf_to_canonical.sh first, or set"
            echo "        DATA_NPZ to point at an existing canonical .npz."
            exit 1
        fi
    done

    # MD config: lookback=50, horizon=50
    python "${PROJECT_DIR}/data/canonical.py" \
        --source    nftsf \
        --train     "$TRAIN_NPY" \
        --test      "$TEST_NPY" \
        --landscape alanine_phi \
        --n_past    50 \
        --n_future  50 \
        --seed      42 \
        --out       "$DATA_NPZ"
    echo "[step 0] Canonical dataset written: $DATA_NPZ"
fi
echo ""

# =============================================================================
# STEP 1 — Training
# =============================================================================
# Resolve per-model config file.
CONFIG="${PROJECT_DIR}/configs/${MODEL}.yaml"
if [[ ! -f "$CONFIG" ]]; then
    echo "[ERROR] Config not found: $CONFIG"
    echo "        Available configs:"
    ls "${PROJECT_DIR}/configs/"
    exit 1
fi

cd "$PROJECT_DIR"

TRAIN_CMD=(
    python train/train.py
    --model     "$MODEL"
    --config    "$CONFIG"
    --data      "$DATA_NPZ"
    --run_id    "$RUN_ID"
    --output_dir "$OUTPUT_DIR"
)
[[ -n "$EPOCHS_OVERRIDE" ]] && TRAIN_CMD+=(--epochs "$EPOCHS_OVERRIDE")

echo "[step 1] Training $MODEL …"
echo "         Command: ${TRAIN_CMD[*]}"
echo ""
"${TRAIN_CMD[@]}"
echo ""
echo "[step 1] Training complete."
echo ""

# =============================================================================
# STEP 2 — Evaluation
# =============================================================================
# train.py writes checkpoints to: $OUTPUT_DIR/{model}/alanine_phi/{run_id}/
CHECKPOINT_DIR="${OUTPUT_DIR}/${MODEL}/alanine_phi/${RUN_ID}"

if [[ -d "$CHECKPOINT_DIR" ]]; then
    echo "[step 2] Evaluating $MODEL (n_samples=$N_EVAL_SAMPLES) …"
    python eval/evaluate.py \
        --checkpoint  "$CHECKPOINT_DIR" \
        --data        "$DATA_NPZ" \
        --n_samples   "$N_EVAL_SAMPLES"
    echo "[step 2] Evaluation complete.  results.npz → $CHECKPOINT_DIR"
else
    echo "[step 2] WARNING: checkpoint directory not found, skipping evaluation."
    echo "         Expected: $CHECKPOINT_DIR"
fi
echo ""

# =============================================================================
# FINISH
# =============================================================================
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
EXIT_STATUS=0

echo "============================================================"
echo " Job finished : $(date)"
echo " Elapsed      : $(( ELAPSED / 3600 ))h $(( (ELAPSED % 3600) / 60 ))m $(( ELAPSED % 60 ))s"
echo " Exit status  : $EXIT_STATUS"
echo " Outputs      : $CHECKPOINT_DIR"
echo "============================================================"

exit $EXIT_STATUS
