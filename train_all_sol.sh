#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
source "$REPO_ROOT/runner.sh"
# =============================================================================
# train_all_sol.sh — Submit one SLURM job per model for a full comparison run
# =============================================================================
# Submits train_sol.sh once for every model listed in MODELS, all sharing the
# same RUN_ID so that viz/comparison.py can auto-discover them and produce
# apples-to-apples comparison plots.
#
# Usage:
#   bash train_all_sol.sh
#
# The RUN_ID defaults to the current timestamp.  Override to resume a run:
#   RUN_ID=run_paper bash train_all_sol.sh
#
# To submit only a subset, set MODELS before calling:
#   MODELS="nftsf tsdiff_q csdi" bash train_all_sol.sh
# =============================================================================

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================

# Shared run identifier — all jobs write to outputs/{model}/alanine_phi/$RUN_ID
RUN_ID="${RUN_ID:-run_$(date +%Y%m%d_%H%M%S)}"

# Conda environment name for NFTSF_paper_pipeline.
# !! CHANGE THIS to your actual environment name !!
CONDA_ENV="${CONDA_ENV:-unified_tsf}"

# Absolute path to the NFTSF_paper_pipeline repo root.
PROJECT_DIR="${PROJECT_DIR:-$REPO_ROOT}"

# Canonical .npz dataset.
DATA_NPZ="${DATA_NPZ:-${PROJECT_DIR}/outputs/canonical/alanine_phi.npz}"

# Source .npy files (only needed if the canonical .npz is absent).
TRAIN_NPY="${TRAIN_NPY:-$(dirname "$REPO_ROOT")/NFTSF_ssh/alanine_phi_train.npy}"
TEST_NPY="${TEST_NPY:-$(dirname "$REPO_ROOT")/NFTSF_ssh/alanine_phi_test.npy}"

# Models to train (space-separated).  Default: all supported models.
MODELS="${MODELS:-nftsf tsdiff_q tsdiff_ms tsdiff_cond csdi ratd nsdiff arima}"

# Per-model SLURM time limits (hh:mm:ss).  Adjust to your cluster experience.
declare -A MODEL_TIME=(
    [nftsf]="1-12:00:00"
    [tsdiff_q]="2-00:00:00"
    [tsdiff_ms]="2-00:00:00"
    [tsdiff_cond]="2-00:00:00"
    [csdi]="2-12:00:00"
    [ratd]="2-00:00:00"
    [nsdiff]="2-00:00:00"
    [arima]="0-00:30:00"
)

# Default time if a model is not listed above.
DEFAULT_TIME="2-00:00:00"

SCRIPT="${PROJECT_DIR}/train_sol.sh"

# =============================================================================
# PRE-FLIGHT CHECKS
# =============================================================================
if [[ ! -f "$SCRIPT" ]]; then
    echo "[ERROR] train_sol.sh not found: $SCRIPT"
    exit 1
fi

# Build canonical dataset once before submitting jobs (jobs inherit it).
if [[ ! -f "$DATA_NPZ" ]]; then
    echo "[pre-flight] Canonical dataset absent.  Building now …"
    for F in "$TRAIN_NPY" "$TEST_NPY"; do
        [[ -f "$F" ]] || { echo "[ERROR] Not found: $F"; exit 1; }
    done

    # Activate env temporarily just to run canonical.py
    module load mamba/latest 2>/dev/null || true
    source activate "$CONDA_ENV" 2>/dev/null || true

    # Derive landscape name from the output .npz filename (e.g. double_well.npz → double_well)
    LANDSCAPE="${LANDSCAPE:-$(basename "$DATA_NPZ" .npz)}"

    # MD config: lookback=50, horizon=50
    PYTHONPATH="${PROJECT_DIR}" "${PYTHON_RUNNER[@]}" "${PROJECT_DIR}/data/canonical.py" \
        --source    nftsf \
        --train     "$TRAIN_NPY" \
        --test      "$TEST_NPY" \
        --landscape "$LANDSCAPE" \
        --n_past    50 \
        --n_future  50 \
        --seed      42 \
        --out       "$DATA_NPZ"
    echo "[pre-flight] Canonical dataset ready: $DATA_NPZ"
fi

mkdir -p "${PROJECT_DIR}/logs"

# =============================================================================
# SUBMIT ONE JOB PER MODEL
# =============================================================================
echo "============================================================"
echo " Submitting full comparison run"
echo " Run ID  : $RUN_ID"
echo " Models  : $MODELS"
echo " Dataset : $DATA_NPZ"
echo "============================================================"

for MODEL in $MODELS; do
    TIME="${MODEL_TIME[$MODEL]:-$DEFAULT_TIME}"

    # ARIMA doesn't need a GPU.
    if [[ "$MODEL" == "arima" ]]; then
        GRES_FLAG=""
        MEM_FLAG="--mem=16G"
    else
        GRES_FLAG="--gres=gpu:a100:1"
        MEM_FLAG="--mem=64G"
    fi

    JOB_NAME="tsf_${MODEL}"

    JOB_ID=$(sbatch \
        --job-name   "$JOB_NAME" \
        --time        "$TIME" \
        $GRES_FLAG \
        $MEM_FLAG \
        --export     "ALL,MODEL=${MODEL},RUN_ID=${RUN_ID},CONDA_ENV=${CONDA_ENV},PROJECT_DIR=${PROJECT_DIR},DATA_NPZ=${DATA_NPZ},TRAIN_NPY=${TRAIN_NPY},TEST_NPY=${TEST_NPY}" \
        "$SCRIPT" \
        | awk '{print $NF}')

    echo "  Submitted: $MODEL  → job $JOB_ID  (time limit: $TIME)"
done

echo ""
echo "============================================================"
echo " All jobs submitted.  Monitor with:"
echo "   squeue -u \$USER"
echo ""
echo " When all jobs finish, generate comparison plots:"
echo "   cd $PROJECT_DIR"
echo "   source activate $CONDA_ENV"
echo "   "${PYTHON_RUNNER[@]}" viz/comparison.py \\"
echo "       --results_dir outputs/ \\"
echo "       --landscape   alanine_phi \\"
echo "       --trajectory_id 42 \\"
echo "       --pdf"
echo "============================================================"
