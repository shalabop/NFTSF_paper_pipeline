#!/bin/bash
#SBATCH --job-name=plot_trajectories
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=0-00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cpu
#
# Automatically discover training data for every landscape and plot 9 random
# trajectories from each one.  Data files are located using the same directory
# layout written by run_full_pipeline.sh:
#
#   <DATA_DIR>/<landscape>_train.npy
#
# If a canonical _train.npy file is not found the script falls back to the
# first *.npy file in DATA_DIR whose name contains the landscape keyword.
# Landscapes with no data are skipped with a warning (non-fatal).
#
# Usage
#   bash run_plot_trajectories.sh                   # local run
#   sbatch run_plot_trajectories.sh                 # SLURM submission
#   DATA_DIR=/custom/path bash run_plot_trajectories.sh   # override data dir

# ============================================================
# CONFIGURATION - edit the variables below if needed
# ============================================================

# Resolve the project directory so the script works from any working directory
# and also when submitted via sbatch (which changes CWD to a spool directory).
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

# Data directory: where run_full_pipeline.sh writes its .npy files.
# Override at runtime:  DATA_DIR=/my/data bash run_plot_trajectories.sh
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/pipeline_results/data}"

# Root output directory; one sub-folder is created per landscape.
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/plots/trajectories}"

DATA_FORMAT="multi_sim"   # all generated datasets use this format
N_PLOTS=9                 # trajectories to show per landscape
SEED=42                   # fixed seed for reproducibility; set to "" to randomise

# Landscapes to process (must match generate_trajectories.py --landscape choices)
LANDSCAPES=("linear_gaussian" "single_well" "double_well")

# Well-position guide lines per landscape.
# Values match the defaults in generate_trajectories.py:
#   double_well : left_well=-1.0, right_well=1.0
#   single_well : single minimum at x_mu=0.0  (shown as one guide line)
#   linear_gaussian : no equilibrium positions
# Set a landscape's entry to "" to suppress the lines for that landscape.
declare -A WELL_POS
WELL_POS["linear_gaussian"]=""
WELL_POS["single_well"]="0.0"
WELL_POS["double_well"]="-1.0 1.0"

# ============================================================
# ENVIRONMENT SETUP
# ============================================================
echo "============================================"
echo "NF-TSF  —  plot random training trajectories"
echo "Job started at : $(date)"
echo "Running on     : $(hostname)"
echo "Job ID         : ${SLURM_JOB_ID:-local}"
echo "Project dir    : $PROJECT_DIR"
echo "Data dir       : $DATA_DIR"
echo "Output dir     : $OUTPUT_DIR"
echo "============================================"
echo ""

mkdir -p logs

# Load mamba / conda only under SLURM (not needed for local runs)
if [ -n "$SLURM_JOB_ID" ]; then
    module load mamba/latest
    source activate nf_tsf
fi
echo "Python : $(which python)"
echo ""

cd "$PROJECT_DIR"

# ============================================================
# AUTO-DISCOVER AND PLOT
# ============================================================
OVERALL_STATUS=0
FOUND=0

for landscape in "${LANDSCAPES[@]}"; do

    echo "--------------------------------------------"
    echo "Landscape : $landscape"

    # 1. Canonical path written by run_full_pipeline.sh
    DATA_FILE="${DATA_DIR}/${landscape}_train.npy"

    # 2. Fallback: first *.npy in DATA_DIR whose name contains the keyword
    if [ ! -f "$DATA_FILE" ]; then
        DATA_FILE=$(find "$DATA_DIR" -maxdepth 1 -name "*${landscape}*.npy" \
                         -type f 2>/dev/null | sort | head -1)
    fi

    # 3. Wider fallback: search one level deeper (e.g. subdirectories)
    if [ ! -f "$DATA_FILE" ]; then
        DATA_FILE=$(find "$DATA_DIR" -name "*${landscape}*.npy" \
                         -type f 2>/dev/null | sort | head -1)
    fi

    if [ ! -f "$DATA_FILE" ]; then
        echo "  WARNING: no data file found for '$landscape' under $DATA_DIR — skipping."
        echo ""
        continue
    fi

    echo "  Data file : $DATA_FILE"

    LANDSCAPE_OUT="${OUTPUT_DIR}/${landscape}"
    mkdir -p "$LANDSCAPE_OUT"

    # Build the command for this landscape
    CMD=(python "${PROJECT_DIR}/plot_random_training_trajectories.py"
        --data_path   "$DATA_FILE"
        --data_format "$DATA_FORMAT"
        --landscape   "$landscape"
        --n_plots     "$N_PLOTS"
        --output_dir  "$LANDSCAPE_OUT"
    )

    if [ -n "$SEED" ]; then
        CMD+=(--seed "$SEED")
    fi

    if [ -n "${WELL_POS[$landscape]}" ]; then
        # shellcheck disable=SC2206  # intentional word-splitting of space-sep values
        CMD+=(--well_positions ${WELL_POS[$landscape]})
    fi

    echo "  Well positions : ${WELL_POS[$landscape]:-none}"
    echo "  Output dir     : $LANDSCAPE_OUT"
    echo "  Running: ${CMD[*]}"
    echo ""

    "${CMD[@]}"
    STATUS=$?

    if [ $STATUS -ne 0 ]; then
        echo "  ERROR: plot failed for $landscape (exit $STATUS)"
        OVERALL_STATUS=$STATUS
    else
        echo "  Done."
        FOUND=$((FOUND + 1))
    fi

    echo ""
done

# ============================================================
# SUMMARY
# ============================================================
echo "============================================"
echo "Finished at : $(date)"
echo "Landscapes plotted : $FOUND / ${#LANDSCAPES[@]}"
echo "Figures saved under: $OUTPUT_DIR"
echo "============================================"

exit $OVERALL_STATUS
