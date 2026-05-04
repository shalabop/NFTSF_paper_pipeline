#!/bin/bash
#SBATCH --job-name=nf_tsf_test
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --export=NONE

# ============================================================
# CONFIGURATION - EDIT THESE VARIABLES
# ============================================================
MODEL_PATH="/path/to/your/trained_model.pth"
NORM_STATS_PATH="/path/to/your/norm_stats.npz"  # Optional, set to "" if not available
DATA_PATH="/path/to/your/test_data.npy"
DATA_FORMAT="multi_sim"  # Options: single_sim, multi_sim, tnf
OUTPUT_DIR="./results/testing_$(date +%Y%m%d_%H%M%S)"

N_PAST=100
N_FUTURE=100
N_SAMPLES=500
N_QUART_TEST=500
RUN_QUANTILE_TESTS=false  # Set to true for full quantile calibration tests
BASIC_EVAL_USE_LAST_WINDOW=false  # Single basic plot trajectory uses tail window of trajectory 0
FULL_EVAL_USE_LAST_WINDOW=true    # Full-set eval uses tail window per trajectory
CONFIG_PATH=""                    # Optional train config_*.json used to match architecture at test time
# Model architecture variant — only needed when CONFIG_PATH is not set.
# When CONFIG_PATH is provided, model_variant is loaded automatically from JSON.
MODEL_VARIANT="${MODEL_VARIANT:-ar}"

# ============================================================
# ENVIRONMENT SETUP
# ============================================================
echo "============================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "============================================"

mkdir -p logs

# Load mamba module and activate conda environment
module load mamba/latest
source activate nf_tsf
echo "Conda env: $CONDA_DEFAULT_ENV"

# Fix environment: downgrade numpy to <2 (required for pre-compiled torch) and ensure seaborn is installed
pip install "numpy<2" seaborn --quiet

echo "Python: $(which python)"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
echo ""

# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# ============================================================
# RUN TESTING
# ============================================================
echo "Starting model evaluation..."
echo "Model: $MODEL_PATH"
echo "Test data: $DATA_PATH"
echo ""

# Build command
CMD="python test_model.py \
    --model_path \"$MODEL_PATH\" \
    --data_path \"$DATA_PATH\" \
    --data_format $DATA_FORMAT \
    --n_past $N_PAST \
    --n_future $N_FUTURE \
    --n_samples $N_SAMPLES \
    --n_quart_test $N_QUART_TEST \
    --output_dir \"$OUTPUT_DIR\" \
    --device auto"

# Add normalization stats if provided
if [ -n "$NORM_STATS_PATH" ] && [ "$NORM_STATS_PATH" != "" ]; then
    CMD="$CMD --norm_stats_path \"$NORM_STATS_PATH\""
fi

# Add quantile tests flag if enabled
if [ "$RUN_QUANTILE_TESTS" = true ]; then
    CMD="$CMD --run_quantile_tests"
fi

# Use the last (n_past + n_future) window for the single basic-eval trajectory
if [ "$BASIC_EVAL_USE_LAST_WINDOW" = true ]; then
    CMD="$CMD --basic_eval_use_last_window"
fi

# Use the last (n_past + n_future) window for each trajectory in full-set eval
if [ "$FULL_EVAL_USE_LAST_WINDOW" = true ]; then
    CMD="$CMD --full_eval_use_last_window"
fi

# Add architecture config if provided
if [ -n "$CONFIG_PATH" ] && [ "$CONFIG_PATH" != "" ]; then
    CMD="$CMD --config_path \"$CONFIG_PATH\""
fi

# Forward model_variant when no config_path is set (config path auto-loads it when present)
if [ -z "$CONFIG_PATH" ] || [ "$CONFIG_PATH" = "" ]; then
    CMD="$CMD --model_variant \"$MODEL_VARIANT\""
fi

# Run the command
eval $CMD

EXIT_STATUS=$?

# ============================================================
# FINISH
# ============================================================
echo ""
echo "============================================"
echo "Job finished at: $(date)"
echo "Exit status: $EXIT_STATUS"
echo "Results saved to: $OUTPUT_DIR"
echo "============================================"

exit $EXIT_STATUS
