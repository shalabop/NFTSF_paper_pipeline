#!/bin/bash
#SBATCH --job-name=nf_tsf_sweep
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --export=NONE
#SBATCH --array=0-5  # Adjust based on number of experiments

# ============================================================
# PARAMETER SWEEP CONFIGURATION
# ============================================================
# This script runs multiple training experiments as a SLURM job array.
# Each array task runs with different hyperparameters.

# Base configuration
DATA_PATH="/path/to/your/data.npy"
DATA_FORMAT="multi_sim"
BASE_OUTPUT_DIR="./results/sweep_$(date +%Y%m%d)"
# Model architecture variant: ar | ar_encoder_full | ar_encoder_light
MODEL_VARIANT="${MODEL_VARIANT:-ar}"

# Define parameter arrays (customize as needed)
N_PAST_VALUES=(50 100 100 150 100 100)
N_FUTURE_VALUES=(50 50 100 100 150 100)
LEARNING_RATES=(1e-3 1e-3 1e-3 1e-3 1e-3 1e-4)
EPOCHS_VALUES=(1000 1000 1000 1000 1000 2000)

# Get parameters for this array task
TASK_ID=$SLURM_ARRAY_TASK_ID
N_PAST=${N_PAST_VALUES[$TASK_ID]}
N_FUTURE=${N_FUTURE_VALUES[$TASK_ID]}
LEARNING_RATE=${LEARNING_RATES[$TASK_ID]}
EPOCHS=${EPOCHS_VALUES[$TASK_ID]}

# Create unique output directory for this experiment
OUTPUT_DIR="${BASE_OUTPUT_DIR}/exp_${TASK_ID}_past${N_PAST}_future${N_FUTURE}_lr${LEARNING_RATE}"
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# ============================================================
# ENVIRONMENT SETUP
# ============================================================
echo "============================================"
echo "Job Array Task: $TASK_ID"
echo "Started at: $(date)"
echo "Node: $(hostname)"
echo "============================================"
echo ""
echo "Parameters:"
echo "  N_past: $N_PAST"
echo "  N_future: $N_FUTURE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Epochs: $EPOCHS"
echo "  Output: $OUTPUT_DIR"
echo ""

# Load mamba module and activate conda environment
module load mamba/latest
source activate nf_tsf
echo "Conda env: $CONDA_DEFAULT_ENV"

# Help PyTorch manage GPU memory more efficiently
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================
# RUN TRAINING
# ============================================================
python train_model.py \
    --data_path "$DATA_PATH" \
    --data_format "$DATA_FORMAT" \
    --n_past $N_PAST \
    --n_future $N_FUTURE \
    --epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --stride 5 \
    --batch_size 4096 \
    --output_dir "$OUTPUT_DIR" \
    --model_name "model_exp${TASK_ID}" \
    --save_interval 200 \
    --use_scheduler \
    --normalize \
    --device auto \
    --model_variant "$MODEL_VARIANT"

EXIT_STATUS=$?

echo ""
echo "============================================"
echo "Task $TASK_ID finished at: $(date)"
echo "Exit status: $EXIT_STATUS"
echo "============================================"

exit $EXIT_STATUS
