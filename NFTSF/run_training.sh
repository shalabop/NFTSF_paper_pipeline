#!/bin/bash
#SBATCH --job-name=nf_tsf_train
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
#
# Adjust the above parameters based on your supercomputer's configuration:
#   --partition: may be 'gpu', 'v100', 'a100', etc.
#   --gres: specify GPU type if needed, e.g., gpu:v100:1, gpu:a100:1
#   --time: adjust based on expected training time
#   --mem: adjust based on data size

# ============================================================
# CONFIGURATION - EDIT THESE VARIABLES
# ============================================================
DATA_PATH="/path/to/your/data.npy"
DATA_FORMAT="multi_sim"  # Options: single_sim, multi_sim, tnf
OUTPUT_DIR="./results/training_$(date +%Y%m%d_%H%M%S)"
MODEL_NAME="nf_tsf_model"

N_PAST=100
N_FUTURE=100
EPOCHS=1000
LEARNING_RATE=1e-3
STRIDE=5
BATCH_SIZE=4096
FLOW_BLOCKS=4
HIDDEN_UNITS=32
HIDDEN_LAYERS="1,2"
WEIGHT_DECAY=1e-5
VAL_FRACTION=0.1
EARLY_STOPPING_PATIENCE=50
# Model architecture variant: ar | ar_encoder_full | ar_encoder_light
MODEL_VARIANT="${MODEL_VARIANT:-ar}"

# ============================================================
# ENVIRONMENT SETUP
# ============================================================
echo "============================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "============================================"

# Create logs directory if it doesn't exist
mkdir -p logs

# Load mamba module and activate conda environment
module load mamba/latest
source activate nf_tsf
echo "Conda env: $CONDA_DEFAULT_ENV"

# Help PyTorch manage GPU memory more efficiently
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Print environment info
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Check GPU availability
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo ""

# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# ============================================================
# RUN TRAINING
# ============================================================
echo "Starting training..."
echo "Data: $DATA_PATH"
echo "Format: $DATA_FORMAT"
echo "N_past: $N_PAST, N_future: $N_FUTURE"
echo "Epochs: $EPOCHS"
echo ""

python train_model.py \
    --data_path "$DATA_PATH" \
    --data_format "$DATA_FORMAT" \
    --n_past $N_PAST \
    --n_future $N_FUTURE \
    --epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --stride $STRIDE \
    --batch_size $BATCH_SIZE \
    --flow_blocks $FLOW_BLOCKS \
    --hidden_units $HIDDEN_UNITS \
    --hidden_layers "$HIDDEN_LAYERS" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --save_interval 100 \
    --use_scheduler \
    --normalize \
    --device auto \
    --weight_decay $WEIGHT_DECAY \
    --val_fraction $VAL_FRACTION \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --model_variant "$MODEL_VARIANT"

# Capture exit status
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
