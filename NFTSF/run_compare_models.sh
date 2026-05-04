#!/bin/bash
#SBATCH --job-name=nftsf_compare
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# NOTE: logs/ must exist before sbatch is called, because SLURM opens these
# files before the script body runs. Run: mkdir -p logs && sbatch run_compare_models.sh
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -G a100:1
#SBATCH -p general
#SBATCH -q public
#SBATCH --export=ALL

# ============================================================
# CONFIGURATION - EDIT THESE VARIABLES
# ============================================================

# Data directories: directory containing {landscape}_test.npy
SINGLE_WELL_DATA_DIR="/path/to/data"
DOUBLE_WELL_DATA_DIR="/path/to/data"
ALANINE_PHI_DATA_DIR="/path/to/data"   # contains alanine_phi_test.npy (from splice_alanine.py)
ALANINE_PSI_DATA_DIR="/path/to/data"   # contains alanine_psi_test.npy (from splice_alanine.py)

# Training directories: directory containing model_*.pth, config_*.json, norm_stats_*.npz
SINGLE_WELL_TRAIN_DIR="/path/to/training/single_well"
DOUBLE_WELL_TRAIN_DIR="/path/to/training/double_well"
ALANINE_PHI_TRAIN_DIR="/path/to/training/alanine_phi"
ALANINE_PSI_TRAIN_DIR="/path/to/training/alanine_psi"

# n_past and n_future for single_well and double_well
# (alanine values are read automatically from their config JSON)
SINGLE_WELL_N_PAST=100
SINGLE_WELL_N_FUTURE=100
DOUBLE_WELL_N_PAST=100
DOUBLE_WELL_N_FUTURE=100

OUTPUT_DIR="./results/comparison_$(date +%Y%m%d_%H%M%S)"
N_SAMPLES=500
N_TRAJ_SHOW=3
ARIMA_WORKERS=${SLURM_CPUS_PER_TASK:-$(nproc)}
SEED=42

# ============================================================
# ENVIRONMENT SETUP
# ============================================================
mkdir -p logs   # must also exist before next submission

echo "============================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "============================================"

module load mamba/latest
source activate nf_tsf
echo "Conda env: $CONDA_DEFAULT_ENV"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Python: $(which python)"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo ""

# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# ============================================================
# RUN COMPARISON
# ============================================================
echo "Starting model comparison..."
echo ""

python compare_models.py \
    --nftsf_dirs \
        "single_well:${SINGLE_WELL_DATA_DIR}:${SINGLE_WELL_TRAIN_DIR}:multi_sim:${SINGLE_WELL_N_PAST}:${SINGLE_WELL_N_FUTURE}" \
        "double_well:${DOUBLE_WELL_DATA_DIR}:${DOUBLE_WELL_TRAIN_DIR}:multi_sim:${DOUBLE_WELL_N_PAST}:${DOUBLE_WELL_N_FUTURE}" \
        "alanine_phi:${ALANINE_PHI_DATA_DIR}:${ALANINE_PHI_TRAIN_DIR}:multi_sim" \
        "alanine_psi:${ALANINE_PSI_DATA_DIR}:${ALANINE_PSI_TRAIN_DIR}:multi_sim" \
    --output_dir "$OUTPUT_DIR" \
    --n_samples $N_SAMPLES \
    --n_traj_show $N_TRAJ_SHOW \
    --arima_workers $ARIMA_WORKERS \
    --seed $SEED \
    --device auto

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
