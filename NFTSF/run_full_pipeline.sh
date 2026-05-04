#!/bin/bash
#SBATCH --job-name=nftsf_pipeline
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -G a30:1
#SBATCH -p public
#SBATCH -q public
# FIX: was --export=NONE which silently dropped every env var set at sbatch call-time
# (e.g. "STRIDE=50 sbatch run_full_pipeline.sh" had zero effect because SLURM never
# forwarded STRIDE into the job).  --export=ALL passes the full submission environment.
#SBATCH --export=ALL
#
# Full NF-TSF pipeline: generate data -> train models -> test models
# for multiple landscapes including single-well and double-well SDE models.
#
# Usage:
#   sbatch run_full_pipeline.sh
#   or locally:
#   bash run_full_pipeline.sh

# ============================================================
# CONFIGURATION
# ============================================================
# Use SLURM_SUBMIT_DIR when running under sbatch (SLURM copies the script to
# a spool directory, so BASH_SOURCE would resolve there instead of the project).
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
BASE_OUTPUT="${PROJECT_DIR}/pipeline_results"
DATA_DIR="${BASE_OUTPUT}/data"

LANDSCAPES=("linear_gaussian" "single_well" "double_well" "alanine_phi" "sw_sle_em" "sw_gle_oe_em" "dw_sle_em" "dw_gle_oe_em")
# Restrict to a subset of landscapes (space-separated list).
# Example: RUN_LANDSCAPES="alanine_phi" sbatch run_full_pipeline.sh
# Default (unset): run all landscapes.
if [ -n "${RUN_LANDSCAPES:-}" ]; then
    read -ra LANDSCAPES <<< "$RUN_LANDSCAPES"
fi
NUM_TRAIN="${NUM_TRAIN:-9000}"
NUM_TEST="${NUM_TEST:-3000}"
DATA_FORMAT="multi_sim"

# Training hyperparameters (overridable via environment variables for local testing)
N_PAST="${N_PAST:-100}"
N_FUTURE="${N_FUTURE:-100}"
EPOCHS="${EPOCHS:-1000}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
STRIDE="${STRIDE:-5}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
FLOW_BLOCKS="${FLOW_BLOCKS:-6}"
HIDDEN_UNITS="${HIDDEN_UNITS:-64}"
HIDDEN_LAYERS="${HIDDEN_LAYERS:-1,2}"
TAIL_BOUND="${TAIL_BOUND:-30}"

# Additional training hyperparameters — also overridable via environment variables.
# Previously these were never read from the environment; they always fell through to
# argparse defaults inside train_model.py (which matched, but prevented override).
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
SCHEDULER_PATIENCE="${SCHEDULER_PATIENCE:-15}"
SCHEDULER_FACTOR="${SCHEDULER_FACTOR:-0.5}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-50}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
SEED="${SEED:-42}"

# Model architecture variant: ar | ar_encoder_full | ar_encoder_light
# Override via: MODEL_VARIANT=ar_encoder_full sbatch run_full_pipeline.sh
MODEL_VARIANT="${MODEL_VARIANT:-ar}"

# Build a run tag from key hyperparameters so each run gets its own folder.
# Model variant is included so ar / ar_encoder_full / ar_encoder_light results
# are stored under separate directories and never overwrite each other.
RUN_TAG="${MODEL_VARIANT}_K${FLOW_BLOCKS}_H${HIDDEN_UNITS}_tb${TAIL_BOUND}_lr${LEARNING_RATE}_ep${EPOCHS}_np${N_PAST}_nf${N_FUTURE}"
TRAIN_DIR="${BASE_OUTPUT}/training/${RUN_TAG}"
TEST_DIR="${BASE_OUTPUT}/testing/${RUN_TAG}"

# Base seed for data generation.
# Each landscape gets its own seed derived from this:
#   train seed = DATA_SEED + landscape_index   (0, 1, 2)
#   test  seed = DATA_SEED + 1000 + landscape_index   (1000, 1001, 1002)
# Override via: DATA_SEED=99 sbatch run_full_pipeline.sh
DATA_SEED="${DATA_SEED:-0}"

# Path to the raw alanine-dipeptide source file (only needed when alanine_phi is
# in LANDSCAPES). Defaults to the data directory so you can simply drop the .npz
# there; override via: ALANINE_NPZ=/path/to/file.npz sbatch run_full_pipeline.sh
ALANINE_NPZ="${ALANINE_NPZ:-${DATA_DIR}/alanine-dipeptide-3x250ns-backbone-dihedrals.npz}"

# Directory containing pre-generated SDE trajectory .npy files
# (sw_sle_em_1_10_x.npy, sw_sle_em_1_10_v.npy, dw_sle_em_1_10_x.npy, etc.).
# These are produced offline by:  python sde/sde_data_gen.py <model_key> 1 3000 10
# Override via: SDE_DATA_DIR=/path/to/sde/files sbatch run_full_pipeline.sh
SDE_DATA_DIR="${SDE_DATA_DIR:-${PROJECT_DIR}/../Data/Trajectories}"
SDE_TEST_ID="${SDE_TEST_ID:-1}"
SDE_SKIP="${SDE_SKIP:-10}"
# Maximum number of SDE windows to keep before splitting (empty = use full dataset).
# Example: MAX_WINDOWS=1000 sbatch run_full_pipeline.sh
MAX_WINDOWS="${MAX_WINDOWS:-}"
# Maximum number of timesteps to keep per trajectory before splitting (empty = full length).
# Example: MAX_TIMESTEPS=1000 sbatch run_full_pipeline.sh
MAX_TIMESTEPS="${MAX_TIMESTEPS:-}"

# ---------------------------------------------------------------------------
# Preprocessed-data mode
# ---------------------------------------------------------------------------
# Set USE_PREPROCESSED_DATA=1 to skip raw SDE file checks and sde_preprocess.py
# for SDE landscapes.  Instead, the pipeline copies already-split files from
# PREPROCESSED_DATA_DIR into DATA_DIR under the names that Phases 2 and 3
# expect (${landscape}_x_train.npy / ${landscape}_x_test.npy).
#
# Required files in PREPROCESSED_DATA_DIR for each SDE landscape:
#   ${landscape}_train.npy   (N_windows, T) position array — same shape as
#   ${landscape}_test.npy    sde_preprocess.py sde_presplit output
#
# If USE_PREPROCESSED_DATA=1 and either file is missing the pipeline aborts
# with an explicit error; it never falls back to raw SDE generation silently.
#
# Example:
#   USE_PREPROCESSED_DATA=1 \
#   PREPROCESSED_DATA_DIR="/path/to/split_data_no_val" \
#   sbatch run_full_pipeline.sh
USE_PREPROCESSED_DATA="${USE_PREPROCESSED_DATA:-}"
PREPROCESSED_DATA_DIR="${PREPROCESSED_DATA_DIR:-${PROJECT_DIR}/split_data_no_val}"

# Testing parameters
N_SAMPLES="${N_SAMPLES:-500}"

# Device for data generation (cpu avoids GPU memory issues during generation)
GEN_DEVICE="cpu"

# Resume configuration (set via environment variables or edit here)
# Example: RESUME_LANDSCAPE=linear_gaussian RESUME_CHECKPOINT=path/to/checkpoint.pth sbatch run_full_pipeline.sh
RESUME_LANDSCAPE="${RESUME_LANDSCAPE:-}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
SKIP_EXISTING_DATA="${SKIP_EXISTING_DATA:-1}"

# ============================================================
# HELPER FUNCTIONS
# ============================================================

# Check if a landscape is an SDE dataset (single-well or double-well)
is_sde_landscape() {
    local landscape="$1"
    case "$landscape" in
        sw_sle_em|sw_gle_oe_em|dw_sle_em|dw_gle_oe_em)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

# ============================================================
# ENVIRONMENT SETUP
# ============================================================
echo "============================================"
echo "NF-TSF Full Pipeline"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Project dir: $PROJECT_DIR"
echo "============================================"
echo ""
echo "--- Resolved hyperparameters (override via env vars before sbatch) ---"
echo "  Model variant:       MODEL_VARIANT=${MODEL_VARIANT}"
echo "  Data:                NUM_TRAIN=${NUM_TRAIN}  NUM_TEST=${NUM_TEST}"
echo "  Architecture:        FLOW_BLOCKS=${FLOW_BLOCKS}  HIDDEN_UNITS=${HIDDEN_UNITS}  HIDDEN_LAYERS=${HIDDEN_LAYERS}  TAIL_BOUND=${TAIL_BOUND}"
echo "  Context/Horizon:     N_PAST=${N_PAST}  N_FUTURE=${N_FUTURE}"
echo "  Optimisation:        EPOCHS=${EPOCHS}  LEARNING_RATE=${LEARNING_RATE}  BATCH_SIZE=${BATCH_SIZE}"
echo "                       WEIGHT_DECAY=${WEIGHT_DECAY}  GRAD_CLIP=${GRAD_CLIP}"
echo "  LR Scheduler:        SCHEDULER_PATIENCE=${SCHEDULER_PATIENCE}  SCHEDULER_FACTOR=${SCHEDULER_FACTOR}"
echo "  Regularisation:      EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE}  VAL_FRACTION=${VAL_FRACTION}"
echo "  Segmentation:        STRIDE=${STRIDE}"
echo "  Reproducibility:     SEED=${SEED}  DATA_SEED=${DATA_SEED}"
echo "  Landscapes:          ${LANDSCAPES[*]}"
echo "  Alanine source:      ALANINE_NPZ=${ALANINE_NPZ}"
echo "  SDE data dir:        SDE_DATA_DIR=${SDE_DATA_DIR}"
echo "  SDE params:          SDE_TEST_ID=${SDE_TEST_ID}  SDE_SKIP=${SDE_SKIP}  MAX_WINDOWS=${MAX_WINDOWS:-<all windows>}  MAX_TIMESTEPS=${MAX_TIMESTEPS:-<full length>}"
echo "  Preprocessed mode:   USE_PREPROCESSED_DATA=${USE_PREPROCESSED_DATA:-0 (disabled)}  PREPROCESSED_DATA_DIR=${PREPROCESSED_DATA_DIR}"
echo "  Testing:             N_SAMPLES=${N_SAMPLES}"
echo "  Run tag:             ${RUN_TAG}"
echo "--------------------------------------------------------------------"
echo ""

mkdir -p logs

# Load mamba module and activate conda environment (only under SLURM)
if [ -n "$SLURM_JOB_ID" ]; then
    module load mamba/latest
    source activate nf_tsf
fi
echo "Conda env: ${CONDA_DEFAULT_ENV:-not set (local run)}"

# Help PyTorch manage GPU memory more efficiently
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Python: $(which python)"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
echo ""

cd "$PROJECT_DIR"

# Create directory structure
mkdir -p "$DATA_DIR"
for landscape in "${LANDSCAPES[@]}"; do
    mkdir -p "${TRAIN_DIR}/${landscape}"
    mkdir -p "${TEST_DIR}/${landscape}"
done

# ============================================================
# PHASE 1: DATA GENERATION
# ============================================================
echo ""
echo "======== PHASE 1: DATA GENERATION ========"
echo ""

LANDSCAPE_IDX=0
for landscape in "${LANDSCAPES[@]}"; do
    TRAIN_DATA="${DATA_DIR}/${landscape}_train.npy"
    TEST_DATA="${DATA_DIR}/${landscape}_test.npy"

    # Each landscape gets a unique seed; train and test are offset by 1000
    # so they never share a seed regardless of DATA_SEED value.
    TRAIN_SEED=$(( DATA_SEED + LANDSCAPE_IDX ))
    TEST_SEED=$(( DATA_SEED + 1000 + LANDSCAPE_IDX ))

    # SDE landscapes: either load from already-preprocessed files (USE_PREPROCESSED_DATA=1)
    # or verify raw trajectory files exist and run sde_preprocess.py to split them.
    # Either path produces ${landscape}_x_train.npy and ${landscape}_x_test.npy in DATA_DIR
    # for Phases 2 and 3 to consume via --data_format sde_presplit.
    if is_sde_landscape "$landscape"; then
        SDE_X_TRAIN="${DATA_DIR}/${landscape}_x_train.npy"
        SDE_X_TEST="${DATA_DIR}/${landscape}_x_test.npy"

        if [ "${USE_PREPROCESSED_DATA:-}" = "1" ]; then
            # ------------------------------------------------------------------
            # Preprocessed-data path: use already-split files from PREPROCESSED_DATA_DIR.
            # Expected filenames: ${landscape}_train.npy / ${landscape}_test.npy
            # ------------------------------------------------------------------
            PRE_TRAIN="${PREPROCESSED_DATA_DIR}/${landscape}_train.npy"
            PRE_TEST="${PREPROCESSED_DATA_DIR}/${landscape}_test.npy"
            echo "--- SDE landscape ${landscape}: preprocessed-data mode (USE_PREPROCESSED_DATA=1) ---"
            echo "  Looking for:"
            echo "    ${PRE_TRAIN}"
            echo "    ${PRE_TEST}"
            MISSING=0
            [ ! -f "$PRE_TRAIN" ] && MISSING=1
            [ ! -f "$PRE_TEST"  ] && MISSING=1
            if [ "$MISSING" = "1" ]; then
                echo "FATAL: USE_PREPROCESSED_DATA=1 but could not find:"
                [ ! -f "$PRE_TRAIN" ] && echo "  ${PRE_TRAIN}"
                [ ! -f "$PRE_TEST"  ] && echo "  ${PRE_TEST}"
                echo "  Ensure PREPROCESSED_DATA_DIR points to a directory containing"
                echo "  ${landscape}_train.npy and ${landscape}_test.npy."
                echo "  Do NOT set USE_PREPROCESSED_DATA=1 to fall back to raw SDE generation."
                exit 1
            fi
            if [ "$SKIP_EXISTING_DATA" = "1" ] && [ -f "$SDE_X_TRAIN" ] && [ -f "$SDE_X_TEST" ]; then
                echo "--- Skipping copy for ${landscape}: target files already exist ---"
                echo "  Train: $SDE_X_TRAIN"
                echo "  Test:  $SDE_X_TEST"
            else
                echo "  Copying to data dir..."
                cp "$PRE_TRAIN" "$SDE_X_TRAIN"
                cp "$PRE_TEST"  "$SDE_X_TEST"
                echo "  Train: $SDE_X_TRAIN"
                echo "  Test:  $SDE_X_TEST"
            fi
        else
            # ------------------------------------------------------------------
            # Raw-SDE path: verify offline-generated files and run sde_preprocess.py.
            # ------------------------------------------------------------------
            SDE_X="${SDE_DATA_DIR}/${landscape}_${SDE_TEST_ID}_${SDE_SKIP}_x.npy"
            SDE_V="${SDE_DATA_DIR}/${landscape}_${SDE_TEST_ID}_${SDE_SKIP}_v.npy"
            echo "--- SDE landscape ${landscape}: checking pre-generated raw files ---"
            if [ ! -f "$SDE_X" ]; then
                echo "FATAL: SDE position file not found for ${landscape}."
                echo "  Expected: ${SDE_X}"
                echo "  Generate it with:"
                echo "    python sde/sde_data_gen.py ${landscape} ${SDE_TEST_ID} 3000 ${SDE_SKIP}"
                echo "  Or set USE_PREPROCESSED_DATA=1 and PREPROCESSED_DATA_DIR if you already"
                echo "  have ${landscape}_train.npy / ${landscape}_test.npy split files."
                exit 1
            fi
            echo "  x file: $SDE_X"
            if [ ! -f "$SDE_V" ]; then
                echo "  [WARNING] v file not found: ${SDE_V}"
                echo "             Pipeline will continue with x-only dataset."
            else
                echo "  v file: $SDE_V"
            fi

            # Preprocess: split (and optionally truncate) the raw trajectories into
            # {landscape}_x_train.npy / _x_val.npy / _x_test.npy in DATA_DIR.
            if [ "$SKIP_EXISTING_DATA" = "1" ] && [ -f "$SDE_X_TRAIN" ] && [ -f "$SDE_X_TEST" ]; then
                echo "--- Skipping SDE preprocessing for ${landscape}: split files already exist ---"
                echo "  Train: $SDE_X_TRAIN"
                echo "  Test:  $SDE_X_TEST"
            else
                MAX_WINDOWS_FLAG=""
                if [ -n "${MAX_WINDOWS}" ]; then
                    MAX_WINDOWS_FLAG="--max_windows ${MAX_WINDOWS}"
                fi
                MAX_TIMESTEPS_FLAG=""
                if [ -n "${MAX_TIMESTEPS}" ]; then
                    MAX_TIMESTEPS_FLAG="--max_timesteps ${MAX_TIMESTEPS}"
                fi
                if [ -n "${MAX_TIMESTEPS}" ] && [ -n "${MAX_WINDOWS}" ]; then
                    echo "--- SDE preprocessing for ${landscape} (max_timesteps=${MAX_TIMESTEPS}, max_windows=${MAX_WINDOWS}) ---"
                elif [ -n "${MAX_TIMESTEPS}" ]; then
                    echo "--- SDE preprocessing for ${landscape} (max_timesteps=${MAX_TIMESTEPS}, all windows) ---"
                elif [ -n "${MAX_WINDOWS}" ]; then
                    echo "--- SDE preprocessing for ${landscape} (full length, max_windows=${MAX_WINDOWS}) ---"
                else
                    echo "--- SDE preprocessing for ${landscape} (full dataset) ---"
                fi
                echo "  Writing split files to: ${DATA_DIR}/"
                python sde_preprocess.py \
                    --model_key "$landscape" \
                    --data_dir "$SDE_DATA_DIR" \
                    --output_dir "$DATA_DIR" \
                    --test_id $SDE_TEST_ID \
                    --skip $SDE_SKIP \
                    --seed $SEED \
                    $MAX_TIMESTEPS_FLAG \
                    $MAX_WINDOWS_FLAG

                if [ $? -ne 0 ]; then
                    echo "FATAL: SDE preprocessing failed for ${landscape}"
                    exit 1
                fi
                echo "  Preprocessing complete: ${SDE_X_TRAIN}"
            fi
        fi

        LANDSCAPE_IDX=$(( LANDSCAPE_IDX + 1 ))
        echo ""
        continue  # skip standard generate_trajectories.py — data is now in DATA_DIR as split files
    fi

    if [ "$SKIP_EXISTING_DATA" = "1" ] && [ -f "$TRAIN_DATA" ] && [ -f "$TEST_DATA" ]; then
        echo "--- Skipping ${landscape}: data files already exist ---"
        echo "  Train: $TRAIN_DATA"
        echo "  Test:  $TEST_DATA"
    elif [ "$landscape" = "alanine_phi" ] || [ "$landscape" = "alanine_psi" ]; then
        # Real MD data: splice pre-recorded trajectories rather than simulate.
        # splice_alanine.py writes both *_train.npy and *_test.npy in one call.
        echo "--- Generating ${landscape} data from source NPZ (splice_alanine.py) ---"
        if [ ! -f "$ALANINE_NPZ" ]; then
            echo "FATAL: alanine source file not found: $ALANINE_NPZ"
            echo "  Place the file there, or set ALANINE_NPZ=/path/to/alanine-dipeptide-3x250ns-backbone-dihedrals.npz"
            exit 1
        fi
        DIHEDRAL="${landscape#alanine_}"   # extracts 'phi' or 'psi' from landscape name
        python splice_alanine.py \
            --input "$ALANINE_NPZ" \
            --output_dir "$DATA_DIR" \
            --dihedral "$DIHEDRAL"

        if [ $? -ne 0 ]; then
            echo "FATAL: Failed to generate ${landscape} data via splice_alanine.py"
            exit 1
        fi
    else
        echo "--- Generating ${landscape} training data (${NUM_TRAIN} trajectories, seed=${TRAIN_SEED}) ---"
        python generate_trajectories.py \
            --landscape "$landscape" \
            --num_sims $NUM_TRAIN \
            --output_path "$TRAIN_DATA" \
            --device "$GEN_DEVICE" \
            --seed $TRAIN_SEED

        if [ $? -ne 0 ]; then
            echo "FATAL: Failed to generate ${landscape} training data"
            exit 1
        fi

        echo "--- Generating ${landscape} test data (${NUM_TEST} trajectories, seed=${TEST_SEED}) ---"
        python generate_trajectories.py \
            --landscape "$landscape" \
            --num_sims $NUM_TEST \
            --output_path "$TEST_DATA" \
            --device "$GEN_DEVICE" \
            --seed $TEST_SEED

        if [ $? -ne 0 ]; then
            echo "FATAL: Failed to generate ${landscape} test data"
            exit 1
        fi
    fi
    echo ""
    LANDSCAPE_IDX=$(( LANDSCAPE_IDX + 1 ))
done

echo "Phase 1 complete. Data files:"
ls -lh "${DATA_DIR}/"
echo ""

# ============================================================
# PHASE 2: TRAINING (one model per landscape)
# ============================================================
echo ""
echo "======== PHASE 2: TRAINING ========"
echo ""

declare -A MODEL_PATHS
declare -A NORM_PATHS
declare -A CONFIG_PATHS

for landscape in "${LANDSCAPES[@]}"; do
    echo "--- Training model on ${landscape} ---"
    OUTDIR="${TRAIN_DIR}/${landscape}"

    mkdir -p "$OUTDIR"

    # Build resume flag if this landscape should resume from checkpoint
    RESUME_FLAG=""
    if [ "$landscape" = "$RESUME_LANDSCAPE" ]; then
        # Auto-detect latest checkpoint if no explicit path given
        if [ -z "$RESUME_CHECKPOINT" ]; then
            LATEST_CKPT=$(ls -t "${OUTDIR}"/checkpoint_epoch_*.pth 2>/dev/null | head -1)
            if [ -n "$LATEST_CKPT" ]; then
                echo "  Auto-detected checkpoint: $LATEST_CKPT"
                RESUME_CHECKPOINT="$LATEST_CKPT"
            fi
        fi
        if [ -n "$RESUME_CHECKPOINT" ]; then
            echo "  Resuming from checkpoint: $RESUME_CHECKPOINT"
            RESUME_FLAG="--resume_checkpoint $RESUME_CHECKPOINT"
        fi
    fi

    # SDE landscapes use pre-split files written by sde_preprocess.py in Phase 1.
    if is_sde_landscape "$landscape"; then
        python train_model.py \
            --data_path "${DATA_DIR}/${landscape}_x_train.npy" \
            --data_format sde_presplit \
            --landscape "$landscape" \
            --n_past $N_PAST \
            --n_future $N_FUTURE \
            --epochs $EPOCHS \
            --learning_rate $LEARNING_RATE \
            --stride $STRIDE \
            --batch_size $BATCH_SIZE \
            --flow_blocks $FLOW_BLOCKS \
            --hidden_units $HIDDEN_UNITS \
            --hidden_layers "$HIDDEN_LAYERS" \
            --tail_bound $TAIL_BOUND \
            --weight_decay $WEIGHT_DECAY \
            --grad_clip $GRAD_CLIP \
            --scheduler_patience $SCHEDULER_PATIENCE \
            --scheduler_factor $SCHEDULER_FACTOR \
            --early_stopping_patience $EARLY_STOPPING_PATIENCE \
            --val_fraction $VAL_FRACTION \
            --seed $SEED \
            --output_dir "$OUTDIR" \
            --model_name "model" \
            --save_interval 100 \
            --use_scheduler \
            --normalize \
            --device auto \
            --model_variant "$MODEL_VARIANT" \
            $RESUME_FLAG
    else
    # FIX: previously only a subset of hyperparameters were forwarded here.
    # weight_decay, grad_clip, scheduler_patience, scheduler_factor,
    # early_stopping_patience, val_fraction, and seed were all missing, so they
    # silently fell back to argparse defaults and could not be overridden by the
    # caller.  Every tunable parameter now has a corresponding --flag.
    python train_model.py \
        --data_path "${DATA_DIR}/${landscape}_train.npy" \
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
        --tail_bound $TAIL_BOUND \
        --weight_decay $WEIGHT_DECAY \
        --grad_clip $GRAD_CLIP \
        --scheduler_patience $SCHEDULER_PATIENCE \
        --scheduler_factor $SCHEDULER_FACTOR \
        --early_stopping_patience $EARLY_STOPPING_PATIENCE \
        --val_fraction $VAL_FRACTION \
        --seed $SEED \
        --output_dir "$OUTDIR" \
        --model_name "model" \
        --save_interval 100 \
        --use_scheduler \
        --normalize \
        --device auto \
        --model_variant "$MODEL_VARIANT" \
        $RESUME_FLAG
    fi

    if [ $? -ne 0 ]; then
        echo "FATAL: Training failed for ${landscape}"
        exit 1
    fi

    # Find the most recently created model, norm_stats, and config files
    MODEL_PATH=$(ls -t "${OUTDIR}"/model_*.pth 2>/dev/null | head -1)
    NORM_PATH=$(ls -t "${OUTDIR}"/norm_stats_*.npz 2>/dev/null | head -1)
    # Config JSON carries the architecture hyperparameters used during training;
    # forwarding it to test_model.py ensures the model is reconstructed correctly.
    CONFIG_PATH=$(ls -t "${OUTDIR}"/config_*.json 2>/dev/null | head -1)

    if [ -z "$MODEL_PATH" ]; then
        echo "FATAL: No model file found in ${OUTDIR}"
        exit 1
    fi

    MODEL_PATHS["$landscape"]="$MODEL_PATH"
    NORM_PATHS["$landscape"]="$NORM_PATH"
    CONFIG_PATHS["$landscape"]="$CONFIG_PATH"

    echo "  Model saved: $MODEL_PATH"
    echo "  Norm stats: $NORM_PATH"
    echo "  Config:     $CONFIG_PATH"
    echo ""
done

echo "Phase 2 complete."
echo ""

# ============================================================
# PHASE 3: TESTING (each model on its own landscape)
# ============================================================
echo ""
echo "======== PHASE 3: TESTING ========"
echo ""

for landscape in "${LANDSCAPES[@]}"; do
    echo "--- Testing model: ${landscape} ---"

    TEST_OUTDIR="${TEST_DIR}/${landscape}"
    mkdir -p "$TEST_OUTDIR"

    # SDE landscapes: use the pre-split test file written by sde_preprocess.py in Phase 1.
    # The same --seed used in preprocessing guarantees this is the held-out 20%.
    if is_sde_landscape "$landscape"; then
        CMD="python test_model.py \
            --model_path \"${MODEL_PATHS[$landscape]}\" \
            --data_path \"${DATA_DIR}/${landscape}_x_test.npy\" \
            --data_format sde_presplit \
            --landscape \"$landscape\" \
            --n_past $N_PAST \
            --n_future $N_FUTURE \
            --n_samples $N_SAMPLES \
            --seed $SEED \
            --output_dir \"$TEST_OUTDIR\" \
            --device auto"
    else
        CMD="python test_model.py \
            --model_path \"${MODEL_PATHS[$landscape]}\" \
            --data_path \"${DATA_DIR}/${landscape}_test.npy\" \
            --data_format $DATA_FORMAT \
            --n_past $N_PAST \
            --n_future $N_FUTURE \
            --n_samples $N_SAMPLES \
            --output_dir \"$TEST_OUTDIR\" \
            --device auto"
    fi

    # Pass the training config so test_model.py reconstructs the correct architecture
    if [ -n "${CONFIG_PATHS[$landscape]}" ]; then
        CMD="$CMD --config_path \"${CONFIG_PATHS[$landscape]}\""
    fi

    # Add norm stats if available
    if [ -n "${NORM_PATHS[$landscape]}" ]; then
        CMD="$CMD --norm_stats_path \"${NORM_PATHS[$landscape]}\""
    fi

    eval $CMD

    if [ $? -ne 0 ]; then
        echo "WARNING: Testing failed for ${landscape}, continuing..."
    fi
    echo ""
done

echo "Phase 3 complete."
echo ""

# ============================================================
# SUMMARY
# ============================================================
echo "============================================"
echo "PIPELINE COMPLETE"
echo "Finished at: $(date)"
echo ""
echo "Run tag: ${RUN_TAG}"
echo "Data:    ${DATA_DIR}/"
echo "Models:  ${TRAIN_DIR}/"
echo "Figures: ${TEST_DIR}/"
echo ""
echo "Output directories:"
for landscape in "${LANDSCAPES[@]}"; do
    echo "  Training: ${TRAIN_DIR}/${landscape}/"
    echo "  Testing:  ${TEST_DIR}/${landscape}/"
done
echo "============================================"
