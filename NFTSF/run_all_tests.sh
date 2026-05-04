#!/bin/bash
# Run test_model.py for all three landscapes using the most recently trained
# model and norm_stats found in each landscape's training output directory.
#
# Usage:
#   bash run_all_tests.sh
#
# Override defaults via environment variables, e.g.:
#   TRAINING_DIR=./my_results/training bash run_all_tests.sh

# ============================================================
# CONFIGURATION — edit these or override via env variables
# ============================================================
TRAINING_DIR="${TRAINING_DIR:-./pipeline_results/training}"
DATA_DIR="${DATA_DIR:-./pipeline_results/data}"
OUTPUT_BASE="${OUTPUT_BASE:-./pipeline_results/testing}"

LANDSCAPES=("linear_gaussian" "single_well" "double_well" "alanine_phi")
# Restrict to a subset of landscapes (space-separated list).
# Example: RUN_LANDSCAPES="alanine_phi" bash run_all_tests.sh
if [ -n "${RUN_LANDSCAPES:-}" ]; then
    read -ra LANDSCAPES <<< "$RUN_LANDSCAPES"
fi

DATA_FORMAT="${DATA_FORMAT:-multi_sim}"
N_PAST="${N_PAST:-100}"
N_FUTURE="${N_FUTURE:-100}"
N_SAMPLES="${N_SAMPLES:-500}"
N_QUART_TEST="${N_QUART_TEST:-500}"
RUN_QUANTILE_TESTS="${RUN_QUANTILE_TESTS:-false}"

# ============================================================
# ENVIRONMENT SETUP  (set ACTIVATE_ENV=false to skip)
# ============================================================
ACTIVATE_ENV="${ACTIVATE_ENV:-true}"

if [ "$ACTIVATE_ENV" = true ]; then
    module load mamba/latest
    source activate nf_tsf
fi

# ============================================================
# RUN TESTS
# ============================================================
echo "============================================"
echo "Run-all-tests started at: $(date)"
echo "Training dir : $TRAINING_DIR"
echo "Data dir     : $DATA_DIR"
echo "Output base  : $OUTPUT_BASE"
echo "============================================"
echo ""

declare -A RESULTS

for landscape in "${LANDSCAPES[@]}"; do
    echo "--------------------------------------------"
    echo "Landscape: $landscape"
    echo "--------------------------------------------"

    # Auto-discover the most recently modified model and norm_stats
    MODEL_PATH=$(ls -t "${TRAINING_DIR}/${landscape}"/model_*.pth 2>/dev/null | head -1)
    NORM_PATH=$(ls -t "${TRAINING_DIR}/${landscape}"/norm_stats_*.npz 2>/dev/null | head -1)
    DATA_PATH="${DATA_DIR}/${landscape}_test.npy"

    if [ -z "$MODEL_PATH" ]; then
        echo "WARNING: no model_*.pth found in '${TRAINING_DIR}/${landscape}' — skipping."
        RESULTS[$landscape]="SKIPPED (no model found)"
        echo ""
        continue
    fi

    if [ ! -f "$DATA_PATH" ]; then
        echo "WARNING: test data not found at '${DATA_PATH}' — skipping."
        RESULTS[$landscape]="SKIPPED (no test data)"
        echo ""
        continue
    fi

    echo "  Model      : $MODEL_PATH"
    echo "  Norm stats : ${NORM_PATH:-<none>}"
    echo "  Test data  : $DATA_PATH"

    OUT_DIR="${OUTPUT_BASE}/${landscape}"
    mkdir -p "$OUT_DIR"

    CMD="python test_model.py \
        --model_path \"$MODEL_PATH\" \
        --data_path \"$DATA_PATH\" \
        --data_format $DATA_FORMAT \
        --n_past $N_PAST \
        --n_future $N_FUTURE \
        --n_samples $N_SAMPLES \
        --n_quart_test $N_QUART_TEST \
        --output_dir \"$OUT_DIR\" \
        --device auto"

    if [ -n "$NORM_PATH" ]; then
        CMD="$CMD --norm_stats_path \"$NORM_PATH\""
    fi

    if [ "$RUN_QUANTILE_TESTS" = true ]; then
        CMD="$CMD --run_quantile_tests"
    fi

    echo ""
    eval $CMD
    EXIT_STATUS=$?

    if [ $EXIT_STATUS -eq 0 ]; then
        RESULTS[$landscape]="OK  → $OUT_DIR"
    else
        RESULTS[$landscape]="FAILED (exit $EXIT_STATUS)"
    fi

    echo ""
done

# ============================================================
# SUMMARY
# ============================================================
echo "============================================"
echo "Summary"
echo "============================================"
for landscape in "${LANDSCAPES[@]}"; do
    printf "  %-20s %s\n" "$landscape" "${RESULTS[$landscape]}"
done
echo ""
echo "Run-all-tests finished at: $(date)"
echo "============================================"
