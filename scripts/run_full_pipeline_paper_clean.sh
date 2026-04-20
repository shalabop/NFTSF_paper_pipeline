#!/usr/bin/env bash
# run_full_pipeline_paper_clean.sh
# =================================
# Shell wrapper for run_full_pipeline_paper_clean.py.
# Edit the CONFIG_FILE variable below and run:
#   bash scripts/run_full_pipeline_paper_clean.sh
# or for a dry run:
#   DRY_RUN=1 bash scripts/run_full_pipeline_paper_clean.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# ---- Configuration ----
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/configs/paper_pipeline_clean.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/paper_clean}"
SEED="${SEED:-42}"
N_SAMPLES="${N_SAMPLES:-500}"
DRY_RUN="${DRY_RUN:-0}"

# ---- Environment ----
ENV_NAME="${ENV_NAME:-unified_tsf}"
if command -v conda &>/dev/null; then
    PYTHON_CMD=(conda run -n "$ENV_NAME" python)
else
    PYTHON_CMD=(python)
fi

echo "========================================"
echo "Paper-clean pipeline started: $(date)"
echo "Config  : $CONFIG_FILE"
echo "Output  : $OUTPUT_DIR"
echo "Seed    : $SEED"
echo "Samples : $N_SAMPLES"
echo "Dry run : $DRY_RUN"
echo "========================================"

EXTRA_ARGS=()
if [ "$DRY_RUN" -eq 1 ]; then
    EXTRA_ARGS+=(--dry_run)
fi

"${PYTHON_CMD[@]}" \
    "${REPO_ROOT}/scripts/run_full_pipeline_paper_clean.py" \
    --config_file "$CONFIG_FILE" \
    --output_dir  "$OUTPUT_DIR" \
    --seed        "$SEED" \
    --n_samples   "$N_SAMPLES" \
    "${EXTRA_ARGS[@]}"

echo "========================================"
echo "Paper-clean pipeline finished: $(date)"
echo "========================================"
