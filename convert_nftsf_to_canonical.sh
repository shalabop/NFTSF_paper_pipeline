#!/usr/bin/env bash
# =============================================================================
# convert_nftsf_to_canonical.sh
# =============================================================================
# Convert the NFTSF_ssh Alanine Dipeptide .npy files to the canonical .npz
# format expected by NFTSF_paper_pipeline.
#
# Usage (from the NFTSF_paper_pipeline directory):
#   bash convert_nftsf_to_canonical.sh
#
# What it does:
#   1. Calls data/canonical.py --source nftsf to merge the train + test .npy
#      files, record the original 80/20 split as index arrays, compute Z-score
#      normalisation statistics from the train trajectories, and write the
#      result to outputs/canonical/alanine_phi.npz.
#   2. Prints the norm_mean and norm_std stored in the canonical .npz so you
#      can compare them against the norm_stats_*.npz produced by NFTSF_ssh
#      (they should agree to < 1e-4 due to PyTorch Bessel vs. NumPy population
#      std; if they differ by more, retrain the NFTSF model through this
#      pipeline rather than reusing the existing checkpoint).
#   3. Prints a shape summary of the canonical file for a quick sanity check.
# =============================================================================

set -euo pipefail

# ── User-configurable paths ──────────────────────────────────────────────────
# Paths to the NFTSF_ssh multi_sim .npy files.
TRAIN_NPY="${TRAIN_NPY:-/home/user/NFTSF_ssh/alanine_phi_train.npy}"
TEST_NPY="${TEST_NPY:-/home/user/NFTSF_ssh/alanine_phi_test.npy}"

# Output canonical .npz
OUT_NPZ="${OUT_NPZ:-outputs/canonical/alanine_phi.npz}"

# Landscape identifier (must match what you pass to train/train.py later)
LANDSCAPE="alanine_phi"

# Prediction horizon metadata (stored in .npz; must match base.yaml)
# MD config: lookback=50, horizon=50
N_PAST=50
N_FUTURE=50

# Seed (must match base.yaml: data.seed)
SEED=42
# ── End of configuration ─────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo " NFTSF → Canonical .npz Conversion"
echo "============================================================"
echo "  Train .npy : $TRAIN_NPY"
echo "  Test  .npy : $TEST_NPY"
echo "  Output     : $OUT_NPZ"
echo "  Landscape  : $LANDSCAPE"
echo "  n_past     : $N_PAST   n_future: $N_FUTURE   seed: $SEED"
echo "------------------------------------------------------------"

# Verify source files exist
for F in "$TRAIN_NPY" "$TEST_NPY"; do
    if [[ ! -f "$F" ]]; then
        echo "[ERROR] File not found: $F"
        exit 1
    fi
done

# Run conversion
conda run -n unified_tsf python data/canonical.py \
    --source    nftsf \
    --train     "$TRAIN_NPY" \
    --test      "$TEST_NPY" \
    --landscape "$LANDSCAPE" \
    --n_past    "$N_PAST" \
    --n_future  "$N_FUTURE" \
    --seed      "$SEED" \
    --out       "$OUT_NPZ"

echo ""
echo "------------------------------------------------------------"
echo " Canonical file inspection"
echo "------------------------------------------------------------"

# Print norm stats + shape summary so the user can compare against
# the norm_stats_*.npz saved by NFTSF_ssh/train_model.py.
conda run -n unified_tsf python - <<'PYEOF'
import sys, numpy as np
from pathlib import Path

npz_path = Path("outputs/canonical/alanine_phi.npz")
if not npz_path.exists():
    sys.exit(f"[ERROR] Expected output not found: {npz_path}")

d = np.load(npz_path, allow_pickle=True)

print(f"  File           : {npz_path}")
print(f"  Keys           : {list(d.files)}")
print(f"  positions      : shape {d['positions'].shape}  dtype {d['positions'].dtype}")
print(f"  positions_raw  : shape {d['positions_raw'].shape}  dtype {d['positions_raw'].dtype}")
print(f"  train_indices  : {len(d['train_indices'])} trajectories")
print(f"  test_indices   : {len(d['test_indices'])} trajectories")
print(f"  landscape      : {d['landscape'].item() if d['landscape'].ndim == 0 else d['landscape']}")
print(f"  norm_mean      : {float(d['norm_mean']):.8f}")
print(f"  norm_std       : {float(d['norm_std']):.8f}")
print()
print("  ► Compare the norm_mean / norm_std above against the values in")
print("    NFTSF_ssh/norm_stats_*.npz (keys: 'mean', 'std').")
print("    Difference < 1e-4 → the existing NFTSF_ssh checkpoint is")
print("    directly reusable.  Larger difference → retrain NFTSF through")
print("    this pipeline (train_sol.sh --model nftsf).")
PYEOF

echo ""
echo "============================================================"
echo " Conversion complete."
echo "============================================================"
