#!/usr/bin/env python
"""
Sanity check for normalized data (Z‑score: subtract mean, divide by std)
Checks that:
  - Training set (first train_val_split steps) has mean ≈ 0 and std ≈ 1.0
  - Test set statistics are finite and not NaN/Inf
  - Metadata keys exist and are consistent
"""

import numpy as np
from pathlib import Path

def check_normalization(dataset_name, data_dir="DATA", tol=1e-2):
    """
    Loads normalized train and test .npz files and performs sanity checks.

    Args:
        dataset_name: name of the dataset (e.g., "double_well")
        data_dir: directory containing the normalized .npz files
        tol: tolerance for mean and std checks (default 0.01)

    Returns:
        bool: True if all checks pass, False otherwise
    """
    train_path = Path(data_dir) / f"{dataset_name}_train.npz"
    test_path  = Path(data_dir) / f"{dataset_name}_test.npz"

    if not train_path.exists() or not test_path.exists():
        print(f"❌ Missing files for {dataset_name}")
        return False

    train_data = np.load(train_path)
    test_data  = np.load(test_path)

    # Extract training portion (first train_val_split steps)
    train_val_pos = train_data["positions"]          # (N, T_total)
    train_val_split = int(train_data["train_val_split"])
    train_pos = train_val_pos[:, :train_val_split]   # (N, train_val_split)

    train_mean = train_pos.mean()
    train_std  = train_pos.std()
    test_pos   = test_data["positions"]
    test_mean  = test_pos.mean()
    test_std   = test_pos.std()

    print(f"\n{dataset_name}")
    print(f"  Training set (first {train_val_split} steps): mean = {train_mean:.6f}, std = {train_std:.6f}")
    print(f"  Test set (full): mean = {test_mean:.6f}, std = {test_std:.6f}")

    # Check training mean and std (should be ≈0 and ≈1)
    ok = True
    if abs(train_mean) < tol:
        print(f"  ✓ Training mean is {train_mean:.6f} ≈ 0")
    else:
        print(f"  ✗ Training mean is {train_mean:.6f}, expected ≈0")
        ok = False

    if abs(train_std - 1.0) < tol:
        print(f"  ✓ Training std is {train_std:.6f} ≈ 1.0")
    else:
        print(f"  ✗ Training std is {train_std:.6f}, expected ≈1.0")
        ok = False

    # Check test set statistics (just finite and non‑zero)
    if np.isfinite(test_std) and test_std > 1e-8:
        print(f"  ✓ Test std is finite and non‑zero ({test_std:.6f})")
    else:
        print(f"  ✗ Test std is invalid ({test_std})")
        ok = False

    if np.isfinite(test_mean):
        print(f"  ✓ Test mean is finite ({test_mean:.6f})")
    else:
        print(f"  ✗ Test mean is not finite")
        ok = False

    # Check metadata keys
    required_keys = ["prediction_length", "context_length", "train_val_split", "time"]
    for key in required_keys:
        if key not in train_data:
            print(f"  ✗ Missing key '{key}' in training file")
            ok = False
        else:
            print(f"  ✓ {key}: {train_data[key]}")

    # Check time array length matches positions columns
    T_total = train_val_pos.shape[1]
    if "time" in train_data and train_data["time"].shape[0] != T_total:
        print(f"  ✗ Time array length ({train_data['time'].shape[0]}) does not match positions columns ({T_total})")
        ok = False
    else:
        print(f"  ✓ Time array matches positions columns ({T_total})")

    # Check for NaN/Inf
    if np.isnan(train_pos).any():
        print(f"  ✗ Training data contains NaN values")
        ok = False
    if np.isinf(train_pos).any():
        print(f"  ✗ Training data contains Inf values")
        ok = False
    if np.isnan(test_pos).any():
        print(f"  ✗ Test data contains NaN values")
        ok = False
    if np.isinf(test_pos).any():
        print(f"  ✗ Test data contains Inf values")
        ok = False
    if ok:
        print(f"  ✓ No NaN or Inf values in training or test data")

    return ok


def main():
    datasets = ["double_well", "single_well", "alanine_phi", "alanine_psi", "linear_gaussian"]
    data_dir = "DATA"   # directory where normalized .npz files are stored

    print("=== Normalization Sanity Check (Z‑score) ===\n")
    print(f"Tolerance for mean and std: {1e-2}")
    print("Checks that the training portion (first train_val_split steps) has mean ≈ 0 and std ≈ 1.0.\n")

    all_passed = True
    for ds in datasets:
        if not check_normalization(ds, data_dir):
            all_passed = False
            print(f"  ❌ {ds} FAILED checks")
        else:
            print(f"  ✅ {ds} passed all checks")

    print("\n" + "="*50)
    if all_passed:
        print("✅ All datasets passed the normalization sanity check.")
    else:
        print("❌ Some datasets failed the sanity check. Please inspect the output above.")
    print("="*50)


if __name__ == "__main__":
    main()