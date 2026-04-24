import argparse
import numpy as np
from pathlib import Path

def sanity_check_raw(datatype, data_dir="DATA", raw_dir="DATA_raw",
                     context_length=None, prediction_length=None):
    """
    Verify that the normalized .npz files are consistent with the original .npy raw data.

    Parameters
    ----------
    datatype : str
        Dataset name (e.g., 'double_well')
    data_dir : str
        Directory containing normalized .npz files (default "DATA")
    raw_dir : str
        Directory containing raw .npy files (train/val/test) (default "DATA_raw")
    context_length : int, optional
        If provided, check that enough history exists before split
    prediction_length : int, optional
        If provided, check that enough forecast steps exist after split
    """
    data_dir = Path(data_dir)
    raw_dir = Path(raw_dir)

    # Expected normalized files (.npz)
    norm_train = data_dir / f"{datatype}_train.npz"
    norm_test = data_dir / f"{datatype}_test.npz"
    norm_combined = data_dir / f"{datatype}.npz"

    # Expected raw files (.npy)
    raw_train = raw_dir / f"{datatype}_train.npy"
    raw_val = raw_dir / f"{datatype}_val.npy"
    raw_test = raw_dir / f"{datatype}_test.npy"

    # Check existence
    for f in [norm_train, norm_test, norm_combined, raw_train, raw_val, raw_test]:
        assert f.exists(), f"Missing file: {f}"
    print("✓ All required files exist")

    # Load raw data (entire trajectories)
    train_raw_all = np.load(raw_train)   # (N_train, T)
    val_raw_all   = np.load(raw_val)     # (N_val, T)
    test_raw_all  = np.load(raw_test)    # (N_test, T)

    # Concatenate train+val to match the combined set used in normalization
    train_val_raw = np.concatenate([train_raw_all, val_raw_all], axis=0)  # (N_train+N_val, T)

    # Load normalized data
    train_norm = np.load(norm_train)
    test_norm = np.load(norm_test)
    combined_norm = np.load(norm_combined)

    # Extract normalized arrays and metadata
    train_pos_norm = train_norm["positions"]          # (N_total, T) where N_total = N_train+N_val
    test_pos_norm = test_norm["positions"]            # (N_test, T)
    train_split = int(train_norm["train_val_split"])  # temporal split index inside train+val
    test_split = int(test_norm["train_test_split"])
    mean = float(train_norm["mean"])
    std = float(train_norm["std"])
    time_train = train_norm["time"]                   # (T,)
    time_test = test_norm["time"]

    print(f"\n=== Data shapes ===")
    print(f"Raw train+val shape: {train_val_raw.shape}")
    print(f"Raw test shape: {test_raw_all.shape}")
    print(f"Normalized train+val shape: {train_pos_norm.shape}")
    print(f"Normalized test shape: {test_pos_norm.shape}")
    print(f"Train split index (forecast start): {train_split}")
    print(f"Test split index: {test_split}")
    print(f"Normalization mean: {mean:.6f}, std: {std:.6f}")
    print(f"Time train length: {len(time_train)}")
    print(f"Time test length: {len(time_test)}")

    # 1. Split validity (temporal splits must be within series length)
    T_train = train_pos_norm.shape[1]
    T_test = test_pos_norm.shape[1]
    assert 0 < train_split < T_train, f"Train split {train_split} out of range [0,{T_train}]"
    assert 0 < test_split < T_test, f"Test split {test_split} out of range [0,{T_test}]"
    print("✓ Split indices within range")

    # 2. Context/prediction length checks (if provided)
    if context_length is not None:
        assert train_split >= context_length, f"Train split {train_split} < context {context_length}"
        assert test_split >= context_length, f"Test split {test_split} < context {context_length}"
        print(f"✓ Context length {context_length} satisfied")
    if prediction_length is not None:
        assert train_split + prediction_length <= T_train, "Train forecast window exceeds series"
        assert test_split + prediction_length <= T_test, "Test forecast window exceeds series"
        print(f"✓ Prediction length {prediction_length} satisfied")

    # 3. Normalization reversibility (sample a few values from normalized train set)
    np.random.seed(42)
    n_check = min(10, train_pos_norm.size)
    idx_traj = np.random.randint(0, train_pos_norm.shape[0], n_check)
    idx_time = np.random.randint(0, train_pos_norm.shape[1], n_check)
    for i, j in zip(idx_traj, idx_time):
        norm_val = train_pos_norm[i, j]
        raw_val = train_val_raw[i, j]   # assumes same ordering
        reconstructed = norm_val * (std + 1e-8) + mean
        assert np.allclose(reconstructed, raw_val, atol=1e-5), \
            f"Norm mismatch at ({i},{j}): {reconstructed:.6f} vs raw {raw_val:.6f}"
    print("✓ Normalization reversible (sample check)")

    # 4. Consistency between combined file and individual normalized files
    assert np.allclose(combined_norm["positions_train"], train_pos_norm), "Combined train mismatch"
    assert np.allclose(combined_norm["positions_test"], test_pos_norm), "Combined test mismatch"
    assert int(combined_norm["train_test_split"]) == test_split, "Split mismatch in combined"
    print("✓ Combined file consistent")

    # 5. Time array length matches series length
    assert len(time_train) == T_train, "Time train length mismatch"
    assert len(time_test) == T_test, "Time test length mismatch"
    # Check monotonic
    assert np.all(np.diff(time_train) > 0), "Time train not monotonic"
    assert np.all(np.diff(time_test) > 0), "Time test not monotonic"
    print("✓ Time arrays correct")

    # 6. Data range sanity (not all zeros, no inf/nan)
    assert not np.allclose(train_pos_norm, 0), "Train positions all zeros"
    assert not np.allclose(test_pos_norm, 0), "Test positions all zeros"
    assert np.isfinite(train_pos_norm).all(), "Train contains NaN/Inf"
    assert np.isfinite(test_pos_norm).all(), "Test contains NaN/Inf"
    print("✓ Data ranges plausible")

    # 7. Verify that the raw test data matches the raw test file (simple check of first few values)
    # This ensures the test split index is within raw test data length.
    # We also check that the number of trajectories in normalized test matches raw test.
    assert test_pos_norm.shape[0] == test_raw_all.shape[0], "Number of test trajectories mismatch"
    # Optionally, check a few values from raw test vs the test_norm after denormalization
    raw_test_sample = test_raw_all[0, :10]
    norm_test_sample = test_pos_norm[0, :10]
    denorm_test = norm_test_sample * (std + 1e-8) + mean
    assert np.allclose(denorm_test, raw_test_sample, atol=1e-5), "Test data mismatch after denormalization"
    print("✓ Test data consistent")

    print("\n✅ All sanity checks passed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity check for normalized data from raw .npy files")
    parser.add_argument("--datatype", "-dt", type=str, required=True, help="Dataset name (e.g., double_well)")
    parser.add_argument("--data_dir", "-d", type=str, default="DATA", help="Directory containing normalized .npz files")
    parser.add_argument("--raw_dir", "-r", type=str, default="DATA_raw", help="Directory containing raw .npy files")
    parser.add_argument("--context_length", "-cl", type=int, default=None, help="Expected context length (optional)")
    parser.add_argument("--prediction_length", "-pl", type=int, default=None, help="Expected prediction length (optional)")
    args = parser.parse_args()

    sanity_check_raw(datatype=args.datatype, data_dir=args.data_dir, raw_dir=args.raw_dir,
                     context_length=args.context_length, prediction_length=args.prediction_length)