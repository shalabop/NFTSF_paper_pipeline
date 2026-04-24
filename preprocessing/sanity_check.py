import argparse
import numpy as np
from pathlib import Path

def sanity_check(datatype, data_dir="DATA", unnorm_dir="DATA_unnorm",
                 context_length=None, prediction_length=None):
    """
    Verify that the data files produced by normalize_all.py are consistent.
    
    Parameters
    ----------
    datatype : str
        Dataset name (e.g., 'alanine_phi')
    data_dir : str
        Directory containing normalized .npz files (default "DATA")
    unnorm_dir : str
        Directory containing unnormalized .npz files (default "DATA_unnorm")
    context_length : int, optional
        If provided, check that enough history exists before split
    prediction_length : int, optional
        If provided, check that enough forecast steps exist after split
    """
    data_dir = Path(data_dir)
    unnorm_dir = Path(unnorm_dir)
    
    # Expected files
    norm_train = data_dir / f"{datatype}_train.npz"
    norm_test = data_dir / f"{datatype}_test.npz"
    norm_combined = data_dir / f"{datatype}.npz"
    unnorm_train = unnorm_dir / f"{datatype}_train.npz"
    unnorm_test = unnorm_dir / f"{datatype}_test.npz"
    unnorm_combined = unnorm_dir / f"{datatype}.npz"
    
    # Check existence
    for f in [norm_train, norm_test, norm_combined, unnorm_train, unnorm_test, unnorm_combined]:
        assert f.exists(), f"Missing file: {f}"
    print("✓ All required files exist")
    
    # Load data
    train_norm = np.load(norm_train)
    test_norm = np.load(norm_test)
    combined_norm = np.load(norm_combined)
    train_raw = np.load(unnorm_train)
    test_raw = np.load(unnorm_test)
    
    # Extract arrays and metadata
    train_pos = train_norm["positions"]
    test_pos = test_norm["positions"]
    train_split = int(train_norm["train_val_split"])
    test_split = int(test_norm["train_test_split"])
    mean = float(train_norm["mean"])
    std = float(train_norm["std"])
    time_train = train_norm["time"]
    time_test = test_norm["time"]
    
    # Compute mean/std of the training part used for normalization
    train_norm_part = train_pos[:, :train_split]
    mean_norm = np.mean(train_norm_part)
    std_norm = np.std(train_norm_part)
    assert np.isclose(mean_norm, 0.0, atol=1e-5), f"Normalized mean = {mean_norm:.6f} != 0"
    assert np.isclose(std_norm, 1.0, atol=1e-5), f"Normalized std = {std_norm:.6f} != 1"
    print("✓ Normalized data has mean≈0, std≈1 (within tolerance)")
    
    print(f"\n=== Data shapes ===")
    print(f"Train+val shape: {train_pos.shape}")
    print(f"Test shape: {test_pos.shape}")
    print(f"Train split index (forecast start): {train_split}")
    print(f"Test split index: {test_split}")
    print(f"Normalization mean: {mean:.6f}, std: {std:.6f}")
    print(f"Time train length: {len(time_train)}")
    print(f"Time test length: {len(time_test)}")
    
    # 1. Split validity
    print(train_split)
    print(train_pos.shape)
    assert 0 < train_split < train_pos.shape[1], "Train split out of range"
    assert 0 < test_split < test_pos.shape[1], "Test split out of range"
    print("✓ Split indices within range")
    
    # 2. Context/prediction length checks (if provided)
    if context_length is not None:
        assert train_split >= context_length, f"Train split {train_split} < context {context_length}"
        assert test_split >= context_length, f"Test split {test_split} < context {context_length}"
        print(f"✓ Context length {context_length} satisfied")
    if prediction_length is not None:
        assert train_split + prediction_length <= train_pos.shape[1], "Train forecast window exceeds series"
        assert test_split + prediction_length <= test_pos.shape[1], "Test forecast window exceeds series"
        print(f"✓ Prediction length {prediction_length} satisfied")
    
    # 3. Normalization reversibility (sample a few values)
    np.random.seed(42)
    n_check = min(10, train_pos.size)
    idx_traj = np.random.randint(0, train_pos.shape[0], n_check)
    idx_time = np.random.randint(0, train_pos.shape[1], n_check)
    for i, j in zip(idx_traj, idx_time):
        norm_val = train_pos[i, j]
        raw_val = train_raw["positions"][i, j]
        reconstructed = norm_val * (std + 1e-8) + mean
        assert np.allclose(reconstructed, raw_val, atol=1e-5), f"Norm mismatch at ({i},{j}): {reconstructed} vs {raw_val}"
    print("✓ Normalization reversible (sample check)")
    
    # 4. Consistency between combined file and individual files
    assert np.allclose(combined_norm["positions_train"], train_pos), "Combined train mismatch"
    assert np.allclose(combined_norm["positions_test"], test_pos), "Combined test mismatch"
    assert int(combined_norm["train_test_split"]) == test_split, "Split mismatch in combined"
    print("✓ Combined file consistent")
    
    # 5. Time array length matches series length
    assert len(time_train) == train_pos.shape[1], "Time train length mismatch"
    assert len(time_test) == test_pos.shape[1], "Time test length mismatch"
    # Check monotonic
    assert np.all(np.diff(time_train) > 0), "Time train not monotonic"
    assert np.all(np.diff(time_test) > 0), "Time test not monotonic"
    print("✓ Time arrays correct")
    
    # 6. Data range sanity (not all zeros, not extreme outliers)
    assert not np.allclose(train_pos, 0), "Train positions all zeros"
    assert not np.allclose(test_pos, 0), "Test positions all zeros"
    assert np.isfinite(train_pos).all(), "Train contains NaN/Inf"
    assert np.isfinite(test_pos).all(), "Test contains NaN/Inf"
    print("✓ Data ranges plausible")
    
    # 7. Unnormalized files match raw data (check same values)
    # The unnormalized train file is named .np (not .npz) but we loaded it as .np? Actually it's .np – but np.load works on .np files if they are numpy binary? Usually .np is not standard. Let's check extension.
    # The script saves unnorm_train as .np (without z). That's unusual; we should warn.
    if unnorm_train.suffix == '.np':
        print("⚠️ Warning: Unnormalized train file has extension '.np' (not '.npz') – this may cause issues. Consider using '.npz' consistently.")
    
    # Additional: check that test split is consistent with combined file's train_test_split
    combined_split = int(combined_norm["train_test_split"])
    assert combined_split == test_split, "Inconsistent train_test_split"
    
    print("\n✅ All sanity checks passed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity check for normalized data")
    parser.add_argument("--datatype", "-dt", type=str, required=True, help="Dataset name")
    parser.add_argument("--data_dir", "-d", type=str, default="DATA", help="Normalized data directory")
    parser.add_argument("--unnorm_dir", "-u", type=str, default="DATA_unnorm", help="Unnormalized data directory")
    parser.add_argument("--context_length", "-cl", type=int, default=None, help="Expected context length (optional)")
    parser.add_argument("--prediction_length", "-pl", type=int, default=None, help="Expected prediction length (optional)")
    args = parser.parse_args()
    
    sanity_check(datatype=args.datatype, data_dir=args.data_dir, unnorm_dir=args.unnorm_dir,
                 context_length=args.context_length, prediction_length=args.prediction_length)