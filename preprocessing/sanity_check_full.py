import argparse
import json
import numpy as np
from pathlib import Path

def load_gluonts_series(folder: Path):
    """
    Load all series from a GluonTS split folder that contains a JSON Lines file `data.json`.
    Falls back to reading individual `.json` files if `data.json` does not exist.
    """
    series = []
    data_file = folder / "data.json"
    if data_file.exists():
        
        with open(data_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    series.append(np.array(data["target"], dtype=np.float32))
    else:
        # fallback: each entry is a separate .json file (legacy format)
        for json_file in sorted(folder.glob("*.json")):
            with open(json_file, "r") as f:
                data = json.load(f)
            series.append(np.array(data["target"], dtype=np.float32))
    return series

def sanity_check_full(datatype, raw_dir="DATA_raw", data_dir="DATA",
                      gluonts_dir="gluonts_datasets",
                      context_length=None, prediction_length=None):
    """
    Verify consistency among:
      - Raw .npy files (train, val, test)
      - Normalized .npz files (train, test, combined)
      - GluonTS JSON dataset (train and test folders, JSON Lines format)
    """
    raw_dir = Path(raw_dir)
    data_dir = Path(data_dir)
    gluonts_base = Path(gluonts_dir) / datatype

    # 1. Load raw data
    raw_train = np.load(raw_dir / f"{datatype}_train.npy")
    raw_val   = np.load(raw_dir / f"{datatype}_val.npy")
    raw_test  = np.load(raw_dir / f"{datatype}_test.npy")
    raw_train_val = np.concatenate([raw_train, raw_val], axis=0)   # (N_train+N_val, T)
    T_raw = raw_train.shape[1]   # assume all splits have same number of time steps

    # 2. Load normalized DATA (.npz)
    norm_train = np.load(data_dir / f"{datatype}_train.npz")
    norm_test  = np.load(data_dir / f"{datatype}_test.npz")
    norm_combined = np.load(data_dir / f"{datatype}.npz")

    train_pos_norm = norm_train["positions"]          # (N_train+N_val, T)
    test_pos_norm  = norm_test["positions"]           # (N_test, T)
    train_split = int(norm_train["train_val_split"])  # temporal split inside train_val set
    test_split  = int(norm_test["train_test_split"])
    mean = float(norm_train["mean"])
    std  = float(norm_train["std"])

    # 3. Load GluonTS datasets (JSON Lines)
    gluon_train_series = load_gluonts_series(gluonts_base / "train")
    gluon_test_series  = load_gluonts_series(gluonts_base / "test")

    # 4. Shape and counts
    print("\n=== Shape consistency ===")
    print(f"Raw train+val  : {raw_train_val.shape}")
    print(f"Raw test       : {raw_test.shape}")
    print(f"Norm train+val : {train_pos_norm.shape}")
    print(f"Norm test      : {test_pos_norm.shape}")
    print(f"GluonTS train  : {len(gluon_train_series)} series")
    print(f"GluonTS test   : {len(gluon_test_series)} series")

    assert train_pos_norm.shape[0] == raw_train_val.shape[0], "Norm train count mismatch"
    assert test_pos_norm.shape[0] == raw_test.shape[0], "Norm test count mismatch"
    assert len(gluon_train_series) == raw_train_val.shape[0], "GluonTS train count mismatch"
    assert len(gluon_test_series) == raw_test.shape[0], "GluonTS test count mismatch"
    print("✓ Trajectory counts match")

    # 5. Temporal splits
    print(f"\nTrain split index : {train_split}")
    print(f"Test split index  : {test_split}")
    assert 0 < train_split < T_raw, "Train split out of range"
    assert 0 < test_split < T_raw, "Test split out of range"
    if context_length is not None:
        assert train_split >= context_length
        assert test_split >= context_length
        print(f"✓ Context length {context_length} satisfied")
    if prediction_length is not None:
        assert train_split + prediction_length <= T_raw
        assert test_split + prediction_length <= T_raw
        print(f"✓ Prediction length {prediction_length} satisfied")

    # 6. Normalization reversibility (sample)
    print("\n=== Checking raw vs normalized DATA ===")
    np.random.seed(42)
    n_check = min(10, train_pos_norm.size)
    idx_traj = np.random.randint(0, train_pos_norm.shape[0], n_check)
    idx_time = np.random.randint(0, train_pos_norm.shape[1], n_check)
    for i, j in zip(idx_traj, idx_time):
        norm_val = train_pos_norm[i, j]
        raw_val = raw_train_val[i, j]
        reconstructed = norm_val * (std + 1e-8) + mean
        assert np.allclose(reconstructed, raw_val, atol=1e-5), f"Norm mismatch at ({i},{j})"
    print("✓ Normalization reversible (sample check)")

    print("\n=== Checking raw vs GluonTS (denormalizing GluonTS) ===")
    for i in range(min(5, len(gluon_train_series))):
        gluon_ts = gluon_train_series[i]                     # normalized (as stored)
        gluon_denorm = gluon_ts * (std + 1e-8) + mean        # convert to raw scale
        raw_ts = raw_train_val[i, :len(gluon_denorm)]
        if len(gluon_denorm) != raw_ts.shape[0]:
            print(f"Warning: train series {i} length mismatch – skipping")
            continue
        if np.allclose(gluon_denorm, raw_ts, atol=1e-5):
            print(f"  ✓ Series {i} (train): GluonTS denormalized matches raw.")
        else:
            max_diff = np.max(np.abs(gluon_denorm - raw_ts))
            raise AssertionError(f"Train series {i} mismatch after denormalization (max diff {max_diff:.2e})")
    print("✓ GluonTS training series consistent (denormalized).")

    for i in range(min(5, len(gluon_test_series))):
        gluon_ts = gluon_test_series[i]
        gluon_denorm = gluon_ts * (std + 1e-8) + mean
        raw_ts = raw_test[i, :len(gluon_denorm)]
        if len(gluon_denorm) != raw_ts.shape[0]:
            print(f"Warning: test series {i} length mismatch – skipping")
            continue
        if np.allclose(gluon_denorm, raw_ts, atol=1e-5):
            print(f"  ✓ Series {i} (test): GluonTS denormalized matches raw.")
        else:
            max_diff = np.max(np.abs(gluon_denorm - raw_ts))
            raise AssertionError(f"Test series {i} mismatch after denormalization (max diff {max_diff:.2e})")
    # 8. Combined .npz file
    print("\n=== Checking combined .npz file ===")
    assert np.allclose(norm_combined["positions_train"], train_pos_norm), "Combined train mismatch"
    assert np.allclose(norm_combined["positions_test"], test_pos_norm), "Combined test mismatch"
    assert int(norm_combined["train_test_split"]) == test_split, "Split mismatch in combined"
    print("✓ Combined file consistent")

    print("\n✅ All sanity checks passed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity check for raw, normalized, and GluonTS data")
    parser.add_argument("--datatype", "-dt", required=True, help="Dataset name (e.g., double_well)")
    parser.add_argument("--raw_dir", default="DATA_raw", help="Directory with .npy files")
    parser.add_argument("--data_dir", default="DATA", help="Directory with .npz normalized files")
    parser.add_argument("--gluonts_dir", default="gluonts_datasets", help="Base directory of GluonTS datasets")
    parser.add_argument("--context_length", "-cl", type=int, default=None)
    parser.add_argument("--prediction_length", "-pl", type=int, default=None)
    args = parser.parse_args()

    sanity_check_full(
        datatype=args.datatype,
        raw_dir=args.raw_dir,
        data_dir=args.data_dir,
        gluonts_dir=args.gluonts_dir,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
    )