import argparse
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Concatenate train and validation .npy files")
    parser.add_argument("--train", "-tr", required=True, help="Path to train.npy")
    parser.add_argument("--val", "-v", required=True, help="Path to val.npy")
    parser.add_argument("--out", "-o", required=True, help="Output .npy path (e.g., train_val.npy)")
    args = parser.parse_args()

    train_path = Path(args.train)
    val_path = Path(args.val)
    out_path = Path(args.out)

    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation file not found: {val_path}")

    train_data = np.load(train_path)
    val_data = np.load(val_path)

    if train_data.ndim != 2 or val_data.ndim != 2:
        raise ValueError("Both files must be 2D arrays (trajectories, time steps)")
    if train_data.shape[1] != val_data.shape[1]:
        raise ValueError(f"Time dimension mismatch: train {train_data.shape[1]} vs val {val_data.shape[1]}")

    combined = np.concatenate([train_data, val_data], axis=0)

    print(f"Train shape: {train_data.shape}")
    print(f"Val shape : {val_data.shape}")
    print(f"Combined : {combined.shape}")

    np.save(out_path, combined)
    print(f"Saved combined array to {out_path}")


if __name__ == "__main__":
    main()