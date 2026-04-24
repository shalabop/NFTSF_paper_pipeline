#!/usr/bin/env python3
"""
visualize_sanity.py
===================
Overlay raw, normalized DATA (denormalized), and GluonTS (denormalized) trajectories
for both train and test splits to verify consistency.
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_gluonts_series(folder: Path):
    """
    Load all series from a GluonTS split folder (JSON Lines format).
    Returns list of numpy arrays (each trajectory).
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
        # fallback: individual .json files
        for json_file in sorted(folder.glob("*.json")):
            with open(json_file, "r") as f:
                data = json.load(f)
            series.append(np.array(data["target"], dtype=np.float32))
    return series

def main():
    parser = argparse.ArgumentParser(description="Overlay raw, DATA, and GluonTS trajectories")
    parser.add_argument("--datatype", "-dt", required=True, help="Dataset name (e.g., double_well)")
    parser.add_argument("--raw_dir", default="DATA_raw", help="Directory with raw .npy files")
    parser.add_argument("--data_dir", default="DATA", help="Directory with normalized .npz files")
    parser.add_argument("--gluonts_dir", default="gluonts_datasets", help="Base directory of GluonTS dataset")
    parser.add_argument("--n_traj", type=int, default=3, help="Number of trajectories to plot per split")
    parser.add_argument("--output", "-o", default="sanity_check_plots.png", help="Output image file")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    data_dir = Path(args.data_dir)
    gluonts_base = Path(args.gluonts_dir) / args.datatype

    # 1. Load raw data
    raw_train = np.load(raw_dir / f"{args.datatype}_train.npy")
    raw_val   = np.load(raw_dir / f"{args.datatype}_val.npy")
    raw_test  = np.load(raw_dir / f"{args.datatype}_test.npy")
    raw_train_val = np.concatenate([raw_train, raw_val], axis=0)   # (N_train+N_val, T)

    # 2. Load normalized DATA (.npz)
    norm_train = np.load(data_dir / f"{args.datatype}_train.npz")
    norm_test  = np.load(data_dir / f"{args.datatype}_test.npz")
    train_pos_norm = norm_train["positions"]      # (N_train+N_val, T)
    test_pos_norm  = norm_test["positions"]       # (N_test, T)
    mean = float(norm_train["mean"])
    std  = float(norm_train["std"])

    # Denormalize DATA
    train_denorm = train_pos_norm * (std + 1e-8) + mean
    test_denorm  = test_pos_norm * (std + 1e-8) + mean

    # 3. Load GluonTS series (normalized) and denormalize
    gluon_train = load_gluonts_series(gluonts_base / "train")
    gluon_test  = load_gluonts_series(gluonts_base / "test")
    gluon_train_denorm = [g * (std + 1e-8) + mean for g in gluon_train]
    gluon_test_denorm  = [g * (std + 1e-8) + mean for g in gluon_test]

    # Check counts
    assert len(gluon_train_denorm) == train_denorm.shape[0], "GluonTS train count mismatch"
    assert len(gluon_test_denorm) == test_denorm.shape[0], "GluonTS test count mismatch"

    # 4. Plot
    n_traj = min(args.n_traj, train_denorm.shape[0], len(gluon_train_denorm))
    fig, axes = plt.subplots(2, n_traj, figsize=(5*n_traj, 8), sharex=False, sharey=False)
    if n_traj == 1:
        axes = axes.reshape(-1, 1)
    # Train plots
    for i in range(n_traj):
        ax = axes[0, i]
        ax.plot(raw_train_val[i], label="Raw", linewidth=2, color="black")
        ax.plot(train_denorm[i], label="DATA (denorm)", linewidth=1.5, linestyle="--", color="blue")
        ax.plot(gluon_train_denorm[i], label="GluonTS (denorm)", linewidth=1.5, linestyle=":", color="red")
        ax.set_title(f"Train trajectory {i}")
        ax.legend()
        ax.grid(alpha=0.3)
    # Test plots
    n_test = min(args.n_traj, test_denorm.shape[0], len(gluon_test_denorm))
    for i in range(n_test):
        ax = axes[1, i]
        ax.plot(raw_test[i], label="Raw", linewidth=2, color="black")
        ax.plot(test_denorm[i], label="DATA (denorm)", linewidth=1.5, linestyle="--", color="blue")
        ax.plot(gluon_test_denorm[i], label="GluonTS (denorm)", linewidth=1.5, linestyle=":", color="red")
        ax.set_title(f"Test trajectory {i}")
        ax.legend()
        ax.grid(alpha=0.3)
    # Hide unused subplots
    for i in range(n_test, n_traj):
        axes[1, i].axis("off")

    fig.suptitle(f"{args.datatype} – Raw vs DATA (denorm) vs GluonTS (denorm)")
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()