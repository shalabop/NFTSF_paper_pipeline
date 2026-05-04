"""
Standalone script to regenerate loss curve plots from a saved loss_history_*.npy file.

Usage:
  # Point directly at a .npy file:
  python plot_loss.py --loss_path ./pipeline_results/training/single_well/loss_history_20240101_120000.npy

  # Or point at a training output directory (auto-picks the most recent .npy):
  python plot_loss.py --training_dir ./pipeline_results/training/single_well

  # Optionally specify where to save the plots (default: same dir as the .npy file):
  python plot_loss.py --training_dir ./pipeline_results/training/single_well --output_dir ./plots
"""

import argparse
import glob
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Plot training loss from a saved loss_history .npy file.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--loss_path", type=str,
                       help="Direct path to a loss_history_*.npy file.")
    group.add_argument("--training_dir", type=str,
                       help="Training output directory; the most recent loss_history_*.npy inside is used.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save the plots (default: same directory as the .npy file).")
    parser.add_argument("--landscape", type=str, default=None,
                        help="Landscape/system name to include in plot titles.")
    return parser.parse_args()


def find_latest_loss_file(directory):
    pattern = os.path.join(directory, "loss_history_*.npy")
    matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not matches:
        print(f"Error: no loss_history_*.npy files found in '{directory}'", file=sys.stderr)
        sys.exit(1)
    return matches[0]


def plot_loss(loss_array, output_dir, landscape_name=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)

    # Linear loss plot
    linear_title = "Training Loss"
    if landscape_name:
        linear_title = f"{landscape_name} — {linear_title}"
    plt.figure(figsize=(10, 6))
    plt.plot(loss_array)
    plt.title(linear_title)
    plt.xlabel("Epoch")
    plt.ylabel("Negative Log Likelihood")
    plt.grid(True, alpha=0.3)
    linear_path = os.path.join(output_dir, f"training_loss_{timestamp}.png")
    plt.savefig(linear_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved linear loss plot to: {linear_path}")

    # Log-scale loss plot — guard against non-positive values
    valid_mask = loss_array > 0
    if valid_mask.any():
        n_skipped = int((~valid_mask).sum())
        if n_skipped:
            print(f"Warning: {n_skipped} non-positive loss value(s) skipped in log-scale plot.")
        log_title = "Training Loss (Log Scale)"
        if landscape_name:
            log_title = f"{landscape_name} — {log_title}"
        plt.figure(figsize=(10, 6))
        plt.plot(np.where(valid_mask)[0], np.log(loss_array[valid_mask]))
        plt.title(log_title)
        plt.xlabel("Epoch")
        plt.ylabel("Log(Loss)")
        plt.grid(True, alpha=0.3)
        log_path = os.path.join(output_dir, f"training_loss_log_{timestamp}.png")
        plt.savefig(log_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved log-scale loss plot to: {log_path}")
    else:
        print("Warning: no positive loss values found; skipping log-scale plot.")


def main():
    args = parse_args()

    if args.loss_path:
        npy_file = args.loss_path
        if not os.path.isfile(npy_file):
            print(f"Error: file not found: '{npy_file}'", file=sys.stderr)
            sys.exit(1)
    else:
        npy_file = find_latest_loss_file(args.training_dir)

    print(f"Loading loss history from: {npy_file}")
    loss_array = np.load(npy_file).astype(float)
    print(f"  {len(loss_array)} epochs, final loss = {loss_array[-1]:.6f}")

    output_dir = args.output_dir if args.output_dir else os.path.dirname(os.path.abspath(npy_file))
    plot_loss(loss_array, output_dir, landscape_name=args.landscape)


if __name__ == "__main__":
    main()
