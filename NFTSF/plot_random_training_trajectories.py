#!/usr/bin/env python3
"""
Plot 9 randomly selected trajectories from already-generated training data.

Use this to visually inspect raw data for any landscape without running the
full training pipeline.

Usage
-----
# Basic (multi_sim format, which is the default for generated data):
    python plot_random_training_trajectories.py --data_path ./data/train.npy

# Specify data format and landscape name for a nicer title:
    python plot_random_training_trajectories.py \
        --data_path ./data/linear_gaussian_train.npy \
        --data_format multi_sim \
        --landscape "Linear Gaussian"

# Plot more/fewer trajectories with a fixed seed:
    python plot_random_training_trajectories.py \
        --data_path ./data/double_well.npy \
        --n_plots 9 \
        --seed 42 \
        --output_dir ./plots

Supported data formats
----------------------
multi_sim  : (Time, 1+NumSims) — column 0 is time, rest are trajectories [default]
single_sim : (Samples, Time, 2) — each sample is (time, value) pairs
tnf        : (T, N, F)         — general time×track×feature format
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend; works on headless servers
import matplotlib.pyplot as plt
from datetime import datetime


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot random training trajectories from saved data.'
    )
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the data .npy file')
    parser.add_argument('--data_format', type=str, default='multi_sim',
                        choices=['single_sim', 'multi_sim', 'tnf'],
                        help='Data format (default: multi_sim)')
    parser.add_argument('--landscape', type=str, default=None,
                        help='Landscape/dataset name used in the plot title '
                             '(defaults to the data filename without extension)')
    parser.add_argument('--n_plots', type=int, default=9,
                        help='Number of trajectories to show (default: 9)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory for the output figure '
                             '(defaults to the same directory as data_path)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible trajectory selection')
    parser.add_argument('--well_positions', nargs='+', type=float, default=None,
                        metavar='Y',
                        help='Y-axis positions of potential well minima to overlay as '
                             'gray dashed lines (e.g. --well_positions -1.0 1.0).')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading  (mirrors train_model.py / test_model.py load_data)
# ---------------------------------------------------------------------------

def load_data(data_path, data_format):
    """Load data and return a 2-D numpy array of shape (Batch, Time)."""
    import torch  # local import so the script has minimal hard dependencies

    print(f"Loading data from: {data_path}")
    data = np.load(data_path, allow_pickle=True)
    print(f"  Raw shape: {data.shape}")

    tensor_data = torch.tensor(data, dtype=torch.float32)

    if data_format == 'single_sim':
        # Shape: (Samples, Time, 2) — value column is index 1
        reshaped = tensor_data[:, :, 1]

    elif data_format == 'multi_sim':
        # Shape: (Time, 1+NumSims) — column 0 is time axis
        positions_only = tensor_data[:, 1:]   # drop time column
        reshaped = positions_only.T           # → (NumSims, Time)

    elif data_format == 'tnf':
        arr = np.asarray(data)
        if arr.ndim == 1:
            arr = arr[:, None, None]
        elif arr.ndim == 2:
            arr = arr[:, None, :]
        data_vals = (arr[:, :, 1:].astype(np.float32)
                     if arr.shape[2] > 1 else arr.astype(np.float32))
        series = data_vals[:, 0, 0].reshape(-1)
        reshaped = torch.tensor(series, dtype=torch.float32).unsqueeze(0)

    else:
        raise ValueError(f"Unknown data format: {data_format!r}")

    result = reshaped.cpu().numpy()
    print(f"  Reshaped to: {result.shape}  (trajectories × time steps)")
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_trajectories(data, n_plots, landscape_name, output_dir, seed=None,
                      well_positions=None):
    """Plot n_plots randomly selected trajectories as a grid."""

    num_available, traj_len = data.shape
    n_plots = min(n_plots, num_available)

    if seed is not None:
        np.random.seed(seed)

    random_indices = np.random.choice(num_available, n_plots, replace=False)

    rows = int(np.ceil(np.sqrt(n_plots)))
    cols = int(np.ceil(n_plots / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten() if n_plots > 1 else [axes]

    time_steps = np.arange(traj_len)

    for i in range(len(axes)):
        ax = axes[i]
        if i >= n_plots:
            ax.axis('off')
            continue

        idx = random_indices[i]
        traj = data[idx]

        # Well lines at zorder=0 — drawn before trajectory so it sits on top.
        if well_positions:
            for j, wp in enumerate(well_positions):
                ax.axhline(y=wp, color='gray', linestyle='--', linewidth=1.2,
                           alpha=0.7, zorder=0,
                           label=f"Well {j + 1}" if i == 0 else '_nolegend_')

        # Y-axis label: use dihedral angle notation for Alanine Dipeptide
        y_label = (r'angle $\varphi$ (rad)'
                   if landscape_name and 'alanine' in landscape_name.lower()
                   else r'Position, $x$')

        ax.plot(time_steps, traj, color='steelblue', linewidth=1.0, alpha=0.85)
        ax.set_title(f"Trajectory {idx}", fontsize=12)
        ax.set_xlabel(r'Step, $N$', fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.grid(True, alpha=0.3)

    if well_positions:
        axes[0].legend(loc='upper left', fontsize=9)
    plt.suptitle(
        f"Random Training Trajectories — {landscape_name}  "
        f"(N={num_available} total, showing {n_plots})",
        fontsize=14,
        y=1.01,
    )
    plt.tight_layout()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(output_dir, f'random_trajectories_{timestamp}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not os.path.exists(args.data_path):
        print(f"ERROR: data file not found: {args.data_path}", file=sys.stderr)
        sys.exit(1)

    # Determine output directory
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(args.data_path))
    os.makedirs(output_dir, exist_ok=True)

    # Determine landscape/dataset name for the plot title
    landscape_name = args.landscape
    if landscape_name is None:
        landscape_name = os.path.splitext(os.path.basename(args.data_path))[0]

    data = load_data(args.data_path, args.data_format)
    plot_trajectories(data, args.n_plots, landscape_name, output_dir,
                      seed=args.seed, well_positions=args.well_positions)


if __name__ == '__main__':
    main()
