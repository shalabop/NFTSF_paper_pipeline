#!/usr/bin/env python3
"""
Reload saved test trajectories and regenerate all test plots.

After running test_model.py, a `test_trajectories_<timestamp>.npz` file is
saved alongside the other output figures.  This script loads that file and
regenerates every test plot without re-running the model.

Usage
-----
# Regenerate all plots into the same directory as the results file:
    python replot_test_results.py \
        --results_path ./test_results/test_trajectories_20240101_120000.npz

# Save into a separate directory (e.g. for a different paper figure style):
    python replot_test_results.py \
        --results_path ./test_results/test_trajectories_20240101_120000.npz \
        --output_dir ./replot_output

Plots regenerated
-----------------
  trajectory_band_*       — single trajectory with 50%/95% confidence bands
  reliability_diagram_*   — quantile calibration (reliability) diagram
  histogram2d_*           — 2-D density heatmap for the single test trajectory
  error_metrics_*         — per-step MAE and CRPS over the forecast horizon
  trajectory_grid_*       — 3×3 grid of trajectories with confidence bands
  histogram2d_grid_*      — 3×3 grid of 2-D density heatmaps
  empirical_coverage_*    — empirical quantile coverage (20%/60%/80% central
                            intervals) vs expected levels; falls back to the
                            1σ/2σ/3σ Gaussian plot for older .npz files

File format
-----------
The .npz file must contain the fields written by test_model.save_test_trajectories:
  ground_truth        : (n_past + n_future,)
  samples             : (n_samples, n_future)
  grid_indices        : (N_grid,)
  grid_ground_truths  : (N_grid, n_past + n_future)
  grid_samples        : (N_grid, n_samples, n_future)
  n_past              : (1,)
  n_future            : (1,)
  grid_y_lo           : (N_grid,)   [optional, written by newer runs]
  grid_y_hi           : (N_grid,)   [optional, written by newer runs]
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime

# Import plotting functions from the main test script.
# Assumes this script lives in the same directory as test_model.py.
try:
    from test_model import (
        plot_trajectory_with_band,
        plot_reliability_diagram,
        plot_histogram2d,
        plot_error_metrics,
        plot_histogram2d_grid,
        plot_trajectory_grid_from_data,
        plot_empirical_coverage,
        plot_empirical_coverage_quantiles,
        ensemble_crps,
    )
except ImportError as exc:
    print(
        f"ERROR: Could not import from test_model.py.  "
        f"Make sure this script is in the same directory.  Details: {exc}",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Regenerate all test plots from a saved .npz results file.'
    )
    parser.add_argument(
        '--results_path', type=str, required=True,
        help='Path to the saved test_trajectories_<timestamp>.npz file'
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Output directory for regenerated figures '
             '(defaults to same directory as results_path)'
    )
    parser.add_argument(
        '--well_positions', nargs='+', type=float, default=None, metavar='Y',
        help='Y-axis positions of potential well minima to overlay as dashed lines '
             '(e.g. --well_positions -1.0 1.0 for a double-well).'
    )
    parser.add_argument(
        '--landscape', type=str, default=None,
        help='Landscape/system name for plot titles '
             '(default: inferred from results filename).'
    )
    parser.add_argument(
        '--plot_future_steps', type=int, default=None,
        help='Number of future steps to display on plots. Defaults to n_future. '
             'Controls display only — model predictions are unaffected.'
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Additional plotting
# ---------------------------------------------------------------------------

def plot_sample_overlays(ground_truth, samples, n_past, n_future, output_dir, timestamp,
                         landscape_name=None):
    """Plot predicted trajectories with 10 overlaid samples."""
    n_show = min(10, samples.shape[0])
    samples_show = samples[:n_show, :n_future]

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    x_all = np.arange(0, n_past + n_future)
    x_future = np.arange(n_past, n_past + n_future)

    ax.plot(x_all[:n_past], ground_truth[:n_past], color='black', linewidth=2.0, label='Past')
    ax.plot(x_future, ground_truth[n_past:n_past + n_future], linestyle='--', color='gray',
            linewidth=2.0, label='Ground Truth Future')

    for i in range(n_show):
        label = 'Predicted Samples' if i == 0 else None
        ax.plot(x_future, samples_show[i], color='tab:blue', alpha=0.7, linewidth=1.2, label=label)

    ax.axvline(x=n_past, color='black', linestyle='--', alpha=0.5)
    title = "10 Predicted Samples"
    if landscape_name:
        title = f"{landscape_name} — {title}"
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(r'Step, $N$', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()

    path = os.path.join(output_dir, f'sample_overlay_{timestamp}.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not os.path.exists(args.results_path):
        print(f"ERROR: results file not found: {args.results_path}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------ load
    print(f"Loading saved test trajectories from: {args.results_path}")
    data = np.load(args.results_path, allow_pickle=True)

    ground_truth       = data['ground_truth']        # (n_past + n_future,)
    samples            = data['samples']             # (n_samples, n_future)
    grid_indices       = data['grid_indices']        # (N_grid,)
    grid_ground_truths = data['grid_ground_truths']  # (N_grid, n_past + n_future)
    grid_samples_arr   = data['grid_samples']        # (N_grid, n_samples, n_future)
    n_past             = int(data['n_past'][0])
    n_future           = int(data['n_future'][0])

    # Reconstruct grid_data list — 5-tuples when y-limits were saved,
    # 3-tuples otherwise (older format compatibility).
    has_ylimits = ('grid_y_lo' in data) and ('grid_y_hi' in data)
    grid_data = []
    for i in range(len(grid_indices)):
        entry = (
            int(grid_indices[i]),
            grid_ground_truths[i],
            grid_samples_arr[i],
        )
        if has_ylimits:
            entry = entry + (float(data['grid_y_lo'][i]),
                             float(data['grid_y_hi'][i]))
        grid_data.append(entry)

    print(f"  n_past={n_past}, n_future={n_future}, "
          f"n_samples={samples.shape[0]}, n_grid={len(grid_data)}")

    # ------------------------------------------- output directory & timestamp
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(args.results_path))
    os.makedirs(output_dir, exist_ok=True)

    # Determine landscape name for plot titles
    landscape_name = args.landscape
    if landscape_name is None:
        landscape_name = os.path.splitext(os.path.basename(args.results_path))[0]

    # Determine display window for future steps
    plot_future_steps = args.plot_future_steps if args.plot_future_steps is not None else n_future

    # Append '_replot' to distinguish regenerated figures from originals.
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') + '_replot'

    # ----------------------------------------------------------------- plots
    y_true_future = ground_truth[n_past:]
    crps_score    = ensemble_crps(y_true_future, samples)
    print(f"  CRPS score (single trajectory): {crps_score:.4f}")

    print("\nRegenerating plots...")
    plot_sample_overlays(
        ground_truth,
        samples,
        n_past,
        n_future,
        output_dir,
        timestamp,
        landscape_name=landscape_name,
    )

    wp = args.well_positions  # shorthand; None when not provided

    # 1. Single trajectory with 50% / 95% confidence bands
    plot_trajectory_with_band(
        ground_truth, samples, n_past, n_future,
        crps_score, output_dir, timestamp,
        well_positions=wp,
        landscape_name=landscape_name,
        plot_future_steps=plot_future_steps,
    )

    # 2. Quantile calibration (reliability) diagram
    plot_reliability_diagram(y_true_future, samples, output_dir, timestamp,
                             landscape_name=landscape_name)

    # 3. 2-D density heatmap for the single test trajectory
    plot_histogram2d(
        samples, ground_truth, n_past, n_future,
        output_dir, timestamp,
        well_positions=wp,
        landscape_name=landscape_name,
        plot_future_steps=plot_future_steps,
    )

    # 4. Per-step MAE and CRPS over the forecast horizon
    plot_error_metrics(
        y_true_future, samples, n_past, n_future,
        output_dir, timestamp,
        landscape_name=landscape_name,
        plot_future_steps=plot_future_steps,
    )

    # 5. Trajectory grid (pre-sampled; no model needed)
    plot_trajectory_grid_from_data(grid_data, n_past, n_future, output_dir, timestamp,
                                   well_positions=wp,
                                   landscape_name=landscape_name,
                                   plot_future_steps=plot_future_steps)

    # 6. 2-D histogram density grid
    plot_histogram2d_grid(grid_data, n_past, n_future, output_dir, timestamp,
                          well_positions=wp,
                          landscape_name=landscape_name,
                          plot_future_steps=plot_future_steps)

    # 7. Empirical coverage plot.
    # If the .npz file contains full-test-set quantile coverage data (written
    # by newer runs of test_model.py), use the quantile-based plot.  Older
    # .npz files that lack this data fall back to the sigma-based plot using
    # just the saved grid trajectories.
    has_fullset_coverage = all(
        f'coverage_per_step_{pct}' in data for pct in [20, 60, 80]
    )
    if has_fullset_coverage:
        coverage_per_step = {pct: data[f'coverage_per_step_{pct}']
                             for pct in [20, 60, 80]}
        n_evaluated = int(data['fullset_n_evaluated'][0])
        plot_empirical_coverage_quantiles(
            coverage_per_step, n_future, n_evaluated, output_dir, timestamp,
            landscape_name=landscape_name,
        )
    else:
        # Fallback: sigma-based coverage from the 9 saved grid trajectories
        plot_empirical_coverage(grid_data, n_past, n_future, output_dir, timestamp,
                                landscape_name=landscape_name)

    print(f"\nAll figures regenerated and saved to: {output_dir}")


if __name__ == '__main__':
    main()
