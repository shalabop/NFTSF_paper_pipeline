#!/usr/bin/env python3
"""
Model testing and evaluation script for NF-TSF.
Converted from Model_Tester__import_data_.ipynb for supercomputer use.

Usage:
    python test_model.py --model_path /path/to/model.pth --data_path /path/to/test_data.npy --output_dir ./results
"""

import os
import sys
import argparse
import json
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless servers
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False

# Local imports
from architecture import create_nfm
from architecture_encoder import preset_stage1, preset_stage2


def parse_args():
    parser = argparse.ArgumentParser(description='Test NF-TSF Model')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model .pth file')
    parser.add_argument('--norm_stats_path', type=str, default=None,
                        help='Path to normalization stats .npz file')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to test data .npy file (not required when --data_format sde)')
    parser.add_argument('--data_format', type=str, default='multi_sim',
                        choices=['single_sim', 'multi_sim', 'tnf', 'sde', 'sde_presplit'],
                        help='Data format: single_sim, multi_sim, tnf, sde, or sde_presplit')

    # SDE-specific parameters (used when --data_format sde)
    parser.add_argument('--sde_model_key', type=str, default=None,
                        help='SDE model key, e.g. sw_sle_em or sw_gle_oe_em '
                             '(required when --data_format sde)')
    parser.add_argument('--sde_data_dir', type=str, default='../Data/Trajectories',
                        help='Directory containing SDE .npy files '
                             '(default: ../Data/Trajectories)')
    parser.add_argument('--sde_test_id', type=int, default=1,
                        help='Parameter-set ID used when generating SDE data (default: 1)')
    parser.add_argument('--sde_skip', type=int, default=10,
                        help='Skip value used when generating SDE data (default: 10)')
    
    # Model architecture parameters (must match training)
    parser.add_argument('--n_past', type=int, default=100,
                        help='Number of past time steps (context)')
    parser.add_argument('--n_future', type=int, default=100,
                        help='Number of future time steps')
    parser.add_argument('--tail_bound', type=float, default=30.0,
                        help='Rational-quadratic spline tail bound for A-RQS flow layers')
    
    # Testing parameters
    parser.add_argument('--n_samples', type=int, default=500,
                        help='Number of sample trajectories to generate')
    parser.add_argument('--n_quart_test', type=int, default=500,
                        help='Number of quantile tests to run')
    parser.add_argument('--run_quantile_tests', action='store_true', default=False,
                        help='Run expensive quantile calibration tests')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./test_results',
                        help='Output directory for results')
    
    # Device parameters
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=29182,
                        help='Random seed')

    parser.add_argument('--well_positions', nargs='+', type=float, default=None,
                        metavar='Y',
                        help='Y-axis positions of potential well minima to overlay as '
                             'gray dashed lines on trajectory and heatmap plots. '
                             'E.g. --well_positions -1.0 1.0 for a symmetric double-well.')

    parser.add_argument('--landscape', type=str, default=None,
                        help='Landscape/system name for plot titles '
                             '(default: inferred from data filename)')

    parser.add_argument('--plot_future_steps', type=int, default=None,
                        help='Number of future steps to display on trajectory and '
                             'density plots. Must be <= n_future. Defaults to n_future. '
                             'Controls display only — model predictions are unaffected.')
    parser.add_argument('--basic_eval_use_last_window', action='store_true', default=True,
                        help='For the single basic-evaluation trajectory only, use the '
                             'last (n_past + n_future) steps from trajectory index 0 '
                             'instead of a random window. DEFAULT: True.')
    parser.add_argument('--full_eval_use_last_window', action='store_true', default=True,
                        help='For full test-set evaluation, use the last '
                             '(n_past + n_future) steps from each trajectory '
                             'instead of a random window. DEFAULT: True.')

    # Architecture config — when provided, flow_blocks/hidden_units/hidden_layers are
    # read from this file so test-time model construction always matches what was trained.
    parser.add_argument('--config_path', type=str, default=None,
                        help='Path to the training config_*.json saved by train_model.py. '
                             'When given, architecture hyperparameters (flow_blocks, '
                             'hidden_units, hidden_layers, model_variant) are loaded from '
                             'this file instead of using hardcoded defaults.')

    parser.add_argument('--model_variant', type=str, default=None,
                        choices=['ar', 'ar_encoder_full', 'ar_encoder_light'],
                        help='Model architecture variant. When --config_path is given, '
                             'model_variant is read from the config JSON. Pass this flag '
                             'explicitly only to override the saved config value. '
                             'Defaults to "ar" when neither config nor flag provides it.')

    return parser.parse_args()


def setup_device(device_arg):
    """Setup and return the compute device."""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    torch.set_default_device(device)
    return device


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_data(data_path, data_format):
    """Load and reshape data based on format."""
    print(f"Loading test data from: {data_path}")
    data = np.load(data_path, allow_pickle=True)
    print(f"  Raw shape: {data.shape}")
    
    tensor_data = torch.tensor(data, dtype=torch.float32)
    
    if data_format == 'single_sim':
        reshaped_data = tensor_data[:, :, 1]
    elif data_format == 'multi_sim':
        positions_only = tensor_data[:, 1:]
        reshaped_data = positions_only.T

    elif data_format == 'sde_presplit':
        # Shape: (N_windows, T) — pre-split output of sde_preprocess.py; already (Batch, Time)
        reshaped_data = tensor_data

    elif data_format == 'tnf':
        arr = np.asarray(data)
        if arr.ndim == 1:
            arr = arr[:, None, None]
        elif arr.ndim == 2:
            arr = arr[:, None, :]
        data_vals = arr[:, :, 1:].astype(np.float32) if arr.shape[2] > 1 else arr.astype(np.float32)
        series = data_vals[:, 0, 0].reshape(-1)
        reshaped_data = torch.tensor(series, dtype=torch.float32).unsqueeze(0)
    
    print(f"  Reshaped to: {reshaped_data.shape} (Batch, Time)")
    return reshaped_data, data


def load_normalization_stats(norm_path):
    """Load normalization statistics."""
    if norm_path and os.path.exists(norm_path):
        stats = np.load(norm_path)
        return {
            'mean': torch.tensor(stats['mean']),
            'std': torch.tensor(stats['std'])
        }
    return None


def get_trajectory(data_tensor, n_extrp, random=True, from_end=False):
    """Select a single trajectory segment from the batch.

    When random=True, picks both a random trajectory and a random start
    position so the visualization reflects a representative window rather
    than always starting from t=0.
    """
    batch_size, time_steps = data_tensor.shape

    if random:
        batch_idx = np.random.randint(0, batch_size)
        start = np.random.randint(0, max(1, time_steps - n_extrp + 1))
    elif from_end:
        batch_idx = 0
        start = max(0, time_steps - n_extrp)
    else:
        batch_idx = 0
        start = 0

    trajectory = data_tensor[batch_idx, start:start + n_extrp]
    return trajectory


def alt_selector(tracks, n_past, n_future, random=False):
    """Alternative selector for TNF format data."""
    full_length = tracks.shape[0]
    num_tracks = tracks.shape[1]
    n_extrp = n_past + n_future
    
    choice2 = 622 if not random else np.random.randint(0, full_length - n_extrp - 1)
    choice1 = 1 if not random else np.random.randint(0, num_tracks)
    
    snippet = tracks[choice2:(choice2 + n_extrp), choice1, 0]
    return snippet


# ==================== METRIC FUNCTIONS ====================

def ensemble_crps(observations, forecasts):
    """
    Computes the Continuous Ranked Probability Score (CRPS) for an ensemble
    forecast, averaged over all forecast time steps.  Lower values indicate
    a better probabilistic forecast.

    The CRPS measures the distance between the predictive cumulative
    distribution function (CDF) F and the observation y:
        CRPS(F, y) = E|X - y| - (1/2) * E|X - X'|

    where X, X' are independent draws from F.  The first term is the mean
    absolute error (MAE) of the ensemble members against the observation;
    the second is the mean absolute difference between all pairs of ensemble
    members (the "energy" or spread term), which penalizes overconfident
    (too narrow) distributions less and underdispersed ones more.

    For an ensemble of M sorted samples x_(1) <= x_(2) <= ... <= x_(M),
    the pairwise spread term has the efficient O(M log M) representation
    using order statistics (Gneiting & Raftery, 2007):
        E|X - X'| = (2 / [M*(M-1)]) * sum_{i=1}^{M-1} w_i * (x_(i+1) - x_(i))

    where the weight w_i = i * (M - i) counts the number of pairs (j, k)
    with j <= i < k, i.e., how many pairwise differences span the gap
    between consecutive sorted samples x_(i) and x_(i+1).

    Implementation:
        mae         = (1/M) * sum_i |x_i - y|           [ensemble MAE]
        diff_i      = x_(i+1) - x_(i)                   [consecutive sorted gaps]
        w_i         = i * (M - i)                        [pair count weights]
        energy      = sum(w_i * diff_i) / [M * (M - 1)] [half the mean pairwise |X-X'|]
        CRPS        = mean over time steps of (mae - energy)

    Args:
        observations: 1-D array of shape (n_future,), ground truth values
        forecasts: 2-D array of shape (n_samples, n_future), ensemble members

    Returns:
        Scalar CRPS averaged over all forecast time steps.
    """
    mae = np.mean(np.abs(forecasts - observations), axis=0)

    n_samples = forecasts.shape[0]
    forecasts_sorted = np.sort(forecasts, axis=0)
    # Consecutive differences between order statistics
    diff = forecasts_sorted[1:] - forecasts_sorted[:-1]
    # Weight w_i = i * (M - i): number of pairs straddling gap i
    w = np.arange(1, n_samples) * np.arange(n_samples - 1, 0, -1)
    # Weighted sum normalized by M*(M-1) gives (1/2)*E|X - X'|
    energy = np.sum(diff * w[:, None], axis=0) / (n_samples * (n_samples - 1))

    return np.mean(mae - energy)


def calculate_coverage_quantiles(y_true, y_samples, quantiles):
    """
    Computes empirical coverage at specified quantile levels to assess
    calibration of the predictive distribution.

    For a well-calibrated probabilistic forecast, the fraction of true
    observations falling below the predicted q-th quantile should equal q.
    That is, if F^{-1}(q) is the predicted quantile at level q, then:
        P(Y <= F^{-1}(q)) = q    (ideal calibration)

    This function estimates the left-hand side empirically:
        coverage(q) = (1 / n_future) * sum_{t=1}^{n_future} 1[ y_t <= Q_q(t) ]

    where Q_q(t) is the q-th percentile of the forecast ensemble at time t,
    and y_t is the observed value.

    Deviations from coverage(q) = q indicate miscalibration: coverage > q
    means the model is underdispersed (too wide); coverage < q means it is
    overdispersed (too narrow) at that quantile level.  Plotting coverage(q)
    vs q yields a reliability (calibration) diagram.

    Args:
        y_true: 1-D array of shape (n_future,), ground truth observations
        y_samples: 2-D array of shape (n_samples, n_future), ensemble forecasts
        quantiles: list/array of quantile levels in [0, 1]

    Returns:
        List of empirical coverage fractions, one per requested quantile.
    """
    empirical_coverage = []
    for q in quantiles:
        q_val = np.percentile(y_samples, q * 100, axis=0)
        coverage = np.mean(y_true <= q_val)
        empirical_coverage.append(coverage)
    return empirical_coverage


def get_metrics_per_step(observations, forecasts):
    """
    Computes two error metrics at each forecast time step:

    1. MAE per step (Median Absolute Error):
        MAE(t) = | median(forecasts(:, t)) - y(t) |

       Uses the ensemble median as the point forecast, which is the optimal
       point predictor under absolute loss.

    2. CRPS per step (see ensemble_crps for full derivation):
        CRPS(t) = (1/M) * sum_i |x_i(t) - y(t)|
                  - (1/[M*(M-1)]) * sum_{i<j} |x_i(t) - x_j(t)|

       Equivalently, using the sorted-order-statistic representation:
        CRPS(t) = mae_term(t) - energy_term(t)

       where mae_term is the ensemble mean absolute error against the
       observation and energy_term is the ensemble spread penalty computed
       via the efficient weighted consecutive-differences formula (see
       ensemble_crps).

    These step-wise metrics reveal how forecast skill evolves with the
    prediction horizon, typically degrading at longer lead times.

    Args:
        observations: 1-D array of shape (n_future,), ground truth values
        forecasts: 2-D array of shape (n_samples, n_future), ensemble members

    Returns:
        crps_per_step: 1-D array of CRPS values, one per future time step
        mae_per_step: 1-D array of MAE values, one per future time step
    """
    # Point forecast: ensemble median (optimal under L1 loss)
    pred_median = np.median(forecasts, axis=0)
    mae_per_step = np.abs(pred_median - observations)

    # CRPS decomposition: reliability term minus sharpness (spread) term
    mae_term = np.mean(np.abs(forecasts - observations), axis=0)

    n_samples = forecasts.shape[0]
    forecasts_sorted = np.sort(forecasts, axis=0)
    diff = forecasts_sorted[1:] - forecasts_sorted[:-1]
    w = np.arange(1, n_samples) * np.arange(n_samples - 1, 0, -1)
    energy_term = np.sum(diff * w[:, None], axis=0) / (n_samples * (n_samples - 1))

    crps_per_step = mae_term - energy_term

    return crps_per_step, mae_per_step


def quantile_test(percent, n_quart_test, n_past, n_future, n_samp_trajs,
                  dynamics_func, model, data, device):
    """
    Runs a Monte Carlo quantile calibration test to assess whether the
    model's predicted central credible intervals have correct empirical
    coverage.

    For a given coverage level P (e.g., 50%), the test constructs the
    symmetric central interval [Q_{(1-P/2)}, Q_{(1+P/2)}] from the
    model's forecast ensemble at each future time step.  For example,
    P = 50% yields [Q_0.25, Q_0.75] (the interquartile range).

    Over N independent test trajectories, the empirical coverage at
    each future step t is:
        coverage(t) = (1/N) * sum_{i=1}^{N} 1[ Q_low^(i)(t) < y^(i)(t) < Q_up^(i)(t) ]

    where y^(i)(t) is the true value and Q_low^(i), Q_up^(i) are the
    predicted quantile bounds for test case i.

    If the model is well-calibrated, coverage(t) should converge to P/100
    for all t as N -> infinity.

    The convergence matrix n_true_mtrx tracks the running average of
    coverage after each iteration i:
        n_true_mtrx[i, t] = (1/(i+1)) * sum_{j=0}^{i} 1[ y^(j)(t) in interval ]

    This allows plotting coverage convergence curves to visually verify
    calibration stability.

    Args:
        percent: target coverage percentage (e.g., 25, 50, 75)
        n_quart_test: number of independent test trajectories (Monte Carlo samples)
        n_past: number of past (context) time steps
        n_future: number of future (forecast) time steps
        n_samp_trajs: number of ensemble members to sample from the model
        dynamics_func: function to select a test trajectory from the data
        model: trained normalizing flow model
        data: full dataset tensor
        device: torch device

    Returns:
        n_true: tensor of shape (n_future,), total count of hits per step
        n_true_mtrx: tensor of shape (n_quart_test, n_future), running
                     average coverage after each test iteration
    """
    n_true = torch.zeros(n_future, dtype=torch.int32, device="cpu")
    n_true_mtrx = torch.zeros((n_quart_test, n_future), dtype=torch.float32, device="cpu")

    # Symmetric central interval bounds:
    #   p_low = 0.5 - P/200,  p_up = 0.5 + P/200
    # e.g., P=50 -> [0.25, 0.75], P=75 -> [0.125, 0.875]
    p_low, p_up = 0.5 - percent/200, 0.5 + percent/200

    for i in tqdm(range(n_quart_test), desc=f"Quantile {percent}%"):
        with torch.no_grad():
            ft = dynamics_func(data, n_past, n_future, random=True).cpu()

            past = ft[:n_past].to(device)
            past_rep = past.repeat(n_samp_trajs, 1)

            sam = model.sample(n_samp_trajs, past_rep)[0]
            sam = sam.cpu()

            del past, past_rep
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Compute lower and upper quantile bounds from ensemble
            ql = torch.quantile(sam.float(), p_low, dim=0)
            qu = torch.quantile(sam.float(), p_up, dim=0)

            # Check if the true future falls within the predicted interval
            tf = ft[n_past:].flatten()
            in_int = (tf > ql) & (tf < qu)

            # Accumulate hit counts and update running average coverage
            n_true += in_int.to(torch.int32)
            n_true_mtrx[i] = n_true.to(torch.float32) / (i + 1)

            del ft, sam, ql, qu, tf, in_int

    return n_true, n_true_mtrx


# ==================== PLOTTING FUNCTIONS ====================

def _draw_well_lines(ax, well_positions, color='gray'):
    """Overlay horizontal dashed lines at potential well y-positions.

    Lines are drawn at zorder=0 so they appear *behind* all trajectory and
    sample data.  Each line is labelled "Well 1", "Well 2", … so that it
    shows up in the subplot legend.

    Parameters
    ----------
    ax            : matplotlib Axes
    well_positions: iterable of floats — y-values of potential minima
    color         : line colour (use 'white' on dark magma heatmaps)
    """
    for j, wp in enumerate(well_positions):
        ax.axhline(y=wp, color=color, linestyle='--', linewidth=1.2,
                   alpha=0.7, zorder=0, label=f"Well {j + 1}")


def _y_label(landscape_name):
    """Return the appropriate y-axis label for the given landscape.

    For Alanine Dipeptide psi angle landscapes the y-axis represents ψ.
    For Alanine Dipeptide phi angle landscapes (or any other alanine landscape)
    the y-axis represents φ.
    For all other landscapes the axis represents a position coordinate.
    """
    if landscape_name:
        n = landscape_name.lower()
        if 'psi' in n:
            return r'angle $\psi$ (rad)'
        if 'alanine' in n:
            return r'angle $\varphi$ (rad)'
    return r'Position, $x$'


def plot_trajectory_with_band(ground_truth, samples, n_past, n_future, crps_score,
                              output_dir, timestamp, y_min=None, y_max=None,
                              well_positions=None, landscape_name=None,
                              plot_future_steps=None):
    """Plot trajectory with 95% confidence band.

    Parameters
    ----------
    plot_future_steps : int or None
        Number of future steps to *display* on the x-axis.  Must be <= n_future.
        Defaults to n_future.  Only the data passed to the plot is cropped —
        the underlying prediction arrays are never modified.
    """
    pfs = min(plot_future_steps, n_future) if plot_future_steps is not None else n_future

    # Slice display data only (underlying arrays unchanged)
    samples_disp = samples[:, :pfs]
    ground_truth_disp = ground_truth[:n_past + pfs]

    n_list_cpu = np.arange(0, n_past + pfs)
    n_list_future = n_list_cpu[n_past:]

    pred_median = np.median(samples_disp, axis=0)
    pred_025 = np.percentile(samples_disp, 2.5, axis=0)
    pred_975 = np.percentile(samples_disp, 97.5, axis=0)
    pred_25 = np.percentile(samples_disp, 25, axis=0)
    pred_75 = np.percentile(samples_disp, 75, axis=0)

    plt.figure(figsize=(10, 6))

    plt.plot(n_list_cpu, ground_truth_disp, color='black', linewidth=2, label='Ground Truth')
    plt.plot(n_list_future, pred_median, color='tab:orange', linewidth=2, label='Median Prediction')
    plt.fill_between(n_list_future, pred_025, pred_975,
                     color='tab:orange', alpha=0.3, label='95% Confidence Band')
    plt.fill_between(n_list_future, pred_25, pred_75,
                     color='tab:blue', alpha=0.3, label='50% Confidence Band')

    # Draw well-position guide lines before other artists so they sit behind.
    if well_positions:
        ax_cur = plt.gca()
        _draw_well_lines(ax_cur, well_positions, color='gray')

    plt.axvline(x=n_past, color='k', linestyle='--', alpha=0.3, label="Forecast Start")

    title = f"Forecast with 50%/95% Coverage (CRPS: {crps_score:.3f})"
    if landscape_name:
        title = f"{landscape_name} — {title}"
    plt.title(title, fontsize=14)
    plt.xlabel(r'Step, $N$', fontsize=12)
    plt.ylabel(_y_label(landscape_name), fontsize=12)

    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)

    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)

    path = os.path.join(output_dir, f'trajectory_band_{timestamp}.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_reliability_diagram(y_true, samples, output_dir, timestamp,
                             landscape_name=None):
    """Plot quantile coverage (reliability) diagram."""

    target_quantiles = np.linspace(0.05, 0.95, 19)
    observed_coverage = calculate_coverage_quantiles(y_true, samples, target_quantiles)

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], 'k--', label="Perfect Calibration")
    plt.plot(target_quantiles, observed_coverage, 'o-', color='purple', label="Model Calibration")

    title = "Quantile Coverage (Reliability Diagram)"
    if landscape_name:
        title = f"{landscape_name} — {title}"
    plt.title(title, fontsize=14)
    plt.xlabel("Expected Coverage (Target Quantile)", fontsize=12)
    plt.ylabel("Observed Coverage", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.5)

    path = os.path.join(output_dir, f'reliability_diagram_{timestamp}.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_histogram2d(samples, ground_truth, n_past, n_future, output_dir, timestamp,
                     y_min=None, y_max=None, well_positions=None, landscape_name=None,
                     plot_future_steps=None):
    """Plot 2D histogram of predictions.

    y_min / y_max default to None, in which case they are computed from the
    sample distribution (1st–99th percentile) expanded to include the full
    ground-truth range so the lime-green line is always visible.

    Parameters
    ----------
    plot_future_steps : int or None
        Number of future steps to display.  Defaults to n_future.
        Only the data passed to the plot is cropped — underlying arrays unchanged.
    """
    pfs = min(plot_future_steps, n_future) if plot_future_steps is not None else n_future

    # Slice for display only
    samples_disp = samples[:, :pfs]
    ground_truth_disp = ground_truth[:n_past + pfs]

    # Compute y-range from display data when not explicitly provided
    if y_min is None or y_max is None:
        y_lo = np.percentile(samples_disp, 1)
        y_hi = np.percentile(samples_disp, 99)
        # Expand to include the FULL ground truth display slice so the
        # conditioning history (past) is always visible — important for
        # landscapes like linear-Gaussian where the past can span a very
        # different value range than the predicted future.
        y_lo = min(y_lo, np.min(ground_truth_disp))
        y_hi = max(y_hi, np.max(ground_truth_disp))
        # Add a small margin so edge values are not cut off
        margin = 0.05 * (y_hi - y_lo) if y_hi > y_lo else 0.5
        if y_min is None:
            y_min = y_lo - margin
        if y_max is None:
            y_max = y_hi + margin

    n_list_cpu = np.arange(0, n_past + pfs)
    n_future_duplicated = np.tile(n_list_cpu[n_past:], (samples_disp.shape[0], 1))

    plt.figure(figsize=(10, 6))
    plt.hist2d(n_future_duplicated.flatten(), samples_disp.flatten(),
               bins=(pfs, 100),
               range=[[n_past, n_past + pfs], [y_min, y_max]],
               cmap='magma', density=True)
    plt.colorbar(label='Density')

    # Well lines in white so they're visible on the dark magma colormap.
    # Drawn before the ground-truth line so they sit behind it.
    if well_positions:
        ax_cur = plt.gca()
        _draw_well_lines(ax_cur, well_positions, color='white')

    plt.plot(n_list_cpu, ground_truth_disp, color='lime', linewidth=2, label='Ground Truth')
    plt.ylim(y_min, y_max)

    plt.legend()
    plt.xlim(0, n_past + pfs)
    plt.xlabel(r'Step, $N$', fontsize=12)
    plt.ylabel(_y_label(landscape_name), fontsize=12)

    title = 'Prediction Distribution Heatmap'
    if landscape_name:
        title = f"{landscape_name} — {title}"
    plt.title(title, fontsize=14)

    path = os.path.join(output_dir, f'histogram2d_{timestamp}.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_error_metrics(y_true, samples, n_past, n_future, output_dir, timestamp,
                       landscape_name=None, plot_future_steps=None):
    """Plot error metrics over time.

    Parameters
    ----------
    plot_future_steps : int or None
        Number of future steps to display.  Defaults to n_future.
        Only the data passed to the plot is cropped — underlying arrays unchanged.
    """
    pfs = min(plot_future_steps, n_future) if plot_future_steps is not None else n_future

    # Slice for display only
    y_true_disp = y_true[:pfs]
    samples_disp = samples[:, :pfs]

    n_list_future = np.arange(n_past, n_past + pfs)
    crps_series, mae_series = get_metrics_per_step(y_true_disp, samples_disp)

    plt.figure(figsize=(10, 6))

    plt.plot(n_list_future, mae_series, color='tab:blue', linestyle='--',
             linewidth=2, label='MAE (Median Error)')
    plt.plot(n_list_future, crps_series, color='tab:red', linestyle='-',
             linewidth=2, label='CRPS (Full Dist)')

    title = "Error Metrics over Time"
    if landscape_name:
        title = f"{landscape_name} — {title}"
    plt.title(title, fontsize=14)
    plt.xlabel("Future Step $N$", fontsize=12)
    plt.ylabel("Error", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    mean_crps = np.mean(crps_series)
    mean_mae = np.mean(mae_series)
    plt.text(0.05, 0.95, f"Avg CRPS: {mean_crps:.3f}\nAvg MAE: {mean_mae:.3f}",
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    path = os.path.join(output_dir, f'error_metrics_{timestamp}.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    return mean_crps, mean_mae


def plot_multi_trajectory_grid(model, data, norm_stats, n_past, n_future,
                               device, output_dir, timestamp, n_plots=9,
                               well_positions=None, landscape_name=None,
                               plot_future_steps=None, use_last_window=False):
    """Generate grid of trajectory predictions.

    Parameters
    ----------
    plot_future_steps : int or None
        Number of future steps to display per subplot.  Defaults to n_future.
        Only the data passed to each subplot is cropped — grid_data returned
        by this function always contains the full prediction arrays.

    Returns
    -------
    grid_data : list of (idx, real_traj_segment, samples_real) tuples
        Sampled data for each subplot, reusable by plot_histogram2d_grid.
        real_traj_segment is the ground-truth window used for this subplot.
        Full-length arrays are stored (not cropped to plot_future_steps).
    """

    # Normalize data if we have stats
    if norm_stats is not None:
        norm_data = (data - norm_stats['mean']) / norm_stats['std']
        mu_np = norm_stats['mean'].cpu().numpy()
        std_np = norm_stats['std'].cpu().numpy()
    else:
        norm_data = data
        mu_np = 0
        std_np = 1

    num_available = data.shape[0]
    traj_len = data.shape[1]
    n_extrp = n_past + n_future
    n_plots = min(n_plots, num_available)
    random_indices = np.random.choice(num_available, n_plots, replace=False)

    rows = int(np.ceil(np.sqrt(n_plots)))
    cols = int(np.ceil(n_plots / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    axes = axes.flatten() if n_plots > 1 else [axes]

    grid_data = []

    for i in range(len(axes)):
        ax = axes[i]
        if i >= n_plots:
            ax.axis('off')
            continue

        idx = random_indices[i]

        # Try up to 3 start positions to avoid spline instability and
        # to show representative segments rather than always t=0.
        samples_norm = None
        start = 0
        last_window_start = max(0, traj_len - n_extrp)
        for attempt in range(3):
            start = last_window_start if use_last_window else np.random.randint(0, max(1, traj_len - n_extrp + 1))
            past_norm = norm_data[idx, start:start + n_past].unsqueeze(0).to(device)
            past_repeat = past_norm.repeat(500, 1)
            with torch.no_grad():
                try:
                    samples_norm = model.sample(500, past_repeat)[0].cpu().numpy()
                    break  # success
                except AssertionError as e:
                    print(f"  WARNING: Sampling failed for trajectory {idx} start={start} "
                          f"(attempt {attempt+1}/3, spline instability): {e}")

        if samples_norm is None:
            ax.set_title(f"Trajectory {idx} [FAILED]")
            continue

        samples_real = (samples_norm * std_np) + mu_np
        real_traj = data[idx, start:start + n_extrp].cpu().numpy()

        median = np.median(samples_real, axis=0)
        lower = np.percentile(samples_real, 2.5, axis=0)
        upper = np.percentile(samples_real, 97.5, axis=0)
        lower_50 = np.percentile(samples_real, 25, axis=0)
        upper_50 = np.percentile(samples_real, 75, axis=0)

        # Compute shared y-range: include past+future GT and 1st–99th percentile
        # of samples.  This same range is stored in grid_data so plot_histogram2d_grid
        # can use identical axis limits for direct comparison.
        y_lo = min(np.percentile(samples_real, 1), np.min(real_traj))
        y_hi = max(np.percentile(samples_real, 99), np.max(real_traj))
        margin = 0.05 * (y_hi - y_lo) if y_hi > y_lo else 0.1
        y_lo -= margin
        y_hi += margin

        grid_data.append((idx, real_traj, samples_real, y_lo, y_hi))

        # Display window: crop to plot_future_steps for rendering only.
        # grid_data stores the full arrays so downstream callers are unaffected.
        pfs = min(plot_future_steps, n_future) if plot_future_steps is not None else n_future
        future_steps_disp = np.arange(n_past, n_past + pfs)
        all_steps_disp = np.arange(0, n_past + pfs)

        median_disp = np.median(samples_real[:, :pfs], axis=0)
        lower_disp = np.percentile(samples_real[:, :pfs], 2.5, axis=0)
        upper_disp = np.percentile(samples_real[:, :pfs], 97.5, axis=0)
        lower_50_disp = np.percentile(samples_real[:, :pfs], 25, axis=0)
        upper_50_disp = np.percentile(samples_real[:, :pfs], 75, axis=0)

        # Well lines at zorder=0 — drawn first so all trajectory data sits on top.
        if well_positions:
            _draw_well_lines(ax, well_positions, color='gray')

        ax.plot(all_steps_disp, real_traj[:n_past + pfs], 'k-', linewidth=1.5, label='Truth')
        ax.plot(future_steps_disp, median_disp, color='tab:orange', linewidth=2, label='Pred Median')
        ax.fill_between(future_steps_disp, lower_disp, upper_disp,
                        color='tab:orange', alpha=0.3, label='95% Band')
        ax.fill_between(future_steps_disp, lower_50_disp, upper_50_disp,
                        color='tab:blue', alpha=0.3, label='50% Band')

        ax.set_title(f"Traj {idx} (t={start})", fontsize=12)
        ax.set_xlabel(r'Step, $N$', fontsize=10)
        ax.set_ylabel(_y_label(landscape_name), fontsize=10)
        ax.set_ylim(y_lo, y_hi)
        ax.axvline(x=n_past, color='k', linestyle='--', alpha=0.3)

    axes[0].legend(loc='upper left')

    suptitle = "Trajectory Grid — Model Predictions"
    if landscape_name:
        suptitle = f"{landscape_name} — {suptitle}"
    plt.suptitle(suptitle, fontsize=14, y=1.01)
    plt.tight_layout()

    path = os.path.join(output_dir, f'trajectory_grid_{timestamp}.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    return grid_data


def plot_histogram2d_grid(grid_data, n_past, n_future, output_dir, timestamp,
                          well_positions=None, landscape_name=None,
                          plot_future_steps=None):
    """Grid of 2D histogram density plots for the same trajectories as trajectory_grid.

    The ground truth line is drawn in lime green, which is clearly visible on
    both the grey (no-data) region before the forecast start and across the full
    range of magma colormap densities (dark purple → bright yellow).

    Parameters
    ----------
    plot_future_steps : int or None
        Number of future steps to display per subplot.  Defaults to n_future.
        Only the data passed to each subplot is cropped — grid_data is not modified.
    """
    pfs = min(plot_future_steps, n_future) if plot_future_steps is not None else n_future

    n_plots = len(grid_data)
    rows = int(np.ceil(np.sqrt(n_plots)))
    cols = int(np.ceil(n_plots / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten() if n_plots > 1 else [axes]

    future_steps_disp = np.arange(n_past, n_past + pfs)
    all_steps_disp = np.arange(0, n_past + pfs)

    for i in range(len(axes)):
        ax = axes[i]
        if i >= n_plots:
            ax.axis('off')
            continue

        # grid_data entries are 5-tuples (idx, real_traj, samples_real, y_lo, y_hi)
        # when produced by plot_multi_trajectory_grid; fall back to 3-tuples for
        # backwards compatibility (e.g. when loading from saved npz files).
        item = grid_data[i]
        if len(item) == 5:
            idx, real_traj, samples_real, y_lo, y_hi = item
        else:
            idx, real_traj, samples_real = item
            # Compute y-range to include both the sample distribution AND the full
            # ground-truth trajectory (past + future).  This is critical for
            # landscapes like linear-Gaussian where the past conditioning steps
            # span a very different value range than the predicted future, which
            # would otherwise cause the lime-green history line to be clipped.
            y_lo = min(np.percentile(samples_real, 1), np.min(real_traj))
            y_hi = max(np.percentile(samples_real, 99), np.max(real_traj))
            margin = 0.05 * (y_hi - y_lo) if y_hi > y_lo else 0.1
            y_lo -= margin
            y_hi += margin

        # Slice samples for display only — grid_data is not modified
        samples_disp = samples_real[:, :pfs]

        # Tile time axis to match flattened display samples
        time_tiled = np.tile(future_steps_disp, (samples_disp.shape[0], 1))

        ax.hist2d(
            time_tiled.flatten(), samples_disp.flatten(),
            bins=(pfs, 100),
            range=[[n_past, n_past + pfs], [y_lo, y_hi]],
            cmap='magma', density=True,
        )

        # Well lines in white so they're legible on the dark magma colormap.
        # Drawn before the ground-truth line so they sit behind it.
        if well_positions:
            _draw_well_lines(ax, well_positions, color='white')

        # Lime green: highly visible on grey background AND across magma density scale
        ax.plot(all_steps_disp, real_traj[:n_past + pfs],
                color='lime', linewidth=1.5, label='Truth')
        ax.axvline(x=n_past, color='white', linestyle='--', alpha=0.5)

        ax.set_xlim(0, n_past + pfs)
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(f"Trajectory {idx}", fontsize=12)
        ax.set_xlabel(r'Step, $N$', fontsize=10)
        ax.set_ylabel(_y_label(landscape_name), fontsize=10)

    axes[0].legend(loc='upper left')

    suptitle = "Prediction Density Grid"
    if landscape_name:
        suptitle = f"{landscape_name} — {suptitle}"
    plt.suptitle(suptitle, fontsize=14, y=1.01)
    plt.tight_layout()

    path = os.path.join(output_dir, f'histogram2d_grid_{timestamp}.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_quantile_convergence(n_true_25, n_true_50, n_true_75, n_future, output_dir, timestamp,
                              landscape_name=None):
    """Plot quantile calibration convergence."""

    plt.figure(figsize=(10, 6))

    plt.axhline(y=25, color='r', linestyle='dashed', alpha=0.5)
    plt.axhline(y=50, color='r', linestyle='dashed', alpha=0.5)
    plt.axhline(y=75, color='r', linestyle='dashed', alpha=0.5)

    plt.plot(100 * np.mean([n_true_25[:, n] for n in range(n_future)], axis=0),
             label='Central 25%')
    plt.plot(100 * np.mean([n_true_50[:, n] for n in range(n_future)], axis=0),
             label='Central 50%')
    plt.plot(100 * np.mean([n_true_75[:, n] for n in range(n_future)], axis=0),
             label='Central 75%')

    plt.legend(loc='upper right', fontsize=11)
    plt.ylabel('Percent', fontsize=12)
    plt.xlabel('Iteration', fontsize=12)

    title = 'Quantile Calibration Convergence'
    if landscape_name:
        title = f"{landscape_name} — {title}"
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)

    path = os.path.join(output_dir, f'quantile_convergence_{timestamp}.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ==================== NEW HELPER FUNCTIONS ====================

def plot_trajectory_grid_from_data(grid_data, n_past, n_future, output_dir, timestamp,
                                   well_positions=None, landscape_name=None,
                                   plot_future_steps=None):
    """Regenerate the trajectory grid plot from pre-sampled grid_data.

    This is a pure plotting function that requires no model — it takes the
    already-sampled data stored in grid_data and produces the same 3×3 grid
    that plot_multi_trajectory_grid generates.  Used by the replot script.

    Parameters
    ----------
    grid_data : list of tuples
        Each tuple is (idx, real_traj, samples_real) or the extended
        (idx, real_traj, samples_real, y_lo, y_hi) format.
    plot_future_steps : int or None
        Number of future steps to display.  Defaults to n_future.
        Only the data passed to each subplot is cropped — grid_data unchanged.
    """
    pfs = min(plot_future_steps, n_future) if plot_future_steps is not None else n_future

    n_plots = len(grid_data)
    rows = int(np.ceil(np.sqrt(n_plots)))
    cols = int(np.ceil(n_plots / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten() if n_plots > 1 else [axes]

    for i in range(len(axes)):
        ax = axes[i]
        if i >= n_plots:
            ax.axis('off')
            continue

        item = grid_data[i]
        if len(item) == 5:
            idx, real_traj, samples_real, y_lo, y_hi = item
        else:
            idx, real_traj, samples_real = item
            y_lo = min(np.percentile(samples_real, 1), np.min(real_traj))
            y_hi = max(np.percentile(samples_real, 99), np.max(real_traj))
            margin = 0.05 * (y_hi - y_lo) if y_hi > y_lo else 0.1
            y_lo -= margin
            y_hi += margin

        # Slice for display only — grid_data not modified
        samples_disp = samples_real[:, :pfs]

        median = np.median(samples_disp, axis=0)
        lower = np.percentile(samples_disp, 2.5, axis=0)
        upper = np.percentile(samples_disp, 97.5, axis=0)
        lower_50 = np.percentile(samples_disp, 25, axis=0)
        upper_50 = np.percentile(samples_disp, 75, axis=0)

        future_steps_disp = np.arange(n_past, n_past + pfs)
        all_steps_disp = np.arange(0, n_past + pfs)

        # Well lines at zorder=0 — drawn first so all trajectory data sits on top.
        if well_positions:
            _draw_well_lines(ax, well_positions, color='gray')

        ax.plot(all_steps_disp, real_traj[:n_past + pfs], 'k-', linewidth=1.5, label='Truth')
        ax.plot(future_steps_disp, median, color='tab:orange', linewidth=2, label='Pred Median')
        ax.fill_between(future_steps_disp, lower, upper,
                        color='tab:orange', alpha=0.3, label='95% Band')
        ax.fill_between(future_steps_disp, lower_50, upper_50,
                        color='tab:blue', alpha=0.3, label='50% Band')

        ax.set_title(f"Trajectory {idx}", fontsize=12)
        ax.set_xlabel(r'Step, $N$', fontsize=10)
        ax.set_ylabel(_y_label(landscape_name), fontsize=10)
        ax.set_ylim(y_lo, y_hi)
        ax.axvline(x=n_past, color='k', linestyle='--', alpha=0.3)

    axes[0].legend(loc='upper left')

    suptitle = "Trajectory Grid — Model Predictions"
    if landscape_name:
        suptitle = f"{landscape_name} — {suptitle}"
    plt.suptitle(suptitle, fontsize=14, y=1.01)
    plt.tight_layout()

    path = os.path.join(output_dir, f'trajectory_grid_{timestamp}.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def save_test_trajectories(ground_truth, sampled_trajectories, grid_data,
                           n_past, n_future, output_dir, timestamp,
                           fullset_results=None):
    """Save all testing trajectories to disk for later replotting.

    Stores a single-trajectory evaluation result plus the full grid of
    trajectories (with their ensemble samples) in a .npz archive so that
    all test plots can be regenerated without re-running the model.

    File layout
    -----------
    ground_truth        : (n_past + n_future,)  — single test trajectory GT
    samples             : (n_samples, n_future) — ensemble predictions for GT
    grid_indices        : (N_grid,)             — trajectory indices in data
    grid_ground_truths  : (N_grid, n_past+n_future) — GT for each grid subplot
    grid_samples        : (N_grid, n_samples, n_future) — ensemble per subplot
    grid_y_lo           : (N_grid,)             — shared y-axis lower bound
    grid_y_hi           : (N_grid,)             — shared y-axis upper bound
    n_past              : scalar
    n_future            : scalar
    coverage_per_step_20/60/80 : (n_future,)  — per-step coverage fractions
                                                 from full-set evaluation
    fullset_n_evaluated : scalar              — number of evaluated trajectories
    fullset_traj_indices          : (N_eval,)              — evaluated trajectory IDs
    fullset_window_starts         : (N_eval,)              — start index used per trajectory
    fullset_window_ground_truths  : (N_eval, n_past+n_future)
    fullset_window_samples        : (N_eval, n_samples, n_future)

    Parameters
    ----------
    fullset_results : optional dict returned by evaluate_full_testset().
                      When provided, per-step coverage data is saved so that
                      replot_test_results.py can regenerate the quantile
                      coverage plot without re-running the model.

    Returns
    -------
    save_path : str — path to the saved .npz file
    """
    grid_indices = np.array([d[0] for d in grid_data])
    grid_ground_truths = np.array([d[1] for d in grid_data])
    grid_samples_arr = np.array([d[2] for d in grid_data])

    save_dict = {
        'ground_truth': ground_truth,
        'samples': sampled_trajectories,
        'grid_indices': grid_indices,
        'grid_ground_truths': grid_ground_truths,
        'grid_samples': grid_samples_arr,
        'n_past': np.array([n_past]),
        'n_future': np.array([n_future]),
    }

    # Include per-subplot y-limits if present (5-tuple format)
    if len(grid_data[0]) == 5:
        save_dict['grid_y_lo'] = np.array([d[3] for d in grid_data])
        save_dict['grid_y_hi'] = np.array([d[4] for d in grid_data])

    # Save full-test-set coverage data so the replot script can regenerate
    # the quantile coverage plot without needing to re-run the model.
    if fullset_results is not None:
        for pct in [20, 50, 60, 80, 90]:
            if pct in fullset_results['coverage_per_step']:
                save_dict[f'coverage_per_step_{pct}'] = \
                    fullset_results['coverage_per_step'][pct]
        save_dict['fullset_n_evaluated'] = \
            np.array([fullset_results['n_evaluated']])
        # Per-step MAE and CRPS from the full test set for use in compare_models
        if 'mae_per_step' in fullset_results:
            save_dict['fullset_mae_per_step']  = fullset_results['mae_per_step']
        if 'crps_per_step' in fullset_results:
            save_dict['fullset_crps_per_step'] = fullset_results['crps_per_step']
        # Save full-set evaluated windows and ensemble predictions so downstream
        # analysis can be performed without re-running inference.
        if 'traj_indices' in fullset_results:
            save_dict['fullset_traj_indices'] = fullset_results['traj_indices']
        if 'window_starts' in fullset_results:
            save_dict['fullset_window_starts'] = fullset_results['window_starts']
        if 'window_ground_truths' in fullset_results:
            save_dict['fullset_window_ground_truths'] = fullset_results['window_ground_truths']
        if 'window_samples' in fullset_results:
            save_dict['fullset_window_samples'] = fullset_results['window_samples']

    save_path = os.path.join(output_dir, f'test_trajectories_{timestamp}.npz')
    np.savez(save_path, **save_dict)
    print(f"Saved test trajectories to: {save_path}")
    return save_path


def plot_empirical_coverage(grid_data, n_past, n_future, output_dir, timestamp,
                            landscape_name=None):
    """Plot empirical σ-coverage vs time and compare to Gaussian expectation.

    For each test trajectory i and time step t, checks whether the ground
    truth falls within k * pred_std of the ensemble mean, where k = 1, 2, 3.
    The fraction of trajectories within each band is plotted as a solid line
    and compared to the ideal Gaussian coverage levels (dashed lines).

    Expected Gaussian coverage
    --------------------------
    1σ : 68.27%   2σ : 95.45%   3σ : 99.73%

    The time-averaged empirical coverage for each sigma level is printed to
    stdout and shown in the plot legend.

    Parameters
    ----------
    grid_data : list of (idx, real_traj, samples_real[, y_lo, y_hi]) tuples
    n_past    : number of conditioning steps (used to slice ground truth)
    n_future  : number of forecast steps
    """
    # Expected Gaussian coverage fractions for k-sigma intervals
    GAUSSIAN_COVERAGE = {1: 0.6827, 2: 0.9545, 3: 0.9973}
    SIGMA_COLORS = {1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green'}

    # Collect ensemble mean / std per trajectory
    y_trues, pred_means, pred_stds = [], [], []
    for item in grid_data:
        idx, real_traj, samples_real = item[0], item[1], item[2]
        y_true = real_traj[n_past:]                    # (n_future,)
        pred_mean = np.mean(samples_real, axis=0)      # (n_future,)
        pred_std = np.std(samples_real, axis=0)        # (n_future,)
        y_trues.append(y_true)
        pred_means.append(pred_mean)
        pred_stds.append(pred_std)

    y_trues = np.array(y_trues)      # (N, n_future)
    pred_means = np.array(pred_means)
    pred_stds = np.array(pred_stds)

    n_trajs = len(grid_data)
    time_steps = np.arange(n_future)

    print(f"\n{'='*50}")
    print("Empirical coverage vs. Gaussian expectation:")
    print(f"  (N={n_trajs} test trajectories)")

    fig, ax = plt.subplots(figsize=(10, 6))

    for k in [1, 2, 3]:
        exp_cov = GAUSSIAN_COVERAGE[k]
        color = SIGMA_COLORS[k]

        # within_k[i, t] = True if |y_true[i,t] - mean[i,t]| <= k * std[i,t]
        within_k = np.abs(y_trues - pred_means) <= k * pred_stds  # (N, n_future)
        coverage_k = np.mean(within_k, axis=0)                     # (n_future,)
        time_avg = float(np.mean(coverage_k))

        print(f"  {k}σ: time-avg empirical = {time_avg:.4f}  "
              f"(Gaussian expected = {exp_cov:.4f})")

        ax.plot(time_steps, coverage_k, color=color, linewidth=2,
                label=rf'Empirical {k}σ  (avg={time_avg:.3f})')
        ax.axhline(y=exp_cov, color=color, linestyle='--', alpha=0.6,
                   label=rf'Expected {k}σ  ({exp_cov:.4f})')

    print('='*50)

    ax.set_xlabel('Future Step', fontsize=12)
    ax.set_ylabel('Fraction Within Predicted Band', fontsize=12)
    base_title = f'Empirical Coverage vs. Gaussian Expectation  (N={n_trajs} trajectories)'
    ax.set_title(
        f"{landscape_name} — {base_title}" if landscape_name else base_title,
        fontsize=14
    )
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, f'empirical_coverage_{timestamp}.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def compute_multi_trajectory_metrics(grid_data, n_past):
    """Compute CRPS and MAE for each grid trajectory, then average.

    CRPS and MAE are first evaluated per trajectory (using the full ensemble
    for CRPS and the ensemble median as the point forecast for MAE), and
    then the per-trajectory scores are averaged.  This gives a more robust
    summary than computing metrics from a single test trajectory.

    Note: ensemble_crps already uses ALL ensemble members (not just the
    median) and is therefore a proper probabilistic accuracy measure.

    Parameters
    ----------
    grid_data : list of (idx, real_traj, samples_real[, ...]) tuples
    n_past    : number of conditioning steps

    Returns
    -------
    mean_crps : float — CRPS averaged over trajectories
    mean_mae  : float — MAE averaged over trajectories
    crps_list : list of per-trajectory CRPS values
    mae_list  : list of per-trajectory MAE values
    """
    crps_list, mae_list = [], []
    for item in grid_data:
        idx, real_traj, samples_real = item[0], item[1], item[2]
        y_true = real_traj[n_past:]                         # (n_future,)
        crps_list.append(ensemble_crps(y_true, samples_real))
        pred_median = np.median(samples_real, axis=0)       # (n_future,)
        mae_list.append(float(np.mean(np.abs(pred_median - y_true))))

    return float(np.mean(crps_list)), float(np.mean(mae_list)), crps_list, mae_list


def evaluate_full_testset(model, data, norm_stats, n_past, n_future,
                          device, n_samples=500, use_last_window=False):
    """
    Evaluate the model over the entire test set in a single pass, computing
    CRPS, MAE, and empirical quantile coverage for each trajectory.

    For each trajectory, a window of length (n_past + n_future) is selected
    (randomly by default, or from the tail when use_last_window=True),
    n_samples forecast trajectories are drawn from the model, and
    per-trajectory CRPS, MAE, and per-step coverage checks are accumulated.
    After all trajectories are processed the per-trajectory scores are
    averaged to give the final reported metrics.

    Quantile intervals evaluated
    ----------------------------
    20% central interval : 40th–60th percentile of predictions
    60% central interval : 20th–80th percentile of predictions
    80% central interval : 10th–90th percentile of predictions

    Parameters
    ----------
    model      : trained normalizing flow model
    data       : (N, T) tensor of test trajectories in original units
    norm_stats : dict with 'mean' and 'std' tensors, or None
    n_past     : number of conditioning steps
    n_future   : number of forecast steps
    device     : torch device
    n_samples  : number of ensemble members to sample per trajectory
    use_last_window : if True, evaluate the last window of each trajectory

    Returns
    -------
    dict with keys:
        crps_list         : per-trajectory CRPS values
        mae_list          : per-trajectory MAE values
        mean_crps         : mean CRPS across all trajectories
        std_crps          : std of CRPS across all trajectories
        mean_mae          : mean MAE across all trajectories
        std_mae           : std of MAE across all trajectories
        coverage_per_step : dict {20: (n_future,), 60: (n_future,), 80: (n_future,)}
                            empirical coverage fractions per step, averaged across trajectories
        traj_indices      : (N_eval,) evaluated trajectory indices in the dataset
        window_starts     : (N_eval,) chosen start index per trajectory
        window_ground_truths : (N_eval, n_past+n_future) selected GT windows
        window_samples    : (N_eval, n_samples, n_future) ensemble predictions
        n_evaluated       : number of trajectories successfully evaluated
        n_failed          : number of trajectories skipped due to spline instability
    """
    # Central interval bounds: (lower_quantile, upper_quantile)
    QUANTILE_INTERVALS = {
        20: (0.40, 0.60),  # 40th–60th percentile → expected 20% coverage
        50: (0.25, 0.75),  # 25th–75th percentile → expected 50% coverage (CI50)
        60: (0.20, 0.80),  # 20th–80th percentile → expected 60% coverage
        80: (0.10, 0.90),  # 10th–90th percentile → expected 80% coverage
        90: (0.05, 0.95),  # 5th–95th percentile  → expected 90% coverage (CI90)
    }

    if norm_stats is not None:
        norm_data = (data - norm_stats['mean']) / norm_stats['std']
        mu_np  = norm_stats['mean'].cpu().numpy()
        std_np = norm_stats['std'].cpu().numpy()
    else:
        norm_data = data
        mu_np  = 0.0
        std_np = 1.0

    n_trajs  = data.shape[0]
    traj_len = data.shape[1]
    n_extrp  = n_past + n_future

    crps_list = []
    mae_list  = []
    # Accumulates (n_future,) float arrays per trajectory for each interval
    coverage_within = {pct: [] for pct in QUANTILE_INTERVALS}
    # Per-step accumulators (one (n_future,) array per trajectory)
    mae_per_step_list  = []
    crps_per_step_list = []
    # Store all evaluated windows and all ensemble predictions.
    traj_indices = []
    window_starts = []
    window_ground_truths = []
    window_samples = []
    n_failed = 0

    print(f"\nEvaluating full test set ({n_trajs} trajectories, "
          f"{n_samples} samples each)...")

    for i in tqdm(range(n_trajs), desc="Full-set evaluation"):
        # Window start position within this trajectory
        if use_last_window:
            start = max(0, traj_len - n_extrp)
        else:
            start = np.random.randint(0, max(1, traj_len - n_extrp + 1))

        # Build normalised context and replicate for batch sampling
        past_norm   = norm_data[i, start:start + n_past].unsqueeze(0).to(device)
        past_repeat = past_norm.repeat(n_samples, 1)

        with torch.no_grad():
            try:
                samples_norm = model.sample(n_samples, past_repeat)[0].cpu().numpy()
            except AssertionError as e:
                n_failed += 1
                del past_norm, past_repeat
                continue  # Skip on spline instability

        # De-normalise back to physical units
        samples_real = (samples_norm * std_np) + mu_np
        real_window = data[i, start:start + n_extrp].cpu().numpy()
        real_future  = data[i, start + n_past:start + n_extrp].cpu().numpy()

        # CRPS: proper probabilistic score over the full ensemble
        crps_list.append(ensemble_crps(real_future, samples_real))

        # MAE: median point forecast absolute error
        pred_median = np.median(samples_real, axis=0)
        mae_list.append(float(np.mean(np.abs(pred_median - real_future))))

        # Per-step MAE and CRPS (accumulated for full-testset metric curves)
        mae_per_step_list.append(np.abs(pred_median - real_future))   # (n_future,)
        n_samp_fs = samples_real.shape[0]
        mae_term  = np.mean(np.abs(samples_real - real_future[None, :]), axis=0)
        fs_sorted = np.sort(samples_real, axis=0)
        diff      = fs_sorted[1:] - fs_sorted[:-1]                    # (n_samp-1, n_future)
        weights   = np.arange(1, n_samp_fs) * np.arange(n_samp_fs - 1, 0, -1)
        spread    = np.sum(diff * weights[:, None], axis=0) / (n_samp_fs ** 2)
        crps_per_step_list.append(mae_term - spread)                   # (n_future,)

        # Per-step coverage: check whether ground truth falls within each
        # central quantile interval at every forecast time step
        for pct, (q_lo, q_hi) in QUANTILE_INTERVALS.items():
            lo     = np.percentile(samples_real, q_lo * 100, axis=0)  # (n_future,)
            hi     = np.percentile(samples_real, q_hi * 100, axis=0)  # (n_future,)
            within = (real_future >= lo) & (real_future <= hi)         # (n_future,) bool
            coverage_within[pct].append(within.astype(float))

        traj_indices.append(i)
        window_starts.append(start)
        window_ground_truths.append(real_window)
        window_samples.append(samples_real)

        del past_norm, past_repeat

    n_evaluated = len(crps_list)
    if n_failed:
        print(f"  WARNING: {n_failed} trajectories skipped due to spline instability.")
    print(f"  Evaluated {n_evaluated} / {n_trajs} trajectories.")
    if n_evaluated == 0:
        raise RuntimeError("All trajectories failed during sampling. "
                           "Model may be numerically unstable.")

    mean_crps = float(np.mean(crps_list))
    std_crps  = float(np.std(crps_list))
    mean_mae  = float(np.mean(mae_list))
    std_mae   = float(np.std(mae_list))

    # Average per-step coverage across trajectories → (n_future,) per interval
    coverage_per_step = {
        pct: np.mean(coverage_within[pct], axis=0)
        for pct in QUANTILE_INTERVALS
    }

    # Average per-step MAE and CRPS across trajectories → (n_future,)
    mae_per_step  = np.mean(mae_per_step_list,  axis=0)
    crps_per_step = np.mean(crps_per_step_list, axis=0)

    return {
        'crps_list':         crps_list,
        'mae_list':          mae_list,
        'mean_crps':         mean_crps,
        'std_crps':          std_crps,
        'mean_mae':          mean_mae,
        'std_mae':           std_mae,
        'coverage_per_step': coverage_per_step,
        'mae_per_step':      mae_per_step,
        'crps_per_step':     crps_per_step,
        'traj_indices':      np.array(traj_indices, dtype=np.int64),
        'window_starts':     np.array(window_starts, dtype=np.int64),
        'window_ground_truths': np.array(window_ground_truths),
        'window_samples':    np.array(window_samples),
        'n_evaluated':       n_evaluated,
        'n_failed':          n_failed,
    }


def plot_empirical_coverage_quantiles(coverage_per_step, n_future, n_trajs,
                                      output_dir, timestamp, landscape_name=None):
    """
    Plot empirical quantile coverage vs expected coverage levels over the
    forecast horizon.

    For each of the three central quantile intervals (20%, 60%, 80%), a
    solid line shows the fraction of test trajectories whose ground truth
    fell within the predicted interval at each forecast time step.  Dashed
    horizontal reference lines mark the ideal (expected) coverage.

    A well-calibrated model produces solid lines close to their corresponding
    dashed reference lines.

    Parameters
    ----------
    coverage_per_step : dict mapping interval_pct -> (n_future,) array of
                        per-step empirical coverage fractions, averaged over
                        all test trajectories
    n_future          : number of forecast steps
    n_trajs           : number of test trajectories evaluated
    output_dir        : directory to save the figure
    timestamp         : string used in the output filename
    """
    INTERVAL_COLORS = {20: 'tab:blue', 60: 'tab:orange', 80: 'tab:green'}

    time_steps = np.arange(n_future)

    print(f"\n{'='*50}")
    print("Empirical quantile coverage vs. expected:")
    print(f"  (N={n_trajs} test trajectories)")

    fig, ax = plt.subplots(figsize=(10, 6))

    for pct in [20, 60, 80]:
        coverage_k = coverage_per_step[pct]      # (n_future,)
        time_avg   = float(np.mean(coverage_k))
        color      = INTERVAL_COLORS[pct]
        expected   = pct / 100.0

        print(f"  {pct}% interval: time-avg empirical = {time_avg:.4f}  "
              f"(expected = {expected:.4f})")

        ax.plot(time_steps, coverage_k, color=color, linewidth=2,
                label=rf'Empirical {pct}%  (avg={time_avg:.3f})')
        ax.axhline(y=expected, color=color, linestyle='--', alpha=0.6,
                   label=rf'Expected {pct}%  ({expected:.2f})')

    print('='*50)

    ax.set_xlabel('Future Step', fontsize=12)
    ax.set_ylabel('Fraction Within Predicted Interval', fontsize=12)
    base_title = f'Empirical Quantile Coverage  (N={n_trajs} trajectories)'
    ax.set_title(
        f"{landscape_name} — {base_title}" if landscape_name else base_title,
        fontsize=14
    )
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, f'empirical_coverage_{timestamp}.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def main():
    args = parse_args()

    # ------------------------------------------------------------------ #
    # SDE validation: check required args early.                          #
    # ------------------------------------------------------------------ #
    if args.data_format == 'sde':
        if args.sde_model_key is None:
            raise ValueError(
                "--sde_model_key is required when --data_format sde.  "
                "Example: --sde_model_key sw_sle_em"
            )
    elif args.data_path is None:
        raise ValueError(
            "--data_path is required when --data_format is not 'sde'."
        )

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = setup_device(args.device)
    if _HAS_SEABORN:
        sns.set_theme()
        sns.set_palette("bright")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Determine landscape name for plot titles.
    # For SDE: append (memory) / (no memory) suffix derived from has_memory.
    landscape_name = args.landscape
    if landscape_name is None:
        if args.data_format == 'sde':
            mem_suffix = '(memory)' if 'gle' in args.sde_model_key else '(no memory)'
            landscape_name = f"{args.sde_model_key} {mem_suffix}"
        else:
            landscape_name = os.path.splitext(os.path.basename(args.data_path))[0]

    # Determine how many future steps to display on plots.
    # Defaults to n_future (full prediction horizon) so existing behaviour
    # is preserved unless --plot_future_steps is explicitly set.
    # NOTE: n_future is not yet available here; set after args are parsed.

    # Model architecture
    n_past = args.n_past
    n_future = args.n_future
    n_extrp = n_past + n_future
    context_size = n_past
    latent_size = n_future

    plot_future_steps = args.plot_future_steps if args.plot_future_steps is not None else n_future
    
    # Load architecture hyperparameters from the saved training config so that
    # the model constructed here always matches the one that was trained.
    # Defaults mirror architecture.py so the script still works without --config_path.
    flow_blocks = 6
    hidden_units = 64
    hidden_layers = '1,2'
    tail_bound = 30.0
    _cfg = {}
    if args.config_path is not None:
        with open(args.config_path) as _cfg_f:
            _cfg = json.load(_cfg_f)
        flow_blocks   = _cfg.get('flow_blocks',   flow_blocks)
        hidden_units  = _cfg.get('hidden_units',  hidden_units)
        hidden_layers = _cfg.get('hidden_layers', hidden_layers)
        tail_bound    = _cfg.get('tail_bound',    tail_bound)
        print(f"\nLoaded architecture from config: flow_blocks={flow_blocks}, "
              f"hidden_units={hidden_units}, hidden_layers={hidden_layers}, "
              f"tail_bound={tail_bound}")
    else:
        print(f"\nNo --config_path provided; using architecture defaults: "
              f"flow_blocks={flow_blocks}, hidden_units={hidden_units}, "
              f"hidden_layers={hidden_layers}, tail_bound={tail_bound}")
    if args.tail_bound != 30.0:
        tail_bound = args.tail_bound
    hidden_layers_list = tuple(int(x) for x in str(hidden_layers).split(','))

    # Resolve model_variant: CLI flag > config JSON > fallback 'ar'.
    config_model_variant = _cfg.get('model_variant', 'ar') if args.config_path else None
    if args.model_variant is not None:
        # Explicit CLI override — warn if it differs from the saved config.
        if config_model_variant is not None and args.model_variant != config_model_variant:
            print(f"WARNING: --model_variant CLI value {args.model_variant!r} differs from "
                  f"config model_variant={config_model_variant!r}. Using CLI value.")
        model_variant = args.model_variant
    elif config_model_variant is not None:
        model_variant = config_model_variant
    else:
        model_variant = 'ar'
        print("No --config_path and no --model_variant provided; "
              "defaulting to 'ar' (old-checkpoint fallback).")
    print(f"Model variant: {model_variant}")

    # Create and load model
    print("\nLoading model...")
    if model_variant == 'ar':
        model = create_nfm(device, latent_size, context_size,
                           K=flow_blocks, hidden_units=hidden_units,
                           hidden_layers_list=hidden_layers_list,
                           tail_bound=tail_bound)
    elif model_variant == 'ar_encoder_full':
        model = preset_stage1(device, n_past=n_past, n_future=n_future, past_dim=1)
    elif model_variant == 'ar_encoder_light':
        model = preset_stage2(device, n_past=n_past, n_future=n_future, past_dim=1)
    else:
        raise ValueError(f"Unknown model_variant: {model_variant!r}")

    # weights_only=False: standard state dicts contain only tensors and are safe;
    # False is required for compatibility with checkpoints saved by older PyTorch.
    state_dict = torch.load(args.model_path, map_location=device, weights_only=False)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            f"Checkpoint is incompatible with model_variant={model_variant!r}, "
            f"n_past={n_past}, n_future={n_future}, "
            f"model_path={args.model_path!r}. "
            "Check --model_variant, --config_path, and checkpoint path."
        ) from e
    model.eval()
    print(f"  Loaded from: {args.model_path}")
    
    # Load normalization stats
    norm_stats = load_normalization_stats(args.norm_stats_path)
    if norm_stats:
        print(f"  Loaded normalization stats from: {args.norm_stats_path}")
    
    # Load test data.
    # For SDE format: reconstruct the deterministic split (same seed as training)
    # and evaluate only on the held-out 20% test trajectories.
    sde_model_key = None
    sde_has_memory = None
    if args.data_format == 'sde':
        from sde_loader import load_sde_dataset, split_sde_dataset
        mem_label = '(memory)' if 'gle' in args.sde_model_key else '(no memory)'
        print(f"\nLoading SDE test split: {args.sde_model_key} {mem_label}")

        sde_dataset = load_sde_dataset(
            model_key=args.sde_model_key,
            data_dir=args.sde_data_dir,
            test_id=args.sde_test_id,
            skip=args.sde_skip,
        )
        # Use the same random_seed as training (args.seed) so the test partition
        # is always the same 20% that training never saw.
        sde_split = split_sde_dataset(sde_dataset, random_seed=args.seed)

        x_test_np     = sde_split['x_test']   # (N_test, T)
        sde_model_key  = sde_split['model_key']
        sde_has_memory = sde_split['has_memory']

        reshaped_data = torch.tensor(x_test_np, dtype=torch.float32)
        raw_data      = x_test_np  # used only for quantile tests, same format works

        print(f"  Test trajectories: {reshaped_data.shape[0]}  "
              f"  model_key={sde_model_key}  has_memory={sde_has_memory}")
    else:
        reshaped_data, raw_data = load_data(args.data_path, args.data_format)

    # Compute normalization from test data if not provided
    if norm_stats is None:
        train_mean = reshaped_data.mean()
        train_std = reshaped_data.std()
        norm_stats = {'mean': train_mean, 'std': train_std}
        print(f"  Computed normalization from test data: mean={train_mean:.4f}, std={train_std:.4f}")
    
    train_mean_cpu = norm_stats['mean'].cpu().numpy()
    train_std_cpu = norm_stats['std'].cpu().numpy()
    
    norm_data = (reshaped_data - norm_stats['mean']) / norm_stats['std']
    
    # ==================== BASIC EVALUATION ====================
    print("\n" + "="*50)
    print("Running basic evaluation...")
    print("="*50)
    
    # Select a test trajectory and sample (retry on spline instability)
    sampled_trajectories = None
    ground_truth = None
    for attempt in range(5):
        full_traj = get_trajectory(
            reshaped_data,
            n_extrp,
            random=not args.basic_eval_use_last_window,
            from_end=args.basic_eval_use_last_window
        )
        full_traj = torch.unsqueeze(full_traj, dim=0)

        past_traj = full_traj[:, :n_past]
        past_norm = (past_traj[0][:n_past] - norm_stats['mean']) / norm_stats['std']
        past_traj_repeat = past_norm.repeat(args.n_samples, 1).to(device)

        with torch.no_grad():
            try:
                sampled_trajectories_model = model.sample(args.n_samples, past_traj_repeat)[0]
                sampled_trajectories = sampled_trajectories_model.cpu().numpy()
                sampled_trajectories = (sampled_trajectories * train_std_cpu) + train_mean_cpu
                ground_truth = full_traj.cpu().numpy().flatten()
                break
            except AssertionError as e:
                print(f"  WARNING: Sampling failed (attempt {attempt+1}/5, spline instability): {e}")

    if sampled_trajectories is None:
        raise RuntimeError("Sampling failed for all 5 attempts. Model may be numerically unstable.")

    y_true_future = ground_truth[n_past:]
    
    # Calculate CRPS
    crps_score = ensemble_crps(y_true_future, sampled_trajectories)
    print(f"\nCRPS Score: {crps_score:.4f}")
    
    # Generate basic plots
    plot_trajectory_with_band(ground_truth, sampled_trajectories, n_past, n_future,
                              crps_score, args.output_dir, timestamp,
                              well_positions=args.well_positions,
                              landscape_name=landscape_name,
                              plot_future_steps=plot_future_steps)

    plot_reliability_diagram(y_true_future, sampled_trajectories, args.output_dir, timestamp,
                             landscape_name=landscape_name)

    plot_histogram2d(sampled_trajectories, ground_truth, n_past, n_future,
                     args.output_dir, timestamp,
                     well_positions=args.well_positions,
                     landscape_name=landscape_name,
                     plot_future_steps=plot_future_steps)

    mean_crps, mean_mae = plot_error_metrics(y_true_future, sampled_trajectories,
                                              n_past, n_future, args.output_dir, timestamp,
                                              landscape_name=landscape_name,
                                              plot_future_steps=plot_future_steps)

    # Multi-trajectory grid with de-normalization; returns sampled data for reuse.
    # grid_data is a list of 5-tuples: (idx, real_traj, samples_real, y_lo, y_hi)
    grid_data = plot_multi_trajectory_grid(model, reshaped_data, norm_stats, n_past, n_future,
                                           device, args.output_dir, timestamp,
                                           well_positions=args.well_positions,
                                           landscape_name=landscape_name,
                                           plot_future_steps=plot_future_steps,
                                           use_last_window=args.full_eval_use_last_window)

    # 2D histogram grid — same trajectories, no re-sampling.  Shared y-limits
    # from grid_data ensure the two grids are directly comparable.
    plot_histogram2d_grid(grid_data, n_past, n_future, args.output_dir, timestamp,
                          well_positions=args.well_positions,
                          landscape_name=landscape_name,
                          plot_future_steps=plot_future_steps)

    # ==================== AGGREGATED METRICS (grid, 9 trajectories) ====================
    # Compute CRPS and MAE for each grid trajectory, then average.
    # ensemble_crps already uses all ensemble members (not just the median),
    # so this is a proper multi-trajectory probabilistic accuracy summary.
    print("\n" + "="*50)
    print(f"Aggregated metrics over {len(grid_data)} grid trajectories:")
    mean_crps_agg, mean_mae_agg, crps_per_traj, mae_per_traj = \
        compute_multi_trajectory_metrics(grid_data, n_past)
    print(f"  Mean CRPS : {mean_crps_agg:.4f}  (per-traj: {[f'{v:.3f}' for v in crps_per_traj]})")
    print(f"  Mean MAE  : {mean_mae_agg:.4f}  (per-traj: {[f'{v:.3f}' for v in mae_per_traj]})")
    print("="*50)

    # ==================== FULL TEST SET EVALUATION ====================
    # Evaluate every trajectory in the test set (not just the 9 grid ones).
    # For each trajectory a random window is selected, n_samples forecast
    # trajectories are drawn, and CRPS/MAE are computed per trajectory.
    # The same pass also accumulates quantile coverage for the coverage plot.
    print("\n" + "="*50)
    print(f"Full test-set evaluation ({reshaped_data.shape[0]} trajectories)...")
    print(f"Window selection mode: {'last window per trajectory' if args.full_eval_use_last_window else 'random window per trajectory'}")
    print("="*50)
    fullset_results = evaluate_full_testset(
        model, reshaped_data, norm_stats, n_past, n_future,
        device, n_samples=args.n_samples,
        use_last_window=args.full_eval_use_last_window
    )
    print(f"\nFull test-set CRPS : {fullset_results['mean_crps']:.4f} "
          f"± {fullset_results['std_crps']:.4f}")
    print(f"Full test-set MAE  : {fullset_results['mean_mae']:.4f} "
          f"± {fullset_results['std_mae']:.4f}")
    print(f"(averaged over {fullset_results['n_evaluated']} trajectories, "
          f"{args.n_samples} samples each)")
    print("="*50)

    # ==================== EMPIRICAL QUANTILE COVERAGE PLOT ====================
    # Uses per-step coverage data already computed in evaluate_full_testset().
    # 20% / 60% / 80% central quantile intervals over the full test set.
    plot_empirical_coverage_quantiles(
        fullset_results['coverage_per_step'],
        n_future, fullset_results['n_evaluated'],
        args.output_dir, timestamp,
        landscape_name=landscape_name,
    )

    # ==================== SAVE TEST TRAJECTORIES ====================
    # Save all trajectory data (GT, ensemble samples, y-limits, and
    # full-test-set coverage) so that all test plots can be regenerated
    # later without re-running the model.
    traj_save_path = save_test_trajectories(
        ground_truth, sampled_trajectories, grid_data,
        n_past, n_future, args.output_dir, timestamp,
        fullset_results=fullset_results
    )

    # ==================== QUANTILE TESTS (Optional) ====================
    if args.run_quantile_tests:
        print("\n" + "="*50)
        print("Running quantile calibration tests...")
        print("="*50)

        n_true_25, n_true_mtrx_25 = quantile_test(
            25, args.n_quart_test, n_past, n_future, args.n_samples,
            alt_selector, model, torch.tensor(raw_data), device
        )

        n_true_50, n_true_mtrx_50 = quantile_test(
            50, args.n_quart_test, n_past, n_future, args.n_samples,
            alt_selector, model, torch.tensor(raw_data), device
        )

        n_true_75, n_true_mtrx_75 = quantile_test(
            75, args.n_quart_test, n_past, n_future, args.n_samples,
            alt_selector, model, torch.tensor(raw_data), device
        )

        plot_quantile_convergence(n_true_mtrx_25, n_true_mtrx_50, n_true_mtrx_75,
                                  n_future, args.output_dir, timestamp,
                                  landscape_name=landscape_name)

    # ==================== SAVE RESULTS SUMMARY ====================
    results = {
        'timestamp': timestamp,
        'model_path': args.model_path,
        'data_path': args.data_path,
        'n_past': n_past,
        'n_future': n_future,
        'n_samples': args.n_samples,
        # SDE metadata (None for non-SDE datasets)
        'sde_model_key': sde_model_key,
        'sde_has_memory': sde_has_memory,
        # Single-trajectory metrics (retained for backwards compatibility)
        'crps_score': float(crps_score),
        'mean_crps': float(mean_crps),
        'mean_mae': float(mean_mae),
        # Multi-trajectory aggregated metrics over the 9 grid trajectories
        'mean_crps_agg': mean_crps_agg,
        'mean_mae_agg': mean_mae_agg,
        'n_grid_trajectories': len(grid_data),
        'crps_per_trajectory': [float(v) for v in crps_per_traj],
        'mae_per_trajectory': [float(v) for v in mae_per_traj],
        # Full test-set metrics (primary reported metrics)
        'fullset_mean_crps': fullset_results['mean_crps'],
        'fullset_std_crps':  fullset_results['std_crps'],
        'fullset_mean_mae':  fullset_results['mean_mae'],
        'fullset_std_mae':   fullset_results['std_mae'],
        'fullset_n_evaluated': fullset_results['n_evaluated'],
        'fullset_n_failed':    fullset_results['n_failed'],
        'test_trajectories_path': traj_save_path,
    }

    results_path = os.path.join(args.output_dir, f'results_summary_{timestamp}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results summary to: {results_path}")

    print("\n" + "="*50)
    print("Testing complete!")
    print(f"Results saved to: {args.output_dir}")
    print("="*50)


if __name__ == '__main__':
    main()
