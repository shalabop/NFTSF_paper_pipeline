#!/usr/bin/env python3
"""
compare_models_from_npz.py
==========================
Run comparison plots using pre-computed .npz result files.

This script:
  1. Loads .npz files (samples, ground_truth, etc.) from different models
  2. Converts them to the format expected by compare_models.py plotting functions
  3. Generates Figures A, B, C, D

Usage
-----
python compare_models_from_npz.py \
    --results \
        double_well:tsdiff_cond:results/tsdiff_cond/double_well.npz \
        double_well:tsdiff_ms:results/tsdiff_ms/double_well.npz \
        double_well:csdi:results/csdi/double_well.npz \
        double_well:ratd:results/ratd/double_well.npz \
    --output_dir ./comparison_figures/ \
    --n_traj_show 3 \
    --seed 42

Result spec format (colon-separated):
  landscape:model_name:npz_path
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import matplotlib
matplotlib.use("Agg")


LANDSCAPE_DISPLAY: dict[str, str] = {
    "single_well":    "Single Well",
    "double_well":    "Double Well",
    "alanine_phi":    r"Alanine $\varphi$ (Phi)",
    "alanine_psi":    r"Alanine $\psi$ (Psi)",
}

COORDINATE_LABEL: dict[str, str] = {
    "single_well":    r"Position $x$",
    "double_well":    r"Position $x$",
    "alanine_phi":    r"angle $\varphi$ (rad)",
    "alanine_psi":    r"angle $\psi$ (rad)",
}


def _land_display(landscape: str) -> str:
    return LANDSCAPE_DISPLAY.get(landscape, landscape.replace("_", " ").title())


def _coord_label(landscape: str) -> str:
    return COORDINATE_LABEL.get(landscape, r"Value")


# Import plotting functions from compare_models.py
# (You'll need to make compare_models.py importable or copy functions here)
def _crps_per_step(ground_truth_future: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    ground_truth_future : (N, n_future)
    samples             : (N, n_samples, n_future)

    Returns
    -------
    crps_t : (n_future,)  — averaged over N trajectories
    """
    N, n_future = ground_truth_future.shape
    n_samp = samples.shape[1]

    crps_t = np.zeros(n_future, dtype=np.float64)
    for t in range(n_future):
        obs = ground_truth_future[:, t]          # (N,)
        fc  = samples[:, :, t]                   # (N, n_samp)
        mae_term = np.mean(np.abs(fc - obs[:, None]), axis=1)  # (N,)

        fc_sorted = np.sort(fc, axis=1)
        diff = fc_sorted[:, 1:] - fc_sorted[:, :-1]           # (N, n_samp-1)
        w = np.arange(1, n_samp) * np.arange(n_samp - 1, 0, -1)
        energy = (diff * w[None, :]).sum(axis=1) / (n_samp * (n_samp - 1))  # (N,)

        crps_t[t] = np.mean(mae_term - energy)
    return crps_t


def _mae_per_step(ground_truth_future: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Median-forecast MAE at each step, averaged over N trajectories."""
    median = np.median(samples, axis=1)                     # (N, n_future)

    return np.abs(ground_truth_future - median).mean(axis=0)  # (n_future,)


def _coverage_per_step(
    ground_truth_future: np.ndarray,
    samples: np.ndarray,
    lo_pct: float,
    hi_pct: float,
) -> np.ndarray:
    """
    Per-step empirical coverage averaged across N trajectories.

    Matches evaluate_full_testset() in test_model.py exactly:
      for each trajectory i:
        - compute CI bounds from that trajectory's samples (axis=0 of (n_samples, n_future))
        - check whether ground truth falls inside at each step  → (n_future,) bool
        - accumulate
      average the per-trajectory boolean arrays across all N trajectories → (n_future,)

    Parameters
    ----------
    ground_truth_future : (N, n_future)
    samples             : (N, n_samples, n_future)
    lo_pct, hi_pct      : percentile bounds (e.g. 25.0, 75.0 for CI50)

    Returns
    -------
    coverage_t : (n_future,)
    """
    N = ground_truth_future.shape[0]
    coverage_within = []
    for i in range(N):
        gt_i   = ground_truth_future[i]          # (n_future,)
        samp_i = samples[i]                      # (n_samples, n_future)
        lo = np.percentile(samp_i, lo_pct, axis=0)   # (n_future,)
        hi = np.percentile(samp_i, hi_pct, axis=0)   # (n_future,)
        within = (gt_i >= lo) & (gt_i <= hi)          # (n_future,) bool
        coverage_within.append(within.astype(float))
    return np.mean(coverage_within, axis=0)      # (n_future,)


def compute_metrics(
    ground_truths: np.ndarray,
    samples: np.ndarray,
    n_past: int,
    fullset_mae:  np.ndarray | None = None,
    fullset_crps: np.ndarray | None = None,
    fullset_ci50: np.ndarray | None = None,
    fullset_ci90: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    Parameters
    ----------
    ground_truths : (N, n_past + n_future)  — visualization grid trajectories
    samples       : (N, n_samples, n_future)
    n_past        : int
    fullset_*     : (n_future,) pre-computed arrays from evaluate_full_testset()
                    over all test trajectories.  When provided these replace the
                    grid-based estimates (which are noisy due to small N).

    Returns
    -------
    dict with keys 'mae', 'crps', 'ci50', 'ci90' — each (n_future,)
    """
    gt_future = ground_truths[:, n_past:]   # (N, n_future)
    return {
        "mae":  fullset_mae  if fullset_mae  is not None else _mae_per_step(gt_future, samples),
        "crps": fullset_crps if fullset_crps is not None else _crps_per_step(gt_future, samples),
        "ci50": fullset_ci50 if fullset_ci50 is not None else _coverage_per_step(gt_future, samples, 25.0, 75.0),
        "ci90": fullset_ci90 if fullset_ci90 is not None else _coverage_per_step(gt_future, samples, 5.0,  95.0),
    }


# ---------------------------------------------------------------------------
# Y-limit helpers
# ---------------------------------------------------------------------------

def _ylim_single(real_traj: np.ndarray, samples: np.ndarray) -> tuple[float, float]:
    """Y-limits for one trajectory/model pair (1st–99th pct of samples + full GT)."""
    y_lo = min(float(np.percentile(samples, 1)), float(np.min(real_traj)))
    y_hi = max(float(np.percentile(samples, 99)), float(np.max(real_traj)))
    margin = 0.05 * (y_hi - y_lo) if y_hi > y_lo else 0.1
    return y_lo - margin, y_hi + margin


def plot_raw_trajectories(
    landscape: str,
    data_tensor: "torch.Tensor",   # (N, T)
    n_past: int,
    n_future: int,
    output_dir: Path,
    seed: int = 42,
) -> None:
    """
    One figure, 3 panels side by side showing randomly selected ground-truth
    trajectories overlaid: panels for 10, 100, and all-N trajectories.
    Each trajectory is a random window of length n_past + n_future.
    Panel titles show the actual count used.
    """
    rng    = np.random.default_rng(seed)
    data   = data_tensor.cpu().numpy()          # (N, T)
    N, T   = data.shape
    n_extrp = n_past + n_future
    x_all  = np.arange(n_extrp)
    counts = [10, 100, N]

    # Pre-sample ALL indices and window starts so that smaller sets are
    # strict subsets of larger ones (consistent windows across panels).
    all_idx    = rng.choice(N, size=N, replace=False)
    all_starts = np.array([
        int(np.random.randint(0, max(1, T - n_extrp + 1)))
        for _ in range(N)
    ])

    # Compute global y-range from the largest set (all-N)
    y_lo, y_hi = float("inf"), float("-inf")
    for i in range(N):
        idx, s = all_idx[i], all_starts[i]
        seg = data[idx, s:s + n_extrp]
        if len(seg) == n_extrp:
            y_lo = min(y_lo, float(seg.min()))
            y_hi = max(y_hi, float(seg.max()))
    margin = 0.05 * (y_hi - y_lo) if y_hi > y_lo else 0.1
    y_lo -= margin
    y_hi += margin

    fig, axes = plt.subplots(1, 3, figsize=(21, 5), sharey=True)

    SINGLE_COLOR = "#4C72B0"

    for ax, count in zip(axes, counts):
        n = min(count, N)
        indices = all_idx[:n]
        starts  = all_starts[:n]

        if n <= 10:
            colors   = plt.get_cmap("tab10")(np.linspace(0, 1, min(n, 10)))
            lw, alpha = 1.5, 1.0
        elif n <= 100:
            colors   = [SINGLE_COLOR] * n
            lw, alpha = 0.7, 0.25
        else:
            colors   = [SINGLE_COLOR] * n
            lw, alpha = 0.5, 0.10

        for i, (idx, s) in enumerate(zip(indices, starts)):
            seg = data[idx, s:s + n_extrp]
            if len(seg) < n_extrp:
                continue
            ax.plot(x_all, seg, color=colors[i] if n <= 10 else SINGLE_COLOR,
                    linewidth=lw, alpha=alpha)

        ax.axvline(x=n_past, color="black", linestyle="--", alpha=0.5, linewidth=1.2)
        ax.set_xlim(0, n_extrp)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel(r"Step $N$", fontsize=10)
        ax.set_title(f"{n} Trajectories", fontsize=12)
        ax.tick_params(labelsize=9)

    axes[0].set_ylabel(_coord_label(landscape), fontsize=10)

    fig.suptitle(
        f"{_land_display(landscape)} — Ground Truth Trajectories",
        fontsize=14, y=1.02,
    )
    plt.tight_layout()

    out_path = output_dir / f"raw_trajectories_{landscape}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[E] Saved: {out_path}")


def plot_predicted_trajectories(
    landscape: str,
    model_name: str,
    model_data: dict,          # {ground_truths, samples, n_past, n_future, ...}
    output_dir: Path,
) -> None:
    """
    Same 3-panel format as plot_raw_trajectories, but showing predicted
    future samples instead of raw ground-truth trajectories.

    For each display trajectory, produces one figure with 3 panels:
      - Panel 1: 1 predicted sample
      - Panel 2: 10 predicted samples
      - Panel 3: all n_samples predicted samples

    Each panel overlays the predicted future lines on the full ground truth
    (past in black solid, future in black dashed).  A vertical dashed line
    marks the past/future split.
    """
    ground_truths = model_data["ground_truths"]   # (N_disp, n_past+n_future)
    samples       = model_data["samples"]          # (N_disp, n_samp, n_future)
    n_past        = model_data["n_past"]
    n_future      = model_data["n_future"]
    n_samp        = samples.shape[1]
    n_extrp       = n_past + n_future
    x_all         = np.arange(n_extrp)
    future_steps  = np.arange(n_past, n_extrp)

    reg   = MODEL_REGISTRY.get(model_name, {"label": model_name, "color": "tab:orange"})
    color = reg["color"]
    label = reg["label"]

    counts = [1, 10, n_samp]

    for traj_idx, traj_label in enumerate(model_data.get("display_labels",
                                          list(range(ground_truths.shape[0])))):
        gt   = ground_truths[traj_idx]           # (n_past+n_future,)
        samp = samples[traj_idx]                  # (n_samp, n_future)

        # Global y-range: all samples + full ground truth
        y_lo = min(float(np.percentile(samp, 1)), float(gt.min()))
        y_hi = max(float(np.percentile(samp, 99)), float(gt.max()))
        margin = 0.05 * (y_hi - y_lo) if y_hi > y_lo else 0.1
        y_lo -= margin
        y_hi += margin

        fig, axes = plt.subplots(1, 3, figsize=(21, 5), sharey=True)

        for ax, count in zip(axes, counts):
            n = min(count, n_samp)

            # Ground truth: past solid, future dashed
            ax.plot(x_all[:n_past],  gt[:n_past],  "k-",  linewidth=2.0)
            ax.plot(x_all[n_past:],  gt[n_past:],  "k--", linewidth=1.5,
                    alpha=0.7, label="Truth (future)")

            # Predicted samples
            if n <= 10:
                colors_s = plt.get_cmap("tab10")(np.linspace(0, 1, min(n, 10)))
                lw, alpha = 1.5, 1.0
            elif n <= 100:
                colors_s = [color] * n
                lw, alpha = 0.7, 0.35
            else:
                colors_s = [color] * n
                lw, alpha = 0.5, 0.15

            for s_idx in range(n):
                c = colors_s[s_idx] if n <= 10 else color
                ax.plot(future_steps, samp[s_idx], color=c,
                        linewidth=lw, alpha=alpha)

            ax.axvline(x=n_past, color="black", linestyle="--",
                       alpha=0.5, linewidth=1.2)
            ax.set_xlim(0, n_extrp)
            ax.set_ylim(y_lo, y_hi)
            ax.set_xlabel(r"Step $N$", fontsize=10)
            ax.set_title(f"{n} Predicted Sample{'s' if n > 1 else ''}", fontsize=12)
            ax.tick_params(labelsize=9)

        axes[0].set_ylabel(_coord_label(landscape), fontsize=10)

        fig.suptitle(
            f"{_land_display(landscape)} — {label} Predicted Trajectories"
            f" (Traj {traj_label})",
            fontsize=14, y=1.02,
        )
        plt.tight_layout()

        out_path = output_dir / f"predicted_trajectories_{landscape}_{model_name}_traj{traj_label}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[F] Saved: {out_path}")


def _global_ylim(model_data: dict[str, dict]) -> tuple[float, float]:
    """
    Compute a single y-axis range that encompasses all display trajectories
    and all models in one comparison figure.  model_data arrays are already
    the display subset (indexed 0..N_display-1).
    """
    g_lo =  float("inf")
    g_hi = float("-inf")
    for mdata in model_data.values():
        n_disp = mdata["ground_truths"].shape[0]
        for col_idx in range(n_disp):
            lo, hi = _ylim_single(
                mdata["ground_truths"][col_idx],
                mdata["samples"][col_idx],
            )
            g_lo = min(g_lo, lo)
            g_hi = max(g_hi, hi)
    return g_lo, g_hi


# ---------------------------------------------------------------------------
# Figure A: Trajectory comparison grid
# ---------------------------------------------------------------------------

def plot_trajectory_comparison_grid(
    landscape: str,
    display_labels: list[int],
    model_data: dict[str, dict],   # model_name → {ground_truths, samples, ...}
    n_past: int,
    n_future: int,
    output_dir: Path,
) -> None:
    """
    Rows = models, columns = selected trajectories.
    Each cell: 90%/50% confidence bands + black ground-truth line.

    model_data arrays are the display subset indexed 0..N_display-1.
    display_labels are the original trajectory numbers (used only for titles).
    """
    n_models = len(model_data)
    n_traj   = len(display_labels)

    fig, axes = plt.subplots(
        n_models, n_traj,
        figsize=(6 * n_traj, 5 * n_models),
        squeeze=False,
    )

    future_steps = np.arange(n_past, n_past + n_future)
    all_steps    = np.arange(0, n_past + n_future)

    # One shared y-range for every subplot in this figure
    g_lo, g_hi = _global_ylim(model_data)

    for row_idx, (model_name, mdata) in enumerate(model_data.items()):
        reg = MODEL_REGISTRY.get(model_name, {"label": model_name, "color": "tab:orange"})
        color  = reg["color"]
        mlabel = reg["label"]

        samples_all       = mdata["samples"]         # (N_disp, n_samp, n_future)
        ground_truths_all = mdata["ground_truths"]   # (N_disp, n_past+n_future)

        # Row label on left-most column
        axes[row_idx, 0].set_ylabel(
            f"{mlabel}\n{_coord_label(landscape)}", fontsize=10
        )

        for col_idx, traj_label in enumerate(display_labels):
            ax = axes[row_idx, col_idx]
            real_traj = ground_truths_all[col_idx]   # (n_past+n_future,)
            samp      = samples_all[col_idx]          # (n_samp, n_future)

            median   = np.median(samp, axis=0)
            lo90     = np.percentile(samp, 5,  axis=0)
            hi90     = np.percentile(samp, 95, axis=0)
            lo50     = np.percentile(samp, 25, axis=0)
            hi50     = np.percentile(samp, 75, axis=0)

            ax.plot(all_steps, real_traj, "k-", linewidth=2.0, label="Truth")
            ax.plot(future_steps, median, color=color, linewidth=2.0, label="Median")
            ax.fill_between(future_steps, lo90, hi90,
                            color="tab:orange", alpha=0.40, label="90% band")
            ax.fill_between(future_steps, lo50, hi50,
                            color="tab:blue", alpha=0.40, label="50% band")
            ax.axvline(x=n_past, color="k", linestyle="--", alpha=0.4)
            ax.set_ylim(g_lo, g_hi)
            ax.set_xlim(0, n_past + n_future)
            ax.set_xlabel(r"Step $N$", fontsize=9)

            if row_idx == 0:
                ax.set_title(f"Trajectory {traj_label}", fontsize=11)

    axes[0, 0].legend(loc="upper left", fontsize=7, framealpha=0.8)

    fig.suptitle(
        f"{_land_display(landscape)} — Trajectory Comparison",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()

    out_path = output_dir / f"trajectory_comparison_{landscape}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[A] Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure B: 2-D histogram comparison grid
# ---------------------------------------------------------------------------

def plot_histogram2d_comparison_grid(
    landscape: str,
    display_labels: list[int],
    model_data: dict[str, dict],
    n_past: int,
    n_future: int,
    output_dir: Path,
) -> None:
    """
    Same layout as Figure A but using magma 2-D density heatmaps.

    model_data arrays are the display subset indexed 0..N_display-1.
    display_labels are the original trajectory numbers (used only for titles).
    """
    n_models = len(model_data)
    n_traj   = len(display_labels)

    fig, axes = plt.subplots(
        n_models, n_traj,
        figsize=(7 * n_traj, 5 * n_models),
        squeeze=False,
    )

    future_steps = np.arange(n_past, n_past + n_future)
    all_steps    = np.arange(0, n_past + n_future)

    # One shared y-range for every subplot in this figure
    g_lo, g_hi = _global_ylim(model_data)

    # Fewer, larger bins (fewer pixels per cell → each one more visible)
    n_x_bins = max(30, n_future // 2)
    n_y_bins = 60

    for row_idx, (model_name, mdata) in enumerate(model_data.items()):
        reg    = MODEL_REGISTRY.get(model_name, {"label": model_name, "color": "tab:orange"})
        mlabel = reg["label"]

        samples_all       = mdata["samples"]         # (N_disp, n_samp, n_future)
        ground_truths_all = mdata["ground_truths"]   # (N_disp, n_past+n_future)

        axes[row_idx, 0].set_ylabel(
            f"{mlabel}\n{_coord_label(landscape)}", fontsize=10
        )

        for col_idx, traj_label in enumerate(display_labels):
            ax        = axes[row_idx, col_idx]
            real_traj = ground_truths_all[col_idx]
            samp      = samples_all[col_idx]          # (n_samp, n_future)

            time_rep = np.tile(future_steps, (samp.shape[0], 1))
            ax.hist2d(
                time_rep.flatten(), samp.flatten(),
                bins=(n_x_bins, n_y_bins),
                range=[[n_past, n_past + n_future], [g_lo, g_hi]],
                cmap="magma", density=True,
                norm=LogNorm(vmin=1e-6),
            )
            ax.plot(all_steps, real_traj, color="lime", linewidth=2.5, label="Truth")
            ax.axvline(x=n_past, color="white", linestyle="--", alpha=0.5)

            ax.set_ylim(g_lo, g_hi)
            ax.set_xlim(0, n_past + n_future)
            ax.set_xlabel(r"Step $N$", fontsize=9)

            if row_idx == 0:
                ax.set_title(f"Trajectory {traj_label}", fontsize=11)

    axes[0, 0].legend(loc="upper left", fontsize=7, framealpha=0.8)

    fig.suptitle(
        f"{_land_display(landscape)} — Prediction Density Comparison",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()

    out_path = output_dir / f"histogram2d_comparison_{landscape}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[B] Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure C: Error metrics grid
# ---------------------------------------------------------------------------

METRIC_META: dict[str, dict] = {
    "mae":  {"title": "MAE",             "ylabel": "MAE",      "ideal": None},
    "crps": {"title": "CRPS",            "ylabel": "CRPS",     "ideal": None},
    "ci50": {"title": "CI50 Coverage",   "ylabel": "Coverage", "ideal": 0.50},
    "ci90": {"title": "CI90 Coverage",   "ylabel": "Coverage", "ideal": 0.90},
}
METRIC_ORDER = ["mae", "crps", "ci50", "ci90"]


def plot_error_metrics_grid(
    landscapes: list[str],
    all_metrics: dict[str, dict[str, dict[str, np.ndarray]]],
    # all_metrics[landscape][model_name][metric_key] = (n_future,) array
    output_dir: Path,
) -> None:
    """
    4 rows (landscapes) × 4 cols (metrics).
    Each cell: per-step line plot, one line per model.
    """
    n_land = len(landscapes)
    n_met  = len(METRIC_ORDER)

    fig, axes = plt.subplots(
        n_land, n_met,
        figsize=(4 * n_met, 3.5 * n_land),
        squeeze=False,
    )

    # Pre-compute per-metric y-range across all landscapes and models so that
    # every cell in the same metric column shares identical y-limits.
    metric_ylims: dict[str, tuple[float, float]] = {}
    for metric_key in METRIC_ORDER:
        col_lo =  float("inf")
        col_hi = float("-inf")
        for landscape in landscapes:
            for met_dict in all_metrics[landscape].values():
                vals = met_dict[metric_key]
                col_lo = min(col_lo, float(np.min(vals)))
                col_hi = max(col_hi, float(np.max(vals)))
        margin = 0.05 * (col_hi - col_lo) if col_hi > col_lo else 0.05
        metric_ylims[metric_key] = (col_lo - margin, col_hi + margin)

    legend_handles: list = []

    for col_idx, metric_key in enumerate(METRIC_ORDER):
        meta   = METRIC_META[metric_key]
        y_lo, y_hi = metric_ylims[metric_key]

        for row_idx, landscape in enumerate(landscapes):
            ax = axes[row_idx, col_idx]
            model_metrics = all_metrics[landscape]

            for m_idx, (model_name, met_dict) in enumerate(model_metrics.items()):
                reg   = MODEL_REGISTRY.get(model_name, {"label": model_name, "color": "tab:gray"})
                color = reg["color"]
                vals  = met_dict[metric_key]
                steps = np.arange(len(vals))
                ax.plot(steps, vals, color=color, linewidth=2.0,
                        label=reg["label"])

                if row_idx == 0 and col_idx == 0:
                    legend_handles.append(
                        Line2D([0], [0], color=color, linewidth=2.0,
                               label=reg["label"])
                    )

            if meta["ideal"] is not None:
                ax.axhline(meta["ideal"], color="black", linestyle="--",
                           linewidth=0.8, alpha=0.6)
                if row_idx == 0 and col_idx == 0:
                    legend_handles.append(
                        Line2D([0], [0], color="black", linestyle="--",
                               linewidth=0.8, label="Ideal")
                    )

            ax.set_ylim(y_lo, y_hi)
            ax.set_xlabel("Forecast step", fontsize=8)
            ax.set_ylabel(meta["ylabel"], fontsize=8)
            ax.grid(True, linestyle=":", alpha=0.4)
            ax.tick_params(labelsize=7)

            # Row label (landscape) on first column
            if col_idx == 0:
                ax.set_ylabel(
                    f"{_land_display(landscape)}\n{meta['ylabel']}", fontsize=8
                )
            # Column title (metric) on first row
            if row_idx == 0:
                ax.set_title(meta["title"], fontsize=10)

    fig.suptitle("Error Metrics Comparison — All Landscapes", fontsize=13, y=1.01)
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=len(legend_handles),
        fontsize=9,
        frameon=False,
    )
    plt.tight_layout()

    out_path = output_dir / "error_metrics_comparison.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[C] Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure D: Tables
# ---------------------------------------------------------------------------

def plot_metric_table(
    metric_key: str,
    landscapes: list[str],
    model_names: list[str],
    all_metrics: dict[str, dict[str, dict[str, np.ndarray]]],
    output_dir: Path,
) -> None:
    """
    One matplotlib table + CSV for *metric_key*.
    Rows = models, columns = landscapes.
    Values = scalar (mean over forecast steps).
    """
    col_labels = [_land_display(l) for l in landscapes]
    row_labels = [
        MODEL_REGISTRY.get(m, {"label": m})["label"] for m in model_names
    ]

    # Build table data
    table_data = []
    for model_name in model_names:
        row = []
        for landscape in landscapes:
            val = float(all_metrics[landscape][model_name][metric_key].mean())
            row.append(f"{val:.4f}")
        table_data.append(row)

    # --- PNG table ---
    fig, ax = plt.subplots(figsize=(max(6, 2.5 * len(landscapes)), 1.5 + 0.5 * len(model_names)))
    ax.axis("off")
    tbl = ax.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 1.8)

    # Color header row and model column
    for (r, c), cell in tbl.get_celld().items():
        if r == 0 or c == -1:
            cell.set_facecolor("#D0D8E8")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#F7F9FC")

    meta  = METRIC_META[metric_key]
    title = f"{meta['title']} — Summary Table (mean over forecast steps)"
    ax.set_title(title, fontsize=12, pad=12)

    png_path = output_dir / f"table_{metric_key}.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[D] Saved: {png_path}")

    # --- CSV ---
    csv_path = output_dir / f"table_{metric_key}.csv"
    with open(csv_path, "w") as f:
        f.write("model," + ",".join(landscapes) + "\n")
        for model_name, row in zip(model_names, table_data):
            f.write(model_name + "," + ",".join(row) + "\n")
    print(f"[D] Saved: {csv_path}")

MODEL_REGISTRY: dict[str, dict] = {
    "nftsf": {"label": "NFTSF", "color": "#1F77B4"},
    "arima": {"label": "ARIMA", "color": "#2CA02C"},
    "tsdiff_cond": {"label": "TSDiff-Cond", "color": "#C71FD6"},
    "tsdiff_ms": {"label": "TSDiff-MS", "color": "#A09E2C"},
    "tsdiff_q": {"label": "TSDiff-Q", "color": "#B41F1F"},
    "csdi": {"label": "CSDI", "color": "#0AF1F1"},
    "ratd": {"label": "RATD", "color": "#FF7F0E"},   #
    "nsdiff": {"label": "NsDiff", "color": "#9467BD"},
    "ccdm": {"label": "CCDM", "color": "#8C564B"},
}

METRIC_META: dict[str, dict] = {
    "mae":  {"title": "MAE",             "ylabel": "MAE",      "ideal": None},
    "crps": {"title": "CRPS",            "ylabel": "CRPS",     "ideal": None},
    "ci50": {"title": "CI50 Coverage",   "ylabel": "Coverage", "ideal": 0.50},
    "ci90": {"title": "CI90 Coverage",   "ylabel": "Coverage", "ideal": 0.90},
}
METRIC_ORDER = ["mae", "crps", "ci50", "ci90"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate comparison figures from pre-computed .npz results"
    )
    p.add_argument(
        "--results", nargs="+", required=True,
        metavar="LANDSCAPE:MODEL:NPZ_PATH",
        help=(
            "Result files per landscape/model combination. "
            "Format: landscape:model_name:path/to/results.npz"
        )
    )
    p.add_argument(
        "--output_dir", default="./comparison_figures",
        help="Directory where all output figures are saved"
    )
    p.add_argument(
        "--n_traj_show", type=int, default=3,
        help="Number of trajectories to display (default: 3)"
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible trajectory selection"
    )
    return p.parse_args()


def parse_result_spec(spec: str) -> dict:
    """
    Parse one --results entry.
    
    Format: landscape:model_name:npz_path
    """
    parts = spec.split(":")
    if len(parts) != 3:
        print(
            f"ERROR: --results entries need landscape:model:npz_path — got: {spec!r}",
            file=sys.stderr,
        )
        sys.exit(1)
    return {
        "landscape": parts[0],
        "model": parts[1],
        "npz_path": parts[2],
    }


def load_npz_result(npz_path: str) -> dict:
    """
    Load .npz result file and convert to standard format.
    
    Returns
    -------
    dict with keys:
        ground_truths : (N, L+H)  # full context + forecast
        samples       : (N, num_samples, H)
        n_past        : int (L)
        n_future      : int (H)
        fullset_mae   : (H,) if available
        fullset_crps  : (H,) if available
        fullset_ci50  : (H,) if available
        fullset_ci90  : (H,) if available
        fullset_N     : int
    """
    data = np.load(npz_path, allow_pickle=True)
    
    # Extract core arrays
    samples = data["samples"]  # Shape varies by model
    ground_truth = data["ground_truth"]  # (N, H)
    #ground_truth = data["full_trajectories"]
    
    # Determine dimensions
    if samples.ndim == 3:
        # Could be (N, H, S) or (N, S, H)
        if samples.shape[1] == ground_truth.shape[1]:
            # (N, H, S) → transpose to (N, S, H)
            samples = samples.transpose(0, 2, 1)
        # Now: (N, S, H)
    
    N, num_samples, H = samples.shape
    
    # Get context length
    if "contexts" in data:
        # RATD format
        contexts = data["contexts"]  # (N, L)
        L = contexts.shape[1]
        # Reconstruct full trajectories
        ground_truths = np.concatenate([contexts, ground_truth], axis=1)  # (N, L+H)
    elif "full_trajectories" in data:
        # Standard format
        full_traj = data["full_trajectories"]  # (N, T_total)
        train_test_split = int(data.get("train_test_split", 800))
        
        # Extract the window that was actually used for forecasting
        # Assuming forecast starts at train_test_split
        L = H  # Assume context = forecast length (adjust if different)
        start = train_test_split - L
        ground_truths = full_traj[:, start:start+L+H]  # (N, L+H)
    else:
        # Fallback: assume L = H
        '''L = H
        # Create dummy context (zeros)
        ground_truths = np.concatenate([
            np.zeros((N, L)),
            ground_truth
        ], axis=1)'''
        raise RuntimeError("qweqwe")
    
    # Check for pre-computed metrics
    result = {
        "ground_truths": ground_truths.astype(np.float32),
        "samples": samples.astype(np.float32),
        "n_past": L,
        "n_future": H,
        "fullset_N": N,
    }
    
    # Add pre-computed metrics if available
    # (These would need to be computed per-step and saved in the .npz)
    # For now, they'll be computed on-the-fly from the display subset
    
    return result


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # Parse result specs
    specs = [parse_result_spec(s) for s in args.results]
    
    # Organize by landscape
    landscape_models = {}
    for spec in specs:
        land = spec["landscape"]
        if land not in landscape_models:
            landscape_models[land] = {}
        landscape_models[land][spec["model"]] = spec["npz_path"]
    
    landscapes = list(landscape_models.keys())
    print(f"Landscapes: {landscapes}")
    
    # Load all results
    print("\n=== Loading .npz results ===")
    all_data = {}  # all_data[landscape][model] = loaded dict
    
    for land in landscapes:
        all_data[land] = {}
        for model, npz_path in landscape_models[land].items():
            print(f"  Loading {land}/{model} from {npz_path}")
            all_data[land][model] = load_npz_result(npz_path)
            N = all_data[land][model]["ground_truths"].shape[0]
            H = all_data[land][model]["n_future"]
            S = all_data[land][model]["samples"].shape[1]
            print(f"    N={N}, H={H}, samples={S}")
    
    # Select display trajectories (common across all models)
    N_min = min(
        data["ground_truths"].shape[0]
        for land_data in all_data.values()
        for data in land_data.values()
    )
    n_show = min(args.n_traj_show, N_min)
    display_labels = rng.choice(N_min, size=n_show, replace=False).tolist()
    print(f"\nDisplay trajectory labels (seed={args.seed}): {display_labels}")
    
    # Subset to display trajectories
    print("\n=== Subsetting to display trajectories ===")
    display_data = {}
    for land in landscapes:
        display_data[land] = {}
        for model, data in all_data[land].items():
            display_data[land][model] = {
                "ground_truths": data["ground_truths"][display_labels],
                "samples": data["samples"][display_labels],
                "display_labels": display_labels,
                "n_past": data["n_past"],
                "n_future": data["n_future"],
                "fullset_N": data["fullset_N"],
            }
    
    # Compute metrics
    print("\n=== Computing metrics ===")
    all_metrics = {}
    
    for land in landscapes:
        all_metrics[land] = {}
        for model, mdata in display_data[land].items():
            n_past = mdata["n_past"]
            
            # Compute metrics on display subset
            # (In production, you'd compute on full testset and save to .npz)
            mets = compute_metrics(
                mdata["ground_truths"],
                mdata["samples"],
                n_past,
                fullset_mae=None,   # Could load from .npz if pre-computed
                fullset_crps=None,
                fullset_ci50=None,
                fullset_ci90=None,
            )
            all_metrics[land][model] = mets
            
            label = MODEL_REGISTRY.get(model, {"label": model})["label"]
            print(
                f"  {label:12s} / {land:15s} — "
                f"MAE={mets['mae'].mean():.4f}  "
                f"CRPS={mets['crps'].mean():.4f}  "
                f"CI50={mets['ci50'].mean():.3f}  "
                f"CI90={mets['ci90'].mean():.3f}"
            )
    
    # Generate figures
    print("\n=== Generating comparison figures ===")
    
    # Figure A & B: per landscape
    for land in landscapes:
        n_past = display_data[land][list(display_data[land].keys())[0]]["n_past"]
        n_future = display_data[land][list(display_data[land].keys())[0]]["n_future"]
        
        print(f"\n  [{land}] Trajectory comparison grid...")
        plot_trajectory_comparison_grid(
            landscape=land,
            display_labels=display_labels,
            model_data=display_data[land],
            n_past=n_past,
            n_future=n_future,
            output_dir=out,
        )
        
        print(f"  [{land}] Histogram comparison grid...")
        plot_histogram2d_comparison_grid(
            landscape=land,
            display_labels=display_labels,
            model_data=display_data[land],
            n_past=n_past,
            n_future=n_future,
            output_dir=out,
        )
    
    # Figure C: error metrics grid
    print("\n  Error metrics comparison grid...")
    plot_error_metrics_grid(
        landscapes=landscapes,
        all_metrics=all_metrics,
        output_dir=out,
    )
    
    # Figure D: tables
    print("\n  Error metric tables...")
    model_order = list(display_data[landscapes[0]].keys())
    for metric_key in METRIC_ORDER:
        plot_metric_table(
            metric_key=metric_key,
            landscapes=landscapes,
            model_names=model_order,
            all_metrics=all_metrics,
            output_dir=out,
        )
    
    print(f"\nAll figures saved to: {out}")


if __name__ == "__main__":
    main()