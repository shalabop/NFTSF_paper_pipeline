#!/usr/bin/env python3
"""
compare_models_from_npz.py
==========================
Generate comparison figures from pre-computed .npz result files.

Usage
-----
python compare_models_from_npz.py \
    --results \
        double_well:tsdiff_q:results/tsdiff/double_well_q_4.npz \
        double_well:tsdiff_ms:results/tsdiff/double_well_mse_05.npz \
        double_well:csdi:results/csdi/double_well.npz \
        double_well:ratd:results/ratd/double_well.npz \
    --data_npz \
        double_well:DATA/double_well_test.npz \
    --output_dir ./comparison_figures/ \
    --n_traj_show 3 \
    --seed 42
"""

import argparse
import sys
import concurrent.futures
import glob
import json
import os
import warnings
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    def _tqdm(it, **kw):  # type: ignore[misc]
        return it


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, dict] = {
    "nftsf":      {"label": "NFTSF",       "color": "#1F77B4"},
    "arima":      {"label": "ARIMA",        "color": "#2CA02C"},
    "tsdiff_cond":{"label": "TSDiff-Cond",  "color": "#C71FD6"},
    "tsdiff_ms":  {"label": "TSDiff-MS",    "color": "#A09E2C"},
    "tsdiff_q":   {"label": "TSDiff-Q",     "color": "#B41F1F"},
    "csdi":       {"label": "CSDI",         "color": "#0AF1F1"},
    "ratd":       {"label": "RATD",         "color": "#FF7F0E"},
    "nsdiff":     {"label": "NsDiff",       "color": "#9467BD"},
    "ccdm":       {"label": "CCDM",         "color": "#8C564B"},
}

# ---------------------------------------------------------------------------
# Landscape display helpers
# ---------------------------------------------------------------------------

LANDSCAPE_DISPLAY: dict[str, str] = {
    "single_well": "Single Well",
    "double_well": "Double Well",
    "alanine_phi": r"Alanine $\varphi$ (Phi)",
    "alanine_psi": r"Alanine $\psi$ (Psi)",
}

COORDINATE_LABEL: dict[str, str] = {
    "single_well": r"Position $x$",
    "double_well": r"Position $x$",
    "alanine_phi": r"Angle $\varphi$ (rad)",
    "alanine_psi": r"Angle $\psi$ (rad)",
}


def _land_display(landscape: str) -> str:
    return LANDSCAPE_DISPLAY.get(landscape, landscape.replace("_", " ").title())


def _coord_label(landscape: str) -> str:
    return COORDINATE_LABEL.get(landscape, r"Value")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate comparison figures from pre-computed .npz results"
    )
    p.add_argument(
        "--results", nargs="+", required=True,
        metavar="LANDSCAPE:MODEL:NPZ_PATH",
        help="Format: landscape:model_name:path/to/results.npz",
    )
    p.add_argument(
        "--output_dir", default="./comparison_figures",
        help="Directory where all output figures are saved",
    )
    p.add_argument(
        "--n_traj_show", type=int, default=3,
        help="Number of trajectories to display (default: 3)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible trajectory selection",
    )
    p.add_argument(
        "--data_npz", nargs="+", default=None,
        metavar="LANDSCAPE:PATH",
        help="Path to _test.npz containing mean/std for denormalization. "
             "Format: landscape:DATA/ds_test.npz",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _load_mean_std(specs: list[str] | None) -> dict[str, tuple[float, float]]:
    """Parse --data_npz entries → {landscape: (mean, std)}."""
    result = {}
    if not specs:
        return result
    for spec in specs:
        parts = spec.split(":")
        if len(parts) != 2:
            print(f"WARNING: --data_npz malformed (expected landscape:path): {spec!r}")
            continue
        land, path = parts
        if not Path(path).exists():
            print(f"WARNING: --data_npz path not found for {land}: {path}")
            continue
        d = np.load(path)
        mean = float(d["mean"])
        std  = float(d["std"])
        result[land] = (mean, std)
        print(f"  [{land}] norm stats: mean={mean:.6f}  std={std:.6f}")
    return result


def _denorm(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Reverse (x - mean) / (std + 1e-8)."""
    return x * (std + 1e-8) + mean


# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------

def parse_result_spec(spec: str) -> dict:
    """Parse one --results entry: landscape:model_name:npz_path"""
    parts = spec.split(":")
    if len(parts) != 3:
        print(
            f"ERROR: --results entries need landscape:model:npz_path — got: {spec!r}",
            file=sys.stderr,
        )
        sys.exit(1)
    return {"landscape": parts[0], "model": parts[1], "npz_path": parts[2]}


def load_npz_result(npz_path: str) -> dict:
    """
    Load .npz result file and convert to standard format.

    Expects keys:
        samples           : (N, H, S) or (N, S, H)
        ground_truth      : (N, H)
        full_trajectories : (N, T_total)
        train_test_split  : int
        prediction_length : int

    Returns dict with:
        ground_truths : (N, L+H)
        samples       : (N, S, H)
        n_past        : int (L)
        n_future      : int (H)
        fullset_N     : int
    """
    data = np.load(npz_path, allow_pickle=True)

    print(f"    Loading {npz_path}")
    print(f"      Keys: {list(data.keys())}")

    samples           = data["samples"]
    ground_truth      = data["ground_truth"]       # (N, H)
    full_trajectories = data["full_trajectories"]  # (N, T_total)

    train_test_split = int(data["train_test_split"])
    H  = int(data["prediction_length"])
    N  = ground_truth.shape[0]

    # Ensure samples shape is (N, S, H)
    if samples.ndim == 3:
        if samples.shape[1] == H:
            # (N, H, S) → (N, S, H)
            samples = samples.transpose(0, 2, 1)
            print(f"      Transposed samples → {samples.shape}")
        elif samples.shape[2] == H:
            pass  # already (N, S, H)
        else:
            print(f"      WARNING: Cannot determine sample format. "
                  f"Shape: {samples.shape}, H={H}")

    N_check, S, H_check = samples.shape
    assert H_check == H, f"Sample H mismatch: {H_check} != {H}"

    # Reconstruct context window [tts-L : tts] and forecast [tts : tts+H]
    L         = H  # assume context_length == prediction_length
    start_idx = train_test_split - L
    end_idx   = train_test_split + H

    if start_idx < 0 or end_idx > full_trajectories.shape[1]:
        print(f"      WARNING: Adjusting window. "
              f"tts={train_test_split}, L={L}, H={H}, "
              f"T={full_trajectories.shape[1]}")
        end_idx   = full_trajectories.shape[1]
        start_idx = end_idx - (L + H)
        if start_idx < 0:
            L         = full_trajectories.shape[1] - H
            start_idx = 0

    ground_truths = full_trajectories[:, start_idx:end_idx]  # (N, L+H)

    print(f"      N={N}, S={S}, L={L}, H={H}")
    print(f"      Window: [{start_idx}:{end_idx}] "
          f"from full_traj shape {full_trajectories.shape}")
    print(f"      ground_truths: {ground_truths.shape}, samples: {samples.shape}")

    assert ground_truths.shape == (N, L + H), \
        f"ground_truths shape mismatch: {ground_truths.shape} != ({N}, {L+H})"
    assert samples.shape == (N, S, H), \
        f"samples shape mismatch: {samples.shape} != ({N}, {S}, {H})"

    # Consistency check
    gt_check = ground_truths[:, L:]
    if not np.allclose(gt_check, ground_truth, rtol=1e-4, atol=1e-4):
        print(f"      WARNING: ground_truth mismatch with full_trajectories extraction")

    return {
        "ground_truths": ground_truths.astype(np.float32),
        "samples":       samples.astype(np.float32),
        "n_past":        L,
        "n_future":      H,
        "fullset_N":     N,
    }


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _crps_per_step(
    ground_truth_future: np.ndarray,  # (N, n_future)
    samples: np.ndarray,              # (N, n_samples, n_future)
) -> np.ndarray:
    """CRPS per forecast step, averaged over N trajectories. Returns (n_future,)."""
    N, n_future = ground_truth_future.shape
    n_samp = samples.shape[1]

    crps_t = np.zeros(n_future, dtype=np.float64)
    for t in range(n_future):
        obs = ground_truth_future[:, t]   # (N,)
        fc  = samples[:, :, t]            # (N, n_samp)

        mae_term = np.mean(np.abs(fc - obs[:, None]), axis=1)  # (N,)

        fc_sorted = np.sort(fc, axis=1)
        diff      = fc_sorted[:, 1:] - fc_sorted[:, :-1]       # (N, n_samp-1)
        w         = np.arange(1, n_samp) * np.arange(n_samp - 1, 0, -1)
        # Correct normalization: ns^2 (consistent with _arima_eval_one)
        energy    = (diff * w[None, :]).sum(axis=1) / (n_samp ** 2)  # (N,)

        crps_t[t] = np.mean(mae_term - energy)
    return crps_t


def _mae_per_step(
    ground_truth_future: np.ndarray,  # (N, n_future)
    samples: np.ndarray,              # (N, n_samples, n_future)
) -> np.ndarray:
    """Median-forecast MAE per step, averaged over N. Returns (n_future,)."""
    median = np.median(samples, axis=1)                        # (N, n_future)
    return np.abs(ground_truth_future - median).mean(axis=0)   # (n_future,)


def _coverage_per_step(
    ground_truth_future: np.ndarray,  # (N, n_future)
    samples: np.ndarray,              # (N, n_samples, n_future)
    lo_pct: float,
    hi_pct: float,
) -> np.ndarray:
    """Per-step empirical coverage averaged over N. Returns (n_future,)."""
    N = ground_truth_future.shape[0]
    coverage_within = []
    for i in range(N):
        gt_i   = ground_truth_future[i]   # (n_future,)
        samp_i = samples[i]               # (n_samples, n_future)
        lo = np.percentile(samp_i, lo_pct, axis=0)
        hi = np.percentile(samp_i, hi_pct, axis=0)
        within = (gt_i >= lo) & (gt_i <= hi)
        coverage_within.append(within.astype(float))
    return np.mean(coverage_within, axis=0)


def compute_metrics(
    ground_truths: np.ndarray,         # (N, n_past + n_future)
    samples: np.ndarray,               # (N, n_samples, n_future)
    n_past: int,
    fullset_mae:  np.ndarray | None = None,
    fullset_crps: np.ndarray | None = None,
    fullset_ci50: np.ndarray | None = None,
    fullset_ci90: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Returns dict with keys 'mae', 'crps', 'ci50', 'ci90' — each (n_future,)."""
    gt_future = ground_truths[:, n_past:]   # (N, n_future)
    return {
        "mae":  fullset_mae  if fullset_mae  is not None
                else _mae_per_step(gt_future, samples),
        "crps": fullset_crps if fullset_crps is not None
                else _crps_per_step(gt_future, samples),
        "ci50": fullset_ci50 if fullset_ci50 is not None
                else _coverage_per_step(gt_future, samples, 25.0, 75.0),
        "ci90": fullset_ci90 if fullset_ci90 is not None
                else _coverage_per_step(gt_future, samples, 5.0,  95.0),
    }


# ---------------------------------------------------------------------------
# Y-limit helpers
# ---------------------------------------------------------------------------

def _ylim_single(
    real_traj: np.ndarray,
    samples: np.ndarray,
) -> tuple[float, float]:
    """Y-limits for one trajectory/model pair (1st–99th pct of samples + full GT)."""
    y_lo = min(float(np.percentile(samples, 1)), float(np.min(real_traj)))
    y_hi = max(float(np.percentile(samples, 99)), float(np.max(real_traj)))
    margin = 0.05 * (y_hi - y_lo) if y_hi > y_lo else 0.1
    return y_lo - margin, y_hi + margin


def _global_ylim(model_data: dict[str, dict]) -> tuple[float, float]:
    """Single y-range encompassing all display trajectories and all models."""
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
# Metric metadata
# ---------------------------------------------------------------------------

METRIC_META: dict[str, dict] = {
    "mae":  {"title": "MAE",           "ylabel": "MAE",      "ideal": None},
    "crps": {"title": "CRPS",          "ylabel": "CRPS",     "ideal": None},
    "ci50": {"title": "CI50 Coverage", "ylabel": "Coverage", "ideal": 0.50},
    "ci90": {"title": "CI90 Coverage", "ylabel": "Coverage", "ideal": 0.90},
}
METRIC_ORDER = ["mae", "crps", "ci50", "ci90"]


# ---------------------------------------------------------------------------
# Figure E: Raw ground-truth trajectories
# ---------------------------------------------------------------------------

def plot_raw_trajectories(
    landscape: str,
    full_trajectories: np.ndarray,   # (N, T)
    n_past: int,
    n_future: int,
    output_dir: Path,
    seed: int = 42,
) -> None:
    """
    Three panels: 10 / 100 / all-N trajectories overlaid.
    Each trajectory is the last (n_past + n_future) steps.
    """
    rng    = np.random.default_rng(seed)
    data   = full_trajectories
    N, T   = data.shape
    n_extrp = n_past + n_future
    x_all   = np.arange(n_extrp)
    counts  = [10, 100, N]

    all_idx    = rng.choice(N, size=N, replace=False)
    all_starts = np.array([
        int(np.random.randint(0, max(1, T - n_extrp + 1)))
        for _ in range(N)
    ])

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

    fig, axes = plt.subplots(1, 3, figsize=(27, 6), sharey=True)
    SINGLE_COLOR = "#4C72B0"

    for ax, count in zip(axes, counts):
        n       = min(count, N)
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
            ax.plot(x_all, seg,
                    color=colors[i] if n <= 10 else SINGLE_COLOR,
                    linewidth=lw, alpha=alpha)

        ax.axvline(x=n_past, color="black", linestyle="--", alpha=0.5, linewidth=1.2)
        ax.set_xlim(0, n_extrp)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel(r"Step $N$", fontsize=12)
        ax.set_title(f"{n} Trajectories", fontsize=13)
        ax.tick_params(labelsize=12)

    axes[0].set_ylabel(_coord_label(landscape), fontsize=13)
    fig.suptitle(
        f"{_land_display(landscape)} — Ground Truth Trajectories",
        fontsize=15, y=1.02,
    )
    plt.tight_layout()
    out_path = output_dir / f"raw_trajectories_{landscape}.png"
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[E] Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure F: Predicted sample overlays (per model, per display trajectory)
# ---------------------------------------------------------------------------

def plot_predicted_trajectories(
    landscape: str,
    model_name: str,
    model_data: dict,
    output_dir: Path,
) -> None:
    """
    Three panels per display trajectory: 1 / 10 / all-S predicted samples.
    """
    ground_truths = model_data["ground_truths"]   # (N_disp, L+H)
    samples       = model_data["samples"]          # (N_disp, S, H)
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

    for traj_idx, traj_label in enumerate(
        model_data.get("display_labels", list(range(ground_truths.shape[0])))
    ):
        gt   = ground_truths[traj_idx]   # (L+H,)
        samp = samples[traj_idx]          # (S, H)

        y_lo = min(float(np.percentile(samp, 1)), float(gt.min()))
        y_hi = max(float(np.percentile(samp, 99)), float(gt.max()))
        margin = 0.05 * (y_hi - y_lo) if y_hi > y_lo else 0.1
        y_lo -= margin
        y_hi += margin

        fig, axes = plt.subplots(1, 3, figsize=(27, 6), sharey=True)

        for ax, count in zip(axes, counts):
            n = min(count, n_samp)

            ax.plot(x_all[:n_past], gt[:n_past], "k-",  linewidth=2.0)
            ax.plot(x_all[n_past:], gt[n_past:], "k--", linewidth=1.5,
                    alpha=0.7, label="Truth (future)")

            if n <= 10:
                colors_s  = plt.get_cmap("tab10")(np.linspace(0, 1, min(n, 10)))
                lw, alpha = 1.5, 1.0
            elif n <= 100:
                colors_s  = [color] * n
                lw, alpha = 0.7, 0.35
            else:
                colors_s  = [color] * n
                lw, alpha = 0.5, 0.15

            for s_idx in range(n):
                c = colors_s[s_idx] if n <= 10 else color
                ax.plot(future_steps, samp[s_idx], color=c,
                        linewidth=lw, alpha=alpha)

            ax.axvline(x=n_past, color="black", linestyle="--",
                       alpha=0.5, linewidth=1.2)
            ax.set_xlim(0, n_extrp)
            ax.set_ylim(y_lo, y_hi)
            ax.set_xlabel(r"Step $N$", fontsize=12)
            ax.set_title(f"{n} Predicted Sample{'s' if n > 1 else ''}", fontsize=13)
            ax.tick_params(labelsize=12)

        axes[0].set_ylabel(_coord_label(landscape), fontsize=13)
        fig.suptitle(
            f"{_land_display(landscape)} — {label} Predicted Trajectories"
            f" (Traj {traj_label})",
            fontsize=15, y=1.02,
        )
        plt.tight_layout()
        out_path = output_dir / (
            f"predicted_trajectories_{landscape}_{model_name}_traj{traj_label}.png"
        )
        fig.savefig(out_path, dpi=600, bbox_inches="tight")
        plt.close(fig)
        print(f"[F] Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure A: Trajectory comparison grid
# ---------------------------------------------------------------------------

def plot_trajectory_comparison_grid(
    landscape: str,
    display_labels: list[int],
    model_data: dict[str, dict],
    n_past: int,
    n_future: int,
    output_dir: Path,
) -> None:
    """
    Rows = models, columns = selected trajectories.
    Each cell: 90%/50% CI bands + ground-truth line.
    Shared y-limits across the entire figure.
    """
    n_models = len(model_data)
    n_traj   = len(display_labels)

    fig, axes = plt.subplots(
        n_models, n_traj,
        figsize=(9 * n_traj, 6 * n_models),
        squeeze=False,
    )

    future_steps = np.arange(n_past, n_past + n_future)
    all_steps    = np.arange(0, n_past + n_future)
    g_lo, g_hi   = _global_ylim(model_data)

    for row_idx, (model_name, mdata) in enumerate(model_data.items()):
        reg    = MODEL_REGISTRY.get(model_name,
                                    {"label": model_name, "color": "tab:orange"})
        color  = reg["color"]
        mlabel = reg["label"]

        samples_all       = mdata["samples"]        # (N_disp, S, H)
        ground_truths_all = mdata["ground_truths"]  # (N_disp, L+H)

        axes[row_idx, 0].set_ylabel(
            f"{mlabel}\n{_coord_label(landscape)}", fontsize=13
        )

        for col_idx, traj_label in enumerate(display_labels):
            ax        = axes[row_idx, col_idx]
            real_traj = ground_truths_all[col_idx]   # (L+H,)
            samp      = samples_all[col_idx]          # (S, H)

            median = np.median(samp, axis=0)
            lo90   = np.percentile(samp,  5, axis=0)
            hi90   = np.percentile(samp, 95, axis=0)
            lo50   = np.percentile(samp, 25, axis=0)
            hi50   = np.percentile(samp, 75, axis=0)

            ax.plot(all_steps,    real_traj, "k-",   linewidth=2.0, label="Truth")
            ax.plot(future_steps, median,    color=color, linewidth=2.0, label="Median")
            ax.fill_between(future_steps, lo90, hi90,
                            color="tab:orange", alpha=0.40, label="90% band")
            ax.fill_between(future_steps, lo50, hi50,
                            color="tab:blue",   alpha=0.40, label="50% band")
            ax.axvline(x=n_past, color="k", linestyle="--", alpha=0.4)
            ax.set_ylim(g_lo, g_hi)
            ax.set_xlim(0, n_past + n_future)
            ax.set_xlabel(r"Step $N$", fontsize=12)
            ax.tick_params(labelsize=12)

            if row_idx == 0:
                ax.set_title(f"Trajectory {traj_label}", fontsize=13, pad=10)

    axes[0, 0].legend(loc="upper left", fontsize=12, framealpha=0.8)
    fig.suptitle(
        f"{_land_display(landscape)} — Trajectory Comparison",
        fontsize=15, y=0.98,
    )
    plt.tight_layout()
    out_path = output_dir / f"trajectory_comparison_{landscape}.png"
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
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
    All subplots share identical bin edges (both x and y) so density
    comparisons across models and trajectories are visually valid.
    Bins are ~2.5× larger than the previous defaults.
    """
    n_models = len(model_data)
    n_traj   = len(display_labels)

    fig, axes = plt.subplots(
        n_models, n_traj,
        figsize=(9 * n_traj, 6 * n_models),
        squeeze=False,
    )

    future_steps = np.arange(n_past, n_past + n_future)
    all_steps    = np.arange(0, n_past + n_future)
    g_lo, g_hi   = _global_ylim(model_data)

    # Shared bin edges — ~2.5× larger bins than before
    # Previous: n_x_bins = max(30, n_future//2), n_y_bins = 60
    # Now:      n_x_bins = max(12, n_future//5), n_y_bins = 24
    n_x_bins = max(12, n_future // 5)
    n_y_bins = 24
    x_edges  = np.linspace(n_past, n_past + n_future, n_x_bins + 1)
    y_edges  = np.linspace(g_lo, g_hi,                n_y_bins + 1)

    for row_idx, (model_name, mdata) in enumerate(model_data.items()):
        reg    = MODEL_REGISTRY.get(model_name,
                                    {"label": model_name, "color": "tab:orange"})
        mlabel = reg["label"]

        samples_all       = mdata["samples"]        # (N_disp, S, H)
        ground_truths_all = mdata["ground_truths"]  # (N_disp, L+H)

        axes[row_idx, 0].set_ylabel(
            f"{mlabel}\n{_coord_label(landscape)}", fontsize=13
        )

        for col_idx, traj_label in enumerate(display_labels):
            ax        = axes[row_idx, col_idx]
            real_traj = ground_truths_all[col_idx]
            samp      = samples_all[col_idx]   # (S, H)

            time_rep = np.tile(future_steps, (samp.shape[0], 1))
            ax.hist2d(
                time_rep.flatten(), samp.flatten(),
                bins=[x_edges, y_edges],          # identical edges every subplot
                cmap="magma", density=True,
                norm=LogNorm(vmin=1e-6),
            )
            ax.plot(all_steps, real_traj, color="lime", linewidth=2.5, label="Truth")
            ax.axvline(x=n_past, color="white", linestyle="--", alpha=0.5)

            ax.set_ylim(g_lo, g_hi)
            ax.set_xlim(0, n_past + n_future)
            ax.set_xlabel(r"Step $N$", fontsize=12)
            ax.tick_params(labelsize=12)

            if row_idx == 0:
                ax.set_title(f"Trajectory {traj_label}", fontsize=13)

    axes[0, 0].legend(loc="upper left", fontsize=12, framealpha=0.8)
    fig.suptitle(
        f"{_land_display(landscape)} — Prediction Density Comparison",
        fontsize=15, y=1.01,
    )
    plt.tight_layout()
    out_path = output_dir / f"histogram2d_comparison_{landscape}.png"
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[B] Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure C: Error metrics grid  (landscapes × metrics)
# ---------------------------------------------------------------------------

def plot_error_metrics_grid(
    landscapes: list[str],
    all_metrics: dict[str, dict[str, dict[str, np.ndarray]]],
    output_dir: Path,
) -> None:
    """
    n_landscapes rows × 4 metric columns.
    Each cell: per-step line plot, one coloured line per model.
    Y-limits shared within each metric column.
    """
    n_land = len(landscapes)
    n_met  = len(METRIC_ORDER)

    fig, axes = plt.subplots(
        n_land, n_met,
        figsize=(9 * n_met, 5 * n_land),
        squeeze=False,
    )

    # Per-metric y-range shared across all landscapes
    metric_ylims: dict[str, tuple[float, float]] = {}
    for metric_key in METRIC_ORDER:
        col_lo =  float("inf")
        col_hi = float("-inf")
        for landscape in landscapes:
            for met_dict in all_metrics[landscape].values():
                vals   = met_dict[metric_key]
                col_lo = min(col_lo, float(np.min(vals)))
                col_hi = max(col_hi, float(np.max(vals)))
        margin = 0.05 * (col_hi - col_lo) if col_hi > col_lo else 0.05
        metric_ylims[metric_key] = (col_lo - margin, col_hi + margin)

    legend_handles: list = []

    for col_idx, metric_key in enumerate(METRIC_ORDER):
        meta       = METRIC_META[metric_key]
        y_lo, y_hi = metric_ylims[metric_key]

        for row_idx, landscape in enumerate(landscapes):
            ax            = axes[row_idx, col_idx]
            model_metrics = all_metrics[landscape]

            for model_name, met_dict in model_metrics.items():
                reg   = MODEL_REGISTRY.get(model_name,
                                           {"label": model_name, "color": "tab:gray"})
                color = reg["color"]
                vals  = met_dict[metric_key]
                steps = np.arange(len(vals))
                ax.plot(steps, vals, color=color, linewidth=2.0, label=reg["label"])

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
            ax.set_xlabel("Forecast step", fontsize=12)
            ax.grid(True, linestyle=":", alpha=0.4)
            ax.tick_params(labelsize=12)

            if col_idx == 0:
                ax.set_ylabel(
                    f"{_land_display(landscape)}\n{meta['ylabel']}", fontsize=12
                )
            else:
                ax.set_ylabel(meta["ylabel"], fontsize=12)

            if row_idx == 0:
                ax.set_title(meta["title"], fontsize=13)

    fig.suptitle("Error Metrics Comparison — All Landscapes", fontsize=15, y=1.01)
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=len(legend_handles),
        fontsize=12,
        frameon=False,
    )
    plt.tight_layout()
    out_path = output_dir / "error_metrics_comparison.png"
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[C] Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure D: Summary tables  — one per dataset, rows=models, cols=all metrics
# ---------------------------------------------------------------------------

def plot_metric_tables(
    landscapes: list[str],
    model_names: list[str],
    all_metrics: dict[str, dict[str, dict[str, np.ndarray]]],
    output_dir: Path,
) -> None:
    """
    One PNG table per landscape.
    Rows  = TSF methods (models).
    Cols  = MAE | CRPS | CI50 | CI90  (all metrics together).

    Also writes a single combined CSV with all landscapes.
    """
    col_labels = [METRIC_META[k]["title"] for k in METRIC_ORDER]

    # ---- combined CSV ----
    csv_path = output_dir / "table_all.csv"
    with open(csv_path, "w") as f:
        f.write("landscape,model," + ",".join(METRIC_ORDER) + "\n")
        for landscape in landscapes:
            for model_name in model_names:
                if model_name not in all_metrics[landscape]:
                    continue
                row_vals = [
                    f"{all_metrics[landscape][model_name][k].mean():.4f}"
                    for k in METRIC_ORDER
                ]
                label = MODEL_REGISTRY.get(model_name, {"label": model_name})["label"]
                f.write(f"{landscape},{label}," + ",".join(row_vals) + "\n")
    print(f"[D] Saved: {csv_path}")

    # ---- one PNG per landscape ----
    for landscape in landscapes:
        row_labels = []
        table_data = []
        for model_name in model_names:
            if model_name not in all_metrics[landscape]:
                continue
            row_labels.append(
                MODEL_REGISTRY.get(model_name, {"label": model_name})["label"]
            )
            table_data.append([
                f"{all_metrics[landscape][model_name][k].mean():.4f}"
                for k in METRIC_ORDER
            ])

        n_rows = len(row_labels)
        n_cols = len(col_labels)

        fig, ax = plt.subplots(
            figsize=(max(9, 2.8 * n_cols), 1.5 + 0.6 * n_rows)
        )
        ax.axis("off")

        tbl = ax.table(
            cellText=table_data,
            rowLabels=row_labels,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(12)
        tbl.scale(1.3, 2.1)

        for (r, c), cell in tbl.get_celld().items():
            if r == 0 or c == -1:
                cell.set_facecolor("#D0D8E8")
                cell.set_text_props(weight="bold", fontsize=12)
            else:
                cell.set_facecolor("#F7F9FC")

        ax.set_title(
            f"{_land_display(landscape)} — Metrics Summary"
            f" (mean over forecast steps)",
            fontsize=13, pad=14,
        )

        png_path = output_dir / f"table_{landscape}.png"
        fig.savefig(png_path, dpi=600, bbox_inches="tight")
        plt.close(fig)
        print(f"[D] Saved: {png_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    rng  = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    out  = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Normalization stats ----
    print("\n=== Normalization stats ===")
    norm_stats = _load_mean_std(args.data_npz)
    if norm_stats:
        for land, (mean, std) in norm_stats.items():
            print(f"  {land}: mean={mean:.6f}, std={std:.6f}")
    else:
        print("  (none provided — raw normalized values will be used)")

    # ---- Parse result specs ----
    specs = [parse_result_spec(s) for s in args.results]

    landscape_models: dict[str, dict[str, str]] = {}
    for spec in specs:
        land = spec["landscape"]
        if land not in landscape_models:
            landscape_models[land] = {}
        landscape_models[land][spec["model"]] = spec["npz_path"]

    landscapes = list(landscape_models.keys())
    print(f"\n=== Landscapes: {landscapes} ===")

    # ---- Load all results and denormalize ----
    print("\n=== Loading .npz results ===")
    all_data: dict[str, dict[str, dict]] = {}

    for land in landscapes:
        all_data[land] = {}
        print(f"\n  [{land}]")
        for model, npz_path in landscape_models[land].items():
            print(f"    {model}:")
            data = load_npz_result(npz_path)

            if land in norm_stats:
                mean, std = norm_stats[land]
                print(f"      Denormalizing: mean={mean:.6f}  std={std:.6f}")
                data["ground_truths"] = _denorm(data["ground_truths"], mean, std)
                data["samples"]       = _denorm(data["samples"],       mean, std)
                print(f"      GT range    : "
                      f"[{data['ground_truths'].min():.4f}, "
                      f"{data['ground_truths'].max():.4f}]")
                print(f"      Sample range: "
                      f"[{data['samples'].min():.4f}, "
                      f"{data['samples'].max():.4f}]")

            all_data[land][model] = data

    # ---- Select display trajectories (consistent across all models/landscapes) ----
    N_min = min(
        d["ground_truths"].shape[0]
        for land_data in all_data.values()
        for d in land_data.values()
    )
    n_show         = min(args.n_traj_show, N_min)
    display_labels = rng.choice(N_min, size=n_show, replace=False).tolist()
    print(f"\n=== Display labels (seed={args.seed}): {display_labels} ===")

    # ---- Compute metrics on FULL dataset ----
    print("\n=== Computing metrics (full testset) ===")
    all_metrics: dict[str, dict[str, dict[str, np.ndarray]]] = {}

    for land in landscapes:
        all_metrics[land] = {}
        for model, data in all_data[land].items():
            n_past = data["n_past"]
            mets   = compute_metrics(
                data["ground_truths"],
                data["samples"],
                n_past,
            )
            all_metrics[land][model] = mets
            label = MODEL_REGISTRY.get(model, {"label": model})["label"]
            print(
                f"  {label:15s} / {land:15s} (N={data['fullset_N']}) — "
                f"MAE={mets['mae'].mean():.4f}  "
                f"CRPS={mets['crps'].mean():.4f}  "
                f"CI50={mets['ci50'].mean():.3f}  "
                f"CI90={mets['ci90'].mean():.3f}"
            )

    # ---- Subset to display trajectories for plotting ----
    print("\n=== Subsetting to display trajectories ===")
    display_data: dict[str, dict[str, dict]] = {}

    for land in landscapes:
        display_data[land] = {}
        for model, data in all_data[land].items():
            display_data[land][model] = {
                "ground_truths": data["ground_truths"][display_labels],
                "samples":       data["samples"][display_labels],
                "display_labels": display_labels,
                "n_past":         data["n_past"],
                "n_future":       data["n_future"],
                "fullset_N":      data["fullset_N"],
            }
            print(f"  {land}/{model}: "
                  f"N_display={len(display_labels)}, "
                  f"L={data['n_past']}, H={data['n_future']}")

    # ---- Generate figures ----
    print("\n=== Generating figures ===")

    model_order = list(display_data[landscapes[0]].keys())

    for land in landscapes:
        n_past   = display_data[land][model_order[0]]["n_past"]
        n_future = display_data[land][model_order[0]]["n_future"]

        # Figure E: raw ground-truth trajectories (from first model's full data)
        first_model_data = all_data[land][model_order[0]]
        print(f"\n  [{land}] Raw trajectory plot...")
        plot_raw_trajectories(
            landscape        = land,
            full_trajectories= first_model_data["ground_truths"],
            n_past           = n_past,
            n_future         = n_future,
            output_dir       = out,
            seed             = args.seed,
        )

        # Figure F: predicted sample overlays per model
        print(f"  [{land}] Predicted trajectory plots...")
        for model_name, mdat in display_data[land].items():
            plot_predicted_trajectories(
                landscape  = land,
                model_name = model_name,
                model_data = mdat,
                output_dir = out,
            )

        # Figure A: trajectory comparison grid
        print(f"  [{land}] Trajectory comparison grid...")
        plot_trajectory_comparison_grid(
            landscape      = land,
            display_labels = display_labels,
            model_data     = display_data[land],
            n_past         = n_past,
            n_future       = n_future,
            output_dir     = out,
        )

        # Figure B: 2-D histogram comparison grid
        print(f"  [{land}] Histogram comparison grid...")
        plot_histogram2d_comparison_grid(
            landscape      = land,
            display_labels = display_labels,
            model_data     = display_data[land],
            n_past         = n_past,
            n_future       = n_future,
            output_dir     = out,
        )

    # Figure C: error metrics grid (all landscapes)
    print("\n  Error metrics comparison grid...")
    plot_error_metrics_grid(
        landscapes  = landscapes,
        all_metrics = all_metrics,
        output_dir  = out,
    )

    # Figure D: summary tables — one per landscape, all metrics as columns
    print("\n  Summary tables...")
    plot_metric_tables(
        landscapes  = landscapes,
        model_names = model_order,
        all_metrics = all_metrics,
        output_dir  = out,
    )

    print(f"\n✓ All figures saved to: {out}")


if __name__ == "__main__":
    main()