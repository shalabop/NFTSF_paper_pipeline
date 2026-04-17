#!/usr/bin/env python3
"""
compare.py
==========================
Generate comparison figures from pre-computed .npz result files.

Figures produced (per landscape unless noted):
  [E] raw_trajectories_{landscape}.png         – ground truth trajectories
  [A] trajectory_comparison_{landscape}.png    – CI bands (50/90%) + median
  [B] histogram2d_comparison_{landscape}.png   – 2D density heatmaps
  [C] error_metrics_comparison.png             – MAE/CRPS/coverage (all landscapes)
  [D] table_{landscape}.png + table_all.csv    – metric summary tables
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

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
    p = argparse.ArgumentParser(description="Generate comparison figures")
    p.add_argument("--results", nargs="+", required=True,
                   metavar="LANDSCAPE:MODEL:NPZ_PATH")
    p.add_argument("--output_dir", default="./comparison_figures")
    p.add_argument("--n_traj_show", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--context_length", "-cl", type=int, default=0)
    p.add_argument("--data_npz", nargs="+", default=None,
                   metavar="LANDSCAPE:PATH")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------
def _load_mean_std(specs: list[str] | None) -> dict[str, tuple[float, float]]:
    result = {}
    if not specs:
        return result
    for spec in specs:
        parts = spec.split(":")
        if len(parts) != 2:
            print(f"WARNING: malformed --data_npz: {spec!r}")
            continue
        land, path = parts
        if not Path(path).exists():
            print(f"WARNING: path not found for {land}: {path}")
            continue
        d = np.load(path)
        result[land] = (float(d["mean"]), float(d["std"]))
        print(f"  [{land}] norm stats: mean={result[land][0]:.6f}  std={result[land][1]:.6f}")
    return result

def _denorm(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    return x * (std + 1e-8) + mean

# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------
def parse_result_spec(spec: str) -> dict:
    parts = spec.split(":")
    if len(parts) != 3:
        print(f"ERROR: --results requires landscape:model:npz_path, got {spec!r}", file=sys.stderr)
        sys.exit(1)
    return {"landscape": parts[0], "model": parts[1], "npz_path": parts[2]}

def load_npz_result(npz_path: str, context_length: int) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    print(f"    Loading {npz_path}")
    print(f"      Keys: {list(data.keys())}")

    samples = data["samples"]
    ground_truth = data["ground_truth"]
    full_trajectories = data["full_trajectories"]
    train_test_split = int(data["train_test_split"])
    H = int(data["prediction_length"])
    N = ground_truth.shape[0]

    # Ensure samples shape (N, S, H)
    if samples.ndim == 3:
        if samples.shape[1] == H:
            samples = samples.transpose(0, 2, 1)
        elif samples.shape[2] != H:
            print(f"      WARNING: sample shape {samples.shape} does not match H={H}")

    N_check, S, H_check = samples.shape
    assert H_check == H, f"Sample H mismatch: {H_check} != {H}"

    #L = H  # context length equals prediction length
    try:
        L = int(data["context_length"])
        print(f'using data["context_length"] = {data["context_length"]}')
    except (KeyError, ValueError):
        L = context_length # Default to prediction length if context length is not specified
        print(f'context_length = {context_length}')
    start_idx = train_test_split - L
    end_idx = train_test_split + H
    if start_idx < 0 or end_idx > full_trajectories.shape[1]:
        print(f"      WARNING: adjusting window")
        end_idx = full_trajectories.shape[1]
        start_idx = end_idx - (L + H)
        if start_idx < 0:
            L = full_trajectories.shape[1] - H
            start_idx = 0

    ground_truths = full_trajectories[:, start_idx:end_idx]  # (N, L+H)
    print(ground_truths[4,4]) 

    print(f"      N={N}, S={S}, L={L}, H={H}")
    print(f"      ground_truths: {ground_truths.shape}, samples: {samples.shape}")

    return {
        "ground_truths": ground_truths.astype(np.float32),
        "samples": samples.astype(np.float32),
        "n_past": L,
        "n_future": H,
        "fullset_N": N,
    }

def _crps_per_step(gt_future: np.ndarray, samples: np.ndarray) -> np.ndarray:
    N, H = gt_future.shape

    coverage_intervals = np.arange(10, 100, 10)  
    crps_t = np.zeros(H)
    
    for t in range(H):
        gt_t = gt_future[:, t]              
        samp_t = samples[:, :, t]         
        mse_vals = []
        for pct in coverage_intervals:
            lo_pct = (100 - pct) / 2.0
            hi_pct = 100 - lo_pct
            lo = np.percentile(samp_t, lo_pct, axis=1)   
            hi = np.percentile(samp_t, hi_pct, axis=1)   
            covered = (gt_t >= lo) & (gt_t <= hi)        
            empirical = np.mean(covered)               
            expected = pct / 100.0
            mse_vals.append((empirical - expected) ** 2)
        crps_t[t] = np.mean(mse_vals)
    return crps_t

def _mae_per_step(gt_future: np.ndarray, samples: np.ndarray) -> np.ndarray:
    median = np.median(samples, axis=1)
    return np.abs(gt_future - median).mean(axis=0)

def _coverage_per_step(gt_future: np.ndarray, samples: np.ndarray, lo_pct: float, hi_pct: float) -> np.ndarray:
    N = gt_future.shape[0]
    cover = []
    for i in range(N):
        lo = np.percentile(samples[i], lo_pct, axis=0)
        hi = np.percentile(samples[i], hi_pct, axis=0)
        cover.append(((gt_future[i] >= lo) & (gt_future[i] <= hi)).astype(float))
    return np.mean(cover, axis=0)

def _coverage_all_intervals(
    gt_future: np.ndarray,   # (N, H)
    samples: np.ndarray,     # (N, S, H)
) -> dict[str, np.ndarray]:
    """
    Compute empirical coverage for CI levels 10%, 20%, ..., 90%.
    Returns dict keyed by e.g. 'ci10', 'ci20', ..., 'ci90'.
    Each value is (H,) array — coverage per forecast step.
    """
    result = {}
    for pct in range(10, 100, 10):
        lo_pct = (100 - pct) / 2.0   # e.g. pct=50 → lo=25, hi=75
        hi_pct = 100 - lo_pct
        key    = f"ci{pct}"
        result[key] = _coverage_per_step(gt_future, samples, lo_pct, hi_pct)
    return result

def compute_metrics(ground_truths: np.ndarray, samples: np.ndarray, n_past: int) -> dict:
    gt_future = ground_truths[:, n_past:]
    metrics = {
        "mae":  _mae_per_step(gt_future, samples),
        "crps": _crps_per_step(gt_future, samples),
    }
    # Add all CI levels ci10 through ci90
    metrics.update(_coverage_all_intervals(gt_future, samples))
    return metrics

# ---------------------------------------------------------------------------
# Y-limit helpers
# ---------------------------------------------------------------------------
def _ylim_single(real_traj: np.ndarray, samples: np.ndarray) -> tuple[float, float]:
    y_lo = min(np.percentile(samples, 1), np.min(real_traj))
    y_hi = max(np.percentile(samples, 99), np.max(real_traj))
    margin = 0.05 * (y_hi - y_lo) if y_hi > y_lo else 0.1
    return y_lo - margin, y_hi + margin

def _global_ylim(model_data: dict[str, dict]) -> tuple[float, float]:
    g_lo, g_hi = float("inf"), float("-inf")
    for mdata in model_data.values():
        n_disp = mdata["ground_truths"].shape[0]
        for col_idx in range(n_disp):
            lo, hi = _ylim_single(mdata["ground_truths"][col_idx], mdata["samples"][col_idx])
            g_lo = min(g_lo, lo)
            g_hi = max(g_hi, hi)
    return g_lo, g_hi

# ---------------------------------------------------------------------------
# Metric metadata
# ---------------------------------------------------------------------------
METRIC_META = {
    "mae":  {"title": "MAE",           "ylabel": "MAE",      "ideal": None},
    "crps": {"title": "CRPS",          "ylabel": "CRPS",     "ideal": None},
}
for _pct in range(10, 100, 10):
    METRIC_META[f"ci{_pct}"] = {
        "title": f"CI{_pct}",
        "ylabel": "Coverage",
        "ideal": _pct / 100.0,
    }

METRIC_ORDER_PLOT = ["mae", "crps", "ci50", "ci90"]
METRIC_ORDER_ALL  = ["mae", "crps"] + [f"ci{p}" for p in range(10, 100, 10)]
METRIC_ORDER   = METRIC_ORDER_ALL  
METRIC_ORDER = ["mae", "crps", "ci50", "ci90"]

# ---------------------------------------------------------------------------
# Figure E: Raw ground-truth trajectories
# ---------------------------------------------------------------------------
def plot_raw_trajectories(landscape: str, full_trajectories: np.ndarray,
                          n_past: int, n_future: int, output_dir: Path, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    data = full_trajectories
    N, T = data.shape
    n_extrp = n_past + n_future
    x_all = np.arange(n_extrp)
    counts = [10, 100, N]

    all_idx = rng.choice(N, size=N, replace=False)
    all_starts = np.array([rng.integers(0, max(1, T - n_extrp + 1)) for _ in range(N)])

    y_lo, y_hi = float("inf"), float("-inf")
    for i in range(N):
        seg = data[all_idx[i], all_starts[i]:all_starts[i] + n_extrp]
        if len(seg) == n_extrp:
            y_lo = min(y_lo, seg.min())
            y_hi = max(y_hi, seg.max())
    margin = 0.05 * (y_hi - y_lo) if y_hi > y_lo else 0.1
    y_lo -= margin
    y_hi += margin

    fig, axes = plt.subplots(1, 3, figsize=(27, 6), sharey=True)
    SINGLE_COLOR = "#4C72B0"

    for ax, count in zip(axes, counts):
        n = min(count, N)
        indices = all_idx[:n]
        starts = all_starts[:n]
        if n <= 10:
            colors = plt.get_cmap("tab10")(np.linspace(0, 1, n))
            lw, alpha = 1.5, 1.0
        elif n <= 100:
            colors = [SINGLE_COLOR] * n
            lw, alpha = 0.7, 0.25
        else:
            colors = [SINGLE_COLOR] * n
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
        ax.set_xlabel(r"Step $N$", fontsize=12)
        ax.set_title(f"{n} Trajectories", fontsize=13)
        ax.tick_params(labelsize=12)

    axes[0].set_ylabel(_coord_label(landscape), fontsize=13)
    fig.suptitle(f"{_land_display(landscape)} — Ground Truth Trajectories", fontsize=15, y=1.02)
    plt.tight_layout()
    out_path = output_dir / f"raw_trajectories_{landscape}.png"
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[E] Saved: {out_path}")

# ---------------------------------------------------------------------------
# Figure A: Trajectory comparison grid (CI bands + median)
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
    For alanine landscapes, y-axis ticks are shown in multiples of π.
    Legend placed inside the first subplot (top left).
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

    # π formatting for alanine
    use_pi_format = landscape in ("alanine_phi", "alanine_psi")
    if use_pi_format:
        from matplotlib.ticker import MultipleLocator, FuncFormatter
        def pi_formatter(x, pos):
            val = round(x / (np.pi/2)) * (np.pi/2)
            if abs(val) < 1e-8:
                return r"$0$"
            num = val / np.pi
            if num == 1:
                return r"$\pi$"
            elif num == -1:
                return r"$-\pi$"
            elif num == 0.5:
                return r"$\frac{\pi}{2}$"
            elif num == -0.5:
                return r"$-\frac{\pi}{2}$"
            else:
                return f"${num:.0f}\\pi$"

    for row_idx, (model_name, mdata) in enumerate(model_data.items()):
        reg    = MODEL_REGISTRY.get(model_name,
                                    {"label": model_name, "color": "tab:orange"})
        color  = reg["color"]
        mlabel = reg["label"]

        samples_all       = mdata["samples"]
        ground_truths_all = mdata["ground_truths"]

        axes[row_idx, 0].set_ylabel(
            f"{mlabel}\n{_coord_label(landscape)}", fontsize=13
        )

        for col_idx, traj_label in enumerate(display_labels):
            ax        = axes[row_idx, col_idx]
            real_traj = ground_truths_all[col_idx]
            samp      = samples_all[col_idx]

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

            if use_pi_format:
                ax.yaxis.set_major_locator(MultipleLocator(np.pi/2))
                ax.yaxis.set_major_formatter(FuncFormatter(pi_formatter))

            if row_idx == 0:
                ax.set_title(f"Trajectory {traj_label}", fontsize=13, pad=10)

            # Place legend in the first subplot (top‑left)
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc='upper left', fontsize=12, framealpha=0.8)

    fig.suptitle(
        f"{_land_display(landscape)} — Trajectory Comparison",
        fontsize=15, y=1,
    )
    plt.subplots_adjust(top=0.93)
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
    Dark background, π‑formatted y‑axis for alanine.
    Legend (Truth) placed inside the first subplot (top left).
    """
    with plt.style.context('dark_background'):
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

        n_x_bins = max(12, n_future // 5)
        n_y_bins = 24
        x_edges  = np.linspace(n_past, n_past + n_future, n_x_bins + 1)
        y_edges  = np.linspace(g_lo, g_hi,                n_y_bins + 1)

        use_pi_format = landscape in ("alanine_phi", "alanine_psi")
        if use_pi_format:
            from matplotlib.ticker import MultipleLocator, FuncFormatter
            def pi_formatter(x, pos):
                val = round(x / (np.pi/2)) * (np.pi/2)
                if abs(val) < 1e-8:
                    return r"$0$"
                num = val / np.pi
                if num == 1:
                    return r"$\pi$"
                elif num == -1:
                    return r"$-\pi$"
                elif num == 0.5:
                    return r"$\frac{\pi}{2}$"
                elif num == -0.5:
                    return r"$-\frac{\pi}{2}$"
                else:
                    return f"${num:.0f}\\pi$"

        for row_idx, (model_name, mdata) in enumerate(model_data.items()):
            reg    = MODEL_REGISTRY.get(model_name,
                                        {"label": model_name, "color": "tab:orange"})
            mlabel = reg["label"]

            samples_all       = mdata["samples"]
            ground_truths_all = mdata["ground_truths"]

            axes[row_idx, 0].set_ylabel(
                f"{mlabel}\n{_coord_label(landscape)}", fontsize=13, color='white'
            )

            for col_idx, traj_label in enumerate(display_labels):
                ax        = axes[row_idx, col_idx]
                real_traj = ground_truths_all[col_idx]
                samp      = samples_all[col_idx]

                time_rep = np.tile(future_steps, (samp.shape[0], 1))
                ax.hist2d(
                    time_rep.flatten(), samp.flatten(),
                    bins=[x_edges, y_edges],
                    cmap="magma", density=True,
                    norm=LogNorm(vmin=1e-6),
                )
                ax.plot(all_steps, real_traj, color="lime", linewidth=2.5, label="Truth")
                ax.axvline(x=n_past, color="white", linestyle="--", alpha=0.5)

                ax.set_ylim(g_lo, g_hi)
                ax.set_xlim(0, n_past + n_future)
                ax.set_xlabel(r"Step $N$", fontsize=12, color='white')
                ax.tick_params(labelsize=12, colors='white')
                ax.set_facecolor('black')

                if use_pi_format:
                    ax.yaxis.set_major_locator(MultipleLocator(np.pi/2))
                    ax.yaxis.set_major_formatter(FuncFormatter(pi_formatter))

                if row_idx == 0:
                    ax.set_title(f"Trajectory {traj_label}", fontsize=13, color='white', pad=10)

                # Legend in top‑left of first subplot
                if row_idx == 0 and col_idx == 0:
                    ax.legend(loc='upper left', fontsize=12, framealpha=0.8,
                              facecolor='black', edgecolor='white', labelcolor='white')

        fig.suptitle(
            f"{_land_display(landscape)} — Prediction Density Comparison",
            fontsize=15, y=1, color='white'
        )
        plt.subplots_adjust(top=0.93)
        plt.tight_layout()
        out_path = output_dir / f"histogram2d_comparison_{landscape}.png"
        fig.savefig(out_path, dpi=600, bbox_inches="tight")
        plt.close(fig)
        print(f"[B] Saved: {out_path}")

# ---------------------------------------------------------------------------
# Figure C: Error metrics grid (landscapes × metrics)
# ---------------------------------------------------------------------------
def plot_error_metrics_grid(
    landscapes: list[str],
    all_metrics: dict[str, dict[str, dict[str, np.ndarray]]],
    output_dir: Path,
) -> None:
    """
    n_landscapes rows × 4 metric columns.
    Legend with title "Methods" placed above the grid, no main figure title.
    """
    n_land = len(landscapes)
    n_met = len(METRIC_ORDER_PLOT)

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

    # Collect handles and labels for legend (Ideal first, then models)
    model_names = list(all_metrics[landscapes[0]].keys())
    legend_handles = []
    legend_labels = []

    # Add Ideal line first
    legend_handles.append(Line2D([0], [0], color="black", linestyle="--", linewidth=0.8))
    legend_labels.append("Ideal")

    # Then add each model
    for model_name in model_names:
        reg = MODEL_REGISTRY.get(model_name, {"label": model_name, "color": "tab:gray"})
        legend_handles.append(Line2D([0], [0], color=reg["color"], linewidth=2.0))
        legend_labels.append(reg["label"])

    # Number of columns for two rows: ceil(n_items/2)
    n_items = len(legend_handles)
    ncol = (n_items + 1) // 2   # two rows

    for col_idx, metric_key in enumerate(METRIC_ORDER_PLOT):
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
                ax.plot(steps, vals, color=color, linewidth=2.0)

            if meta["ideal"] is not None:
                ax.axhline(meta["ideal"], color="black", linestyle="--",
                           linewidth=0.8, alpha=0.6)

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

    # Legend placed above the grid, no main title
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15),
        ncol=ncol,
        title="Methods",
        title_fontsize=12,
        fontsize=12,
        frameon=True,
        edgecolor='gray',
        facecolor='white'
    )
    plt.subplots_adjust(top=0.88)  # make room for legend above

    # No suptitle
    plt.tight_layout()
    out_path = output_dir / "error_metrics_comparison.png"
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[C] Saved: {out_path}")

# ---------------------------------------------------------------------------
# Figure D: Summary tables
# ---------------------------------------------------------------------------
def plot_metric_tables(landscapes, model_names, all_metrics, output_dir):
    col_labels = [METRIC_META[k]["title"] for k in METRIC_ORDER_ALL]
    ideal_vals = {k: METRIC_META[k]["ideal"] for k in METRIC_ORDER_ALL}

    # CSV
    csv_path = output_dir / "table_all.csv"
    with open(csv_path, "w") as f:
        f.write("landscape,model," + ",".join(METRIC_ORDER_ALL) + "\n")
        for landscape in landscapes:
            for model_name in model_names:
                if model_name not in all_metrics[landscape]:
                    continue
                row_vals = [
                    f"{all_metrics[landscape][model_name][k].mean():.4f}"
                    for k in METRIC_ORDER_ALL
                ]
                label = MODEL_REGISTRY.get(model_name, {"label": model_name})["label"]
                f.write(f"{landscape},{label}," + ",".join(row_vals) + "\n")
    print(f"[D] Saved: {csv_path}")

    for landscape in landscapes:
        row_labels, table_data, metric_values = [], [], []
        for model_name in model_names:
            if model_name not in all_metrics[landscape]:
                continue
            row_labels.append(
                MODEL_REGISTRY.get(model_name, {"label": model_name})["label"])
            vals = [
                all_metrics[landscape][model_name][k].mean()
                for k in METRIC_ORDER_ALL
            ]
            metric_values.append(vals)
            table_data.append([f"{v:.4f}" for v in vals])

        # Best value per column
        best_indices = {}
        if metric_values:
            for col_idx, metric_key in enumerate(METRIC_ORDER_ALL):
                col_vals = [metric_values[r][col_idx] for r in range(len(metric_values))]
                ideal = ideal_vals[metric_key]
                if ideal is None:
                    best_indices[col_idx] = int(np.argmin(col_vals))
                else:
                    best_indices[col_idx] = int(
                        np.argmin(np.abs(np.array(col_vals) - ideal)))

        n_rows = len(row_labels)
        n_cols = len(col_labels)
        # Wider figure to accommodate all CI columns
        fig, ax = plt.subplots(
            figsize=(max(9, 1.8 * n_cols), 1.5 + 0.6 * n_rows))
        ax.axis("off")
        tbl = ax.table(
            cellText=table_data,
            rowLabels=row_labels,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)   # slightly smaller to fit all columns
        tbl.scale(1.2, 2.0)

        for (r, c), cell in tbl.get_celld().items():
            if r == 0 or c == -1:
                cell.set_facecolor("#D0D8E8")
                cell.set_text_props(weight="bold", fontsize=10)
            else:
                cell.set_facecolor("#F7F9FC")
                if r > 0 and c >= 0 and c in best_indices and best_indices[c] == r - 1:
                    cell.set_facecolor("#FFF3B0")   # yellow highlight for best
                    cell.set_text_props(weight="bold", fontsize=10)

        ax.set_title(
            f"{_land_display(landscape)} — Metrics Summary (mean over forecast steps)",
            fontsize=13, pad=14)
        png_path = output_dir / f"table_{landscape}.png"
        fig.savefig(png_path, dpi=600, bbox_inches="tight")
        plt.close(fig)
        print(f"[D] Saved: {png_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("\n=== Normalization stats ===")
    norm_stats = _load_mean_std(args.data_npz)

    specs = [parse_result_spec(s) for s in args.results]
    landscape_models: dict[str, dict[str, str]] = {}
    for spec in specs:
        land = spec["landscape"]
        landscape_models.setdefault(land, {})[spec["model"]] = spec["npz_path"]
    landscapes = list(landscape_models.keys())
    print(f"\n=== Landscapes: {landscapes} ===")

    print("\n=== Loading .npz results ===")
    all_data: dict[str, dict[str, dict]] = {}
    for land in landscapes:
        all_data[land] = {}
        print(f"\n  [{land}]")
        for model, npz_path in landscape_models[land].items():
            print(f"    {model}:")
            data = load_npz_result(npz_path, args.context_length)
            if land in norm_stats:
                mean, std = norm_stats[land]
                print(f"      Denormalizing: mean={mean:.6f}  std={std:.6f}")
                data["ground_truths"] = _denorm(data["ground_truths"], mean, std)
                data["samples"] = _denorm(data["samples"], mean, std)
            all_data[land][model] = data

    # Select display trajectories
    N_min = min(d["ground_truths"].shape[0] for land_data in all_data.values() for d in land_data.values())
    n_show = min(args.n_traj_show, N_min)
    display_labels = rng.choice(N_min, size=n_show, replace=False).tolist()
    print(f"\n=== Display labels (seed={args.seed}): {display_labels} ===")

    # Compute metrics on full dataset
    print("\n=== Computing metrics (full testset) ===")
    all_metrics: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for land in landscapes:
        all_metrics[land] = {}
        for model, data in all_data[land].items():
            mets = compute_metrics(data["ground_truths"], data["samples"], data["n_past"])
            all_metrics[land][model] = mets
            label = MODEL_REGISTRY.get(model, {"label": model})["label"]
            print(f"  {label:15s} / {land:15s} (N={data['fullset_N']}) — "
                  f"MAE={mets['mae'].mean():.4f}  CRPS={mets['crps'].mean():.4f}  "
                  f"CI50={mets['ci50'].mean():.3f}  CI90={mets['ci90'].mean():.3f}")

    # Subset for display
    print("\n=== Subsetting to display trajectories ===")
    display_data: dict[str, dict[str, dict]] = {}
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

    # Generate figures
    print("\n=== Generating figures ===")
    model_order = list(display_data[landscapes[0]].keys())

    for land in landscapes:
        n_past = display_data[land][model_order[0]]["n_past"]
        n_future = display_data[land][model_order[0]]["n_future"]

        # Figure E: raw ground truth trajectories
        first_model_data = all_data[land][model_order[0]]
        print(f"\n  [{land}] Raw trajectory plot...")
        plot_raw_trajectories(land, first_model_data["ground_truths"],
                              n_past, n_future, out, args.seed)

        # Figure A: trajectory comparison grid (CI bands)
        print(f"  [{land}] Trajectory comparison grid...")
        plot_trajectory_comparison_grid(land, display_labels, display_data[land],
                                        n_past, n_future, out)

        # Figure B: 2D histogram grid
        print(f"  [{land}] Histogram comparison grid...")
        plot_histogram2d_comparison_grid(land, display_labels, display_data[land],
                                         n_past, n_future, out)

    # Figure C: error metrics grid
    print("\n  Error metrics comparison grid...")
    plot_error_metrics_grid(landscapes, all_metrics, out)

    # Figure D: summary tables
    print("\n  Summary tables...")
    plot_metric_tables(landscapes, model_order, all_metrics, out)

    print(f"\n✓ All figures saved to: {out}")

if __name__ == "__main__":
    main()