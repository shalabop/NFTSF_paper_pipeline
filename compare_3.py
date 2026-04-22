#!/usr/bin/env python3
"""
compare.py – Generate all figures and a single summary table with only the requested metrics.
Assumes samples in .npz files have shape (N, H, S) (trajectories × forecast horizon × samples).
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
    p = argparse.ArgumentParser(description="Generate comparison figures and summary table")
    p.add_argument("--results", nargs="+", required=True,
                   metavar="LANDSCAPE:MODEL:NPZ_PATH")
    p.add_argument("--output_dir", default="./comparison_figures")
    p.add_argument("--n_traj_show", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--context_length", "-cl", type=int, default=0,
                   help="Context length (used if not stored in .npz)")
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
# Result loading – assumes samples shape (N, H, S)
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

    samples = data["samples"]                     # (N, H, S)
    full_trajectories = data["full_trajectories"]
    train_test_split = int(data["train_test_split"])
    H = int(data["prediction_length"])
    N = samples.shape[0]

    if samples.ndim != 3 or samples.shape[1] != H:
        raise ValueError(f"Expected samples shape (N, H, S) with H={H}, got {samples.shape}")
    S = samples.shape[2]

    try:
        L = int(data["context_length"])
        print(f'using data["context_length"] = {data["context_length"]}')
    except (KeyError, ValueError):
        L = context_length
        print(f'context_length = {context_length}')

    start_idx = train_test_split - L
    end_idx   = train_test_split + H
    if start_idx < 0 or end_idx > full_trajectories.shape[1]:
        print(f"      WARNING: adjusting window")
        end_idx = full_trajectories.shape[1]
        start_idx = end_idx - (L + H)
        if start_idx < 0:
            L = full_trajectories.shape[1] - H
            start_idx = 0

    ground_truths = full_trajectories[:, start_idx:end_idx]   # (N, L+H)

    # Optional: inference time
    time_elapsed = float(data["time_elapsed"]) if "time_elapsed" in data else np.nan

    print(f"      N={N}, S={S}, L={L}, H={H}")
    print(f"      ground_truths: {ground_truths.shape}, samples: {samples.shape}")

    return {
        "ground_truths": ground_truths.astype(np.float32),
        "samples": samples.astype(np.float32),      # (N, H, S)
        "n_past": L,
        "n_future": H,
        "fullset_N": N,
        "time_elapsed": time_elapsed,
    }

# ---------------------------------------------------------------------------
# Metric functions (all assume samples shape (N, H, S))
# ---------------------------------------------------------------------------
def _mae_median(gt_future: np.ndarray, samples: np.ndarray) -> float:
    median = np.median(samples, axis=2)          # (N, H)
    return float(np.abs(gt_future - median).mean())

def _mae_mean(gt_future: np.ndarray, samples: np.ndarray) -> float:
    mean_forecast = np.mean(samples, axis=2)     # (N, H)
    return float(np.abs(gt_future - mean_forecast).mean())

def _mae_sample(gt_future: np.ndarray, samples: np.ndarray) -> float:
    # samples: (N, H, S), gt_future: (N, H) -> broadcast to (N, H, S)
    return float(np.abs(samples - gt_future[:, :, np.newaxis]).mean())

def _central_coverage(gt_future: np.ndarray, samples: np.ndarray, pct: int) -> float:
    """Central coverage for a given percentage (0..100)."""
    if pct == 0:
        return 0.0
    if pct == 100:
        return 1.0
    lo_pct = (100 - pct) / 2.0
    hi_pct = 100 - lo_pct
    lo = np.percentile(samples, lo_pct, axis=2)   # (N, H)
    hi = np.percentile(samples, hi_pct, axis=2)
    covered = (gt_future >= lo) & (gt_future <= hi)
    return float(np.mean(covered))

def _isce_central(gt_future: np.ndarray, samples: np.ndarray) -> float:
    """Integrated squared calibration error for central intervals (0%–100%, step 1%)."""
    H = gt_future.shape[1]
    pcts = np.arange(0, 101, 1)   # 0,1,...,100
    isce_t = np.zeros(H)
    for t in range(H):
        gt_t = gt_future[:, t]
        samp_t = samples[:, t, :]                # (N, S)
        err = 0.0
        for pct in pcts:
            if pct == 0:
                empirical = 0.0
            elif pct == 100:
                empirical = 1.0
            else:
                lo_pct = (100 - pct) / 2.0
                hi_pct = 100 - lo_pct
                lo = np.percentile(samp_t, lo_pct, axis=1)
                hi = np.percentile(samp_t, hi_pct, axis=1)
                covered = (gt_t >= lo) & (gt_t <= hi)
                empirical = np.mean(covered)
            expected = pct / 100.0
            err += (empirical - expected) ** 2
        isce_t[t] = err / len(pcts)
    return float(np.mean(isce_t))

def _lower_coverage(gt_future: np.ndarray, samples: np.ndarray, q: float) -> float:
    """One‑sided lower coverage for quantile q (0..1)."""
    if q == 0.0:
        return 0.0
    if q == 1.0:
        return 1.0
    q_quantile = np.quantile(samples, q, axis=2)   # (N, H)
    below = (gt_future <= q_quantile)
    return float(np.mean(below))

def _isce_lower(gt_future: np.ndarray, samples: np.ndarray) -> float:
    """Integrated squared calibration error for lower quantiles (0.00–1.00, step 0.01)."""
    H = gt_future.shape[1]
    qs = np.arange(0.0, 1.01, 0.01)   # 0.00,0.01,...,1.00
    isce_t = np.zeros(H)
    for t in range(H):
        gt_t = gt_future[:, t]
        samp_t = samples[:, t, :]
        err = 0.0
        for q in qs:
            if q == 0.0:
                empirical = 0.0
            elif q == 1.0:
                empirical = 1.0
            else:
                q_val = np.quantile(samp_t, q, axis=1)
                empirical = np.mean(gt_t < q_val)
            err += (empirical - q) ** 2
        isce_t[t] = err / len(qs)
    return float(np.mean(isce_t))

def compute_metrics(ground_truths: np.ndarray, samples: np.ndarray, n_past: int,
                    time_elapsed: float) -> dict:
    gt_future = ground_truths[:, n_past:]   # (N, H)
    metrics = {
        "mae_median": _mae_median(gt_future, samples),
        "mae_mean":   _mae_mean(gt_future, samples),
        "mae_sample": _mae_sample(gt_future, samples),
        "ci50": _central_coverage(gt_future, samples, 50),
        "ci90": _central_coverage(gt_future, samples, 90),
        "isce_central": _isce_central(gt_future, samples),
        "q50": _lower_coverage(gt_future, samples, 0.5),
        "q90": _lower_coverage(gt_future, samples, 0.9),
        "isce_lower": _isce_lower(gt_future, samples),
        "time_elapsed": time_elapsed,
    }
    return metrics

# ---------------------------------------------------------------------------
# Y-limit helpers for plots
# ---------------------------------------------------------------------------
def _ylim_single(real_traj: np.ndarray, samples: np.ndarray) -> tuple[float, float]:
    y_lo = min(np.min(real_traj), np.percentile(samples, 1))
    y_hi = max(np.max(real_traj), np.percentile(samples, 99))
    margin = 0.05 * (y_hi - y_lo) if y_hi > y_lo else 0.1
    return y_lo - margin, y_hi + margin

def _global_ylim(model_data: dict[str, dict]) -> tuple[float, float]:
    g_lo, g_hi = float("inf"), float("-inf")
    for mdata in model_data.values():
        n_disp = mdata["ground_truths"].shape[0]
        for col_idx in range(n_disp):
            real_traj = mdata["ground_truths"][col_idx]
            samp = mdata["samples"][col_idx]
            lo, hi = _ylim_single(real_traj, samp)
            g_lo = min(g_lo, lo)
            g_hi = max(g_hi, hi)
    return g_lo, g_hi

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
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
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
        reg    = MODEL_REGISTRY.get(model_name, {"label": model_name, "color": "tab:orange"})
        color  = reg["color"]
        mlabel = reg["label"]

        samples_all       = mdata["samples"]
        ground_truths_all = mdata["ground_truths"]

        axes[row_idx, 0].set_ylabel(f"{mlabel}\n{_coord_label(landscape)}", fontsize=13)

        for col_idx, traj_label in enumerate(display_labels):
            ax = axes[row_idx, col_idx]
            real_traj = ground_truths_all[col_idx]
            samp = samples_all[col_idx]

            median = np.median(samp, axis=1)
            lo90   = np.percentile(samp,  5, axis=1)
            hi90   = np.percentile(samp, 95, axis=1)
            lo50   = np.percentile(samp, 25, axis=1)
            hi50   = np.percentile(samp, 75, axis=1)

            ax.plot(all_steps,    real_traj, "k-",   linewidth=2.0, label="Truth")
            ax.plot(future_steps, median,    color=color, linewidth=2.0, label="Median")
            ax.fill_between(future_steps, lo90, hi90, color="tab:orange", alpha=0.40, label="90% band")
            ax.fill_between(future_steps, lo50, hi50, color="tab:blue",   alpha=0.40, label="50% band")
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

            if row_idx == 0 and col_idx == 0:
                ax.legend(loc='upper left', fontsize=12, framealpha=0.8)

    fig.suptitle(f"{_land_display(landscape)} — Trajectory Comparison", fontsize=15, y=1)
    plt.subplots_adjust(top=0.93)
    plt.tight_layout()
    out_path = output_dir / f"trajectory_comparison_{landscape}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
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
            reg    = MODEL_REGISTRY.get(model_name, {"label": model_name, "color": "tab:orange"})
            mlabel = reg["label"]

            samples_all       = mdata["samples"]
            ground_truths_all = mdata["ground_truths"]

            axes[row_idx, 0].set_ylabel(f"{mlabel}\n{_coord_label(landscape)}", fontsize=13, color='white')

            for col_idx, traj_label in enumerate(display_labels):
                ax = axes[row_idx, col_idx]
                real_traj = ground_truths_all[col_idx]
                samp = samples_all[col_idx]

                samp_for_hist = samp.T
                time_rep = np.tile(future_steps, (samp_for_hist.shape[0], 1))
                ax.hist2d(
                    time_rep.flatten(), samp_for_hist.flatten(),
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

                if row_idx == 0 and col_idx == 0:
                    ax.legend(loc='upper left', fontsize=12, framealpha=0.8,
                              facecolor='black', edgecolor='white', labelcolor='white')

        fig.suptitle(f"{_land_display(landscape)} — Prediction Density Comparison", fontsize=15, y=1, color='white')
        plt.subplots_adjust(top=0.93)
        plt.tight_layout()
        out_path = output_dir / f"histogram2d_comparison_{landscape}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[B] Saved: {out_path}")

# ---------------------------------------------------------------------------
# Figure C: Error metrics grid (MAE_median, ISCE_central, CI50, CI90)
# ---------------------------------------------------------------------------
def plot_error_metrics_grid(
    landscapes: list[str],
    all_metrics: dict[str, dict[str, dict]],
    output_dir: Path,
) -> None:
    n_land = len(landscapes)
    n_met = 4  # mae_median, isce_central, ci50, ci90

    fig, axes = plt.subplots(n_land, n_met, figsize=(9 * n_met, 5 * n_land), squeeze=False)

    metric_keys = ["mae_median", "isce_central", "ci50", "ci90"]
    metric_ylims = {}
    for mk in metric_keys:
        lo, hi = float("inf"), float("-inf")
        for land in landscapes:
            for met_dict in all_metrics[land].values():
                vals = met_dict[mk]
                if isinstance(vals, np.ndarray):
                    vals = vals.mean()
                lo = min(lo, vals)
                hi = max(hi, vals)
        margin = 0.05 * (hi - lo) if hi > lo else 0.05
        metric_ylims[mk] = (lo - margin, hi + margin)

    model_names = list(all_metrics[landscapes[0]].keys())
    legend_handles = [Line2D([0], [0], color="black", linestyle="--", linewidth=0.8)]
    legend_labels = ["Ideal"]
    for mn in model_names:
        reg = MODEL_REGISTRY.get(mn, {"label": mn, "color": "tab:gray"})
        legend_handles.append(Line2D([0], [0], color=reg["color"], linewidth=2.0))
        legend_labels.append(reg["label"])
    ncol = (len(legend_handles) + 1) // 2

    for col_idx, mk in enumerate(metric_keys):
        meta = {"title": mk.replace("_", " ").title(), "ylabel": "Value"}
        if mk == "ci50":
            ideal = 0.5
        elif mk == "ci90":
            ideal = 0.9
        else:
            ideal = None
        y_lo, y_hi = metric_ylims[mk]

        for row_idx, land in enumerate(landscapes):
            ax = axes[row_idx, col_idx]
            model_metrics = all_metrics[land]

            for model_name, met_dict in model_metrics.items():
                reg = MODEL_REGISTRY.get(model_name, {"label": model_name, "color": "tab:gray"})
                color = reg["color"]
                vals = met_dict[mk]
                if isinstance(vals, np.ndarray):
                    vals = vals.mean()
                ax.bar(model_name, vals, color=color, alpha=0.7)
                ax.tick_params(axis='x', rotation=45)

            if ideal is not None:
                ax.axhline(ideal, color="black", linestyle="--", linewidth=0.8, alpha=0.6)

            ax.set_ylim(y_lo, y_hi)
            ax.set_ylabel(meta["ylabel"], fontsize=12)
            ax.set_title(meta["title"], fontsize=13)
            ax.grid(True, linestyle=":", alpha=0.4)

        # Set xlabel only for bottom row
        for ax in axes[-1, :]:
            ax.set_xlabel("Model", fontsize=12)

    fig.legend(handles=legend_handles, labels=legend_labels,
               loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=ncol,
               title="Methods", title_fontsize=12, fontsize=12,
               frameon=True, edgecolor='gray', facecolor='white')
    plt.subplots_adjust(top=0.88)
    plt.tight_layout()
    out_path = output_dir / "error_metrics_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[C] Saved: {out_path}")

# ---------------------------------------------------------------------------
# Figure D: Single summary table (only requested metrics)
# ---------------------------------------------------------------------------
def plot_summary_table(landscapes, model_names, all_metrics, output_dir):
    col_headers = [
        "Model", "MAE (med)", "MAE (mean)", "MAE (sample)",
        "CI50", "CI90", "ISCE_central", "Q50", "Q90", "ISCE_lower", "Time (s)"
    ]
    metric_keys = ["mae_median", "mae_mean", "mae_sample",
                   "ci50", "ci90", "isce_central", "q50", "q90", "isce_lower", "time_elapsed"]

    # CSV
    csv_path = output_dir / "summary_table.csv"
    with open(csv_path, "w") as f:
        f.write("landscape,model," + ",".join(metric_keys) + "\n")
        for land in landscapes:
            for model_name in model_names:
                if model_name not in all_metrics[land]:
                    continue
                mets = all_metrics[land][model_name]
                label = MODEL_REGISTRY.get(model_name, {"label": model_name})["label"]
                row = [land, label]
                for k in metric_keys:
                    val = mets[k]
                    if isinstance(val, np.ndarray):
                        val = val.mean()
                    row.append(f"{val:.6f}" if k != "time_elapsed" else f"{val:.2f}")
                f.write(",".join(row) + "\n")
    print(f"CSV saved: {csv_path}")

    # PNG table per landscape
    for land in landscapes:
        rows = []
        numeric_vals = []
        for model_name in model_names:
            if model_name not in all_metrics[land]:
                continue
            mets = all_metrics[land][model_name]
            label = MODEL_REGISTRY.get(model_name, {"label": model_name})["label"]
            vals = []
            for k in metric_keys:
                v = mets[k]
                if isinstance(v, np.ndarray):
                    v = v.mean()
                vals.append(v)
            numeric_vals.append(vals)
            # Format for display
            row = [label]
            row.append(f"{vals[0]:.4f}")   # MAE median
            row.append(f"{vals[1]:.4f}")   # MAE mean
            row.append(f"{vals[2]:.4f}")   # MAE sample
            row.append(f"{vals[3]:.3f}")   # CI50
            row.append(f"{vals[4]:.3f}")   # CI90
            row.append(f"{vals[5]:.4f}")   # ISCE_central
            row.append(f"{vals[6]:.3f}")   # Q50
            row.append(f"{vals[7]:.3f}")   # Q90
            row.append(f"{vals[8]:.4f}")   # ISCE_lower
            row.append(f"{vals[9]:.2f}")   # Time (s)
            rows.append(row)

        if not rows:
            continue

        # Determine best indices per column
        best_indices = {}
        # For each column (1-based index after model name)
        for col_idx in range(1, len(rows[0])):
            col_vals = [float(row[col_idx]) for row in rows]
            # Which column?
            if col_idx in [1,2,3,6,9,10]:   # MAE_median, MAE_mean, MAE_sample, ISCE_central, ISCE_lower, time
                best_idx = int(np.argmin(col_vals))
            elif col_idx == 4:   # CI50
                best_idx = int(np.argmin(np.abs(np.array(col_vals) - 0.5)))
            elif col_idx == 5:   # CI90
                best_idx = int(np.argmin(np.abs(np.array(col_vals) - 0.9)))
            elif col_idx == 7:   # Q50
                best_idx = int(np.argmin(np.abs(np.array(col_vals) - 0.5)))
            elif col_idx == 8:   # Q90
                best_idx = int(np.argmin(np.abs(np.array(col_vals) - 0.9)))
            else:
                best_idx = None
            if best_idx is not None:
                best_indices[col_idx] = best_idx

        fig, ax = plt.subplots(figsize=(14, 2 + len(rows)))
        ax.axis("off")
        table = ax.table(cellText=rows, colLabels=col_headers, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.8)

        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_facecolor("#D0D8E8")
                cell.set_text_props(weight="bold")
            else:
                cell.set_facecolor("#F7F9FC")
                if c in best_indices and best_indices[c] == r - 1:
                    cell.set_facecolor("#90EE90")
                    cell.set_text_props(weight="bold")

        ax.set_title(f"{_land_display(land)} — Summary Metrics", fontsize=13, pad=14)
        png_path = output_dir / f"summary_table_{land}.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
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
    all_metrics: dict[str, dict[str, dict]] = {}
    for land in landscapes:
        all_metrics[land] = {}
        for model, data in all_data[land].items():
            mets = compute_metrics(data["ground_truths"], data["samples"],
                                   data["n_past"], data["time_elapsed"])
            all_metrics[land][model] = mets
            label = MODEL_REGISTRY.get(model, {"label": model})["label"]
            print(f"  {label:15s} / {land:15s} (N={data['fullset_N']}) — "
                  f"MAE_med={mets['mae_median']:.4f}  MAE_mean={mets['mae_mean']:.4f}  "
                  f"MAE_sample={mets['mae_sample']:.4f}  CI50={mets['ci50']:.3f}  "
                  f"CI90={mets['ci90']:.3f}  ISCE_cen={mets['isce_central']:.4f}  "
                  f"Q50={mets['q50']:.3f}  Q90={mets['q90']:.3f}  ISCE_low={mets['isce_lower']:.4f}  "
                  f"Time={mets['time_elapsed']:.2f}s")

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

        first_model_data = all_data[land][model_order[0]]
        print(f"\n  [{land}] Raw trajectory plot...")
        plot_raw_trajectories(land, first_model_data["ground_truths"],
                              n_past, n_future, out, args.seed)

        print(f"  [{land}] Trajectory comparison grid...")
        plot_trajectory_comparison_grid(land, display_labels, display_data[land],
                                        n_past, n_future, out)

        print(f"  [{land}] Histogram comparison grid...")
        plot_histogram2d_comparison_grid(land, display_labels, display_data[land],
                                         n_past, n_future, out)

    print("\n  Error metrics comparison grid...")
    plot_error_metrics_grid(landscapes, all_metrics, out)

    print("\n  Summary table...")
    plot_summary_table(landscapes, model_order, all_metrics, out)

    print(f"\n✓ All outputs saved to: {out}")

if __name__ == "__main__":
    main()