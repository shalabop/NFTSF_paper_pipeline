#!/usr/bin/env python3
"""
compare_full.py – Generate all figures (trajectory comparison, histograms, error metrics grid)
plus summary table with MAE, CI50, CI90, ISCE, and CRPS decomposition.
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

# Import CRPS decomposition
from metrics.crps_decomposition import crps_ensemble, crps_decomposition

DPI = 600

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
    p = argparse.ArgumentParser()
    p.add_argument("--results", nargs="+", required=True,
                   help="Format: LANDSCAPE:MODEL:NPZ_PATH")
    p.add_argument("--output_dir", default="./comparison")
    p.add_argument("--n_traj_show", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--context_length", "-cl", type=int, default=0,
                   help="Fallback context length if not stored in .npz")
    p.add_argument("--data_npz", nargs="+", required=True,
                   metavar="LANDSCAPE:PATH", help="Normalization stats")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------
def _load_mean_std(specs):
    result = {}
    for spec in specs:
        parts = spec.split(":")
        if len(parts) != 2:
            raise ValueError(f"data_npz must be LANDSCAPE:PATH, got {spec}")
        land, path = parts
        d = np.load(path)
        result[land] = (float(d["mean"]), float(d["std"]))
        print(f"  [{land}] norm stats: mean={result[land][0]:.6f}  std={result[land][1]:.6f}")
    return result

def _denorm(x, mean, std):
    return x * (std + 1e-8) + mean

# ---------------------------------------------------------------------------
# Result loading – assumes samples shape (N, H, S)
# ---------------------------------------------------------------------------
def load_npz_result(npz_path: str, context_length_fallback: int, mean=None, std=None):
    data = np.load(npz_path, allow_pickle=True)
    samples = data["samples"]                     # (N, H, S)
    full_trajectories = data["full_trajectories"]
    train_test_split = int(data["train_test_split"])
    H = int(data["prediction_length"])
    N = samples.shape[0]
    if samples.ndim != 3 or samples.shape[1] != H:
        raise ValueError(f"Expected (N, H, S) with H={H}, got {samples.shape}")
    S = samples.shape[2]
    try:
        L = int(data["context_length"])
    except (KeyError, ValueError):
        L = context_length_fallback
    start_idx = train_test_split - L
    end_idx   = train_test_split + H
    if start_idx < 0 or end_idx > full_trajectories.shape[1]:
        print(f"WARNING: adjusting window for {npz_path}")
        end_idx = full_trajectories.shape[1]
        start_idx = end_idx - (L + H)
        if start_idx < 0:
            L = full_trajectories.shape[1] - H
            start_idx = 0
    ground_truths = full_trajectories[:, start_idx:end_idx]   # (N, L+H)
    if mean is not None and std is not None:
        ground_truths = _denorm(ground_truths, mean, std)
        samples = _denorm(samples, mean, std)
    return {
        "ground_truths": ground_truths.astype(np.float32),
        "samples": samples.astype(np.float32),
        "n_past": L,
        "n_future": H,
        "fullset_N": N,
        "pred_len": H,
    }

# ---------------------------------------------------------------------------
# Metrics (all assume samples shape (N, H, S))
# ---------------------------------------------------------------------------
def _mae_median_per_step(gt_future, samples):
    median = np.median(samples, axis=2)
    return np.abs(gt_future - median).mean(axis=0)

def _mae_mean_per_step(gt_future, samples):
    mean_forecast = np.mean(samples, axis=2)
    return np.abs(gt_future - mean_forecast).mean(axis=0)

def _mae_sample_per_step(gt_future, samples):
    return np.abs(samples - gt_future[:, :, np.newaxis]).mean(axis=(0, 1))

def _ci_coverage_per_step(gt_future, samples, pct):
    if pct == 0:
        return np.zeros(gt_future.shape[1])
    if pct == 100:
        return np.ones(gt_future.shape[1])
    lo = (100 - pct) / 2.0
    hi = 100 - lo
    lo_q = np.percentile(samples, lo, axis=2)
    hi_q = np.percentile(samples, hi, axis=2)
    covered = (gt_future >= lo_q) & (gt_future <= hi_q)
    return np.mean(covered, axis=0)

def _isce_central_per_step(gt_future, samples):
    H = gt_future.shape[1]
    pcts = np.arange(0, 101, 1)
    isce_t = np.zeros(H)
    for t in range(H):
        gt_t = gt_future[:, t]
        samp_t = samples[:, t, :]
        err = 0.0
        for pct in pcts:
            if pct == 0:
                emp = 0.0
            elif pct == 100:
                emp = 1.0
            else:
                lo = (100 - pct) / 2.0
                hi = 100 - lo
                lo_q = np.percentile(samp_t, lo, axis=1)
                hi_q = np.percentile(samp_t, hi, axis=1)
                emp = np.mean((gt_t >= lo_q) & (gt_t <= hi_q))
            err += (emp - pct/100.0)**2
        isce_t[t] = err / len(pcts)
    return isce_t

def compute_metrics(ground_truths, samples, n_past):
    gt_future = ground_truths[:, n_past:]   # (N, H)
    # Flatten for CRPS decomposition: (N*H, S) and (N*H,)
    N, H = gt_future.shape
    samples_flat = samples.reshape(-1, samples.shape[2])  # (N*H, S)
    obs_flat = gt_future.reshape(-1)                      # (N*H,)
    # CRPS decomposition
    crps_dict = crps_decomposition(samples_flat, obs_flat, sample_axis=-1, return_per_rank=False)
    crps_val = crps_dict["crps"]
    reliability = crps_dict["reliability"]
    resolution = crps_dict["resolution"]
    uncertainty = crps_dict["uncertainty"]

    metrics = {
        "mae_median_step": _mae_median_per_step(gt_future, samples),
        "mae_mean_step":   _mae_mean_per_step(gt_future, samples),
        "mae_sample_step": _mae_sample_per_step(gt_future, samples),
        "ci50_step": _ci_coverage_per_step(gt_future, samples, 50),
        "ci90_step": _ci_coverage_per_step(gt_future, samples, 90),
        "isce_step": _isce_central_per_step(gt_future, samples),
        "crps": crps_val,
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
    }
    # Scalar means for table
    metrics["mae_median"] = np.mean(metrics["mae_median_step"])
    metrics["mae_mean"]   = np.mean(metrics["mae_mean_step"])
    metrics["mae_sample"] = np.mean(metrics["mae_sample_step"])
    metrics["ci50"] = np.mean(metrics["ci50_step"])
    metrics["ci90"] = np.mean(metrics["ci90_step"])
    metrics["isce_mean"] = np.mean(metrics["isce_step"])
    return metrics

# ---------------------------------------------------------------------------
# Figure E: Raw ground-truth trajectories
# ---------------------------------------------------------------------------
def plot_raw_trajectories(landscape, full_trajectories, n_past, n_future, output_dir, seed=42):
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

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3), sharey=True)
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
            if len(seg) < n_extrp: continue
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
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[E] Saved: {out_path}")

# ---------------------------------------------------------------------------
# Figure A: Trajectory comparison grid (CI bands + median)
# ---------------------------------------------------------------------------
def plot_trajectory_comparison_grid(landscape, display_labels, model_data, n_past, n_future, output_dir):
    n_models = len(model_data)
    n_traj = len(display_labels)
    fig, axes = plt.subplots(n_models, n_traj, figsize=(5*n_traj, 3.5*n_models), squeeze=False)
    future_steps = np.arange(n_past, n_past + n_future)
    all_steps = np.arange(0, n_past + n_future)
    g_lo, g_hi = float("inf"), float("-inf")
    for mdata in model_data.values():
        for gt in mdata["ground_truths"]:
            g_lo = min(g_lo, gt.min())
            g_hi = max(g_hi, gt.max())
    margin = 0.05 * (g_hi - g_lo) if g_hi > g_lo else 0.1
    g_lo -= margin; g_hi += margin

    use_pi_format = landscape in ("alanine_phi", "alanine_psi")
    if use_pi_format:
        from matplotlib.ticker import MultipleLocator, FuncFormatter
        def pi_formatter(x, pos):
            val = round(x / (np.pi/2)) * (np.pi/2)
            if abs(val) < 1e-8: return r"$0$"
            num = val / np.pi
            if num == 1: return r"$\pi$"
            elif num == -1: return r"$-\pi$"
            elif num == 0.5: return r"$\frac{\pi}{2}$"
            elif num == -0.5: return r"$-\frac{\pi}{2}$"
            else: return f"${num:.0f}\\pi$"

    for row_idx, (model_name, mdata) in enumerate(model_data.items()):
        reg = MODEL_REGISTRY.get(model_name, {"label": model_name, "color": "tab:orange"})
        color = reg["color"]
        mlabel = reg["label"]
        samples_all = mdata["samples"]
        ground_truths_all = mdata["ground_truths"]
        axes[row_idx, 0].set_ylabel(f"{mlabel}\n{_coord_label(landscape)}", fontsize=13)
        for col_idx in range(n_traj):
            ax = axes[row_idx, col_idx]
            real_traj = ground_truths_all[col_idx]
            samp = samples_all[col_idx]
            median = np.median(samp, axis=1)
            lo90 = np.percentile(samp, 5, axis=1)
            hi90 = np.percentile(samp, 95, axis=1)
            lo50 = np.percentile(samp, 25, axis=1)
            hi50 = np.percentile(samp, 75, axis=1)
            ax.plot(all_steps, real_traj, "k-", linewidth=2.0, label="Truth")
            ax.plot(future_steps, median, color=color, linewidth=2.0, label="Median")
            ax.fill_between(future_steps, lo90, hi90, color="tab:orange", alpha=0.40, label="90% band")
            ax.fill_between(future_steps, lo50, hi50, color="tab:blue", alpha=0.40, label="50% band")
            ax.axvline(x=n_past, color="k", linestyle="--", alpha=0.4)
            ax.set_ylim(g_lo, g_hi)
            ax.set_xlim(0, n_past + n_future)
            ax.set_xlabel(r"Step $N$", fontsize=12)
            ax.tick_params(labelsize=12)
            if use_pi_format:
                ax.yaxis.set_major_locator(MultipleLocator(np.pi/2))
                ax.yaxis.set_major_formatter(FuncFormatter(pi_formatter))
            if row_idx == 0:
                ax.set_title(f"Trajectory {col_idx+1}", fontsize=13, pad=10)
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc='upper left', fontsize=12, framealpha=0.8)
    fig.suptitle(f"{_land_display(landscape)} — Trajectory Comparison", fontsize=15, y=1)
    plt.subplots_adjust(top=0.93)
    plt.tight_layout()
    out_path = output_dir / f"trajectory_comparison_{landscape}.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[A] Saved: {out_path}")

# ---------------------------------------------------------------------------
# Figure B: 2-D histogram comparison grid
# ---------------------------------------------------------------------------
def plot_histogram2d_comparison_grid(landscape, display_labels, model_data, n_past, n_future, output_dir):
    with plt.style.context('dark_background'):
        n_models = len(model_data)
        n_traj = len(display_labels)
        fig, axes = plt.subplots(n_models, n_traj, figsize=(5*n_traj, 3.5*n_models), squeeze=False)
        future_steps = np.arange(n_past, n_past + n_future)
        all_steps = np.arange(0, n_past + n_future)
        g_lo, g_hi = float("inf"), float("-inf")
        for mdata in model_data.values():
            for gt in mdata["ground_truths"]:
                g_lo = min(g_lo, gt.min())
                g_hi = max(g_hi, gt.max())
        margin = 0.05 * (g_hi - g_lo) if g_hi > g_lo else 0.1
        g_lo -= margin; g_hi += margin
        n_x_bins = max(12, n_future // 5)
        n_y_bins = 24
        x_edges = np.linspace(n_past, n_past + n_future, n_x_bins + 1)
        y_edges = np.linspace(g_lo, g_hi, n_y_bins + 1)
        use_pi_format = landscape in ("alanine_phi", "alanine_psi")
        if use_pi_format:
            from matplotlib.ticker import MultipleLocator, FuncFormatter
            def pi_formatter(x, pos):
                val = round(x / (np.pi/2)) * (np.pi/2)
                if abs(val) < 1e-8: return r"$0$"
                num = val / np.pi
                if num == 1: return r"$\pi$"
                elif num == -1: return r"$-\pi$"
                elif num == 0.5: return r"$\frac{\pi}{2}$"
                elif num == -0.5: return r"$-\frac{\pi}{2}$"
                else: return f"${num:.0f}\\pi$"
        for row_idx, (model_name, mdata) in enumerate(model_data.items()):
            reg = MODEL_REGISTRY.get(model_name, {"label": model_name, "color": "tab:orange"})
            mlabel = reg["label"]
            samples_all = mdata["samples"]
            ground_truths_all = mdata["ground_truths"]
            axes[row_idx, 0].set_ylabel(f"{mlabel}\n{_coord_label(landscape)}", fontsize=13, color='white')
            for col_idx in range(n_traj):
                ax = axes[row_idx, col_idx]
                real_traj = ground_truths_all[col_idx]
                samp = samples_all[col_idx]
                samp_for_hist = samp.T
                time_rep = np.tile(future_steps, (samp_for_hist.shape[0], 1))
                ax.hist2d(time_rep.flatten(), samp_for_hist.flatten(),
                          bins=[x_edges, y_edges], cmap="magma", density=True, norm=LogNorm(vmin=1e-6))
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
                    ax.set_title(f"Trajectory {col_idx+1}", fontsize=13, color='white', pad=10)
                if row_idx == 0 and col_idx == 0:
                    ax.legend(loc='upper left', fontsize=12, framealpha=0.8,
                              facecolor='black', edgecolor='white', labelcolor='white')
        fig.suptitle(f"{_land_display(landscape)} — Prediction Density Comparison", fontsize=15, y=1, color='white')
        plt.subplots_adjust(top=0.93)
        plt.tight_layout()
        out_path = output_dir / f"histogram2d_comparison_{landscape}.png"
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"[B] Saved: {out_path}")

# ---------------------------------------------------------------------------
# Figure C: Error metrics grid (MAE_median, ISCE*1000, CI50, CI90)
# ---------------------------------------------------------------------------
def plot_error_metrics_grid(land, length, model_names, all_metrics, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(8, 5))
    metric_keys = [("mae_median_step", "MAE (median)"),
                   ("isce_step", "ISCE"),
                   ("ci50_step", "CI50"),
                   ("ci90_step", "CI90")]
    for ax, (key, title) in zip(axes.flatten(), metric_keys):
        for mn in model_names:
            label = MODEL_REGISTRY.get(mn, {"label": mn})["label"]
            color = MODEL_REGISTRY.get(mn, {"color": "tab:gray"})["color"]
            vals = all_metrics[mn][key]
            if key == "isce_step":
                vals = vals * 1000
                title = "ISCE (×1000)"
            steps = np.arange(1, len(vals) + 1)
            ax.plot(steps, vals, label=label, color=color, linewidth=2)
        if "CI" in title:
            ideal = 0.5 if "50" in title else 0.9
            ax.axhline(ideal, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("Forecast step", fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"{_land_display(land)} (L/H={length}) — Error Metrics", fontsize=14)
    plt.tight_layout()
    out_path = output_dir / f"error_metrics_{land}_len{length}.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[C] Saved: {out_path}")

# ---------------------------------------------------------------------------
# Figure D: Summary table (with best values highlighted)
# ---------------------------------------------------------------------------
def plot_summary_table(land, length, model_names, all_metrics, output_dir):
    col_headers = [
        "Model",
        "MAE (med)", "MAE (mean)", "MAE (sample)",
        "CI50", "CI90", "ISCE (*1000)",
        "CRPS", "Reliability", "Resolution", "Uncertainty"
    ]
    rows = []
    numeric_vals = []
    for mn in model_names:
        label = MODEL_REGISTRY.get(mn, {"label": mn})["label"]
        mets = all_metrics[mn]
        rows.append([
            label,
            f"{mets['mae_median']:.4f}",
            f"{mets['mae_mean']:.4f}",
            f"{mets['mae_sample']:.4f}",
            f"{mets['ci50']:.3f}",
            f"{mets['ci90']:.3f}",
            f"{mets['isce_mean']*1000:.4f}",
            f"{mets['crps']:.4f}",
            f"{mets['reliability']:.4f}",
            f"{mets['resolution']:.4f}",
            f"{mets['uncertainty']:.4f}",
        ])
        numeric_vals.append([
            mets['mae_median'], mets['mae_mean'], mets['mae_sample'],
            mets['ci50'], mets['ci90'], mets['isce_mean']*1000,
            mets['crps'], mets['reliability'], mets['resolution'], mets['uncertainty']
        ])
    # Determine best indices per column (lower better for most; CI50/CI90 closest to nominal)
    best_idx = []
    for col in range(len(numeric_vals[0])):
        col_vals = [row[col] for row in numeric_vals]
        if col in [0,1,2,6,7,8,9]:  # MAEs, CRPS, Reliability, Resolution, Uncertainty? Actually Uncertainty is not lower better; but we treat as lower? Uncertainty is fixed by data, not model performance; we might not highlight. We'll skip highlighting for Uncertainty.
            best = int(np.argmin(col_vals))
        elif col == 3:  # CI50
            best = int(np.argmin(np.abs(np.array(col_vals) - 0.5)))
        elif col == 4:  # CI90
            best = int(np.argmin(np.abs(np.array(col_vals) - 0.9)))
        else:
            best = None
        best_idx.append(best)
    fig, ax = plt.subplots(figsize=(12, 2 + len(rows)))  # wider for more columns
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=col_headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.1, 1.5)
    for (r,c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#D0D8E8")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#F7F9FC")
            if c >= 1 and best_idx[c-1] is not None and best_idx[c-1] == r-1:
                cell.set_facecolor("#90EE90")
                cell.set_text_props(weight="bold")
    ax.set_title(f"{_land_display(land)} (L/H={length}) — Summary Metrics", fontsize=13, pad=14)
    png_path = output_dir / f"summary_{land}_len{length}.png"
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[D] Saved: {png_path}")
    # CSV
    csv_path = output_dir / f"summary_{land}_len{length}.csv"
    with open(csv_path, "w") as f:
        f.write(",".join(col_headers) + "\n")
        for row in rows:
            f.write(",".join(row) + "\n")
    print(f"CSV saved: {csv_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    print("\n=== Normalization stats ===")
    norm_stats = _load_mean_std(args.data_npz)

    # Parse result specs: LANDSCAPE:MODEL:NPZ_PATH
    raw_specs = []
    for spec in args.results:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(f"Expected LANDSCAPE:MODEL:NPZ_PATH, got {spec}")
        land, model, path = parts
        raw_specs.append((land, model, path))

    # Load each result, group by (land, pred_len)
    by_land_len = {}
    for land, model, path in raw_specs:
        mean, std = norm_stats.get(land, (None, None))
        data = load_npz_result(path, args.context_length, mean, std)
        pred_len = data["pred_len"]
        key = (land, pred_len)
        by_land_len.setdefault(key, {})[model] = data

    # Process each (land, length) group
    for (land, length), models_data in by_land_len.items():
        print(f"\n=== {land}, prediction length = {length} ===")
        model_names = list(models_data.keys())
        # Compute metrics for each model
        all_metrics = {}
        for model, data in models_data.items():
            mets = compute_metrics(data["ground_truths"], data["samples"], data["n_past"])
            all_metrics[model] = mets
        # Print summary
        for mn in model_names:
            label = MODEL_REGISTRY.get(mn, {"label": mn})["label"]
            mets = all_metrics[mn]
            print(f"  {label:12s}: MAE_med={mets['mae_median']:.4f} MAE_mean={mets['mae_mean']:.4f} "
                  f"MAE_samp={mets['mae_sample']:.4f} CI50={mets['ci50']:.3f} CI90={mets['ci90']:.3f} "
                  f"ISCE*1000={mets['isce_mean']*1000:.4f} CRPS={mets['crps']:.4f} "
                  f"Rel={mets['reliability']:.4f} Res={mets['resolution']:.4f} Unc={mets['uncertainty']:.4f}")

        # Select trajectories to display
        N_min = min(data["fullset_N"] for data in models_data.values())
        n_show = min(args.n_traj_show, N_min)
        rng = np.random.default_rng(args.seed)
        display_labels = rng.choice(N_min, size=n_show, replace=False).tolist()

        # Prepare display data for figures A and B
        display_data = {}
        for model, data in models_data.items():
            display_data[model] = {
                "ground_truths": data["ground_truths"][display_labels],
                "samples": data["samples"][display_labels],
                "n_past": data["n_past"],
                "n_future": data["n_future"],
            }
        first = next(iter(display_data.values()))
        n_past = first["n_past"]
        n_future = first["n_future"]

        # Figures
        first_model_data = next(iter(models_data.values()))
        plot_raw_trajectories(land, first_model_data["ground_truths"], n_past, n_future, out_dir, args.seed)
        plot_trajectory_comparison_grid(land, display_labels, display_data, n_past, n_future, out_dir)
        plot_histogram2d_comparison_grid(land, display_labels, display_data, n_past, n_future, out_dir)
        plot_error_metrics_grid(land, length, model_names, all_metrics, out_dir)
        plot_summary_table(land, length, model_names, all_metrics, out_dir)

    print("\n✓ All outputs saved.")

if __name__ == "__main__":
    main()