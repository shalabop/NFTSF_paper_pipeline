#!/usr/bin/env python3
"""
plot_per_trajectory.py
======================
For a single test trajectory, generate per model:
  - Cropped ground truth + CI bands
  - Cropped ground truth + forecast samples
  - Cropped ground truth + 2D histogram
All plots use the same cropped window (context+forecast) with a vertical line at the forecast start.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

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
# Landscape helpers
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
# Result loading – returns cropped window (context+forecast) as ground_truths
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
        "pred_len": H,
    }

# ---------------------------------------------------------------------------
# Plotting functions (all use the cropped window)
# ---------------------------------------------------------------------------
def plot_ci(land, model_name, ground_truth, n_past, n_future, samples, output_dir):
    """Ground truth (cropped) + CI bands."""
    reg = MODEL_REGISTRY.get(model_name, {"label": model_name, "color": "tab:orange"})
    color = reg["color"]
    mlabel = reg["label"]
    all_steps = np.arange(0, n_past + n_future)
    future_steps = np.arange(n_past, n_past + n_future)
    median = np.median(samples, axis=1)
    lo90 = np.percentile(samples, 5, axis=1)
    hi90 = np.percentile(samples, 95, axis=1)
    lo50 = np.percentile(samples, 25, axis=1)
    hi50 = np.percentile(samples, 75, axis=1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(all_steps, ground_truth, "k-", linewidth=2.0, label="Truth")
    ax.plot(future_steps, median, color=color, linewidth=2.0, label="Median")
    ax.fill_between(future_steps, lo90, hi90, color="tab:orange", alpha=0.40, label="90% band")
    ax.fill_between(future_steps, lo50, hi50, color="tab:blue", alpha=0.40, label="50% band")
    ax.axvline(x=n_past, color="k", linestyle="--", alpha=0.4, label="Forecast start")
    ax.set_xlabel(r"Step $N$")
    ax.set_ylabel(_coord_label(land))
    ax.set_title(f"{_land_display(land)} — {mlabel} (CI bands)")
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    out_path = output_dir / f"trajectory_ci_{model_name}_{land}.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

def plot_samples(land, model_name, ground_truth, n_past, n_future, samples, output_dir, n_display=50):
    """Ground truth (cropped) + random forecast samples."""
    reg = MODEL_REGISTRY.get(model_name, {"label": model_name, "color": "tab:orange"})
    color = reg["color"]
    mlabel = reg["label"]
    all_steps = np.arange(0, n_past + n_future)
    future_steps = np.arange(n_past, n_past + n_future)
    S = samples.shape[1]
    n_show = min(n_display, S)
    idx = np.random.choice(S, n_show, replace=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(all_steps, ground_truth, "k-", linewidth=2.0, label="Truth")
    for i in idx:
        ax.plot(future_steps, samples[:, i], color=color, alpha=0.2, linewidth=0.8)
    ax.axvline(x=n_past, color="k", linestyle="--", alpha=0.4, label="Forecast start")
    ax.set_xlabel(r"Step $N$")
    ax.set_ylabel(_coord_label(land))
    ax.set_title(f"{_land_display(land)} — {mlabel} (forecast samples)")
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    out_path = output_dir / f"trajectory_samples_{model_name}_{land}.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

def plot_histogram(land, model_name, ground_truth, n_past, n_future, samples, output_dir):
    """2D histogram (dark background) over forecast region, with ground truth line over cropped window."""
    reg = MODEL_REGISTRY.get(model_name, {"label": model_name, "color": "tab:orange"})
    mlabel = reg["label"]
    with plt.style.context('dark_background'):
        fig, ax = plt.subplots(figsize=(10, 4))
        future_steps = np.arange(n_past, n_past + n_future)
        all_steps = np.arange(0, n_past + n_future)
        y_vals = ground_truth
        g_lo = np.min(y_vals)
        g_hi = np.max(y_vals)
        margin = 0.05 * (g_hi - g_lo) if g_hi > g_lo else 0.1
        g_lo -= margin
        g_hi += margin
        n_x_bins = max(12, n_future // 5)
        n_y_bins = 24
        x_edges = np.linspace(n_past, n_past + n_future, n_x_bins + 1)
        y_edges = np.linspace(g_lo, g_hi, n_y_bins + 1)
        samples_T = samples.T  # (S, H)
        time_rep = np.tile(future_steps, (samples_T.shape[0], 1))
        ax.hist2d(time_rep.flatten(), samples_T.flatten(),
                  bins=[x_edges, y_edges], cmap="magma", density=True,
                  norm=LogNorm(vmin=1e-6))
        ax.plot(all_steps, ground_truth, color="lime", linewidth=2.5, label="Truth")
        ax.axvline(x=n_past, color="white", linestyle="--", alpha=0.5, label="Forecast start")
        ax.set_ylim(g_lo, g_hi)
        ax.set_xlim(0, n_past + n_future)
        ax.set_xlabel(r"Step $N$")
        ax.set_ylabel(_coord_label(land))
        ax.set_title(f"{_land_display(land)} — {mlabel} density")
        ax.legend(loc='upper left')
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
        out_path = output_dir / f"trajectory_hist2d_{model_name}_{land}.svg"
        fig.savefig(out_path, format="svg", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", required=True,
                        help="Format: LANDSCAPE:MODEL:NPZ_PATH")
    parser.add_argument("--output_dir", default="./comparison")
    parser.add_argument("--context_length", "-cl", type=int, default=0)
    parser.add_argument("--data_npz", nargs="+", required=True,
                        metavar="LANDSCAPE:PATH", help="Normalization stats")
    parser.add_argument("--traj_index", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    plot_dir = out_dir / "per_trajectory_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Normalization stats ===")
    norm_stats = _load_mean_std(args.data_npz)

    specs = []
    for spec in args.results:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(f"Expected LANDSCAPE:MODEL:NPZ_PATH, got {spec}")
        land, model, path = parts
        specs.append((land, model, path))

    by_land = {}
    for land, model, path in specs:
        mean, std = norm_stats.get(land, (None, None))
        data = load_npz_result(path, args.context_length, mean, std)
        by_land.setdefault(land, {})[model] = data

    for land, models_data in by_land.items():
        print(f"\n=== Landscape: {land} ===")
        any_model = next(iter(models_data.values()))
        n_past = any_model["n_past"]
        n_future = any_model["n_future"]

        for model_name, data in models_data.items():
            ground_truth = data["ground_truths"][args.traj_index]  # (L+H,)
            samples = data["samples"][args.traj_index]             # (H, S)

            plot_ci(land, model_name, ground_truth, n_past, n_future, samples, plot_dir)
            plot_samples(land, model_name, ground_truth, n_past, n_future, samples, plot_dir)
            plot_histogram(land, model_name, ground_truth, n_past, n_future, samples, plot_dir)

    print("\n✓ All per‑trajectory plots saved.")

if __name__ == "__main__":
    main()