"""
viz/comparison.py
=================
Publication-quality unified comparison visualizations.

Automatically discovers all trained model results in a structured outputs/
directory and generates two types of plots:

1. Per-trajectory comparison plot
   - Ground truth + predictions from all models on the same axes.
   - Per-model error annotation (MAE, RMSE for the selected trajectory).
   - Title, axis labels, and colors auto-generated from metadata.

2. Aggregate error comparison plots
   - 2×2 grid: CRPS(t), MAE(t), CI90 coverage(t), CI50 coverage(t).
   - Bar chart of mean CRPS and mean MAE per model.

Usage
-----
python viz/comparison.py \\
    --results_dir outputs/ \\
    --landscape   alanine_phi \\
    --trajectory_id 42

Optional flags
--------------
--results_dir   Root output directory (default: outputs/)
--landscape     Landscape identifier (required).
--trajectory_id Which test trajectory to plot (default: 0).
--out_dir       Where to save figures (default: results_dir/landscape/).
--pdf           Also save PDF versions.
--dpi           Figure DPI (default: 150).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# Coordinate label map — extend here to support new landscapes.
# ---------------------------------------------------------------------------

COORDINATE_LABEL_MAP: dict[str, str] = {
    "alanine_phi":      r"$\varphi$ (Phi) [degrees]",
    "alanine_psi":      r"$\psi$ (Psi) [degrees]",
    "double_well":      r"Position $x$",
    "double_well_bi":   r"Position $x$",
    "single_well":      r"Position $x$",
    "linear_gaussian":  r"Position $x$",
    "linear":           r"Position $x$",
    "sine":             r"Amplitude",
}

# Known-model color palette (from tsf_models-adam/plot_metrics.py).
_KNOWN_COLORS: dict[str, str] = {
    "nftsf":       "#2CA02C",   # green  (new — NFTSF)
    "arima":       "#4878CF",   # steel blue
    "tsdiff_q":    "#D65F5F",   # red
    "tsdiff_ms":   "#6ACC65",   # light green
    "tsdiff_cond": "#B47CC7",   # purple
    "csdi":        "#F0901E",   # orange
    "ratd":        "#77BEDB",   # light blue
    "nsdiff":      "#C4AD66",   # tan/gold
}

# Line styles cycled for additional contrast when printing in greyscale.
_LINE_STYLES = ["-", "--", "-.", ":", (0, (5, 2)), (0, (3, 1, 1, 1))]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_results(
    results_dir: Path,
    landscape: str,
) -> dict[str, np.lib.npyio.NpzFile]:
    """
    Scan *results_dir* for ``results.npz`` files matching *landscape*.

    Expected path pattern::

        results_dir / {model_name} / {landscape} / {run_id} / results.npz

    Returns a dict mapping model_name → loaded NpzFile.
    Multiple runs of the same model are resolved by taking the latest run_id
    (lexicographic sort).
    """
    results_dir = Path(results_dir)
    found: dict[str, Path] = {}

    for npz in sorted(results_dir.rglob("results.npz")):
        parts = npz.relative_to(results_dir).parts
        # Expected: model / landscape / run_id / results.npz
        if len(parts) < 4:
            continue
        m_name, land, run_id = parts[0], parts[1], parts[2]
        if land != landscape:
            continue
        # Keep latest run_id per model.
        if m_name not in found or run_id > found[m_name].parent.name:
            found[m_name] = npz

    if not found:
        print(f"[viz] No results found under {results_dir} for landscape '{landscape}'.",
              file=sys.stderr)
        return {}

    loaded: dict[str, np.lib.npyio.NpzFile] = {}
    for name, path in sorted(found.items()):
        try:
            loaded[name] = np.load(path, allow_pickle=True)
            print(f"[viz] Found: {name}  ({path})")
        except Exception as e:
            print(f"[viz] WARNING: could not load {path}: {e}", file=sys.stderr)
    return loaded


# ---------------------------------------------------------------------------
# Color / style helpers
# ---------------------------------------------------------------------------

def _model_color(name: str) -> str:
    if name in _KNOWN_COLORS:
        return _KNOWN_COLORS[name]
    import random as _rng
    r = _rng.Random(name)
    return f"#{r.randint(0, 0xFFFFFF):06x}"


def _model_style(idx: int) -> str:
    return _LINE_STYLES[idx % len(_LINE_STYLES)]


def _y_label(landscape: str) -> str:
    return COORDINATE_LABEL_MAP.get(landscape, r"Value")


def _landscape_display(landscape: str) -> str:
    """Human-readable landscape name for plot titles."""
    mapping = {
        "alanine_phi":  "Alanine Dipeptide \u2014 \u03c6 (Phi)",
        "alanine_psi":  "Alanine Dipeptide \u2014 \u03c8 (Psi)",
        "double_well":  "Double Well",
        "single_well":  "Single Well",
        "linear_gaussian": "Linear Gaussian",
        "linear":       "Linear AR(1)",
        "sine":         "Sine / Multi-frequency",
    }
    return mapping.get(landscape, landscape.replace("_", " ").title())


# ---------------------------------------------------------------------------
# Plot 1: Per-trajectory comparison
# ---------------------------------------------------------------------------

def plot_trajectory_comparison(
    results: dict[str, np.lib.npyio.NpzFile],
    landscape: str,
    trajectory_id: int,
    out_dir: Path,
    save_pdf: bool = False,
    dpi: int = 150,
) -> None:
    """
    Plot ground truth and all-model predictions for a single test trajectory.

    The figure contains:
    - Main panel: time-series lines (ground truth + model medians + 90% CI band).
    - Annotation box: per-model MAE and RMSE for this trajectory.
    """
    n_future = None
    for name, data in results.items():
        gt = data["ground_truth"]
        if trajectory_id >= len(gt):
            print(f"[viz] WARNING: trajectory_id {trajectory_id} ≥ N_test={len(gt)} "
                  f"for model '{name}'.  Clamping to {len(gt) - 1}.")
            trajectory_id = len(gt) - 1
        n_future = gt.shape[1]
        break

    if n_future is None:
        print("[viz] No data to plot.", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    time_steps = np.arange(n_future)
    legend_handles: list = []

    metric_lines: list[str] = []

    for idx, (name, data) in enumerate(sorted(results.items())):
        gt      = data["ground_truth"][trajectory_id]   # (T,)
        samples = data["samples"][trajectory_id]         # (T, S)

        median   = np.median(samples, axis=-1)
        ci90_lo  = np.percentile(samples, 5,  axis=-1)
        ci90_hi  = np.percentile(samples, 95, axis=-1)

        color = _model_color(name)
        ls    = _model_style(idx)

        ax.plot(time_steps, median, color=color, linestyle=ls, linewidth=1.5,
                label=name)
        ax.fill_between(time_steps, ci90_lo, ci90_hi,
                        color=color, alpha=0.12)

        # Per-trajectory metrics.
        traj_mae  = float(np.abs(gt - median).mean())
        traj_rmse = float(np.sqrt(((gt - median) ** 2).mean()))
        metric_lines.append(f"{name:14s}  MAE={traj_mae:.4f}  RMSE={traj_rmse:.4f}")

        legend_handles.append(
            Line2D([0], [0], color=color, linestyle=ls, linewidth=1.5, label=name)
        )

    # Ground truth (drawn last so it's on top).
    gt_all = next(iter(results.values()))["ground_truth"][trajectory_id]
    ax.plot(time_steps, gt_all, color="black", linewidth=2.0,
            linestyle="-", label="Ground truth", zorder=10)
    legend_handles.insert(0,
        Line2D([0], [0], color="black", linewidth=2.0, label="Ground truth"))

    # Metric annotation box.
    metric_text = "Per-trajectory metrics:\n" + "\n".join(metric_lines)
    ax.text(
        0.01, 0.98, metric_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=7,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85,
                  edgecolor="gray"),
    )

    ax.set_xlabel("Forecast time step", fontsize=11)
    ax.set_ylabel(_y_label(landscape), fontsize=11)
    title = f"{_landscape_display(landscape)} \u2014 Trajectory {trajectory_id} Comparison"
    ax.set_title(title, fontsize=12)
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=8, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / f"comparison_traj{trajectory_id}.png"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    print(f"[viz] Saved: {png_path}")

    if save_pdf:
        pdf_path = out_dir / f"comparison_traj{trajectory_id}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"[viz] Saved: {pdf_path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2a: Aggregate 2×2 metric grid
# ---------------------------------------------------------------------------

def plot_aggregate_metrics(
    results: dict[str, np.lib.npyio.NpzFile],
    landscape: str,
    out_dir: Path,
    save_pdf: bool = False,
    dpi: int = 150,
) -> None:
    """
    2×2 grid of CRPS(t), MAE(t), CI90 coverage(t), CI50 coverage(t) —
    one line per model.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    axes = axes.flatten()

    legend_handles: list = []

    for idx, (name, data) in enumerate(sorted(results.items())):
        n_future = data["crps_t"].shape[0]
        time     = np.arange(n_future)
        color    = _model_color(name)
        ls       = _model_style(idx)

        axes[0].plot(time, data["crps_t"], color=color, linestyle=ls, linewidth=1.2)
        axes[1].plot(time, data["mae_t"],  color=color, linestyle=ls, linewidth=1.2)
        axes[2].plot(time, data["ci90_t"], color=color, linestyle=ls, linewidth=1.2)
        axes[3].plot(time, data["ci50_t"], color=color, linestyle=ls, linewidth=1.2)

        legend_handles.append(
            Line2D([0], [0], color=color, linestyle=ls, linewidth=1.2, label=name)
        )

    # Ideal coverage reference lines.
    axes[2].axhline(0.90, linestyle="--", color="black", linewidth=0.8, alpha=0.6)
    axes[3].axhline(0.50, linestyle="--", color="black", linewidth=0.8, alpha=0.6)
    legend_handles.append(
        Line2D([0], [0], color="black", linestyle="--", linewidth=0.8, label="Ideal")
    )

    for ax, title, ylabel in zip(
        axes,
        ["CRPS(t)", "MAE(t)", "CI90 Coverage(t)", "CI50 Coverage(t)"],
        ["CRPS", "MAE", "Coverage", "Coverage"],
    ):
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Forecast step", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.4)

    fig.suptitle(f"{_landscape_display(landscape)} \u2014 Aggregate Metrics",
                 fontsize=13)
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=min(len(legend_handles), 6),
        fontsize=8,
        frameon=False,
    )
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / "aggregate_metrics.png"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    print(f"[viz] Saved: {png_path}")

    if save_pdf:
        pdf_path = out_dir / "aggregate_metrics.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"[viz] Saved: {pdf_path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2b: Aggregate bar chart
# ---------------------------------------------------------------------------

def plot_aggregate_bar(
    results: dict[str, np.lib.npyio.NpzFile],
    landscape: str,
    out_dir: Path,
    save_pdf: bool = False,
    dpi: int = 150,
) -> None:
    """
    Bar chart comparing mean CRPS and mean MAE per model.
    """
    model_names = sorted(results.keys())
    mean_crps = [float(results[m]["crps_t"].mean()) for m in model_names]
    mean_mae  = [float(results[m]["mae_t"].mean())  for m in model_names]
    colors    = [_model_color(m) for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(x, mean_crps, width=width * 2, color=colors, edgecolor="white", linewidth=0.5)
    axes[0].set_title("Mean CRPS", fontsize=11)
    axes[0].set_ylabel("CRPS", fontsize=9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, rotation=30, ha="right", fontsize=8)
    axes[0].grid(axis="y", linestyle=":", alpha=0.5)

    axes[1].bar(x, mean_mae, width=width * 2, color=colors, edgecolor="white", linewidth=0.5)
    axes[1].set_title("Mean MAE", fontsize=11)
    axes[1].set_ylabel("MAE", fontsize=9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names, rotation=30, ha="right", fontsize=8)
    axes[1].grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle(
        f"{_landscape_display(landscape)} \u2014 Mean Error Comparison",
        fontsize=12,
    )
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / "aggregate_bar.png"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    print(f"[viz] Saved: {png_path}")

    if save_pdf:
        pdf_path = out_dir / "aggregate_bar.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"[viz] Saved: {pdf_path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate unified model comparison plots."
    )
    p.add_argument("--results_dir",    default="outputs/",
                   help="Root output directory containing model subdirs.")
    p.add_argument("--landscape",      required=True,
                   help="Landscape identifier, e.g. 'alanine_phi'.")
    p.add_argument("--trajectory_id",  type=int, default=0,
                   help="Test trajectory index for per-trajectory plot.")
    p.add_argument("--out_dir",        default=None,
                   help="Output directory for figures "
                        "(default: results_dir/landscape/).")
    p.add_argument("--pdf",            action="store_true",
                   help="Also save PDF versions of all figures.")
    p.add_argument("--dpi",            type=int, default=150)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    results_dir = Path(args.results_dir)
    out_dir     = Path(args.out_dir) if args.out_dir else (
        results_dir / args.landscape / "plots"
    )

    results = discover_results(results_dir, args.landscape)
    if not results:
        sys.exit(1)

    print(f"\n[viz] Generating plots for landscape '{args.landscape}' "
          f"with {len(results)} model(s): {', '.join(sorted(results))}\n")

    plot_trajectory_comparison(
        results=results,
        landscape=args.landscape,
        trajectory_id=args.trajectory_id,
        out_dir=out_dir,
        save_pdf=args.pdf,
        dpi=args.dpi,
    )

    plot_aggregate_metrics(
        results=results,
        landscape=args.landscape,
        out_dir=out_dir,
        save_pdf=args.pdf,
        dpi=args.dpi,
    )

    plot_aggregate_bar(
        results=results,
        landscape=args.landscape,
        out_dir=out_dir,
        save_pdf=args.pdf,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
