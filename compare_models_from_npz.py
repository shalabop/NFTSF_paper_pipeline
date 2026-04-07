#!/usr/bin/env python3
"""
compare_models_from_npz.py
==========================
Generate comparison figures from pre-computed .npz result files.

INTERNAL CONVENTION (enforced by load_npz_result):
    samples       : (N, H, S)  — N trajectories, H forecast steps, S ensemble members
    ground_truths : (N, L+H)   — L context steps followed by H forecast steps
    ground_truth  : (N, H)     — forecast horizon only (used for metrics)

All metric functions operate on this convention.

Usage
-----
python compare_models_from_npz.py \
    --results \
        double_well:arima:results/arima/double_well.npz \
        double_well:csdi:results/csdi/double_well.npz \
        double_well:nf:results/nf/double_well.npz \
    --output_dir ./comparison_figures/ \
    --n_traj_show 7 \
    --seed 42
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
# Landscape / coordinate helpers
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
    "alanine_phi": r"angle $\varphi$ (rad)",
    "alanine_psi": r"angle $\psi$ (rad)",
}

def _land_display(landscape: str) -> str:
    return LANDSCAPE_DISPLAY.get(landscape, landscape.replace("_", " ").title())

def _coord_label(landscape: str) -> str:
    return COORDINATE_LABEL.get(landscape, r"Value")


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, dict] = {
    "nf":          {"label": "NFTSF",       "color": "#1F77B4"},
    "arima":       {"label": "ARIMA",        "color": "#2CA02C"},
    "tsdiff_cond": {"label": "TSDiff-Cond",  "color": "#C71FD6"},
    "tsdiff_ms":   {"label": "TSDiff-MS",    "color": "#A09E2C"},
    "tsdiff_q":    {"label": "TSDiff-Q",     "color": "#B41F1F"},
    "csdi":        {"label": "CSDI",         "color": "#0AF1F1"},
    "ratd":        {"label": "RATD",         "color": "#FF7F0E"},
    "nsdiff":      {"label": "NsDiff",       "color": "#9467BD"},
    "ccdm":        {"label": "CCDM",         "color": "#8C564B"},
}

METRIC_META: dict[str, dict] = {
    "mae":  {"title": "MAE",           "ylabel": "MAE",      "ideal": None},
    "crps": {"title": "CRPS",          "ylabel": "CRPS",     "ideal": None},
    "ci50": {"title": "CI50 Coverage", "ylabel": "Coverage", "ideal": 0.50},
    "ci90": {"title": "CI90 Coverage", "ylabel": "Coverage", "ideal": 0.90},
}
METRIC_ORDER = ["mae", "crps", "ci50", "ci90"]


# ---------------------------------------------------------------------------
# Metric functions — all expect samples shape (N, H, S)
# ---------------------------------------------------------------------------

def _mae_per_step(ground_truth_future: np.ndarray,
                  samples: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    ground_truth_future : (N, H)
    samples             : (N, H, S)

    Returns
    -------
    mae_t : (H,)  — MAE of median forecast, averaged over N
    """
    median = np.median(samples, axis=2)                      # (N, H)
    return np.abs(ground_truth_future - median).mean(axis=0) # (H,)


def _crps_per_step(ground_truth_future: np.ndarray,
                   samples: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    ground_truth_future : (N, H)
    samples             : (N, H, S)

    Returns
    -------
    crps_t : (H,)  — CRPS averaged over N trajectories
    """
    N, H = ground_truth_future.shape
    S = samples.shape[2]

    crps_t = np.zeros(H, dtype=np.float64)
    for t in range(H):
        obs = ground_truth_future[:, t]       # (N,)
        fc  = samples[:, t, :]                # (N, S)

        mae_term = np.mean(np.abs(fc - obs[:, None]), axis=1)  # (N,)

        fc_sorted = np.sort(fc, axis=1)                        # (N, S)
        diff      = fc_sorted[:, 1:] - fc_sorted[:, :-1]       # (N, S-1)
        w         = np.arange(1, S) * np.arange(S - 1, 0, -1) # (S-1,)
        energy    = (diff * w[None, :]).sum(axis=1) / (S * (S - 1))  # (N,)

        crps_t[t] = np.mean(mae_term - energy)
    return crps_t


def _coverage_per_step(ground_truth_future: np.ndarray,
                        samples: np.ndarray,
                        lo_pct: float,
                        hi_pct: float) -> np.ndarray:
    """
    Parameters
    ----------
    ground_truth_future : (N, H)
    samples             : (N, H, S)
    lo_pct, hi_pct      : percentile bounds, e.g. 25.0 / 75.0 for CI50

    Returns
    -------
    coverage_t : (H,)  — per-step empirical coverage averaged over N
    """
    N = ground_truth_future.shape[0]
    coverage_within = []
    for i in range(N):
        gt_i   = ground_truth_future[i]       # (H,)
        samp_i = samples[i]                   # (H, S)
        lo = np.percentile(samp_i, lo_pct, axis=1)   # (H,)  ← axis=1 over S
        hi = np.percentile(samp_i, hi_pct, axis=1)   # (H,)
        within = (gt_i >= lo) & (gt_i <= hi)           # (H,) bool
        coverage_within.append(within.astype(float))
    return np.mean(coverage_within, axis=0)   # (H,)


def compute_metrics(ground_truths: np.ndarray,
                    samples: np.ndarray,
                    n_past: int,
                    fullset_mae:  np.ndarray | None = None,
                    fullset_crps: np.ndarray | None = None,
                    fullset_ci50: np.ndarray | None = None,
                    fullset_ci90: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """
    Parameters
    ----------
    ground_truths : (N, L+H)
    samples       : (N, H, S)
    n_past        : L
    fullset_*     : (H,) pre-computed full-testset arrays (optional)

    Returns
    -------
    dict with keys 'mae', 'crps', 'ci50', 'ci90' — each (H,)
    """
    gt_future = ground_truths[:, n_past:]     # (N, H)
    return {
        "mae":  fullset_mae  if fullset_mae  is not None else _mae_per_step(gt_future, samples),
        "crps": fullset_crps if fullset_crps is not None else _crps_per_step(gt_future, samples),
        "ci50": fullset_ci50 if fullset_ci50 is not None else _coverage_per_step(gt_future, samples, 25.0, 75.0),
        "ci90": fullset_ci90 if fullset_ci90 is not None else _coverage_per_step(gt_future, samples,  5.0, 95.0),
    }


# ---------------------------------------------------------------------------
# NPZ loader — enforces (N, H, S) convention
# ---------------------------------------------------------------------------

def load_npz_result(npz_path: str) -> dict:
    """
    Load one model's .npz result and return a standardised dict.

    Internal convention enforced here:
        samples       : (N, H, S)
        ground_truth  : (N, H)        — forecast horizon only
        ground_truths : (N, L+H)      — context + forecast (for plotting)
        n_past  (L)   : int
        n_future (H)  : int

    Handles all models:
        ARIMA / CSDI / TSDiff-MS/Q/Cond  → save samples as (N, H, S)
        NF                                → saves samples as (N, H, S) + contexts (N, L)

    train_test_split is the time index where forecasting STARTS (e.g. 900).
    """
    data = np.load(npz_path, allow_pickle=True)

    print(f"    Keys: {sorted(data.files)}")

    # ── 1. Authoritative H from saved prediction_length ───────────────────
    H = int(data["prediction_length"])

    # ── 2. ground_truth → always exactly (N, H) ──────────────────────────
    gt_raw = data["ground_truth"]               # may be (N, H) or (N, T-tts)
    if gt_raw.shape[1] != H:
        # Some models saved full_traj[:,tts:] which is longer than H
        gt_raw = gt_raw[:, :H]                  # take first H steps
    N = gt_raw.shape[0]

    # ── 3. samples → enforce (N, H, S) ───────────────────────────────────
    s_raw = data["samples"]
    if s_raw.ndim != 3:
        raise ValueError(f"Expected 3D samples, got shape {s_raw.shape}")

    _, d1, d2 = s_raw.shape

    if d1 == H and d2 != H:
        # Already (N, H, S) ✅
        samples = s_raw
    elif d2 == H and d1 != H:
        # Stored as (N, S, H) → transpose to (N, H, S)
        samples = s_raw.transpose(0, 2, 1)
        print(f"      Transposed samples (N,S,H)→(N,H,S): {s_raw.shape} → {samples.shape}")
    elif d1 == H and d2 == H:
        # Ambiguous (H == S): use std heuristic
        # Samples should vary MORE across S than across time steps.
        # For shape (N, H, S): std over axis=1 (time) should be large,
        #                       std over axis=2 (samples) should be moderate.
        # For shape (N, S, H): std over axis=1 (samples) moderate,
        #                       std over axis=2 (time) large.
        # We check: if std(axis=2) > std(axis=1) the last axis varies more →
        # last axis = S → already (N, H, S).
        std_last  = float(s_raw[:10].std(axis=2).mean())  # std across last dim
        std_mid   = float(s_raw[:10].std(axis=1).mean())  # std across middle dim
        if std_last >= std_mid:
            samples = s_raw                               # (N, H, S) ✅
            print(f"      Ambiguous shape {s_raw.shape}: kept as (N,H,S) [std heuristic]")
        else:
            samples = s_raw.transpose(0, 2, 1)            # (N, H, S)
            print(f"      Ambiguous shape {s_raw.shape}: transposed to (N,H,S) [std heuristic]")
    else:
        raise ValueError(
            f"Cannot determine orientation: samples={s_raw.shape}, H={H}  ({npz_path})"
        )

    # samples is now (N, H, S) ✅
    S = samples.shape[2]

    # ── 4. train_test_split ───────────────────────────────────────────────
    tts = int(data["train_test_split"]) if "train_test_split" in data else None

    # ── 5. Reconstruct ground_truths (N, L+H) for plotting ───────────────
    L = H   # assume context_length == prediction_length

    if "contexts" in data:
        # NF saves context explicitly
        contexts = data["contexts"]              # (N, L)
        L = contexts.shape[1]
        ground_truths = np.concatenate([contexts, gt_raw], axis=1)  # (N, L+H)

    elif "full_trajectories" in data:
        full_traj = data["full_trajectories"]    # (N, T)
        T_total   = full_traj.shape[1]
        _tts      = tts if tts is not None else T_total - H
        start     = _tts - L

        if start < 0:
            pad           = np.zeros((N, -start), dtype=np.float32)
            ctx           = np.concatenate([pad, full_traj[:, :_tts]], axis=1)
        else:
            ctx           = full_traj[:, start:_tts]          # (N, L)

        ground_truths = np.concatenate([ctx, gt_raw], axis=1) # (N, L+H)

    else:
        # Fallback — zero context
        print(f"      WARNING: no contexts or full_trajectories found; using zero context")
        ctx           = np.zeros((N, L), dtype=np.float32)
        ground_truths = np.concatenate([ctx, gt_raw], axis=1)

    # ── 6. Assertions ─────────────────────────────────────────────────────
    assert samples.shape      == (N, H, S),   f"samples: {samples.shape} ≠ ({N},{H},{S})"
    assert ground_truths.shape == (N, L + H), f"ground_truths: {ground_truths.shape} ≠ ({N},{L+H})"

    print(f"      N={N}  L={L}  H={H}  S={S}  tts={tts}")

    return {
        "ground_truths":    ground_truths.astype(np.float32),  # (N, L+H)
        "ground_truth":     gt_raw.astype(np.float32),         # (N, H)
        "samples":          samples.astype(np.float32),        # (N, H, S)
        "n_past":           L,
        "n_future":         H,
        "n_samples":        S,
        "N":                N,
        "train_test_split": tts,
    }


# ---------------------------------------------------------------------------
# Y-limit helpers
# ---------------------------------------------------------------------------

def _ylim_single(real_traj: np.ndarray,
                 samples: np.ndarray) -> tuple[float, float]:
    """
    samples : (H, S)  — single trajectory's ensemble
    real_traj: (L+H,)
    """
    y_lo = min(float(np.percentile(samples, 1)), float(np.min(real_traj)))
    y_hi = max(float(np.percentile(samples, 99)), float(np.max(real_traj)))
    margin = 0.05 * (y_hi - y_lo) if y_hi > y_lo else 0.1
    return y_lo - margin, y_hi + margin


def _global_ylim(model_data: dict[str, dict]) -> tuple[float, float]:
    g_lo, g_hi = float("inf"), float("-inf")
    for mdata in model_data.values():
        n_disp = mdata["ground_truths"].shape[0]
        for i in range(n_disp):
            lo, hi = _ylim_single(
                mdata["ground_truths"][i],
                mdata["samples"][i],       # (H, S)
            )
            g_lo = min(g_lo, lo)
            g_hi = max(g_hi, hi)
    return g_lo, g_hi


# ---------------------------------------------------------------------------
# Figure A: Trajectory comparison grid
# ---------------------------------------------------------------------------

def plot_trajectory_comparison_grid(landscape: str,
                                     display_labels: list[int],
                                     model_data: dict[str, dict],
                                     n_past: int,
                                     n_future: int,
                                     output_dir: Path) -> None:
    """
    Rows = models, columns = selected trajectories.
    samples expected as (N, H, S) — median/percentiles over axis=2.
    """
    n_models = len(model_data)
    n_traj   = len(display_labels)

    fig, axes = plt.subplots(n_models, n_traj,
                             figsize=(6 * n_traj, 5 * n_models),
                             squeeze=False)

    future_steps = np.arange(n_past, n_past + n_future)
    all_steps    = np.arange(0, n_past + n_future)
    g_lo, g_hi   = _global_ylim(model_data)

    for row_idx, (model_name, mdata) in enumerate(model_data.items()):
        reg    = MODEL_REGISTRY.get(model_name, {"label": model_name, "color": "tab:orange"})
        color  = reg["color"]
        mlabel = reg["label"]

        samples_all       = mdata["samples"]        # (N, H, S)
        ground_truths_all = mdata["ground_truths"]  # (N, L+H)

        axes[row_idx, 0].set_ylabel(f"{mlabel}\n{_coord_label(landscape)}", fontsize=10)

        for col_idx, traj_label in enumerate(display_labels):
            ax        = axes[row_idx, col_idx]
            real_traj = ground_truths_all[col_idx]  # (L+H,)
            samp      = samples_all[col_idx]         # (H, S)

            # percentiles over S axis (axis=1 of (H, S))
            median = np.median(samp, axis=1)                    # (H,)
            lo90   = np.percentile(samp, 5,  axis=1)            # (H,)
            hi90   = np.percentile(samp, 95, axis=1)
            lo50   = np.percentile(samp, 25, axis=1)
            hi50   = np.percentile(samp, 75, axis=1)

            ax.plot(all_steps, real_traj, "k-", linewidth=2.0, label="Truth")
            ax.plot(future_steps, median, color=color, linewidth=2.0, label="Median")
            ax.fill_between(future_steps, lo90, hi90,
                            color="tab:orange", alpha=0.40, label="90% band")
            ax.fill_between(future_steps, lo50, hi50,
                            color="tab:blue",   alpha=0.40, label="50% band")
            ax.axvline(x=n_past, color="k", linestyle="--", alpha=0.4)
            ax.set_ylim(g_lo, g_hi)
            ax.set_xlim(0, n_past + n_future)
            ax.set_xlabel(r"Step $N$", fontsize=9)

            if row_idx == 0:
                ax.set_title(f"Trajectory {traj_label}", fontsize=11)

    axes[0, 0].legend(loc="upper left", fontsize=7, framealpha=0.8)
    fig.suptitle(f"{_land_display(landscape)} — Trajectory Comparison", fontsize=13, y=1.01)
    plt.tight_layout()

    out_path = output_dir / f"trajectory_comparison_{landscape}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[A] Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure B: 2-D histogram comparison grid
# ---------------------------------------------------------------------------

def plot_histogram2d_comparison_grid(landscape: str,
                                      display_labels: list[int],
                                      model_data: dict[str, dict],
                                      n_past: int,
                                      n_future: int,
                                      output_dir: Path) -> None:
    """
    samples expected as (N, H, S) — flattened per-trajectory for hist2d.
    """
    n_models = len(model_data)
    n_traj   = len(display_labels)

    fig, axes = plt.subplots(n_models, n_traj,
                             figsize=(7 * n_traj, 5 * n_models),
                             squeeze=False)

    future_steps = np.arange(n_past, n_past + n_future)
    all_steps    = np.arange(0, n_past + n_future)
    g_lo, g_hi   = _global_ylim(model_data)
    n_x_bins     = max(30, n_future // 2)
    n_y_bins     = 60

    for row_idx, (model_name, mdata) in enumerate(model_data.items()):
        reg    = MODEL_REGISTRY.get(model_name, {"label": model_name, "color": "tab:orange"})
        mlabel = reg["label"]

        samples_all       = mdata["samples"]        # (N, H, S)
        ground_truths_all = mdata["ground_truths"]  # (N, L+H)

        axes[row_idx, 0].set_ylabel(f"{mlabel}\n{_coord_label(landscape)}", fontsize=10)

        for col_idx, traj_label in enumerate(display_labels):
            ax        = axes[row_idx, col_idx]
            real_traj = ground_truths_all[col_idx]  # (L+H,)
            samp      = samples_all[col_idx]         # (H, S)

            # samp.T is (S, H); tile future_steps to match
            time_rep = np.tile(future_steps, (samp.shape[1], 1))  # (S, H)

            ax.hist2d(
                time_rep.flatten(), samp.T.flatten(),   # samp.T = (S, H) → flatten
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
    fig.suptitle(f"{_land_display(landscape)} — Prediction Density Comparison",
                 fontsize=13, y=1.01)
    plt.tight_layout()

    out_path = output_dir / f"histogram2d_comparison_{landscape}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[B] Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure C: Error metrics grid
# ---------------------------------------------------------------------------

def plot_error_metrics_grid(landscapes: list[str],
                             all_metrics: dict[str, dict[str, dict[str, np.ndarray]]],
                             output_dir: Path) -> None:
    n_land = len(landscapes)
    n_met  = len(METRIC_ORDER)

    fig, axes = plt.subplots(n_land, n_met,
                             figsize=(5 * n_met, 3.5 * n_land),
                             squeeze=False,
                             constrained_layout=True)

    metric_ylims: dict[str, tuple[float, float]] = {}
    for metric_key in METRIC_ORDER:
        col_lo, col_hi = float("inf"), float("-inf")
        for landscape in landscapes:
            for met_dict in all_metrics[landscape].values():
                vals = met_dict[metric_key]
                col_lo = min(col_lo, float(np.min(vals)))
                col_hi = max(col_hi, float(np.max(vals)))
        margin = 0.05 * (col_hi - col_lo) if col_hi > col_lo else 0.05
        metric_ylims[metric_key] = (col_lo - margin, col_hi + margin)

    legend_handles: list = []

    for col_idx, metric_key in enumerate(METRIC_ORDER):
        meta       = METRIC_META[metric_key]
        y_lo, y_hi = metric_ylims[metric_key]

        for row_idx, landscape in enumerate(landscapes):
            ax = axes[row_idx, col_idx]

            for model_name, met_dict in all_metrics[landscape].items():
                reg   = MODEL_REGISTRY.get(model_name, {"label": model_name, "color": "tab:gray"})
                color = reg["color"]
                vals  = met_dict[metric_key]
                steps = np.arange(len(vals))
                ax.plot(steps, vals, color=color, linewidth=2.0, label=reg["label"])

                if row_idx == 0 and col_idx == 0:
                    legend_handles.append(
                        Line2D([0], [0], color=color, linewidth=2.0, label=reg["label"])
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
            ax.grid(True, linestyle=":", alpha=0.4)
            ax.tick_params(labelsize=7)

            if col_idx == 0:
                ax.set_ylabel(f"{_land_display(landscape)}\n{meta['ylabel']}", fontsize=8)
            if row_idx == 0:
                ax.set_title(meta["title"], fontsize=10)

    fig.legend(handles=legend_handles, loc="center left",
               bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=True)
    fig.suptitle("Error Metrics Comparison — All Landscapes", fontsize=13, y=1.02)

    out_path = output_dir / "error_metrics_comparison.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[C] Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure D: Summary tables
# ---------------------------------------------------------------------------

def plot_metric_table(metric_key: str,
                      landscapes: list[str],
                      model_names: list[str],
                      all_metrics: dict[str, dict[str, dict[str, np.ndarray]]],
                      output_dir: Path) -> None:
    col_labels = [_land_display(l) for l in landscapes]
    row_labels = [MODEL_REGISTRY.get(m, {"label": m})["label"] for m in model_names]

    table_data = []
    for model_name in model_names:
        row = []
        for landscape in landscapes:
            val = float(all_metrics[landscape][model_name][metric_key].mean())
            row.append(f"{val:.4f}")
        table_data.append(row)

    fig, ax = plt.subplots(figsize=(max(6, 2.5 * len(landscapes)),
                                    1.5 + 0.5 * len(model_names)))
    ax.axis("off")
    tbl = ax.table(cellText=table_data, rowLabels=row_labels, colLabels=col_labels,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 1.8)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0 or c == -1:
            cell.set_facecolor("#D0D8E8")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#F7F9FC")

    meta = METRIC_META[metric_key]
    ax.set_title(f"{meta['title']} — Summary Table (mean over forecast steps)",
                 fontsize=12, pad=12)

    png_path = output_dir / f"table_{metric_key}.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[D] Saved: {png_path}")

    csv_path = output_dir / f"table_{metric_key}.csv"
    with open(csv_path, "w") as f:
        f.write("model," + ",".join(landscapes) + "\n")
        for model_name, row in zip(model_names, table_data):
            f.write(model_name + "," + ",".join(row) + "\n")
    print(f"[D] Saved: {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate comparison figures from pre-computed .npz results"
    )
    p.add_argument("--results", nargs="+", required=True,
                   metavar="LANDSCAPE:MODEL:NPZ_PATH")
    p.add_argument("--output_dir", default="./comparison_figures")
    p.add_argument("--n_traj_show", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def parse_result_spec(spec: str) -> dict:
    parts = spec.split(":")
    if len(parts) != 3:
        print(f"ERROR: need landscape:model:npz_path — got: {spec!r}", file=sys.stderr)
        sys.exit(1)
    return {"landscape": parts[0], "model": parts[1], "npz_path": parts[2]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    rng  = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    out  = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    specs = [parse_result_spec(s) for s in args.results]

    # Organise by landscape
    landscape_models: dict[str, dict[str, str]] = {}
    for sp in specs:
        landscape_models.setdefault(sp["landscape"], {})[sp["model"]] = sp["npz_path"]
    landscapes = list(landscape_models.keys())
    print(f"Landscapes: {landscapes}")

    # ── Load ──────────────────────────────────────────────────────────────
    print("\n=== Loading .npz results ===")
    all_data: dict[str, dict[str, dict]] = {}

    for land in landscapes:
        all_data[land] = {}
        for model, npz_path in landscape_models[land].items():
            print(f"  {land}/{model}  ←  {npz_path}")
            result = load_npz_result(npz_path)
            all_data[land][model] = result
            print(f"      loaded: N={result['N']}  H={result['n_future']}  "
                  f"S={result['n_samples']}  L={result['n_past']}")

    # ── Select display trajectories ───────────────────────────────────────
    N_min   = min(d["N"] for ld in all_data.values() for d in ld.values())
    n_show  = min(args.n_traj_show, N_min)
    display_labels = rng.choice(N_min, size=n_show, replace=False).tolist()
    print(f"\nDisplay trajectory labels (seed={args.seed}): {display_labels}")

    # ── Build display_data (pass full arrays; plotting uses display_labels) ─
    display_data: dict[str, dict[str, dict]] = {}
    for land in landscapes:
        display_data[land] = {}
        for model, data in all_data[land].items():
            display_data[land][model] = {
                "ground_truths": data["ground_truths"],  # (N, L+H)
                "samples":       data["samples"],         # (N, H, S)
                "display_labels": display_labels,
                "n_past":        data["n_past"],
                "n_future":      data["n_future"],
                "N":             data["N"],
            }

    # ── Compute metrics on full N ──────────────────────────────────────────
    print("\n=== Computing metrics (full N) ===")
    all_metrics: dict[str, dict[str, dict[str, np.ndarray]]] = {}

    for land in landscapes:
        all_metrics[land] = {}
        for model, mdata in display_data[land].items():
            n_past = mdata["n_past"]
            mets = compute_metrics(
                mdata["ground_truths"],
                mdata["samples"],
                n_past,
            )
            all_metrics[land][model] = mets
            label = MODEL_REGISTRY.get(model, {"label": model})["label"]
            print(
                f"  {label:14s} / {land:15s} — "
                f"MAE={mets['mae'].mean():.4f}  "
                f"CRPS={mets['crps'].mean():.4f}  "
                f"CI50={mets['ci50'].mean():.3f}  "
                f"CI90={mets['ci90'].mean():.3f}"
            )

    # ── Subset ground_truths to display_labels for plotting ───────────────
    print("\n=== Subsetting display trajectories for plots ===")
    plot_data: dict[str, dict[str, dict]] = {}
    for land in landscapes:
        plot_data[land] = {}
        for model, mdata in display_data[land].items():
            plot_data[land][model] = {
                "ground_truths":  mdata["ground_truths"][display_labels],   # (n_show, L+H)
                "samples":        mdata["samples"][display_labels],          # (n_show, H, S)
                "display_labels": display_labels,
                "n_past":         mdata["n_past"],
                "n_future":       mdata["n_future"],
            }

    # ── Generate figures ───────────────────────────────────────────────────
    print("\n=== Generating figures ===")
    for land in landscapes:
        n_past   = plot_data[land][list(plot_data[land].keys())[0]]["n_past"]
        n_future = plot_data[land][list(plot_data[land].keys())[0]]["n_future"]

        print(f"\n  [{land}] Trajectory comparison grid...")
        plot_trajectory_comparison_grid(
            landscape=land, display_labels=display_labels,
            model_data=plot_data[land], n_past=n_past, n_future=n_future,
            output_dir=out,
        )

        print(f"  [{land}] Histogram comparison grid...")
        plot_histogram2d_comparison_grid(
            landscape=land, display_labels=display_labels,
            model_data=plot_data[land], n_past=n_past, n_future=n_future,
            output_dir=out,
        )

    print("\n  Error metrics comparison grid...")
    plot_error_metrics_grid(landscapes=landscapes, all_metrics=all_metrics, output_dir=out)

    print("\n  Error metric tables...")
    model_order = list(display_data[landscapes[0]].keys())
    for metric_key in METRIC_ORDER:
        plot_metric_table(
            metric_key=metric_key, landscapes=landscapes,
            model_names=model_order, all_metrics=all_metrics, output_dir=out,
        )

    print(f"\nAll figures saved to: {out}")


if __name__ == "__main__":
    main()