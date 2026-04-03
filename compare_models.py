#!/usr/bin/env python3
"""
compare_models.py
=================
Multi-model, multi-landscape comparison figures for NFTSF vs ARIMA
(and any future models).

This script:
  1. Loads the raw test data .npy file and NFTSF model .pth for each landscape,
     runs full-testset inference internally to compute accurate metrics.
  2. Runs ARIMA inline on the same display context windows.
  3. Produces four types of figures:

     Figure A — Trajectory comparison grid (one per landscape)
         n_models rows × 3 trajectory columns.
         Each cell: 90%/50% confidence bands + ground truth.

     Figure B — 2-D histogram comparison grid (one per landscape)
         Same layout as Figure A using magma density heatmaps.

     Figure C — Error metrics grid (one figure, all landscapes)
         4 rows (landscapes) × 4 cols (MAE, CRPS, CI50, CI90).
         Each cell: per-step line plot, one line per model.

     Figure D — Error metric tables (one per metric: MAE, CRPS, CI50, CI90)
         Rows = models, columns = landscapes.
         Saved as both PNG (matplotlib table) and CSV.

Usage
-----
python compare_models.py \\
    --nftsf_data \\
        single_well:sw_test.npy:sw_model.pth:sw_config.json:sw_norm.npz:multi_sim:100:100 \\
        double_well:dw_test.npy:dw_model.pth:::multi_sim:100:100 \\
        alanine_phi:phi_test.npy:phi_model.pth:phi_config.json::tnf:50:50 \\
        alanine_psi:psi_test.npy:psi_model.pth:::tnf:50:50 \\
    --output_dir ./comparison_figures/ \\
    --n_traj_show 3 \\
    --n_samples 200 \\
    --seed 42

Spec format (colon-separated, first three fields required):
  landscape:data_path:model_path[:config_path[:norm_path[:data_format[:n_past[:n_future]]]]]
  - config_path  : training config JSON (architecture params); empty → use defaults
  - norm_path    : normalization stats .npz; empty → no normalisation
  - data_format  : multi_sim | single_sim | tnf  (default: multi_sim)
  - n_past       : context steps (default: 100)
  - n_future     : forecast steps (default: 100)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import os
import sys
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
# Path resolution — NFTSF_ssh ships architecture.py; insert its root so
# this script can import create_nfm without modifying either upstream repo.
# ---------------------------------------------------------------------------
_NFTSF_SSH = Path(__file__).resolve().parent.parent / "NFTSF_ssh"
if str(_NFTSF_SSH) not in sys.path:
    sys.path.insert(0, str(_NFTSF_SSH))

from architecture import create_nfm

# ---------------------------------------------------------------------------
# Model registry — add new models here.
# Keys: model name string; values: display label and colour.
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Landscape display helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate multi-model comparison figures from NFTSF + ARIMA."
    )
    p.add_argument(
        "--nftsf_data", nargs="+", default=None,
        metavar="LANDSCAPE:DATA:MODEL[:CONFIG[:NORM[:FORMAT[:N_PAST[:N_FUTURE]]]]]",
        help=(
            "Explicit file paths per landscape.  "
            "Format: landscape:data.npy:model.pth[:config.json[:norm.npz[:data_format[:n_past[:n_future]]]]]  "
            "Omit optional fields with an empty string, e.g. 'single_well:sw.npy:sw.pth:::multi_sim:100:100'."
        ),
    )
    p.add_argument(
        "--nftsf_dirs", nargs="+", default=None,
        metavar="LANDSCAPE:DATA_DIR:TRAINING_DIR[:FORMAT[:N_PAST[:N_FUTURE]]]",
        help=(
            "Auto-discover the latest model/config/norm files from directories.  "
            "Format: landscape:data_dir:training_dir[:format[:n_past[:n_future]]]  "
            "Finds {data_dir}/{landscape}_test.npy and the newest model_*.pth, "
            "config_*.json, norm_stats_*.npz in training_dir.  "
            "n_past/n_future default to values in the config JSON if not specified."
        ),
    )
    p.add_argument(
        "--output_dir", default="./comparison_figures",
        help="Directory where all output figures and tables are saved.",
    )
    p.add_argument(
        "--n_traj_show", type=int, default=3,
        help="Number of trajectories to display in the comparison grids (default 3).",
    )
    p.add_argument(
        "--n_samples", type=int, default=200,
        help="Ensemble samples per trajectory for NFTSF (default 200).",
    )
    p.add_argument(
        "--arima_n_samples", type=int, default=None,
        help=(
            "Ensemble samples for ARIMA (default: same as --n_samples). "
            "Set lower (e.g. 100) to speed up ARIMA without affecting NFTSF quality."
        ),
    )
    p.add_argument(
        "--no_arima", action="store_true", default=False,
        help="Skip ARIMA entirely — produce NFTSF-only figures (much faster).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible trajectory selection.",
    )
    p.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "cpu"],
        help="Torch device for NFTSF inference (default: auto).",
    )
    p.add_argument("--arima_max_p", type=int, default=5)
    p.add_argument("--arima_max_q", type=int, default=5)
    p.add_argument("--arima_max_d", type=int, default=2)
    p.add_argument(
        "--arima_workers", type=int, default=os.cpu_count() or 1,
        help="Parallel worker processes for ARIMA fitting (default: all CPUs).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# NFTSF data loading and full-testset inference
# ---------------------------------------------------------------------------

def _parse_nftsf_spec(spec: str) -> dict:
    """
    Parse one --nftsf_data entry.

    Format: landscape:data_path:model_path[:config_path[:norm_path[:data_format[:n_past[:n_future]]]]]
    Empty fields (::) mean "use default".
    """
    parts = spec.split(":")
    if len(parts) < 3:
        print(
            f"ERROR: --nftsf_data entries need at least landscape:data:model — got: {spec!r}",
            file=sys.stderr,
        )
        sys.exit(1)
    return {
        "landscape":   parts[0],
        "data_path":   parts[1],
        "model_path":  parts[2],
        "config_path": parts[3] if len(parts) > 3 and parts[3] else None,
        "norm_path":   parts[4] if len(parts) > 4 and parts[4] else None,
        "data_format": parts[5] if len(parts) > 5 and parts[5] else "multi_sim",
        "n_past":      int(parts[6]) if len(parts) > 6 and parts[6] else 100,
        "n_future":    int(parts[7]) if len(parts) > 7 and parts[7] else 100,
    }


def _latest_file(directory: str, pattern: str) -> str | None:
    """Return the most-recently-modified file matching *pattern* in *directory*, or None."""
    matches = sorted(
        glob.glob(os.path.join(directory, pattern)),
        key=os.path.getmtime,
    )
    return matches[-1] if matches else None


def _parse_nftsf_dirs_spec(spec: str) -> dict:
    """
    Parse one --nftsf_dirs entry and auto-discover the latest files.

    Format: landscape:data_dir:training_dir[:format[:n_past[:n_future]]]

    Discovery rules:
      data_path   = data_dir/{landscape}_test.npy  (falls back to newest *test*.npy)
      model_path  = newest model_*.pth  in training_dir
      config_path = newest config_*.json in training_dir
      norm_path   = newest norm_stats_*.npz in training_dir  (optional)
      n_past / n_future read from config JSON when not given on CLI
    """
    parts = spec.split(":")
    if len(parts) < 3:
        print(
            f"ERROR: --nftsf_dirs entries need landscape:data_dir:training_dir — got: {spec!r}",
            file=sys.stderr,
        )
        sys.exit(1)

    landscape    = parts[0]
    data_dir     = parts[1]
    training_dir = parts[2]
    data_format  = parts[3] if len(parts) > 3 and parts[3] else "multi_sim"
    n_past_arg   = int(parts[4]) if len(parts) > 4 and parts[4] else None
    n_future_arg = int(parts[5]) if len(parts) > 5 and parts[5] else None

    # ---- data file ----
    data_path = os.path.join(data_dir, f"{landscape}_test.npy")
    if not os.path.exists(data_path):
        data_path = _latest_file(data_dir, "*test*.npy")

    # ---- model / config / norm ----
    model_path  = _latest_file(training_dir, "model_*.pth")
    config_path = _latest_file(training_dir, "config_*.json")
    norm_path   = _latest_file(training_dir, "norm_stats_*.npz")  # may be None

    # ---- n_past / n_future from config ----
    n_past, n_future = 100, 100
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        n_past   = cfg.get("n_past",   n_past)
        n_future = cfg.get("n_future", n_future)
    if n_past_arg   is not None:
        n_past   = n_past_arg
    if n_future_arg is not None:
        n_future = n_future_arg

    # ---- validate required paths ----
    if not data_path or not os.path.exists(data_path):
        print(f"ERROR [{landscape}]: no test data found in {data_dir}", file=sys.stderr)
        sys.exit(1)
    if not model_path:
        print(f"ERROR [{landscape}]: no model_*.pth found in {training_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"  [{landscape}] auto-discovered:")
    print(f"    data   : {data_path}")
    print(f"    model  : {model_path}")
    print(f"    config : {config_path}")
    print(f"    norm   : {norm_path or '(none)'}")
    print(f"    format={data_format}  n_past={n_past}  n_future={n_future}")

    return {
        "landscape":   landscape,
        "data_path":   data_path,
        "model_path":  model_path,
        "config_path": config_path,
        "norm_path":   norm_path,
        "data_format": data_format,
        "n_past":      n_past,
        "n_future":    n_future,
    }


def _load_test_data(data_path: str, data_format: str) -> torch.Tensor:
    """Load .npy test data → (N_traj, T) float32 tensor."""
    raw = np.load(data_path, allow_pickle=True)
    t   = torch.tensor(np.asarray(raw), dtype=torch.float32)
    if data_format == "single_sim":
        return t[:, :, 1]
    elif data_format == "multi_sim":
        return t[:, 1:].T
    elif data_format == "tnf":
        arr = np.asarray(raw)
        if arr.ndim == 1:
            arr = arr[:, None, None]
        elif arr.ndim == 2:
            arr = arr[:, None, :]
        vals = arr[:, :, 1:].astype(np.float32) if arr.shape[2] > 1 else arr.astype(np.float32)
        series = vals[:, 0, 0].reshape(-1)
        return torch.tensor(series, dtype=torch.float32).unsqueeze(0)
    else:
        raise ValueError(f"Unknown data_format: {data_format!r}")


def _load_norm_stats(norm_path: str | None) -> dict | None:
    """Load normalisation stats .npz → dict with 'mean'/'std' tensors, or None."""
    if not norm_path or not Path(norm_path).exists():
        return None
    s = np.load(norm_path)
    return {"mean": torch.tensor(s["mean"]), "std": torch.tensor(s["std"])}


def _load_nftsf_model(
    model_path: str,
    config_path: str | None,
    n_past: int,
    n_future: int,
    device: torch.device,
):
    """Create NFTSF model, load weights, set to eval mode."""
    flow_blocks  = 6
    hidden_units = 64
    hidden_layers = "1,2"
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            cfg = json.load(f)
        flow_blocks   = cfg.get("flow_blocks",   flow_blocks)
        hidden_units  = cfg.get("hidden_units",  hidden_units)
        hidden_layers = cfg.get("hidden_layers", hidden_layers)
    hidden_layers_list = tuple(int(x) for x in str(hidden_layers).split(","))
    model = create_nfm(
        device, n_future, n_past,
        K=flow_blocks, hidden_units=hidden_units,
        hidden_layers_list=hidden_layers_list,
    )
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=False)
    )
    model.eval()
    return model


def run_nftsf_full_inference(
    landscape: str,
    data_tensor: torch.Tensor,  # (N, T)
    model,
    norm_stats: dict | None,
    n_past: int,
    n_future: int,
    n_samples: int,
    display_labels: list[int],  # which trajectory indices to collect for visualization
    device: torch.device,
) -> dict:
    """
    Run NFTSF inference over ALL N trajectories.

    * Computes per-step MAE, CRPS, CI50 and CI90 averaged over the full test
      set — these are used for Figure C (accurate, low-variance estimates).
    * Collects full (ground_truth, samples) arrays only for *display_labels*
      trajectories — these are used for Figures A & B (visualization subset).

    Returns
    -------
    dict with keys:
        ground_truths : (N_display, n_past+n_future)  — display subset
        samples       : (N_display, n_samples, n_future)
        n_past        : int
        n_future      : int
        fullset_mae   : (n_future,)   averaged over all N
        fullset_crps  : (n_future,)
        fullset_ci50  : (n_future,)
        fullset_ci90  : (n_future,)
        fullset_N     : int
    """
    n_trajs, traj_len = data_tensor.shape
    n_extrp  = n_past + n_future
    disp_set = set(display_labels)

    if norm_stats is not None:
        mu_np  = norm_stats["mean"].cpu().numpy()
        std_np = norm_stats["std"].cpu().numpy()
        norm_tensor = (data_tensor - norm_stats["mean"]) / norm_stats["std"]
    else:
        mu_np, std_np = 0.0, 1.0
        norm_tensor = data_tensor

    # Per-step accumulators
    mae_steps:  list[np.ndarray] = []
    crps_steps: list[np.ndarray] = []
    cov_ci50:   list[np.ndarray] = []
    cov_ci90:   list[np.ndarray] = []

    # Visualization storage
    disp_gts:  dict[int, np.ndarray] = {}
    disp_samp: dict[int, np.ndarray] = {}

    n_failed = 0
    for i in _tqdm(range(n_trajs), desc=f"NFTSF [{landscape}]"):
        start = np.random.randint(0, max(1, traj_len - n_extrp + 1))

        past_norm   = norm_tensor[i, start:start + n_past].unsqueeze(0).to(device)
        past_repeat = past_norm.repeat(n_samples, 1)

        with torch.no_grad():
            try:
                samp_norm = model.sample(n_samples, past_repeat)[0].cpu().numpy()
            except AssertionError:
                n_failed += 1
                del past_norm, past_repeat
                continue

        samp_real   = (samp_norm * std_np) + mu_np          # (n_samples, n_future)
        real_future = data_tensor[i, start + n_past:start + n_extrp].cpu().numpy()

        # Per-step MAE
        mae_steps.append(np.abs(np.median(samp_real, axis=0) - real_future))

        # Per-step CRPS
        mae_term = np.mean(np.abs(samp_real - real_future[None, :]), axis=0)
        s_sorted = np.sort(samp_real, axis=0)
        diff     = s_sorted[1:] - s_sorted[:-1]
        ns       = n_samples
        weights  = np.arange(1, ns) * np.arange(ns - 1, 0, -1)
        spread   = np.sum(diff * weights[:, None], axis=0) / ns ** 2
        crps_steps.append(mae_term - spread)

        # Per-step coverage
        for cov_list, (lo_pct, hi_pct) in [(cov_ci50, (25.0, 75.0)), (cov_ci90, (5.0, 95.0))]:
            lo = np.percentile(samp_real, lo_pct, axis=0)
            hi = np.percentile(samp_real, hi_pct, axis=0)
            cov_list.append(((real_future >= lo) & (real_future <= hi)).astype(float))

        # Collect visualization data
        if i in disp_set:
            real_full   = data_tensor[i, start:start + n_extrp].cpu().numpy()
            disp_gts[i]  = real_full
            disp_samp[i] = samp_real

        del past_norm, past_repeat

    n_eval = len(mae_steps)
    if n_failed:
        print(f"  WARNING: {n_failed}/{n_trajs} trajectories skipped (spline instability)")
    print(f"  NFTSF [{landscape}]: evaluated {n_eval}/{n_trajs} trajectories")

    # Pack display arrays in the order given by display_labels
    valid_labels = [lb for lb in display_labels if lb in disp_gts]
    if len(valid_labels) < len(display_labels):
        missing = set(display_labels) - set(valid_labels)
        print(f"  WARNING: display trajectories {missing} failed during inference — skipping.")

    disp_gt_arr  = np.array([disp_gts[lb]  for lb in valid_labels])  # (N_disp, n_past+n_future)
    disp_samp_arr = np.array([disp_samp[lb] for lb in valid_labels]) # (N_disp, n_samples, n_future)

    return {
        "ground_truths": disp_gt_arr,
        "samples":       disp_samp_arr,
        "display_labels": valid_labels,
        "n_past":        n_past,
        "n_future":      n_future,
        "fullset_mae":   np.mean(mae_steps,  axis=0),
        "fullset_crps":  np.mean(crps_steps, axis=0),
        "fullset_ci50":  np.mean(cov_ci50,   axis=0),
        "fullset_ci90":  np.mean(cov_ci90,   axis=0),
        "fullset_N":     n_eval,
    }


# ---------------------------------------------------------------------------
# ARIMA evaluation
# ---------------------------------------------------------------------------

def _arima_predict_one(
    context: np.ndarray,
    n_future: int,
    n_samples: int,
    max_p: int,
    max_q: int,
    max_d: int,
) -> np.ndarray:
    """
    Fit auto_arima to *context* and draw *n_samples* simulated paths.

    Returns
    -------
    samples : (n_samples, n_future) float32
    """
    try:
        import pmdarima as pm
    except ImportError:
        raise ImportError(
            "pmdarima is required for ARIMA forecasting.  "
            "Install with:  pip install pmdarima"
        )

    series = context.astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arima = pm.auto_arima(
            series,
            max_p=max_p,
            max_q=max_q,
            max_d=max_d,
            seasonal=False,
            error_action="ignore",
            suppress_warnings=True,
            stepwise=True,
        )

    out = np.empty((n_samples, n_future), dtype=np.float32)
    for s in range(n_samples):
        try:
            sim = arima.simulate(
                nsimulations=n_future,
                repetitions=1,
                initial_values=series,
            ).flatten()
            out[s] = sim.astype(np.float32)
        except Exception:
            fc, ci = arima.predict(n_periods=n_future, return_conf_int=True)
            std = (ci[:, 1] - ci[:, 0]) / (2 * 1.96)
            out[s] = (fc + np.random.randn(n_future) * std).astype(np.float32)
    return out


def run_arima(
    ground_truths: np.ndarray,
    n_past: int,
    n_future: int,
    n_samples: int,
    max_p: int = 5,
    max_q: int = 5,
    max_d: int = 2,
) -> np.ndarray:
    """
    Fit and predict ARIMA for every trajectory in *ground_truths*.

    Parameters
    ----------
    ground_truths : (N, n_past + n_future)
    n_past, n_future : context / forecast lengths
    n_samples : ensemble size

    Returns
    -------
    arima_samples : (N, n_samples, n_future) float32
    """
    N = ground_truths.shape[0]
    all_samples = np.empty((N, n_samples, n_future), dtype=np.float32)
    for i in range(N):
        context = ground_truths[i, :n_past]
        print(f"    ARIMA fitting trajectory {i+1}/{N} ...", end="\r", flush=True)
        all_samples[i] = _arima_predict_one(
            context, n_future, n_samples, max_p, max_q, max_d
        )
    print()   # newline after \r progress
    return all_samples


def _arima_eval_one(args_tuple):
    """
    Top-level worker function (picklable) for ProcessPoolExecutor.

    Parameters
    ----------
    args_tuple : (context_np, future_np, n_future, n_samples, max_p, max_q, max_d,
                  collect_samples)
        context_np      : (n_past,) float32 context window
        future_np       : (n_future,) float32 ground truth future
        collect_samples : bool — whether to return the full samples array

    Returns
    -------
    (mae_step, crps_step, ci50_step, ci90_step, samples_or_None)
        Each metric array is (n_future,) float32.
        samples_or_None is (n_samples, n_future) float32 if collect_samples else None.
    """
    context_np, future_np, n_future, n_samples, max_p, max_q, max_d, collect = args_tuple

    try:
        import pmdarima as pm
    except ImportError:
        raise ImportError("pmdarima is required.  pip install pmdarima")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arima = pm.auto_arima(
            context_np.astype(np.float64),
            max_p=max_p, max_q=max_q, max_d=max_d,
            seasonal=False, error_action="ignore",
            suppress_warnings=True, stepwise=True,
        )

    out = np.empty((n_samples, n_future), dtype=np.float32)
    for s in range(n_samples):
        try:
            sim = arima.simulate(
                nsimulations=n_future, repetitions=1,
                initial_values=context_np.astype(np.float64),
            ).flatten()
            out[s] = sim.astype(np.float32)
        except Exception:
            fc, ci = arima.predict(n_periods=n_future, return_conf_int=True)
            std = (ci[:, 1] - ci[:, 0]) / (2 * 1.96)
            out[s] = (fc + np.random.randn(n_future) * std).astype(np.float32)

    # Per-step MAE (median forecast)
    mae_step = np.abs(np.median(out, axis=0) - future_np).astype(np.float32)

    # Per-step CRPS
    mae_term = np.mean(np.abs(out - future_np[None, :]), axis=0)
    s_sorted = np.sort(out, axis=0)
    diff     = s_sorted[1:] - s_sorted[:-1]
    ns       = n_samples
    weights  = np.arange(1, ns) * np.arange(ns - 1, 0, -1)
    spread   = np.sum(diff * weights[:, None], axis=0) / ns ** 2
    crps_step = (mae_term - spread).astype(np.float32)

    # Per-step coverage
    lo50 = np.percentile(out, 25.0, axis=0)
    hi50 = np.percentile(out, 75.0, axis=0)
    lo90 = np.percentile(out,  5.0, axis=0)
    hi90 = np.percentile(out, 95.0, axis=0)
    ci50_step = ((future_np >= lo50) & (future_np <= hi50)).astype(np.float32)
    ci90_step = ((future_np >= lo90) & (future_np <= hi90)).astype(np.float32)

    return mae_step, crps_step, ci50_step, ci90_step, (out if collect else None)


def run_arima_full_eval(
    landscape: str,
    data_tensor: "torch.Tensor",   # (N, T)
    n_past: int,
    n_future: int,
    n_samples: int,
    max_p: int,
    max_q: int,
    max_d: int,
    n_workers: int,
    display_labels: list[int],
) -> dict:
    """
    Evaluate ARIMA over ALL N trajectories in the test set in parallel.

    Mirrors run_nftsf_full_inference exactly: every trajectory uses a randomly
    chosen window, and the same display_labels are collected for visualisation.

    Returns
    -------
    dict with keys:
        ground_truths : (N_display, n_past+n_future)  — display subset
        samples       : (N_display, n_samples, n_future)
        n_past        : int
        n_future      : int
        fullset_mae   : (n_future,) averaged over all N
        fullset_crps  : (n_future,)
        fullset_ci50  : (n_future,)
        fullset_ci90  : (n_future,)
        fullset_N     : int
    """
    n_trajs, traj_len = data_tensor.shape
    n_extrp  = n_past + n_future
    disp_set = set(display_labels)
    data_np  = data_tensor.cpu().numpy()

    # Build work items: (context, future, n_future, n_samples, ..., collect)
    starts = np.random.randint(0, max(1, traj_len - n_extrp + 1), size=n_trajs)
    work   = []
    for i in range(n_trajs):
        s   = int(starts[i])
        ctx = data_np[i, s:s + n_past].astype(np.float32)
        fut = data_np[i, s + n_past:s + n_extrp].astype(np.float32)
        work.append((ctx, fut, n_future, n_samples, max_p, max_q, max_d, i in disp_set))

    print(f"  ARIMA [{landscape}]: fitting {n_trajs} trajectories "
          f"with {n_workers} workers ...")

    mae_list  = []
    crps_list = []
    ci50_list = []
    ci90_list = []
    disp_gts:  dict[int, np.ndarray] = {}
    disp_samp: dict[int, np.ndarray] = {}

    n_workers_eff = min(n_workers, n_trajs)
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers_eff) as pool:
        for i, result in enumerate(_tqdm(
            pool.map(_arima_eval_one, work, chunksize=max(1, n_trajs // (n_workers_eff * 4))),
            total=n_trajs, desc=f"ARIMA [{landscape}]",
        )):
            mae_s, crps_s, ci50_s, ci90_s, samp = result
            mae_list.append(mae_s)
            crps_list.append(crps_s)
            ci50_list.append(ci50_s)
            ci90_list.append(ci90_s)
            if samp is not None:
                s   = int(starts[i])
                gt  = data_np[i, s:s + n_extrp].astype(np.float32)
                disp_gts[i]  = gt
                disp_samp[i] = samp

    n_eval = len(mae_list)
    print(f"  ARIMA [{landscape}]: evaluated {n_eval}/{n_trajs} trajectories")

    valid_labels  = [lb for lb in display_labels if lb in disp_gts]
    disp_gt_arr   = np.array([disp_gts[lb]  for lb in valid_labels])
    disp_samp_arr = np.array([disp_samp[lb] for lb in valid_labels])

    return {
        "ground_truths":  disp_gt_arr,
        "samples":        disp_samp_arr,
        "display_labels": valid_labels,
        "n_past":         n_past,
        "n_future":       n_future,
        "fullset_mae":    np.mean(mae_list,  axis=0),
        "fullset_crps":   np.mean(crps_list, axis=0),
        "fullset_ci50":   np.mean(ci50_list, axis=0),
        "fullset_ci90":   np.mean(ci90_list, axis=0),
        "fullset_N":      n_eval,
    }


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    rng  = np.random.default_rng(args.seed)
    np.random.seed(args.seed)   # also seed legacy numpy RNG used in inference loop
    out  = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Resolve torch device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # -------------------------------------------------- Parse landscape specs
    if not args.nftsf_data and not args.nftsf_dirs:
        print("ERROR: provide --nftsf_data or --nftsf_dirs (or both).", file=sys.stderr)
        sys.exit(1)

    specs: list[dict] = []
    if args.nftsf_data:
        specs += [_parse_nftsf_spec(s) for s in args.nftsf_data]
    if args.nftsf_dirs:
        print("\n=== Auto-discovering files from directories ===")
        specs += [_parse_nftsf_dirs_spec(s) for s in args.nftsf_dirs]

    landscapes = [s["landscape"] for s in specs]

    # ---------------------------------- Load data tensors (cheap, gets N sizes)
    print("\n=== Loading test data ===")
    data_tensors: dict[str, torch.Tensor] = {}
    for sp in specs:
        land = sp["landscape"]
        print(f"  [{land}] {sp['data_path']}  (format={sp['data_format']})")
        data_tensors[land] = _load_test_data(sp["data_path"], sp["data_format"])
        print(f"    shape: {tuple(data_tensors[land].shape)}")

    # ---------------------------------- Select display trajectories (seeded)
    N_min  = min(t.shape[0] for t in data_tensors.values())
    n_show = min(args.n_traj_show, N_min)
    display_labels: list[int] = rng.choice(N_min, size=n_show, replace=False).tolist()
    print(f"\nDisplay trajectory labels (seed={args.seed}): {display_labels}")

    # ---------------------------------- Raw trajectory overlay plots
    print("\n=== Generating raw trajectory plots ===")
    for sp in specs:
        plot_raw_trajectories(
            landscape=sp["landscape"],
            data_tensor=data_tensors[sp["landscape"]],
            n_past=sp["n_past"],
            n_future=sp["n_future"],
            output_dir=out,
            seed=args.seed,
        )

    # ---------------------------------- NFTSF full-testset inference
    print("\n=== Running NFTSF full-testset inference ===")
    nftsf_data: dict[str, dict] = {}
    for sp in specs:
        land = sp["landscape"]
        print(f"  Loading model [{land}] from {sp['model_path']}")
        norm_stats = _load_norm_stats(sp["norm_path"])
        model = _load_nftsf_model(
            sp["model_path"], sp["config_path"],
            sp["n_past"], sp["n_future"], device,
        )
        nftsf_data[land] = run_nftsf_full_inference(
            landscape=land,
            data_tensor=data_tensors[land],
            model=model,
            norm_stats=norm_stats,
            n_past=sp["n_past"],
            n_future=sp["n_future"],
            n_samples=args.n_samples,
            display_labels=display_labels,
            device=device,
        )
        # Update display_labels in case some failed (intersection of valid labels)
        display_labels = nftsf_data[land]["display_labels"]
        del model   # free GPU memory before next landscape

    # ------------------------------------------------------------------ ARIMA
    # ARIMA runs only on the display subset (fitting ARIMA for all N is too slow).
    # ------------------------------------------------------------------ ARIMA
    arima_n_samples = args.arima_n_samples if args.arima_n_samples is not None else args.n_samples
    arima_data: dict[str, dict] = {}

    if args.no_arima:
        print("\n=== Skipping ARIMA (--no_arima) ===")
    else:
        print(
            f"\n=== Running ARIMA inference "
            f"(full test set, {args.arima_workers} workers, {arima_n_samples} samples) ==="
        )
        for land in landscapes:
            ndat = nftsf_data[land]
            arima_data[land] = run_arima_full_eval(
                landscape=land,
                data_tensor=data_tensors[land],
                n_past=ndat["n_past"],
                n_future=ndat["n_future"],
                n_samples=arima_n_samples,
                max_p=args.arima_max_p,
                max_q=args.arima_max_q,
                max_d=args.arima_max_d,
                n_workers=args.arima_workers,
                display_labels=display_labels,
            )
            display_labels = arima_data[land]["display_labels"]

    # ---------------------------------------------------- Compute metrics
    print("\n=== Computing metrics ===")
    all_metrics: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    model_order = ["nftsf"] + ([] if args.no_arima else ["arima"])

    for land in landscapes:
        all_metrics[land] = {}
        n_past = nftsf_data[land]["n_past"]

        model_bundles: dict[str, dict] = {"nftsf": nftsf_data[land]}
        if not args.no_arima:
            model_bundles["arima"] = arima_data[land]

        for model_name in model_order:
            mdat = model_bundles[model_name]
            mets = compute_metrics(
                mdat["ground_truths"], mdat["samples"], n_past,
                fullset_mae=mdat.get("fullset_mae"),
                fullset_crps=mdat.get("fullset_crps"),
                fullset_ci50=mdat.get("fullset_ci50"),
                fullset_ci90=mdat.get("fullset_ci90"),
            )
            all_metrics[land][model_name] = mets
            label = MODEL_REGISTRY.get(model_name, {"label": model_name})["label"]
            fs_n  = mdat.get("fullset_N")
            src   = f"N={fs_n}" if fs_n else f"N={mdat['ground_truths'].shape[0]} (display)"
            print(
                f"  {label:6s} / {land:15s} ({src}) — "
                f"MAE={mets['mae'].mean():.4f}  "
                f"CRPS={mets['crps'].mean():.4f}  "
                f"CI50={mets['ci50'].mean():.3f}  "
                f"CI90={mets['ci90'].mean():.3f}"
            )

    # ---------------------------------------- Figures A & B (per landscape)
    print("\n=== Generating trajectory & histogram comparison grids ===")
    for land in landscapes:
        n_past   = nftsf_data[land]["n_past"]
        n_future = nftsf_data[land]["n_future"]

        model_data: dict[str, dict] = {"nftsf": nftsf_data[land]}
        if not args.no_arima:
            model_data["arima"] = arima_data[land]

        # Figure F: predicted sample overlays (raw-trajectory-style, per model)
        print(f"\n=== Generating predicted trajectory plots [{land}] ===")
        for mname, mdat in model_data.items():
            mdat_with_labels = {**mdat, "display_labels": display_labels}
            plot_predicted_trajectories(
                landscape=land,
                model_name=mname,
                model_data=mdat_with_labels,
                output_dir=out,
            )

        plot_trajectory_comparison_grid(
            landscape=land,
            display_labels=display_labels,
            model_data=model_data,
            n_past=n_past,
            n_future=n_future,
            output_dir=out,
        )

        plot_histogram2d_comparison_grid(
            landscape=land,
            display_labels=display_labels,
            model_data=model_data,
            n_past=n_past,
            n_future=n_future,
            output_dir=out,
        )

    # ---------------------------------------- Figure C: error metrics grid
    print("\n=== Generating error metrics comparison grid ===")
    plot_error_metrics_grid(
        landscapes=landscapes,
        all_metrics=all_metrics,
        output_dir=out,
    )

    # ---------------------------------------- Figure D: tables
    print("\n=== Generating error metric tables ===")
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
