#!/usr/bin/env python3
"""
generate_tables_only.py – Same CLI as compare.py, outputs tables with best values shaded.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

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
# Argument parsing (identical to compare.py)
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate summary tables with best values highlighted")
    p.add_argument("--results", nargs="+", required=True,
                   metavar="LANDSCAPE:MODEL:NPZ_PATH")
    p.add_argument("--output_dir", default="./comparison_figures")
    p.add_argument("--n_traj_show", type=int, default=3)   # ignored
    p.add_argument("--seed", type=int, default=42)         # ignored
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
    samples = data["samples"]
    full_trajectories = data["full_trajectories"]
    train_test_split = int(data["train_test_split"])
    H = int(data["prediction_length"])

    if samples.ndim == 3 and samples.shape[1] == H:
        samples = samples.transpose(0, 2, 1)

    try:
        L = int(data["context_length"])
    except (KeyError, ValueError):
        L = context_length

    start_idx = train_test_split - L
    end_idx   = train_test_split + H
    ground_truths = full_trajectories[:, start_idx:end_idx]

    return {
        "ground_truths": ground_truths.astype(np.float32),
        "samples": samples.astype(np.float32),
        "n_past": L,
        "n_future": H,
        "fullset_N": ground_truths.shape[0],
    }

# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------
def _mae_median(gt_future: np.ndarray, samples: np.ndarray) -> np.ndarray:
    median = np.median(samples, axis=1)
    return np.abs(gt_future - median).mean(axis=0)

def _mae_mean(gt_future: np.ndarray, samples: np.ndarray) -> np.ndarray:
    mean = np.mean(samples, axis=1)
    return np.abs(gt_future - mean).mean(axis=0)

def _mae_sample(gt_future: np.ndarray, samples: np.ndarray) -> np.ndarray:
    return np.abs(samples - gt_future[:, np.newaxis, :]).mean(axis=(0, 1))

def _crps_coverage(gt_future: np.ndarray, samples: np.ndarray) -> np.ndarray:
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

def compute_metrics(ground_truths: np.ndarray, samples: np.ndarray, n_past: int) -> dict:
    gt_future = ground_truths[:, n_past:]
    return {
        "mae_median": _mae_median(gt_future, samples),
        "mae_mean":   _mae_mean(gt_future, samples),
        "mae_sample": _mae_sample(gt_future, samples),
        "crps":       _crps_coverage(gt_future, samples),
    }

# ---------------------------------------------------------------------------
# Table generation with best value shading
# ---------------------------------------------------------------------------
def generate_tables(landscapes, model_names, all_metrics, output_dir):
    # CSV (no shading, just raw values)
    csv_path = output_dir / "table_all.csv"
    with open(csv_path, "w") as f:
        f.write("landscape,model,mae_median,mae_mean,mae_sample,crps\n")
        for land in landscapes:
            for model_name in model_names:
                if model_name not in all_metrics[land]:
                    continue
                mets = all_metrics[land][model_name]
                label = MODEL_REGISTRY.get(model_name, {"label": model_name})["label"]
                row = [
                    land,
                    label,
                    f"{mets['mae_median'].mean():.4f}",
                    f"{mets['mae_mean'].mean():.4f}",
                    f"{mets['mae_sample'].mean():.4f}",
                    f"{mets['crps'].mean():.4f}",
                ]
                f.write(",".join(row) + "\n")
    print(f"CSV saved: {csv_path}")

    # PNG tables per landscape with best values highlighted
    for land in landscapes:
        # Collect rows and numeric values
        rows = []       # list of [label, mae_median, mae_mean, mae_sample, crps] as strings
        values = []     # list of [mae_median, mae_mean, mae_sample, crps] as floats
        model_labels = []
        for model_name in model_names:
            if model_name not in all_metrics[land]:
                continue
            mets = all_metrics[land][model_name]
            label = MODEL_REGISTRY.get(model_name, {"label": model_name})["label"]
            model_labels.append(label)
            vals = [
                mets["mae_median"].mean(),
                mets["mae_mean"].mean(),
                mets["mae_sample"].mean(),
                mets["crps"].mean(),
            ]
            values.append(vals)
            rows.append([label] + [f"{v:.4f}" for v in vals])

        if not rows:
            continue

        # Determine best indices per column (lower is better for all metrics)
        best_indices = {}
        num_cols = len(values[0])  # 4 metrics
        for col in range(num_cols):
            col_vals = [v[col] for v in values]
            best_idx = int(np.argmin(col_vals))
            best_indices[col] = best_idx

        # Create table with highlighting
        fig, ax = plt.subplots(figsize=(8, 2 + len(rows)))
        ax.axis("off")

        col_labels = ["Model", "MAE (median)", "MAE (mean)", "MAE (sample)", "CRPS"]
        table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)

        # Apply shading: header row (row 0) already has default color; data rows start at row 1
        for (r, c), cell in table.get_celld().items():
            if r == 0:   # header row
                cell.set_facecolor("#D0D8E8")
                cell.set_text_props(weight="bold")
            else:
                cell.set_facecolor("#F7F9FC")
                # c=0 is the model name column; metrics start at c=1
                if c >= 1:
                    metric_col = c - 1
                    if metric_col in best_indices and best_indices[metric_col] == r - 1:
                        cell.set_facecolor("#90EE90")   # light green
                        cell.set_text_props(weight="bold")

        ax.set_title(f"{_land_display(land)} — Metrics Summary (mean over forecast steps)", fontsize=13, pad=14)
        png_path = output_dir / f"table_{land}.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Table PNG saved: {png_path}")

def _land_display(landscape: str) -> str:
    """Simple display name for landscape."""
    return landscape.replace("_", " ").title()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
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

    print("\n=== Computing metrics (full testset) ===")
    all_metrics: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for land in landscapes:
        all_metrics[land] = {}
        for model, data in all_data[land].items():
            mets = compute_metrics(data["ground_truths"], data["samples"], data["n_past"])
            all_metrics[land][model] = mets
            label = MODEL_REGISTRY.get(model, {"label": model})["label"]
            print(f"  {label:15s} / {land:15s} (N={data['fullset_N']}) — "
                  f"MAE_med={mets['mae_median'].mean():.4f}  MAE_mean={mets['mae_mean'].mean():.4f}  "
                  f"MAE_sample={mets['mae_sample'].mean():.4f}  CRPS={mets['crps'].mean():.4f}")

    print("\n=== Generating tables ===")
    model_order = list(next(iter(all_metrics.values())).keys())
    generate_tables(landscapes, model_order, all_metrics, out)

    print(f"\n✓ Tables saved to: {out}")

if __name__ == "__main__":
    main()