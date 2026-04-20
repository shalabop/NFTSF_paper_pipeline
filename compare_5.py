#!/usr/bin/env python3
"""
table_q50_q90_only.py – Generate table with Q50 and Q90 coverage only.
Best values: Q50 closest to 0.5, Q90 closest to 0.9.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Q50/Q90 coverage table")
    p.add_argument("--results", nargs="+", required=True,
                   metavar="LANDSCAPE:MODEL:NPZ_PATH")
    p.add_argument("--output_dir", default="./comparison_figures")
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
# Metrics (only Q50 and Q90 coverage)
# ---------------------------------------------------------------------------
def _quantile_coverage(gt_future: np.ndarray, samples: np.ndarray, q: float) -> float:
    """Fraction of ground truth values below the q‑quantile of the samples."""
    q_quantile = np.quantile(samples, q, axis=1)   # (N, H)
    below = (gt_future < q_quantile)
    return float(np.mean(below))

def compute_metrics(ground_truths: np.ndarray, samples: np.ndarray, n_past: int) -> dict:
    gt_future = ground_truths[:, n_past:]
    return {
        "q50_cov": _quantile_coverage(gt_future, samples, 0.5),
        "q90_cov": _quantile_coverage(gt_future, samples, 0.9),
    }

# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------
def _land_display(landscape: str) -> str:
    return landscape.replace("_", " ").title()

def save_table(rows, col_labels, title, output_path, best_funcs):
    fig, ax = plt.subplots(figsize=(8, 2 + len(rows)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    best_indices = {}
    num_cols = len(rows[0]) - 1
    for col in range(num_cols):
        func = best_funcs[col]
        col_vals = [float(row[col+1]) for row in rows]
        best_idx = func(col_vals)
        best_indices[col] = best_idx

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#D0D8E8")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#F7F9FC")
            if c >= 1 and (c-1) in best_indices and best_indices[c-1] == r-1:
                cell.set_facecolor("#90EE90")
                cell.set_text_props(weight="bold")

    ax.set_title(title, fontsize=13, pad=14)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")

def generate_tables(landscapes, model_names, all_metrics, output_dir):
    # CSV
    csv_path = output_dir / "table_q50_q90.csv"
    with open(csv_path, "w") as f:
        f.write("landscape,model,q50_cov,q90_cov\n")
        for land in landscapes:
            for model_name in model_names:
                if model_name not in all_metrics[land]:
                    continue
                mets = all_metrics[land][model_name]
                label = MODEL_REGISTRY.get(model_name, {"label": model_name})["label"]
                row = [
                    land, label,
                    f"{mets['q50_cov']:.4f}",
                    f"{mets['q90_cov']:.4f}",
                ]
                f.write(",".join(row) + "\n")
    print(f"CSV saved: {csv_path}")

    # PNG table per landscape
    for land in landscapes:
        rows = []
        for model_name in model_names:
            if model_name not in all_metrics[land]:
                continue
            mets = all_metrics[land][model_name]
            label = MODEL_REGISTRY.get(model_name, {"label": model_name})["label"]
            rows.append([
                label,
                f"{mets['q50_cov']:.4f}",
                f"{mets['q90_cov']:.4f}",
            ])
        if rows:
            def best_q50(vals):
                return int(np.argmin(np.abs(np.array(vals) - 0.5)))
            def best_q90(vals):
                return int(np.argmin(np.abs(np.array(vals) - 0.9)))
            best_funcs = [best_q50, best_q90]
            save_table(
                rows,
                ["Model", "Q50 Coverage", "Q90 Coverage"],
                f"{_land_display(land)} — Quantile Coverage",
                output_dir / f"table_q50_q90_{land}.png",
                best_funcs
            )

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
    all_metrics: dict[str, dict[str, dict[str, float]]] = {}
    for land in landscapes:
        all_metrics[land] = {}
        for model, data in all_data[land].items():
            mets = compute_metrics(data["ground_truths"], data["samples"], data["n_past"])
            all_metrics[land][model] = mets
            label = MODEL_REGISTRY.get(model, {"label": model})["label"]
            print(f"  {label:15s} / {land:15s} (N={data['fullset_N']}) — "
                  f"Q50_cov={mets['q50_cov']:.4f}  Q90_cov={mets['q90_cov']:.4f}")

    print("\n=== Generating tables ===")
    model_order = list(next(iter(all_metrics.values())).keys())
    generate_tables(landscapes, model_order, all_metrics, out)

    print(f"\n✓ Tables saved to: {out}")

if __name__ == "__main__":
    main()