#!/usr/bin/env python3
"""
scripts/test_model_paper_clean.py
===================================
Clean single-model evaluation script for paper use.

This is the recommended entry point for generating per-method evaluation
outputs consumed by compare_models_paper_clean.py.  It supersedes
eval/evaluate.py for the paper path; legacy scripts remain untouched.

Key differences from eval/evaluate.py
---------------------------------------
- Computes CRPS decomposition (Reliability / Resolution / Uncertainty) via
  metrics/crps_decomposition.py instead of properscoring.
- Times only the inference/sampling stage via metrics/timing_metrics.SampleTimer.
- Emits a stable JSON output schema (no raw samples saved — keeps files small).
- All config is explicit: no hidden globals, no hardcoded chunk sizes.
- Seed is set and logged before any random operation.

Output schema
-------------
Writes ``<output_dir>/result_<method>_<dataset>_<run_id>.json`` with keys
matching the stable paper-column names in compare_models_paper_clean.py.

See docs/paper_clean_path.md for scientific framing.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from metrics.metrics_probabilistic import compute_all
from metrics.timing_metrics import SampleTimer, append_timing_record


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _run_inference_timed(
    model,
    context,
    *,
    n_samples: int,
    chunk_size: int,
    method: str,
    dataset: str,
    horizon: int,
    device: str,
    notes: str | None,
) -> tuple[np.ndarray, dict]:
    """
    Run chunked inference inside SampleTimer.

    Returns (samples array (N, T, S), timing_record dict).
    """
    import numpy as np

    N = context.shape[0]
    all_samples: list[np.ndarray] = []

    # Warm-up: one small forward pass to force JIT / CUDA kernel compilation.
    try:
        _ = model.predict(context[:1], n_samples=max(2, min(n_samples, 4)))
    except Exception:
        pass

    with SampleTimer(
        method=method,
        dataset=dataset,
        horizon=horizon,
        n_samples=n_samples,
        n_items=N,
        device=device,
        notes=notes,
    ) as timer:
        for start in range(0, N, chunk_size):
            chunk = context[start : start + chunk_size]
            s = model.predict(chunk, n_samples=n_samples)  # (B, T, S)
            all_samples.append(s)

    samples = np.concatenate(all_samples, axis=0)  # (N, T, S)
    return samples, timer.record


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_paper_clean(
    *,
    checkpoint: Path,
    data_path: Path,
    method: str,
    dataset: str,
    system: str | None,
    n_samples: int,
    chunk_size: int,
    seed: int,
    output_dir: Path,
    run_id: str,
    notes: str | None,
    timing_log: Path | None,
) -> dict:
    """
    Full evaluation for one method/dataset pair.

    Returns the result dict (also written to disk).
    """
    import torch

    _set_seed(seed)

    # ---- Config ----
    config_path = checkpoint / "config.json"
    if not config_path.exists():
        sys.exit(f"[test_model_paper_clean] config.json not found in {checkpoint}")
    with open(config_path) as f:
        config = json.load(f)

    n_past   = int(config["data"]["n_past"])
    n_future = int(config["data"]["n_future"])
    landscape = config.get("landscape", dataset)

    device_str = config["training"].get("device", "auto")
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[test_model_paper_clean] method={method}  dataset={dataset}  "
          f"device={device_str}  seed={seed}  n_samples={n_samples}")
    print(f"  checkpoint : {checkpoint}")
    print(f"  data       : {data_path}")

    # ---- Build test tensors ----
    _SDE_MODELS = {"sw_sle_em", "sw_gle_oe_em"}
    model_name = config["model"]["name"].lower()

    if model_name in _SDE_MODELS:
        from data.sde_loader import load_sde_dataset, split_sde_dataset
        from data.preprocessing import normalize
        from data.splitter import build_test_windows

        seed_data = int(config["data"]["seed"])
        sde_cfg   = config.get("sde", {})
        norm_mean = float(sde_cfg.get("norm_mean", 0.0))
        norm_std  = float(sde_cfg.get("norm_std",  1.0))
        raw = load_sde_dataset(model_name, data_path)
        splits = split_sde_dataset(raw, random_seed=seed_data)
        x_test = normalize(splits["x_test"].astype(np.float32), norm_mean, norm_std)
        ctx_np, tgt_np = build_test_windows(x_test, n_past, n_future)
        context = torch.tensor(ctx_np, dtype=torch.float32)
        ground_truth = tgt_np.astype(np.float64)
    else:
        from data.adapters.nftsf_adapter import get_nftsf_test_tensors
        context, gt_t = get_nftsf_test_tensors(data_path, n_past, n_future)
        ground_truth = gt_t.numpy().astype(np.float64)

    N_test = context.shape[0]
    print(f"  test cases : {N_test}  horizon : {n_future}")

    # ---- Load model ----
    if model_name in _SDE_MODELS:
        from models.nftsf import NFTSFModel
        model = NFTSFModel(config)
    else:
        from models.registry import get_model
        model = get_model(model_name)(config)

    # Find checkpoint file.
    for name in ("model_best.pth", "model_final.pth",
                 "model_best.ckpt", "model_final.ckpt"):
        p = checkpoint / name
        if p.exists():
            ckpt_file = p
            break
    else:
        matches = sorted(checkpoint.rglob("*.pth")) + sorted(checkpoint.rglob("*.ckpt"))
        if not matches:
            sys.exit(f"[test_model_paper_clean] No checkpoint file in {checkpoint}")
        ckpt_file = matches[-1]

    model.load(ckpt_file)
    inner = getattr(model, "model", None)
    if inner is not None:
        inner.eval()
    print(f"  ckpt file  : {ckpt_file}")

    # ---- Timed inference ----
    samples, timing_rec = _run_inference_timed(
        model, context,
        n_samples=n_samples,
        chunk_size=chunk_size,
        method=method,
        dataset=dataset,
        horizon=n_future,
        device=device_str,
        notes=notes,
    )
    # samples: (N_test, n_future, n_samples)
    print(f"  samples shape: {samples.shape}")

    # ---- Metrics ----
    # Flatten (N, T, S) to (N*T, S) for decomposition — compute per-step means after.
    # We compute decomposition over all (case, step) pairs jointly and also
    # compute mean per-step CRPS for the paper summary.
    print("  computing metrics …")

    # Per-step CRPS (mean over cases) for the summary line.
    from metrics.crps_decomposition import crps_ensemble as _crps_ens
    crps_per_step = np.array([
        _crps_ens(samples[:, t, :], ground_truth[:, t])
        for t in range(n_future)
    ])  # (T,)

    # Global decomposition (aggregate over all cases × steps).
    samples_flat   = samples.reshape(-1, n_samples)        # (N*T, S)
    obs_flat       = ground_truth.reshape(-1)               # (N*T,)
    metrics_dict   = compute_all(
        samples_flat, obs_flat,
        sample_axis=-1,
        compute_calibration=True,
    )

    elapsed_sec    = timing_rec["elapsed_sec"]
    sec_per_sample = timing_rec["sec_per_sample"]

    # ---- Print summary ----
    print(f"\n  === {method} / {dataset} ===")
    print(f"  CRPS        : {metrics_dict['crps']:.4f}")
    print(f"  Reliability : {metrics_dict['crps_reliability']:.4f}  (0=perfect calib)")
    print(f"  Resolution  : {metrics_dict['crps_resolution']:.4f}  (higher=sharper)")
    print(f"  Uncertainty : {metrics_dict['crps_uncertainty']:.4f}  (obs property)")
    print(f"  PIT mean/std: {metrics_dict['calibration_pit_mean']:.4f} / "
          f"{metrics_dict['calibration_pit_std']:.4f}")
    print(f"  Rank χ²     : {metrics_dict['rank_hist_chi2']:.2f}")
    print(f"  Elapsed     : {elapsed_sec:.2f}s  ({sec_per_sample:.6f} s/sample)")

    # ---- Build output record ----
    result = {
        # Identity
        "method":         method,
        "dataset":        dataset,
        "system":         system,
        "horizon":        n_future,
        "n_samples":      n_samples,
        "n_items":        N_test,
        "run_id":         run_id,
        "timestamp_utc":  datetime.now(timezone.utc).isoformat(),
        "checkpoint":     str(ckpt_file),
        "seed":           seed,
        # Metrics
        "crps":                     metrics_dict["crps"],
        "crps_reliability":         metrics_dict["crps_reliability"],
        "crps_resolution":          metrics_dict["crps_resolution"],
        "crps_uncertainty":         metrics_dict["crps_uncertainty"],
        "calibration_pit_mean":     metrics_dict["calibration_pit_mean"],
        "calibration_pit_std":      metrics_dict["calibration_pit_std"],
        "rank_hist_chi2":           metrics_dict["rank_hist_chi2"],
        "crps_per_step":            crps_per_step.tolist(),
        # Timing
        "elapsed_sec":    elapsed_sec,
        "sec_per_sample": sec_per_sample,
        "device":         timing_rec["device"],
        "device_name":    timing_rec["device_name"],
        "torch_version":  timing_rec["torch_version"],
    }

    # ---- Write output JSON ----
    output_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"result_{method}_{dataset}_{run_id}.json"
    out_path = output_dir / out_name
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Written: {out_path}")

    # ---- Append to timing log ----
    if timing_log is not None:
        append_timing_record(timing_rec, timing_log)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Paper-clean single-model evaluation."
    )
    p.add_argument("--checkpoint", required=True,
                   help="Run directory containing config.json + model checkpoint.")
    p.add_argument("--data", required=True,
                   help="Canonical .npz or SDE data directory.")
    p.add_argument("--method",  required=True,
                   help="Method label for output (e.g. nftsf, tsdiff_q).")
    p.add_argument("--dataset", required=True,
                   help="Dataset label for output (e.g. single_well).")
    p.add_argument("--system", default=None,
                   help="Optional physical-system label (e.g. sw_sle_em).")
    p.add_argument("--n_samples", type=int, default=500)
    p.add_argument("--chunk_size", type=int, default=32,
                   help="Inference batch size (default 32).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="outputs/paper_clean",
                   help="Directory for result JSON.")
    p.add_argument("--run_id", default=None)
    p.add_argument("--notes", default=None)
    p.add_argument("--timing_log", default=None,
                   help="JSONL file to append timing record.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    evaluate_paper_clean(
        checkpoint=Path(args.checkpoint),
        data_path=Path(args.data),
        method=args.method,
        dataset=args.dataset,
        system=args.system,
        n_samples=args.n_samples,
        chunk_size=args.chunk_size,
        seed=args.seed,
        output_dir=Path(args.output_dir),
        run_id=run_id,
        notes=args.notes,
        timing_log=Path(args.timing_log) if args.timing_log else None,
    )


if __name__ == "__main__":
    main()
