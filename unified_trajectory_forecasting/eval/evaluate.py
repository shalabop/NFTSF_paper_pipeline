"""
eval/evaluate.py
================
Canonical evaluation script for the NFTSF comparison pipeline.

For any trained model checkpoint, this script:
  1. Loads the model and its training config.
  2. Builds the test set from the canonical .npz (20 % held-out trajectories).
  3. Runs probabilistic inference to obtain ``(N_test, n_future, n_samples)``
     forecast samples.
  4. Computes per-step metrics: MAE, RMSE, CRPS, CI90 and CI50 coverage.
  5. Saves everything in a canonical ``results.npz`` for use by
     ``viz/comparison.py``.

Usage
-----
python eval/evaluate.py \\
    --checkpoint outputs/nftsf/alanine_phi/run_001/ \\
    --data       outputs/canonical/alanine_phi.npz \\
    --n_samples  500

The ``--checkpoint`` argument should point to the run directory that
contains ``config.json`` and at least one of:
  model_best.pth | model_final.pth | model_best.ckpt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure project root is importable.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Metrics (mirrors tsf_models-adam/metrics.py but extended with RMSE)
# ---------------------------------------------------------------------------

def _crps_per_step(
    ground_truth: np.ndarray,
    samples: np.ndarray,
) -> np.ndarray:
    """
    CRPS per forecast step, averaged over test trajectories.

    Parameters
    ----------
    ground_truth : (N, T) float32
    samples      : (N, T, S) float32

    Returns
    -------
    crps_t : (T,) float32
    """
    try:
        import properscoring as ps
    except ImportError as exc:
        raise ImportError(
            "properscoring is required for CRPS.  "
            "Install with: pip install properscoring"
        ) from exc

    N, T, S = samples.shape
    crps_all = np.array([
        [ps.crps_ensemble(ground_truth[i, t], samples[i, t, :]) for t in range(T)]
        for i in range(N)
    ])   # (N, T)
    return crps_all.mean(axis=0)   # (T,)


def _mae_per_step(
    ground_truth: np.ndarray,
    samples: np.ndarray,
) -> np.ndarray:
    """MAE of the median forecast per step, averaged over trajectories."""
    median = np.median(samples, axis=-1)   # (N, T)
    return np.abs(ground_truth - median).mean(axis=0)   # (T,)


def _rmse_per_step(
    ground_truth: np.ndarray,
    samples: np.ndarray,
) -> np.ndarray:
    """RMSE of the median forecast per step."""
    median = np.median(samples, axis=-1)
    return np.sqrt(((ground_truth - median) ** 2).mean(axis=0))


def _ci_coverage_per_step(
    ground_truth: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Fraction of ground-truth values inside CI per step."""
    inside = (ground_truth >= lower) & (ground_truth <= upper)
    return inside.mean(axis=0)   # (T,)


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def _find_checkpoint(run_dir: Path) -> Path:
    """Return the best available model checkpoint in *run_dir*."""
    for name in ("model_best.pth", "model_final.pth",
                 "model_best.ckpt", "model_final.ckpt"):
        p = run_dir / name
        if p.exists():
            return p
    # Fallback: any .pth or .ckpt.
    for suffix in ("*.pth", "*.ckpt"):
        matches = list(run_dir.rglob(suffix))
        if matches:
            return sorted(matches)[-1]
    raise FileNotFoundError(
        f"No model checkpoint found in {run_dir}.  "
        "Expected model_best.pth / model_final.pth / *.ckpt"
    )


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

def evaluate(
    run_dir: Path,
    data_path: Path,
    n_samples: int = 500,
) -> Path:
    """
    Run evaluation and write ``results.npz`` to *run_dir*.

    Parameters
    ----------
    run_dir    : Training output directory containing ``config.json`` and checkpoint.
    data_path  : Canonical ``.npz`` dataset.
    n_samples  : Number of forecast samples per test trajectory.

    Returns
    -------
    Path to the written ``results.npz``.
    """
    run_dir   = Path(run_dir)
    data_path = Path(data_path)

    # ---- Load config ----
    config_path = run_dir / "config.json"
    if not config_path.exists():
        sys.exit(f"[evaluate] config.json not found in {run_dir}")
    with open(config_path) as f:
        config = json.load(f)

    model_name = config["model"]["name"].lower()
    n_past     = int(config["data"]["n_past"])
    n_future   = int(config["data"]["n_future"])
    landscape  = config.get("landscape", "unknown")
    run_id     = config.get("run_id", run_dir.name)

    device_str = config["training"].get("device", "auto")
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[evaluate] Model: {model_name}  Run: {run_id}  Device: {device_str}")

    # ---- Build test tensors ----
    from data.adapters.nftsf_adapter import get_nftsf_test_tensors
    from data.canonical import load_canonical

    canon = load_canonical(data_path)
    landscape = str(canon.get("landscape", landscape))

    context, ground_truth_t = get_nftsf_test_tensors(data_path, n_past, n_future)
    N_test = context.shape[0]
    print(f"[evaluate] Test set: {N_test} trajectories, "
          f"context={n_past} steps, horizon={n_future} steps")

    # ---- Instantiate and load model ----
    from models.registry import get_model
    ModelClass = get_model(model_name)
    model = ModelClass(config)

    ckpt_path = _find_checkpoint(run_dir)
    print(f"[evaluate] Loading checkpoint: {ckpt_path}")
    model.load(ckpt_path)

    # ---- Run inference ----
    print(f"[evaluate] Generating {n_samples} forecast samples …")
    inner = getattr(model, "model", None)
    if inner is not None:
        inner.eval()

    # Process in chunks to avoid OOM.
    chunk = 32
    all_samples: list[np.ndarray] = []
    for start in range(0, N_test, chunk):
        ctx_chunk = context[start : start + chunk]
        s_chunk = model.predict(ctx_chunk, n_samples=n_samples)   # (B, n_future, S)
        all_samples.append(s_chunk)
    samples = np.concatenate(all_samples, axis=0)   # (N_test, n_future, n_samples)

    ground_truth = ground_truth_t.numpy()   # (N_test, n_future)

    # ---- Compute metrics ----
    print("[evaluate] Computing metrics …")
    crps_t = _crps_per_step(ground_truth, samples)
    mae_t  = _mae_per_step(ground_truth, samples)
    rmse_t = _rmse_per_step(ground_truth, samples)

    ci90_lower = np.percentile(samples, 5,  axis=-1)   # (N_test, n_future)
    ci90_upper = np.percentile(samples, 95, axis=-1)
    ci50_lower = np.percentile(samples, 25, axis=-1)
    ci50_upper = np.percentile(samples, 75, axis=-1)

    ci90_t = _ci_coverage_per_step(ground_truth, ci90_lower, ci90_upper)
    ci50_t = _ci_coverage_per_step(ground_truth, ci50_lower, ci50_upper)

    # ---- Write results.npz ----
    out_path = run_dir / "results.npz"
    np.savez(
        out_path,
        samples=samples.astype(np.float32),
        ground_truth=ground_truth.astype(np.float32),
        ci90_lower=ci90_lower.astype(np.float32),
        ci90_upper=ci90_upper.astype(np.float32),
        ci50_lower=ci50_lower.astype(np.float32),
        ci50_upper=ci50_upper.astype(np.float32),
        crps_t=crps_t.astype(np.float32),
        mae_t=mae_t.astype(np.float32),
        rmse_t=rmse_t.astype(np.float32),
        ci90_t=ci90_t.astype(np.float32),
        ci50_t=ci50_t.astype(np.float32),
        model_name=np.array(model_name),
        landscape=np.array(landscape),
        run_id=np.array(run_id),
    )
    print(f"[evaluate] Results saved to {out_path}")

    # ---- Print summary ----
    print(f"\n  Mean CRPS : {crps_t.mean():.4f}")
    print(f"  Mean MAE  : {mae_t.mean():.4f}")
    print(f"  Mean RMSE : {rmse_t.mean():.4f}")
    print(f"  CI90 cov  : {ci90_t.mean():.3f}  (ideal 0.90)")
    print(f"  CI50 cov  : {ci50_t.mean():.3f}  (ideal 0.50)")

    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained model checkpoint.")
    p.add_argument("--checkpoint", required=True,
                   help="Path to run directory (contains config.json + *.pth)")
    p.add_argument("--data", required=True,
                   help="Path to canonical .npz dataset file")
    p.add_argument("--n_samples", type=int, default=500,
                   help="Number of forecast samples per test trajectory")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate(
        run_dir=Path(args.checkpoint),
        data_path=Path(args.data),
        n_samples=args.n_samples,
    )
