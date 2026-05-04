"""
metrics_crps_decomp.py
======================
Raw CRPS and Hersbach (2000) CRPS decomposition for ensemble forecasts.

Background
----------
Raw CRPS conflates calibration quality with sharpness and discrimination,
making it difficult to diagnose *why* a model scores well or poorly.  This
module reports the Hersbach decomposition alongside raw CRPS, so each aspect
can be assessed separately.  Direct calibration diagnostics (quantile
calibration error, interval coverage) are computed via calibration_diagnostics.py
and returned in the same output dict.

CRPS decomposition identity
---------------------------
    CRPS ≈ Reliability − Resolution + Uncertainty        (Hersbach 2000, eq. 21)

    Reliability (RELI):  calibration penalty — lower is better.
                         RELI = 0 for a perfectly calibrated ensemble.
    Resolution  (RESOL): discrimination skill vs. the climatological baseline —
                         higher is better.  A constant-climatology forecast has
                         RESOL = 0.
    Uncertainty (UNCE):  inherent variability of the observations; independent
                         of the forecast.  Cannot be reduced by a better model.

The sanity check CRPS ≈ RELI − RESOL + UNCE is logged (not raised) with a
configurable tolerance.

Notation (Hersbach 2000)
------------------------
For a single forecast-observation pair with M sorted ensemble members
x_(1) ≤ x_(2) ≤ … ≤ x_(M) and observation y:

  Bin i (i = 0, …, M) spans [x_(i), x_(i+1)) with x_(0) = −∞, x_(M+1) = +∞.
  Nominal CDF value at bin i: g_i = i / M.

  α_{n,i} = portion of bin i strictly below y (width between x_(i) and min(y, x_(i+1)))
  β_{n,i} = portion of bin i strictly above y (width between max(y, x_(i)) and x_(i+1))

  CRPS_n = Σ_i  α_{n,i} · g_i²  +  β_{n,i} · (1 − g_i)²

Averaged over N samples and T time steps:
  ᾱ_i, β̄_i  → from these, w_i = ᾱ_i + β̄_i,  o_i = β̄_i / w_i

  RELI  = Σ_i  w_i · (g_i − o_i)²
  RESOL = Σ_i  w_i · [ō·(1 − ō) − o_i·(1 − o_i)]
  UNCE  = Σ_i  w_i · ō·(1 − ō)        where  ō = Σ_i w_i·o_i / Σ_i w_i

Assumptions
-----------
- Ensemble members are exchangeable (i.i.d. draws from the predictive distribution).
- CRPS uses the standard (biased) estimator: E|X − y| − ½ E|X − X'|, consistent
  with the Hersbach rank-based formula and with properscoring.crps_ensemble.
  This differs from the "fair CRPS" (Ferro et al. 2008) by a factor of M/(M−1)
  in the spread term; for M ≥ 50 the difference is < 2%.
- No finite-ensemble-size correction is applied to the decomposition terms.
- Ties in the ensemble are broken by the sort order (no special handling).
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

from calibration_diagnostics import calibration_diagnostics

logger = logging.getLogger(__name__)

# Relative tolerance for the decomposition sanity check.
_SANITY_RTOL = 1e-4


# ---------------------------------------------------------------------------
# Core Hersbach computation — single time step
# ---------------------------------------------------------------------------

def _hersbach_one_step(
    y: np.ndarray,          # (N,)
    x_sorted: np.ndarray,   # (N, M) sorted ascending along axis=1
) -> tuple[float, float, float, float]:
    """
    Return (crps, reliability, resolution, uncertainty) for one time step.

    Implements Hersbach (2000) equations 19-23 in vectorised form.
    All four values are averages over N forecast-observation pairs.
    """
    N, M = x_sorted.shape
    y_col = y[:, np.newaxis]  # (N, 1) for broadcasting

    # ---- Interior bins i = 1, …, M−1 ----
    # Bin i spans [x_(i), x_(i+1)) using 0-based indices in x_sorted.
    x_lo = x_sorted[:, :-1]   # (N, M-1)   left edges of interior bins
    x_hi = x_sorted[:, 1:]    # (N, M-1)   right edges
    alpha_int = np.maximum(np.minimum(y_col, x_hi) - x_lo, 0.0)  # (N, M-1)
    beta_int  = np.maximum(x_hi - np.maximum(y_col, x_lo), 0.0)  # (N, M-1)

    # ---- Boundary bin 0: (−∞, x_(1)) ----
    # α₀ = 0 always; (0/M)² = 0 so α₀ contributes nothing.
    beta_0 = np.maximum(x_sorted[:, 0] - y, 0.0)   # (N,)

    # ---- Boundary bin M: (x_(M), +∞) ----
    # β_M = 0 always; (1 − M/M)² = 0 so β_M contributes nothing.
    alpha_M = np.maximum(y - x_sorted[:, -1], 0.0)  # (N,)

    # ---- Average over N ----
    # Stack to (N, M+1): [bin0, bin1, …, bin(M-1), binM]
    ab = np.column_stack([beta_0,  alpha_int.mean(0)])  # just a scratch var; handled below
    # Compute directly as averages:
    ab_alpha = np.empty(M + 1)
    ab_beta  = np.empty(M + 1)
    ab_alpha[0]    = 0.0
    ab_alpha[1:M]  = alpha_int.mean(axis=0)
    ab_alpha[M]    = alpha_M.mean()
    ab_beta[0]     = beta_0.mean()
    ab_beta[1:M]   = beta_int.mean(axis=0)
    ab_beta[M]     = 0.0

    # ---- Nominal CDF values g_i = i/M ----
    g = np.arange(M + 1, dtype=np.float64) / M   # (M+1,)

    # ---- Raw CRPS ----
    crps = float(np.dot(ab_alpha, g ** 2) + np.dot(ab_beta, (1.0 - g) ** 2))

    # ---- Effective weights and empirical hit rate ----
    w = ab_alpha + ab_beta                                 # (M+1,)
    w_total = w.sum()
    # o_i = β̄_i / w_i  (the "effective" observed frequency below quantile g_i)
    # Where w_i = 0, fall back to g_i so (g_i − o_i)² = 0 (no contribution).
    with np.errstate(invalid="ignore", divide="ignore"):
        o = np.where(w > 0.0, ab_beta / w, g)

    # Weighted mean hit rate ō
    o_bar = float((w * o).sum() / w_total) if w_total > 0.0 else 0.5

    # ---- Decomposition ----
    reli  = float(np.dot(w, (g - o) ** 2))
    resol = float(np.dot(w, o_bar * (1.0 - o_bar) - o * (1.0 - o)))
    unce  = float(w_total * o_bar * (1.0 - o_bar))

    return crps, reli, resol, unce


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def crps_decomposition(
    ground_truth: np.ndarray,
    samples: np.ndarray,
    sanity_rtol: float = _SANITY_RTOL,
    include_calibration_diagnostics: bool = True,
) -> dict:
    """
    Compute raw CRPS and Hersbach (2000) decomposition, per time step and averaged.

    Parameters
    ----------
    ground_truth : (N, T) float — observed trajectories.
    samples      : (N, T, M) float — ensemble forecast trajectories.
    sanity_rtol  : Relative tolerance for the identity check
                   |CRPS − (RELI − RESOL + UNCE)| / max(|CRPS|, ε) ≤ sanity_rtol.
    include_calibration_diagnostics : If True, append output of
                   calibration_diagnostics.calibration_diagnostics().

    Returns
    -------
    dict with three top-level sections:

    "raw_crps"
        float — mean CRPS averaged over trajectories and time steps.
    "raw_crps_per_step"
        (T,) float32 — per-step CRPS averaged over trajectories.

    "crps_decomposition"
        "reliability"          : float
        "resolution"           : float
        "uncertainty"          : float
        "reconstructed_total"  : float  — RELI − RESOL + UNCE (sanity check)
        "sanity_check_passed"  : bool
        "reliability_per_step" : (T,) float32
        "resolution_per_step"  : (T,) float32
        "uncertainty_per_step" : (T,) float32

    "calibration_diagnostics"
        As returned by calibration_diagnostics.calibration_diagnostics(), or
        None if include_calibration_diagnostics=False.
    """
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    samples      = np.asarray(samples,      dtype=np.float64)

    if ground_truth.ndim != 2:
        raise ValueError(f"ground_truth must be (N, T); got {ground_truth.shape}")
    if samples.ndim != 3:
        raise ValueError(f"samples must be (N, T, M); got {samples.shape}")
    if samples.shape[:2] != ground_truth.shape:
        raise ValueError(
            f"samples shape {samples.shape[:2]} does not match "
            f"ground_truth shape {ground_truth.shape}"
        )

    N, T, M = samples.shape

    crps_t  = np.empty(T)
    reli_t  = np.empty(T)
    resol_t = np.empty(T)
    unce_t  = np.empty(T)

    # Sort ensemble once along member axis.
    x_sorted = np.sort(samples, axis=2)   # (N, T, M)

    for t in range(T):
        crps_t[t], reli_t[t], resol_t[t], unce_t[t] = _hersbach_one_step(
            ground_truth[:, t], x_sorted[:, t, :]
        )

    mean_crps  = float(crps_t.mean())
    mean_reli  = float(reli_t.mean())
    mean_resol = float(resol_t.mean())
    mean_unce  = float(unce_t.mean())
    reconstructed = mean_reli - mean_resol + mean_unce

    # Sanity check: CRPS ≈ RELI − RESOL + UNCE
    denom = max(abs(mean_crps), 1e-12)
    rel_err = abs(reconstructed - mean_crps) / denom
    sanity_ok = bool(rel_err <= sanity_rtol)
    if not sanity_ok:
        logger.warning(
            "CRPS decomposition sanity check failed: "
            "CRPS=%.6f, RELI−RESOL+UNCE=%.6f, rel_err=%.2e (tol=%.2e)",
            mean_crps, reconstructed, rel_err, sanity_rtol,
        )

    cal_diag = None
    if include_calibration_diagnostics:
        cal_diag = calibration_diagnostics(
            ground_truth.astype(np.float32),
            samples.astype(np.float32),
        )

    return {
        "raw_crps": mean_crps,
        "raw_crps_per_step": crps_t.astype(np.float32),
        "crps_decomposition": {
            "reliability":          mean_reli,
            "resolution":           mean_resol,
            "uncertainty":          mean_unce,
            "reconstructed_total":  reconstructed,
            "sanity_check_passed":  sanity_ok,
            "reliability_per_step":  reli_t.astype(np.float32),
            "resolution_per_step":  resol_t.astype(np.float32),
            "uncertainty_per_step": unce_t.astype(np.float32),
        },
        "calibration_diagnostics": cal_diag,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_metrics(
    metrics: dict,
    output_dir: Path,
    prefix: str = "metrics",
) -> dict[str, Path]:
    """
    Save the output of :func:`crps_decomposition` to *output_dir*.

    Scalars and calibration metadata → ``<prefix>_scalars.json``
    Per-step arrays                  → ``<prefix>_per_step.npz``

    Returns a dict mapping "json" and "npz" to the written paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Collect scalar fields ----
    scalars: dict = {
        "raw_crps": metrics["raw_crps"],
    }
    decomp = metrics.get("crps_decomposition", {})
    for key in ("reliability", "resolution", "uncertainty",
                "reconstructed_total", "sanity_check_passed"):
        if key in decomp:
            scalars[f"crps_decomposition.{key}"] = decomp[key]

    cal = metrics.get("calibration_diagnostics")
    if cal is not None:
        scalars["calibration_diagnostics.qce"] = cal.get("qce")
        mc = cal.get("marginal_coverage", {})
        scalars["calibration_diagnostics.marginal_coverage"] = mc
        scalars["calibration_diagnostics.interval_coverage_errors"] = \
            cal.get("interval_coverage_errors", {})

    json_path = output_dir / f"{prefix}_scalars.json"
    with open(json_path, "w") as f:
        json.dump(scalars, f, indent=2, default=float)

    # ---- Collect per-step arrays ----
    arrays: dict[str, np.ndarray] = {
        "raw_crps_per_step": metrics["raw_crps_per_step"],
    }
    if decomp:
        arrays["reliability_per_step"]  = decomp["reliability_per_step"]
        arrays["resolution_per_step"]   = decomp["resolution_per_step"]
        arrays["uncertainty_per_step"]  = decomp["uncertainty_per_step"]

    # Per-step coverage from calibration_diagnostics (if present)
    if cal is not None:
        mc = cal.get("marginal_coverage", {})
        for k, v in mc.items():
            if "per_step" in k and isinstance(v, np.ndarray):
                arrays[k] = v

    npz_path = output_dir / f"{prefix}_per_step.npz"
    np.savez(npz_path, **arrays)

    return {"json": json_path, "npz": npz_path}


def load_metrics(output_dir: Path, prefix: str = "metrics") -> dict:
    """
    Reload scalar and per-step metrics saved by :func:`save_metrics`.
    Returns a flat dict merging both files.
    """
    output_dir = Path(output_dir)
    result: dict = {}

    json_path = output_dir / f"{prefix}_scalars.json"
    if json_path.exists():
        with open(json_path) as f:
            result.update(json.load(f))

    npz_path = output_dir / f"{prefix}_per_step.npz"
    if npz_path.exists():
        with np.load(npz_path) as d:
            for k in d.files:
                result[k] = d[k]

    return result
