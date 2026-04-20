"""
metrics/metrics_probabilistic.py
==================================
Thin, clean façade for all probabilistic metrics used by the paper path.

This module is the single import point for paper scripts.  It delegates to
crps_decomposition.py and provides PIT/rank-histogram scaffolding.

Motivation: docs/paper_clean_path.md.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from metrics.crps_decomposition import crps_ensemble, crps_decomposition


# ---------------------------------------------------------------------------
# Aggregate metric bundle
# ---------------------------------------------------------------------------

def compute_all(
    forecast_samples: np.ndarray,
    observations: np.ndarray,
    *,
    sample_axis: int = -1,
    compute_calibration: bool = True,
    n_pit_bins: int = 10,
) -> dict[str, Any]:
    """
    Compute the full probabilistic metric bundle for one method/dataset pair.

    Parameters
    ----------
    forecast_samples : (N_cases, ..., M) or any shape with sample_axis of length M.
    observations     : shape equal to forecast_samples with sample_axis removed.
    sample_axis      : ensemble member axis.
    compute_calibration : if True, compute PIT histogram and rank histogram stats.
    n_pit_bins       : number of bins for PIT histogram (default 10 = deciles).

    Returns
    -------
    dict with stable paper-column keys (see scripts/compare_models_paper_clean.py).
    """
    decomp = crps_decomposition(
        forecast_samples, observations, sample_axis=sample_axis
    )

    result: dict[str, Any] = {
        "crps":             decomp["crps"],
        "crps_reliability": decomp["reliability"],
        "crps_resolution":  decomp["resolution"],
        "crps_uncertainty": decomp["uncertainty"],
        "n_samples":        decomp["n_samples"],
        "n_cases":          decomp["n_cases"],
        "n_nan_cases":      decomp["n_nan_cases"],
        # Calibration scaffolds — computed below if requested.
        "calibration_pit_mean": None,
        "calibration_pit_std":  None,
        "rank_hist_chi2":       None,
    }

    if compute_calibration and decomp["n_cases"] > 0:
        pit_mean, pit_std = _pit_summary(
            forecast_samples, observations,
            sample_axis=sample_axis, n_bins=n_pit_bins,
        )
        result["calibration_pit_mean"] = pit_mean
        result["calibration_pit_std"]  = pit_std
        result["rank_hist_chi2"]       = _rank_hist_chi2(
            forecast_samples, observations, sample_axis=sample_axis
        )

    return result


# ---------------------------------------------------------------------------
# PIT (Probability Integral Transform) histogram summary
# ---------------------------------------------------------------------------

def _pit_summary(
    forecast_samples: np.ndarray,
    observations: np.ndarray,
    *,
    sample_axis: int = -1,
    n_bins: int = 10,
) -> tuple[float, float]:
    """
    Compute mean and std of the PIT histogram bin counts (normalised to sum=1).

    For a perfectly calibrated forecast the PIT is uniform on [0,1]; the
    normalised histogram has mean = 1/n_bins and std ≈ 0.

    Returns
    -------
    (pit_mean, pit_std) — floats.  Returns (nan, nan) on failure.
    """
    try:
        fc = np.asarray(forecast_samples, dtype=np.float64)
        obs = np.asarray(observations, dtype=np.float64)
        if sample_axis != -1 and sample_axis != fc.ndim - 1:
            fc = np.moveaxis(fc, sample_axis, -1)
        fc_2d = fc.reshape(-1, fc.shape[-1])
        obs_1d = obs.reshape(-1)
        valid = ~(np.isnan(obs_1d) | np.any(np.isnan(fc_2d), axis=1))
        if not np.any(valid):
            return float("nan"), float("nan")
        fc_v, obs_v = fc_2d[valid], obs_1d[valid]
        # PIT value: fraction of ensemble members ≤ observation.
        pit = np.mean(fc_v <= obs_v[:, None], axis=1)  # (N,) in [0,1]
        counts, _ = np.histogram(pit, bins=n_bins, range=(0.0, 1.0))
        norm = counts / counts.sum()
        return float(np.mean(norm)), float(np.std(norm))
    except Exception:
        return float("nan"), float("nan")


# ---------------------------------------------------------------------------
# Rank histogram chi-squared statistic
# ---------------------------------------------------------------------------

def _rank_hist_chi2(
    forecast_samples: np.ndarray,
    observations: np.ndarray,
    *,
    sample_axis: int = -1,
) -> float:
    """
    Chi-squared statistic for departure from a flat rank histogram.

    For a perfectly calibrated M-member ensemble the rank of the observation
    among M+1 bins is uniformly distributed.  The chi-squared stat measures
    the departure from flatness.  A large value indicates miscalibration.

    Returns the chi-squared statistic (not normalised by dof), or nan on failure.
    """
    try:
        fc = np.asarray(forecast_samples, dtype=np.float64)
        obs = np.asarray(observations, dtype=np.float64)
        if sample_axis != -1 and sample_axis != fc.ndim - 1:
            fc = np.moveaxis(fc, sample_axis, -1)
        fc_2d = fc.reshape(-1, fc.shape[-1])
        obs_1d = obs.reshape(-1)
        valid = ~(np.isnan(obs_1d) | np.any(np.isnan(fc_2d), axis=1))
        if not np.any(valid):
            return float("nan")
        fc_v, obs_v = fc_2d[valid], obs_1d[valid]
        M = fc_v.shape[1]
        N = fc_v.shape[0]
        # Rank of observation: 0 if obs < all members, M if obs > all members.
        ranks = np.sum(fc_v < obs_v[:, None], axis=1)   # (N,) in 0..M
        counts = np.bincount(ranks, minlength=M + 1).astype(float)
        expected = N / (M + 1)
        chi2 = float(np.sum((counts - expected) ** 2 / expected))
        return chi2
    except Exception:
        return float("nan")
