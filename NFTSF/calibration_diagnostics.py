"""
calibration_diagnostics.py
==========================
Direct calibration metrics for ensemble / sample-based probabilistic forecasts.

A forecast is well-calibrated if, for every nominal level p, the fraction of
observations falling inside the p-prediction interval equals p.  These metrics
assess that property directly, without conflating calibration with sharpness.

Inputs
------
ground_truth : (N, T) float32  — N trajectories × T time steps
samples      : (N, T, M) float32 — M ensemble members per (trajectory, step)

The metrics are averaged over both N and T unless noted.
"""

from __future__ import annotations

import numpy as np

# Nominal central-interval levels used by all functions by default.
STANDARD_LEVELS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)

# Quantile levels for QCE: p = 0.01, 0.02, ..., 0.99
_QCE_LEVELS = np.linspace(0.01, 0.99, 99)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _empirical_quantile_coverage(
    ground_truth: np.ndarray,
    samples: np.ndarray,
    quantile_level: float,
) -> float:
    """
    Fraction of (trajectory, step) pairs where y <= Q_p(samples).

    For a perfectly calibrated forecast, this should equal *quantile_level*.
    """
    q = np.percentile(samples, 100.0 * quantile_level, axis=-1)  # (N, T)
    return float(np.mean(ground_truth <= q))


def _central_interval_bounds(
    samples: np.ndarray,
    nominal_level: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Lower and upper percentile bounds for a symmetric central interval.

    For nominal_level=0.90, returns the 5th and 95th percentiles.
    """
    tail = (1.0 - nominal_level) / 2.0
    lower = np.percentile(samples, 100.0 * tail,          axis=-1)  # (N, T)
    upper = np.percentile(samples, 100.0 * (1.0 - tail),  axis=-1)  # (N, T)
    return lower, upper


# ---------------------------------------------------------------------------
# Public metrics
# ---------------------------------------------------------------------------

def marginal_coverage(
    ground_truth: np.ndarray,
    samples: np.ndarray,
    nominal_levels: tuple[float, ...] = STANDARD_LEVELS,
) -> dict[str, float]:
    """
    Empirical coverage of symmetric central prediction intervals.

    For each nominal level p, the central interval spans [Q_{(1-p)/2}, Q_{1-(1-p)/2}].
    Coverage is the fraction of observations inside that interval, averaged over
    all (trajectory, step) pairs.

    Returns
    -------
    dict mapping "ci_{pct}" -> empirical_coverage, e.g. "ci_90" -> 0.87.
    Also includes "nominal_{pct}" -> p for easy plotting.
    """
    result: dict[str, float] = {}
    for p in nominal_levels:
        lower, upper = _central_interval_bounds(samples, p)
        inside = (ground_truth >= lower) & (ground_truth <= upper)
        empirical = float(inside.mean())
        label = f"{round(p * 100):d}"
        result[f"ci_{label}_nominal"] = p
        result[f"ci_{label}_empirical"] = empirical
        result[f"ci_{label}_error"] = empirical - p
    return result


def quantile_calibration_error(
    ground_truth: np.ndarray,
    samples: np.ndarray,
    quantile_levels: np.ndarray = _QCE_LEVELS,
) -> float:
    """
    Mean squared quantile calibration error (QCE).

    QCE = mean_{p in levels} (empirical_coverage(p) - p)^2

    A QCE of 0 means the forecast is perfectly calibrated at every quantile level.
    QCE is always >= 0; lower is better.

    The default 99-level grid (0.01 to 0.99) approximates the continuous integral.
    """
    errors_sq = np.array([
        (_empirical_quantile_coverage(ground_truth, samples, p) - p) ** 2
        for p in quantile_levels
    ])
    return float(errors_sq.mean())


def interval_coverage_errors(
    ground_truth: np.ndarray,
    samples: np.ndarray,
    nominal_levels: tuple[float, ...] = STANDARD_LEVELS,
) -> dict[str, float]:
    """
    Signed coverage error for each standard central interval.

    ICE(p) = empirical_coverage(p) - p.
    Positive: the interval is too wide (over-dispersed).
    Negative: the interval is too narrow (under-dispersed / over-confident).
    """
    result: dict[str, float] = {}
    for p in nominal_levels:
        lower, upper = _central_interval_bounds(samples, p)
        inside = (ground_truth >= lower) & (ground_truth <= upper)
        ice = float(inside.mean()) - p
        label = f"{round(p * 100):d}"
        result[f"ice_{label}"] = ice
    return result


def calibration_diagnostics(
    ground_truth: np.ndarray,
    samples: np.ndarray,
    nominal_levels: tuple[float, ...] = STANDARD_LEVELS,
) -> dict:
    """
    Full suite of direct calibration metrics.

    Parameters
    ----------
    ground_truth : (N, T) float32
    samples      : (N, T, M) float32
    nominal_levels : Central interval levels to evaluate.

    Returns
    -------
    dict with top-level keys:
      "marginal_coverage"  : dict — empirical vs. nominal per level (see marginal_coverage)
      "qce"                : float — mean squared quantile calibration error
      "interval_coverage_errors" : dict — signed ICE per level (see interval_coverage_errors)
    """
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    samples      = np.asarray(samples,      dtype=np.float64)

    if ground_truth.ndim != 2:
        raise ValueError(f"ground_truth must be (N, T); got shape {ground_truth.shape}")
    if samples.ndim != 3:
        raise ValueError(f"samples must be (N, T, M); got shape {samples.shape}")
    if samples.shape[:2] != ground_truth.shape:
        raise ValueError(
            f"samples shape {samples.shape[:2]} does not match "
            f"ground_truth shape {ground_truth.shape}"
        )

    return {
        "marginal_coverage": marginal_coverage(ground_truth, samples, nominal_levels),
        "qce": quantile_calibration_error(ground_truth, samples),
        "interval_coverage_errors": interval_coverage_errors(
            ground_truth, samples, nominal_levels
        ),
    }


def per_step_coverage(
    ground_truth: np.ndarray,
    samples: np.ndarray,
    nominal_levels: tuple[float, ...] = (0.50, 0.90),
) -> dict[str, np.ndarray]:
    """
    Per-time-step empirical coverage for selected central intervals.

    Returns
    -------
    dict mapping "ci_{pct}_per_step" -> (T,) array.
    """
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    samples      = np.asarray(samples,      dtype=np.float64)
    result: dict[str, np.ndarray] = {}
    for p in nominal_levels:
        lower, upper = _central_interval_bounds(samples, p)
        inside = (ground_truth >= lower) & (ground_truth <= upper)
        label = f"{round(p * 100):d}"
        result[f"ci_{label}_per_step"] = inside.mean(axis=0).astype(np.float32)
    return result
