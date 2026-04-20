"""
metrics/crps_decomposition.py
==============================
Sample/ensemble-based Hersbach-style CRPS decomposition.

Scientific interpretation
-------------------------
CRPS = Reliability − Resolution + Uncertainty  (Hersbach 2000)

  Reliability  measures miscalibration of the forecast CDF: how far the
               empirical rank distribution deviates from uniform.
               Lower reliability → better calibration.

  Resolution   measures the forecast's ability to discriminate between
               outcomes beyond climatology: sharper useful forecasts score
               higher resolution.

  Uncertainty  is a property of the verifying observations alone (climatology
               spread); it does not depend on the forecast.

The decomposition lets the paper separate calibration failure (high
Reliability) from sharpness failure (low Resolution) — important when
comparing NFTSF against diffusion-style methods with different
miscalibration signatures.

Reference: Hersbach H. (2000), WAF 15, 559–570.

Budget identity (exact within this estimator)
---------------------------------------------
Let phi_kn = lower fraction of inner bin k for case n (see below), and
psi_kn = 1 - phi_kn.  Define g_bar_k = mean bin width, psi_bar_k = mean(psi_kn),
o_c = sum_k(g_bar_k * psi_bar_k) / sum_k(g_bar_k).

Then:
  CRPS_decomp = sum_k g_bar_k*(p_k^2*phi_bar_k + (1-p_k)^2*psi_bar_k) + tails

  Reliability = CRPS_decomp - Resolution + Uncertainty   →  (decomp CRPS, not fair estimator)
  Resolution  = sum_k g_bar_k * (psi_bar_k - o_c)^2
  Uncertainty = o_c * (1 - o_c) * sum_k g_bar_k

  Reliability − Resolution + Uncertainty = CRPS_decomp  (algebraically exact; see docstring).

The reported "crps" key uses the INDEPENDENT fair (energy-score) estimator and
will differ slightly from crps_decomp for finite M.

NaN policy
----------
Cases where any sample or the observation is NaN are excluded before all
accumulation.  The count is reported in "n_nan_cases".
"""
from __future__ import annotations

from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def crps_ensemble(
    forecast_samples: np.ndarray,
    observations: np.ndarray,
    *,
    sample_axis: int = -1,
) -> float:
    """
    Total CRPS via the fair (unbiased) ensemble energy-score estimator.

        CRPS = E|X - y| - (M/(M-1)) * 0.5 * E|X - X'|

    Parameters
    ----------
    forecast_samples : array with ensemble axis of length M.
    observations     : array broadcastable to forecast_samples minus sample_axis.
    sample_axis      : ensemble-member axis (default -1).

    Returns
    -------
    Scalar mean CRPS over all non-NaN cases, or np.nan if all NaN.
    """
    fc  = np.asarray(forecast_samples, dtype=np.float64)
    obs = np.asarray(observations,     dtype=np.float64)

    if sample_axis != -1 and sample_axis != fc.ndim - 1:
        fc = np.moveaxis(fc, sample_axis, -1)

    M = fc.shape[-1]
    if M < 2:
        raise ValueError(f"Need at least 2 ensemble members; got M={M}.")

    obs_b    = np.expand_dims(obs, axis=-1)
    mae_term = np.nanmean(np.abs(fc - obs_b), axis=-1)

    # Fair spread via sorted-array identity (Ferro et al. 2008):
    #   fair_spread = sum_k (2k-M-1)*x_(k) / (M*(M-1))   k = 1..M (1-indexed)
    fc_sorted = np.sort(fc, axis=-1)
    k         = np.arange(1, M + 1, dtype=np.float64)
    weights   = 2.0 * k - M - 1.0          # (M,)
    spread    = (fc_sorted * weights).sum(axis=-1) / (M * (M - 1))

    crps_per_case = mae_term - spread

    obs_nan = np.isnan(obs)
    fc_nan  = np.any(np.isnan(fc), axis=-1)
    mask    = ~(obs_nan | fc_nan)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(crps_per_case[mask]))


def crps_decomposition(
    forecast_samples: np.ndarray,
    observations: np.ndarray,
    *,
    sample_axis: int = -1,
    return_per_rank: bool = False,
) -> dict[str, Any]:
    """
    Hersbach (2000) CRPS decomposition into Reliability, Resolution, Uncertainty.

    For each inner bin k (0-indexed, k = 0 … M-2, nominal CDF p_k = (k+1)/M):

      phi_kn = 1            if obs_n ≤ x_{k,n}   (obs below bin — forecast too high)
             = (obs-x_k)/g  if x_k < obs < x_{k+1}  (obs inside bin, lower fraction)
             = 0            if obs_n ≥ x_{k+1,n}  (obs above bin)
      psi_kn = 1 - phi_kn

      g_bar_k  = mean(x_{k+1,n} - x_{k,n})   over all N cases
      phi_bar_k = mean(phi_kn);  psi_bar_k = mean(psi_kn)

    CRPS_decomp = tails + sum_k g_bar_k * (p_k^2 * phi_bar_k + (1-p_k)^2 * psi_bar_k)
    o_c         = sum_k g_bar_k * psi_bar_k / sum_k g_bar_k
    Reliability = CRPS_decomp - Resolution + Uncertainty  (derived, non-negative)
    Resolution  = sum_k g_bar_k * (psi_bar_k - o_c)^2
    Uncertainty = o_c * (1 - o_c) * sum_k g_bar_k

    Budget: Reliability − Resolution + Uncertainty = CRPS_decomp  (algebraically exact).

    Parameters
    ----------
    forecast_samples : any shape with sample_axis of length M; other axes are cases.
    observations     : shape = forecast_samples with sample_axis removed.
    sample_axis      : ensemble member axis (default -1).
    return_per_rank  : include per-rank arrays alpha/beta/g in output.

    Returns
    -------
    dict with keys: crps, crps_decomp, reliability, resolution, uncertainty,
                    n_samples, n_cases, n_nan_cases.
    (plus alpha, beta, g if return_per_rank=True)

    Caveats
    -------
    - "crps" uses the independent fair estimator; "crps_decomp" uses the
      Hersbach integral estimator.  They agree in expectation as M, N → ∞.
    - For M < 10, binning artefacts in reliability/resolution grow.
    - Reliability is clamped to ≥ 0 to avoid tiny negative float artefacts.
    """
    fc  = np.asarray(forecast_samples, dtype=np.float64)
    obs = np.asarray(observations,     dtype=np.float64)

    if sample_axis != -1 and sample_axis != fc.ndim - 1:
        fc = np.moveaxis(fc, sample_axis, -1)

    M = fc.shape[-1]
    if M < 2:
        raise ValueError(f"Need at least 2 ensemble members; got M={M}.")

    fc_2d  = fc.reshape(-1, M)
    obs_1d = obs.reshape(-1)

    if fc_2d.shape[0] != obs_1d.shape[0]:
        raise ValueError(
            f"Shape mismatch: {fc_2d.shape[0]} forecast cases vs "
            f"{obs_1d.shape[0]} observations."
        )

    # ---- NaN masking ----
    obs_nan = np.isnan(obs_1d)
    fc_nan  = np.any(np.isnan(fc_2d), axis=1)
    valid   = ~(obs_nan | fc_nan)
    n_nan   = int(np.sum(~valid))
    N       = int(np.sum(valid))

    _nan: dict[str, Any] = {
        "crps": float("nan"), "crps_decomp": float("nan"),
        "reliability": float("nan"), "resolution": float("nan"),
        "uncertainty": float("nan"),
        "n_samples": M, "n_cases": 0, "n_nan_cases": n_nan,
    }
    if N == 0:
        if return_per_rank:
            _nan.update({"alpha": np.full(M + 1, np.nan),
                         "beta":  np.full(M + 1, np.nan),
                         "g":     np.full(M + 1, np.nan)})
        return _nan

    fc_v   = fc_2d[valid]   # (N, M)
    obs_v  = obs_1d[valid]  # (N,)

    fc_sorted = np.sort(fc_v, axis=1)  # (N, M)

    # ---- Tail contributions ----
    left_tail  = float(np.mean(np.maximum(fc_sorted[:, 0]  - obs_v, 0.0)))
    right_tail = float(np.mean(np.maximum(obs_v - fc_sorted[:, -1], 0.0)))

    # ---- Inner bins (k = 0 … M-2) ----
    x_lo = fc_sorted[:, :-1]   # (N, M-1) left boundary of each bin
    x_hi = fc_sorted[:, 1:]    # (N, M-1) right boundary
    y    = obs_v[:, None]      # (N, 1)

    g_kn     = x_hi - x_lo    # (N, M-1) per-case bin widths
    safe_g   = np.where(g_kn > 0, g_kn, 1.0)

    # phi_kn: upper fraction — 1 when obs is ABOVE bin (y >= x_hi), 0 when below.
    # psi_kn: lower fraction — 1 when obs is BELOW bin (y <= x_lo), 0 when above.
    # phi + psi = 1 always.  Both include contributions from all N cases.
    #
    # Integral derivation: for bin with CDF value p_k = (k+1)/M,
    #   crps_k = g_k * [p_k^2 * phi_k + (1-p_k)^2 * psi_k]
    # where phi=1(obs above) captures the "obs above, CDF too low" error and
    # psi=1(obs below) captures the "obs below, CDF too high" error.
    below = (y <= x_lo)
    above = (y >= x_hi)
    phi_kn = np.where(above, 1.0,
             np.where(below, 0.0,
             (y - x_lo) / safe_g))   # (N, M-1): 1 when obs above bin
    psi_kn = 1.0 - phi_kn            # (N, M-1): 1 when obs below bin

    g_bar    = g_kn.mean(axis=0)    # (M-1,)
    phi_bar  = phi_kn.mean(axis=0)  # (M-1,) ≈ P(obs > x_k) for calibrated ≈ 1-p_k
    psi_bar  = psi_kn.mean(axis=0)  # (M-1,) ≈ P(obs < x_k) for calibrated ≈ p_k

    # Nominal CDF in bin k: (k+1)/M  (there are k+1 members ≤ x in interval (x_k, x_{k+1}))
    p_inner = np.arange(1, M, dtype=np.float64) / M   # (M-1,)

    # ---- CRPS_decomp ----
    crps_inner  = g_bar * (p_inner**2 * phi_bar + (1.0 - p_inner)**2 * psi_bar)
    crps_decomp = left_tail + right_tail + float(crps_inner.sum())

    # ---- Resolution, Uncertainty ----
    # o_c = climatological observed frequency (fraction of obs "below" rank boundary),
    # weighted average of psi_bar_k by bin width.
    G_total = float(g_bar.sum())
    o_c     = float((g_bar * psi_bar).sum() / G_total) if G_total > 0 else 0.5

    resolution  = float((g_bar * (psi_bar - o_c)**2).sum())
    uncertainty = o_c * (1.0 - o_c) * G_total

    # ---- Reliability ----
    # RELI_k = g_k * (p_k - psi_bar_k)^2  (departure of observed from nominal freq.)
    # For calibrated ensemble: psi_bar_k ≈ p_k → RELI ≈ 0.
    # Tail contributions enter RELI because left/right outlier cases represent
    # calibration failures; this preserves the budget identity:
    #   sum_k (RELI_k - RESOL_k + UNCE_k) + tails = CRPS_decomp  (algebraically exact).
    reli_inner  = float((g_bar * (p_inner - psi_bar)**2).sum())
    reliability = max(0.0, reli_inner + left_tail + right_tail)

    # ---- Fair CRPS (independent estimator; preferred for paper tables) ----
    crps_total = crps_ensemble(fc_v, obs_v, sample_axis=-1)

    result: dict[str, Any] = {
        "crps":        crps_total,
        "crps_decomp": crps_decomp,
        "reliability": reliability,
        "resolution":  resolution,
        "uncertainty": uncertainty,
        "n_samples":   M,
        "n_cases":     N,
        "n_nan_cases": n_nan,
    }

    if return_per_rank:
        # Expose full (M+1)-length vectors in Hersbach indexing (0 = left tail, M = right tail).
        alpha_full       = np.zeros(M + 1)
        beta_full        = np.zeros(M + 1)
        g_full           = np.zeros(M + 1)
        alpha_full[1:M]  = phi_bar
        beta_full[1:M]   = psi_bar
        g_full[1:M]      = g_bar
        alpha_full[0]    = left_tail   / max(left_tail  + right_tail, 1e-30)
        beta_full[M]     = right_tail  / max(left_tail  + right_tail, 1e-30)
        result["alpha"]  = alpha_full
        result["beta"]   = beta_full
        result["g"]      = g_full

    return result
