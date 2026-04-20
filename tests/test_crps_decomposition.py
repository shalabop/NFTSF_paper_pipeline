"""
tests/test_crps_decomposition.py
==================================
Fast unit tests for metrics/crps_decomposition.py.

Tests
-----
(a) Identity / calibration check
    A perfectly calibrated ensemble (samples drawn from the same distribution
    as the observations) should yield near-zero Reliability.

(b) Shape and NaN-handling
    Verify output shapes, key presence, and NaN-exclusion counting.

(c) Budget check
    Verify that Reliability − Resolution + Uncertainty ≈ CRPS (via the
    decomposition's internal CRPS) within a documented tolerance.
    The budget identity holds exactly in the large-M, large-N limit;
    for finite ensembles a tolerance of ~5 % of mean CRPS is expected.

All tests run in < 2 seconds on a CPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from metrics.crps_decomposition import crps_ensemble, crps_decomposition


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_calibrated(N: int = 300, M: int = 50, seed: int = 0) -> tuple:
    """
    Synthetic perfectly-calibrated ensemble.

    Both observations and ensemble members are drawn i.i.d. from N(0, 1).
    In expectation, the empirical CDF of the ensemble matches the true CDF,
    so Reliability should be near zero.
    """
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal(N)
    fc  = rng.standard_normal((N, M))
    return fc, obs


def _make_overconfident(N: int = 300, M: int = 50, seed: int = 1) -> tuple:
    """
    Over-confident (too narrow) ensemble: samples from N(0, 0.2), obs from N(0, 1).
    Should yield high Reliability (miscalibrated).
    """
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal(N)
    fc  = rng.normal(0, 0.2, size=(N, M))
    return fc, obs


# ---------------------------------------------------------------------------
# (a) Identity / calibration check
# ---------------------------------------------------------------------------

class TestCalibrationIdentity:
    def test_calibrated_reliability_near_zero(self):
        """Perfectly calibrated ensemble → Reliability ≈ 0."""
        fc, obs = _make_calibrated(N=500, M=100, seed=42)
        result = crps_decomposition(fc, obs, sample_axis=-1)

        # Reliability should be small relative to CRPS.
        assert result["reliability"] >= 0.0, "Reliability must be non-negative."
        rel_fraction = result["reliability"] / max(result["crps"], 1e-9)
        assert rel_fraction < 0.20, (
            f"Reliability fraction {rel_fraction:.3f} too large for calibrated ensemble "
            f"(reliability={result['reliability']:.4f}, crps={result['crps']:.4f})"
        )

    def test_overconfident_reliability_positive(self):
        """Over-confident ensemble should have higher Reliability than calibrated."""
        fc_cal,  obs_cal  = _make_calibrated(N=500, M=50, seed=10)
        fc_over, obs_over = _make_overconfident(N=500, M=50, seed=10)

        res_cal  = crps_decomposition(fc_cal,  obs_cal,  sample_axis=-1)
        res_over = crps_decomposition(fc_over, obs_over, sample_axis=-1)

        assert res_over["reliability"] > res_cal["reliability"], (
            "Over-confident ensemble should have higher Reliability "
            f"(got over={res_over['reliability']:.4f}, cal={res_cal['reliability']:.4f})"
        )


# ---------------------------------------------------------------------------
# (b) Shape and NaN-handling
# ---------------------------------------------------------------------------

class TestShapeAndNaN:
    def test_output_keys_present(self):
        fc, obs = _make_calibrated(N=50, M=20)
        result = crps_decomposition(fc, obs)
        required = {"crps", "crps_decomp", "reliability", "resolution",
                    "uncertainty", "n_samples", "n_cases", "n_nan_cases"}
        assert required.issubset(result.keys()), (
            f"Missing keys: {required - result.keys()}"
        )

    def test_n_samples_and_n_cases(self):
        N, M = 80, 30
        fc, obs = _make_calibrated(N=N, M=M)
        result = crps_decomposition(fc, obs, sample_axis=-1)
        assert result["n_samples"] == M
        assert result["n_cases"]   == N
        assert result["n_nan_cases"] == 0

    def test_nan_obs_excluded(self):
        N, M = 100, 20
        fc, obs = _make_calibrated(N=N, M=M, seed=5)
        obs_with_nan = obs.copy()
        obs_with_nan[:10] = np.nan
        result = crps_decomposition(fc, obs_with_nan)
        assert result["n_nan_cases"] == 10
        assert result["n_cases"]     == N - 10

    def test_nan_fc_excluded(self):
        N, M = 100, 20
        fc, obs = _make_calibrated(N=N, M=M, seed=6)
        fc_with_nan = fc.copy()
        fc_with_nan[5, 3] = np.nan   # one bad ensemble member → whole case excluded
        result = crps_decomposition(fc_with_nan, obs)
        assert result["n_nan_cases"] == 1
        assert result["n_cases"]     == N - 1

    def test_all_nan_returns_nan(self):
        N, M = 20, 10
        fc  = np.full((N, M), np.nan)
        obs = np.full(N, np.nan)
        result = crps_decomposition(fc, obs)
        assert np.isnan(result["crps"])
        assert np.isnan(result["reliability"])

    def test_per_rank_return(self):
        fc, obs = _make_calibrated(N=50, M=15)
        result = crps_decomposition(fc, obs, return_per_rank=True)
        assert "alpha" in result and "beta" in result and "g" in result
        assert len(result["alpha"]) == 15 + 1  # M+1

    def test_raises_on_single_member(self):
        fc  = np.random.randn(50, 1)
        obs = np.random.randn(50)
        with pytest.raises(ValueError, match="at least 2 ensemble members"):
            crps_decomposition(fc, obs)

    def test_multidim_cases(self):
        """Accept (N, T, M) input — cases axis is N*T after flatten."""
        rng = np.random.default_rng(99)
        N, T, M = 20, 10, 30
        fc  = rng.standard_normal((N, T, M))
        obs = rng.standard_normal((N, T))
        result = crps_decomposition(fc, obs, sample_axis=-1)
        assert result["n_cases"] == N * T
        assert result["n_samples"] == M


# ---------------------------------------------------------------------------
# (c) Budget check: Reliability − Resolution + Uncertainty ≈ CRPS
# ---------------------------------------------------------------------------

class TestBudget:
    def test_budget_identity_calibrated(self):
        """
        Reliability − Resolution + Uncertainty = CRPS_decomp (algebraically exact).

        The budget uses "crps_decomp" (Hersbach integral estimator), not
        "crps" (the independent fair/energy-score estimator).  The two
        estimators differ slightly for finite M; this is documented in the
        module docstring.
        """
        rng = np.random.default_rng(0)
        N, M = 1000, 50
        obs = rng.standard_normal(N)
        fc  = rng.standard_normal((N, M))
        result = crps_decomposition(fc, obs, sample_axis=-1)

        budget      = result["reliability"] - result["resolution"] + result["uncertainty"]
        crps_decomp = result["crps_decomp"]

        # Budget is algebraically exact; any residual is pure floating-point error.
        assert abs(budget - crps_decomp) < 1e-10, (
            f"Budget identity violated: "
            f"R-Res+U={budget:.6f}, crps_decomp={crps_decomp:.6f}, "
            f"diff={abs(budget-crps_decomp):.2e}"
        )

    def test_budget_identity_overconfident(self):
        """Budget identity holds for strongly miscalibrated ensembles too."""
        rng = np.random.default_rng(7)
        N, M = 800, 40
        obs = rng.standard_normal(N)
        fc  = rng.normal(0, 0.3, size=(N, M))
        result = crps_decomposition(fc, obs, sample_axis=-1)

        budget      = result["reliability"] - result["resolution"] + result["uncertainty"]
        crps_decomp = result["crps_decomp"]

        assert abs(budget - crps_decomp) < 1e-10, (
            f"Budget identity violated (overconfident): "
            f"R-Res+U={budget:.6f}, crps_decomp={crps_decomp:.6f}"
        )

    def test_crps_ensemble_close_to_decomp_crps(self):
        """
        crps_ensemble (fair estimator) and crps_decomp (Hersbach integral)
        should agree within 5 % for large N, M.

        They use different estimators so exact equality is not expected.
        """
        rng = np.random.default_rng(3)
        N, M = 2000, 100
        obs = rng.standard_normal(N)
        fc  = rng.standard_normal((N, M))

        result    = crps_decomposition(fc, obs, sample_axis=-1)
        crps_fair = result["crps"]
        crps_dec  = result["crps_decomp"]

        rel_diff = abs(crps_fair - crps_dec) / max(crps_fair, 1e-9)
        assert rel_diff < 0.05, (
            f"Fair CRPS and decomp CRPS differ by more than 5%: "
            f"fair={crps_fair:.4f}, decomp={crps_dec:.4f}, rel={rel_diff:.3f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
