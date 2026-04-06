"""
models/arima.py
===============
Auto-ARIMA baseline wrapper using the pmdarima library.

ARIMA has no gradient training; it fits one model per trajectory at
prediction time.  ``training_step`` and ``validation_step`` are no-ops
that return zero.

Mirrors the approach in tsf_models-adam/run_auto_arima.py but integrated
into the unified BaseModel interface.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from models.base import BaseModel


class ARIMAModel(BaseModel):
    """Auto-ARIMA probabilistic forecasting baseline."""

    def __init__(self, config: dict) -> None:
        mc = config["model"]
        dc = config["data"]

        self._name    = "arima"
        self.n_past   = int(dc["n_past"])
        self.n_future = int(dc["n_future"])
        self.max_p    = int(mc.get("max_p", 5))
        self.max_q    = int(mc.get("max_q", 5))
        self.max_d    = int(mc.get("max_d", 2))
        self.seasonal = bool(mc.get("seasonal", False))
        # Cache fitted models keyed by trajectory index.
        self._fitted: dict = {}

    # -----------------------------------------------------------------------
    # Forward — not meaningful for ARIMA; returns the point forecast.
    # -----------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        preds: list[np.ndarray] = []
        for i in range(x.shape[0]):
            model = self._fit_one(x[i].numpy())
            fc, _ = model.predict(n_periods=self.n_future, return_conf_int=False)
            preds.append(fc)
        return torch.tensor(np.stack(preds), dtype=torch.float32)

    # -----------------------------------------------------------------------
    # Training / validation — ARIMA fits at inference time; no-ops here.
    # -----------------------------------------------------------------------

    def training_step(self, batch: tuple) -> torch.Tensor:
        return torch.tensor(0.0)

    def validation_step(self, batch: tuple) -> torch.Tensor:
        return torch.tensor(0.0)

    # -----------------------------------------------------------------------
    # Probabilistic inference
    # -----------------------------------------------------------------------

    def predict(
        self,
        context: torch.Tensor,
        n_samples: int = 500,
    ) -> np.ndarray:
        """
        Fit ARIMA to each context series and draw ``n_samples`` sample paths.

        Parameters
        ----------
        context   : (B, n_past) tensor.
        n_samples : Number of sample paths per trajectory.

        Returns
        -------
        (B, n_future, n_samples) float32 ndarray.
        """
        try:
            import pmdarima as pm
        except ImportError as exc:
            raise ImportError(
                "pmdarima is required for the ARIMA model.  "
                "Install with: pip install pmdarima"
            ) from exc

        B = context.shape[0]
        all_samples = np.empty((B, self.n_future, n_samples), dtype=np.float32)

        for i in range(B):
            series = context[i].numpy().astype(np.float64)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                arima = pm.auto_arima(
                    series,
                    max_p=self.max_p,
                    max_q=self.max_q,
                    max_d=self.max_d,
                    seasonal=self.seasonal,
                    error_action="ignore",
                    suppress_warnings=True,
                    stepwise=True,
                )

            # Draw sample paths by simulating from the fitted model.
            for s in range(n_samples):
                try:
                    sim = arima.simulate(
                        nsimulations=self.n_future,
                        repetitions=1,
                        initial_values=series,
                    ).flatten()
                    all_samples[i, :, s] = sim.astype(np.float32)
                except Exception:
                    # Fallback: use point forecast + Gaussian noise.
                    fc, ci = arima.predict(
                        n_periods=self.n_future, return_conf_int=True
                    )
                    std = (ci[:, 1] - ci[:, 0]) / (2 * 1.96)
                    all_samples[i, :, s] = (
                        fc + np.random.randn(self.n_future) * std
                    ).astype(np.float32)

        return all_samples

    # -----------------------------------------------------------------------
    # Checkpoint I/O — ARIMA state is trivially pickled / ignored.
    # -----------------------------------------------------------------------

    def save(self, path: Path) -> None:
        import pickle
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._fitted, f)

    def load(self, path: Path) -> None:
        import pickle
        with open(Path(path), "rb") as f:
            self._fitted = pickle.load(f)

    # -----------------------------------------------------------------------

    def _fit_one(self, series: np.ndarray):
        import pmdarima as pm
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pm.auto_arima(
                series.astype(np.float64),
                max_p=self.max_p,
                max_q=self.max_q,
                max_d=self.max_d,
                seasonal=self.seasonal,
                error_action="ignore",
                suppress_warnings=True,
                stepwise=True,
            )
