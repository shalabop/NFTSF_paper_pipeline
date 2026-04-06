"""
models/ratd.py
==============
Wrapper around RATD (Reference-Aided Temporal Diffusion) from
tsf_models-adam/models/RATD/.

Imports RATD_Forecasting from the upstream repo via sys.path insertion.
No source files are modified.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from models.base import BaseModel

_RATD_ROOT = (
    Path(__file__).resolve().parents[1].parent
    / "tsf_models-adam"
    / "models"
    / "RATD"
).as_posix()
if _RATD_ROOT not in sys.path:
    sys.path.insert(0, _RATD_ROOT)


class RATDModel(BaseModel):
    """RATD Forecasting model wrapper."""

    def __init__(self, config: dict) -> None:
        mc = config["model"]
        dc = config["data"]
        tc = config["training"]

        self._name    = "ratd"
        self.n_past   = int(dc["n_past"])
        self.n_future = int(dc["n_future"])

        device_str = tc.get("device", "auto")
        if device_str == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)

        from main_model import RATD_Forecasting  # from tsf_models-adam RATD

        ratd_config = {
            "channels":              int(mc.get("channels", 64)),
            "layers":                int(mc.get("layers", 4)),
            "nheads":                int(mc.get("nheads", 8)),
            "num_steps":             int(mc.get("diffusion_steps", 50)),
            "beta_start":            float(mc.get("beta_start", 0.0001)),
            "beta_end":              float(mc.get("beta_end", 0.5)),
            "schedule":              mc.get("schedule", "quad"),
            "target_strategy":       mc.get("target_strategy", "mix"),
            "unconditional_rate":    float(mc.get("unconditional_rate", 0.1)),
            "target_dim":            1,
        }

        self.model = RATD_Forecasting(
            config=ratd_config,
            device=self.device,
            target_dim=1,
            context_length=self.n_past,
            prediction_length=self.n_future,
        ).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        samples = self.predict(x, n_samples=50)
        return torch.tensor(np.median(samples, axis=-1), dtype=torch.float32)

    def training_step(self, batch: tuple) -> torch.Tensor:
        context, target = batch
        context = context.unsqueeze(-1).to(self.device)
        target  = target.unsqueeze(-1).to(self.device)
        loss, _ = self.model(context, target, is_train=1)
        return loss

    def validation_step(self, batch: tuple) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            loss = self.training_step(batch)
        self.model.train()
        return loss

    def predict(
        self,
        context: torch.Tensor,
        n_samples: int = 500,
    ) -> np.ndarray:
        """Returns (B, n_future, n_samples) float32 ndarray."""
        self.model.eval()
        context = context.unsqueeze(-1).to(self.device)
        with torch.no_grad():
            samples_raw = self.model.impute(context, n_samples=n_samples)
            samples = samples_raw[..., 0].permute(0, 2, 1).cpu().numpy()
        self.model.train()
        return samples.astype(np.float32)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: Path) -> None:
        state = torch.load(Path(path), map_location=self.device)
        self.model.load_state_dict(state)
