"""
models/csdi.py
==============
Wrapper around CSDI (Conditional Score-Based Diffusion for Imputation /
Forecasting) from architectures/CSDI/.

Imports CSDI_Forecasting via sys.path insertion.  No source files are modified.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from models.base import BaseModel

_CSDI_ROOT = (
    Path(__file__).resolve().parents[1]
    / "architectures"
    / "CSDI"
).as_posix()
if _CSDI_ROOT not in sys.path:
    sys.path.insert(0, _CSDI_ROOT)


class CSDIModel(BaseModel):
    """CSDI Forecasting model wrapper."""

    def __init__(self, config: dict) -> None:
        mc = config["model"]
        dc = config["data"]
        tc = config["training"]

        self._name    = "csdi"
        self.n_past   = int(dc["n_past"])
        self.n_future = int(dc["n_future"])

        device_str = tc.get("device", "auto")
        if device_str == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)

        from main_model import CSDI_Forecasting  # from tsf_models-adam CSDI

        csdi_config = {
            "channels":                       int(mc.get("channels", 64)),
            "layers":                         int(mc.get("layers", 4)),
            "nheads":                         int(mc.get("nheads", 8)),
            "diffusion_embedding_dim":        int(mc.get("diffusion_step_embed_dim_in", 128)),
            "diffusion_step_embed_dim_mid":   int(mc.get("diffusion_step_embed_dim_mid", 512)),
            "diffusion_step_embed_dim_out":   int(mc.get("diffusion_step_embed_dim_out", 512)),
            "num_steps":                      int(mc.get("diffusion_steps", 50)),
            "beta_start":                     float(mc.get("beta_start", 0.0001)),
            "beta_end":                       float(mc.get("beta_end", 0.5)),
            "schedule":                       mc.get("schedule", "quad"),
            "target_dim":                     1,   # univariate
        }

        self.model = CSDI_Forecasting(
            config=csdi_config,
            device=self.device,
            target_dim=1,
            context_length=self.n_past,
            prediction_length=self.n_future,
        ).to(self.device)

        self._lr = float(tc.get("learning_rate", 1e-3))

    # -----------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        samples = self.predict(x, n_samples=50)
        return torch.tensor(np.median(samples, axis=-1), dtype=torch.float32)

    def training_step(self, batch: tuple) -> torch.Tensor:
        context, target = batch
        context = context.unsqueeze(-1).to(self.device)   # (B, n_past, 1)
        target  = target.unsqueeze(-1).to(self.device)    # (B, n_future, 1)
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
        context = context.unsqueeze(-1).to(self.device)   # (B, n_past, 1)
        with torch.no_grad():
            # CSDI impute() returns (B, n_samples, n_future, D)
            samples_raw = self.model.impute(
                context,
                n_samples=n_samples,
            )
            # Shape: (B, n_samples, n_future, 1) → (B, n_future, n_samples)
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
