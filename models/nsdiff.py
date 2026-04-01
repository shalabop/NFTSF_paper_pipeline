"""
models/nsdiff.py
================
Wrapper around NsDiff from architectures/NsDiff/.

Imports the NsDiff model via sys.path insertion.  No source files are modified.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from models.base import BaseModel

_NSDIFF_ROOTS = [
    (
        Path(__file__).resolve().parents[1]
        / "architectures"
        / "NsDiff"
    ).as_posix(),
    (
        Path(__file__).resolve().parents[1]
        / "architectures"
        / "NsDiff"
        / "src"
    ).as_posix(),
]
for _root in _NSDIFF_ROOTS:
    if _root not in sys.path:
        sys.path.insert(0, _root)


class NsDiffModel(BaseModel):
    """NsDiff (Neural SDE Diffusion) wrapper."""

    def __init__(self, config: dict) -> None:
        mc = config["model"]
        dc = config["data"]
        tc = config["training"]

        self._name    = "nsdiff"
        self.n_past   = int(dc["n_past"])
        self.n_future = int(dc["n_future"])

        device_str = tc.get("device", "auto")
        if device_str == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)

        # NsDiff is at architectures/NsDiff/src/models/NsDiff.py
        # architectures/NsDiff/ is on sys.path, so use the package path.
        from NsDiff.src.models.NsDiff import NsDiff

        self.model = NsDiff(
            context_length=self.n_past,
            prediction_length=self.n_future,
            hidden_dim=int(mc.get("hidden_dim", 64)),
            num_layers=int(mc.get("num_layers", 3)),
            num_steps=int(mc.get("diffusion_steps", 50)),
            beta_start=float(mc.get("beta_start", 0.0001)),
            beta_end=float(mc.get("beta_end", 0.5)),
        ).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        samples = self.predict(x, n_samples=50)
        return torch.tensor(np.median(samples, axis=-1), dtype=torch.float32)

    def training_step(self, batch: tuple) -> torch.Tensor:
        context, target = batch
        loss = self.model.get_loss(
            context.to(self.device),
            target.to(self.device),
        )
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
        context = context.to(self.device)
        with torch.no_grad():
            # Repeat context for each sample.
            ctx_rep = context.unsqueeze(1).expand(-1, n_samples, -1)   # (B, S, n_past)
            B, S, _ = ctx_rep.shape
            ctx_flat = ctx_rep.reshape(B * S, -1)                       # (B*S, n_past)
            pred_flat = self.model.sample(ctx_flat)                     # (B*S, n_future)
            pred = pred_flat.reshape(B, S, self.n_future)               # (B, S, n_future)
            result = pred.permute(0, 2, 1).cpu().numpy()                # (B, n_future, S)
        self.model.train()
        return result.astype(np.float32)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: Path) -> None:
        state = torch.load(Path(path), map_location=self.device)
        self.model.load_state_dict(state)
