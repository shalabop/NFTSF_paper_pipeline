"""
models/nftsf.py
===============
Wrapper around the NFTSF Conditional Normalizing Flow model.

This wrapper imports ``architecture.create_nfm`` from the NFTSF_ssh
repository via sys.path insertion.  The architecture source is never
modified — only the surrounding training / inference scaffolding is
unified here.

Architecture summary (from NFTSF_ssh/architecture.py)
------------------------------------------------------
- Base distribution: DiagGaussian(n_future)
- K flow blocks, each containing:
    - len(hidden_layers_list) Autoregressive RQS transforms
    - 1 LU Linear Permute layer
- Context (x_past) is fed to every spline conditioner network.
- Training objective: negative mean log-likelihood.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from models.base import BaseModel
from models.architecture import create_nfm


class NFTSFModel(BaseModel):
    """Conditional Normalizing Flow for trajectory prediction."""

    def __init__(self, config: dict) -> None:
        self._name = "nftsf"
        mc = config["model"]
        dc = config["data"]

        device_str = config["training"].get("device", "auto")
        if device_str == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)

        self.n_past   = int(dc["n_past"])
        self.n_future = int(dc["n_future"])

        hidden_layers = mc.get("hidden_layers", [1, 2])
        if isinstance(hidden_layers, str):
            hidden_layers = [int(h) for h in hidden_layers.split(",")]

        self.model = create_nfm(
            device=self.device,
            latent_size=self.n_future,
            context_size=self.n_past,
            K=int(mc.get("flow_blocks", 6)),
            hidden_units=int(mc.get("hidden_units", 64)),
            hidden_layers_list=tuple(hidden_layers),
            tail_bound=float(mc.get("tail_bound", 30)),
        )

    # -----------------------------------------------------------------------
    # Forward (point prediction via flow mean / mode)
    # -----------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the mean of 256 forecast samples as a point prediction."""
        with torch.no_grad():
            samples = self.model.sample(256, context=x.to(self.device))
        return samples.mean(dim=0)   # (B, n_future)

    # -----------------------------------------------------------------------
    # Training / validation steps
    # -----------------------------------------------------------------------

    def training_step(self, batch: tuple) -> torch.Tensor:
        """Negative mean log-likelihood over a mini-batch."""
        context, target = batch
        context = context.to(self.device)
        target  = target.to(self.device)
        loss = -self.model.log_prob(target, context).mean()
        return loss

    def validation_step(self, batch: tuple) -> torch.Tensor:
        """Same as training_step but without gradient tracking."""
        self.model.eval()
        with torch.no_grad():
            loss = self.training_step(batch)
        self.model.train()
        return loss

    # -----------------------------------------------------------------------
    # Probabilistic inference
    # -----------------------------------------------------------------------

    def predict(
        self,
        context: torch.Tensor,
        n_samples: int = 500,
    ) -> np.ndarray:
        """
        Draw ``n_samples`` forecast trajectories for each context in the batch.

        Parameters
        ----------
        context   : (B, n_past) tensor.
        n_samples : Number of sample paths.

        Returns
        -------
        samples : (B, n_future, n_samples) float32 ndarray.
        """
        self.model.eval()
        context = context.to(self.device)
        B = context.shape[0]
        all_samples: list[np.ndarray] = []

        with torch.no_grad():
            # Sample in one call when B is small; chunk for large batches
            # to avoid GPU OOM.
            chunk = 64
            for s in range(0, n_samples, chunk):
                n = min(chunk, n_samples - s)
                # model.sample returns (n, B, n_future) or (n*B, n_future)?
                # normflows ConditionalNormalizingFlow.sample(num_samples, context)
                # returns (num_samples, *event_shape) when context is broadcast.
                # We call it once per batch item to be safe.
                per_item = []
                for b in range(B):
                    ctx_b = context[b : b + 1].expand(n, -1)   # (n, n_past)
                    s_b = self.model.sample(n, context=ctx_b)   # (n, n_future)
                    per_item.append(s_b.cpu().numpy())          # (n, n_future)
                # per_item: list of B arrays each (n, n_future)
                all_samples.append(
                    np.stack(per_item, axis=0)   # (B, n, n_future)
                )

        # Concatenate over sample chunks: (B, n_samples, n_future)
        result = np.concatenate(all_samples, axis=1)   # (B, n_samples, n_future)
        self.model.train()
        return result.transpose(0, 2, 1).astype(np.float32)   # (B, n_future, n_samples)

    # -----------------------------------------------------------------------
    # Checkpoint I/O
    # -----------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: Path) -> None:
        state = torch.load(Path(path), map_location=self.device)
        self.model.load_state_dict(state)
