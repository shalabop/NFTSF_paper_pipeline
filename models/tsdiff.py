"""
models/tsdiff.py
================
Wrapper around the TSDiff family of diffusion models from
architectures/unconditional_time_series_diffusion/.

Covers three variants selected by ``config["model"]["name"]``:
  tsdiff_q    — unconditional TSDiff with DDPM quantile guidance
  tsdiff_ms   — unconditional TSDiff with DDIM multi-step guidance
  tsdiff_cond — conditional TSDiff (forecast via masking)

Imports via sys.path insertion.  No architecture files are modified.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from models.base import BaseModel

# ---------------------------------------------------------------------------
# Add architectures/unconditional_time_series_diffusion/src to sys.path.
# ---------------------------------------------------------------------------
_TSFDIFF_ROOT = (
    Path(__file__).resolve().parents[1]
    / "architectures"
    / "unconditional_time_series_diffusion"
    / "src"
).as_posix()
if _TSFDIFF_ROOT not in sys.path:
    sys.path.insert(0, _TSFDIFF_ROOT)


class TSDiffModel(BaseModel):
    """
    Thin wrapper around TSDiff / TSDiffCond for the unified pipeline.

    The heavy lifting (diffusion forward, loss, sampling) is delegated to
    the upstream TSDiff / TSDiffCond PyTorch-Lightning modules.  This
    wrapper adapts their interface to BaseModel and handles the GluonTS
    training engine interaction via ``engine_lightning.py``.
    """

    def __init__(self, config: dict) -> None:
        mc = config["model"]
        dc = config["data"]
        tc = config["training"]

        self._name  = mc["name"].lower()   # "tsdiff_q" | "tsdiff_ms" | "tsdiff_cond"
        self.n_past   = int(dc["n_past"])
        self.n_future = int(dc["n_future"])

        device_str = tc.get("device", "auto")
        if device_str == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)

        # Import TSDiff internals (deferred to avoid ImportError when GluonTS
        # is not installed and only NFTSF / ARIMA are being used).
        import uncond_ts_diff.configs as diffusion_configs
        from uncond_ts_diff.model.diffusion.tsdiff import TSDiff
        from uncond_ts_diff.model.diffusion.tsdiff_cond import TSDiffCond

        diff_cfg_name = mc.get("diffusion_config", "diffusion_small_config")
        diff_cfg = getattr(diffusion_configs, diff_cfg_name)

        is_cond = self._name == "tsdiff_cond"
        ModelClass = TSDiffCond if is_cond else TSDiff

        self.model: torch.nn.Module = ModelClass(
            **diff_cfg,
            freq="1T",
            use_features=bool(mc.get("use_features", False)),
            use_lags=bool(mc.get("use_lags", False)),
            normalization=mc.get("normalization", "none"),
            context_length=self.n_past,
            prediction_length=self.n_future,
            lr=float(tc.get("learning_rate", 1e-3)),
            init_skip=bool(mc.get("init_skip", False)),
        )
        self.model.to(self.device)

        # Store variant-specific inference settings.
        self._guidance      = mc.get("guidance", "ddpm")
        self._num_inf_steps = int(mc.get("num_inference_steps", 50))
        self._guidance_scale = float(mc.get("guidance_scale", 1.0))

    # -----------------------------------------------------------------------
    # Forward (point prediction)
    # -----------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        samples = self.predict(x, n_samples=50)   # (B, n_future, 50)
        return torch.tensor(np.median(samples, axis=-1), dtype=torch.float32)

    # -----------------------------------------------------------------------
    # Training / validation (delegated to PL via engine_lightning)
    # -----------------------------------------------------------------------

    def training_step(self, batch: tuple) -> torch.Tensor:
        """
        Diffusion training step.

        NOTE: the GluonTS / Lightning engine calls the PL LightningModule's
        ``training_step`` directly.  This shim is provided to satisfy the
        BaseModel interface and is used by the custom loop engine as a
        fallback — it may not be called during normal Lightning training.
        """
        past_target, past_observed, future_target = batch
        output = self.model.training_step(
            {
                "past_target": past_target.to(self.device),
                "past_observed_values": past_observed.to(self.device),
                "future_target": future_target.to(self.device),
            },
            batch_idx=0,
        )
        return output["loss"] if isinstance(output, dict) else output

    def validation_step(self, batch: tuple) -> torch.Tensor:
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
        Draw ``n_samples`` forecast trajectories.

        Parameters
        ----------
        context   : (B, n_past) tensor.
        n_samples : Number of sample paths.

        Returns
        -------
        (B, n_future, n_samples) float32 ndarray.
        """
        from uncond_ts_diff.sampler import DDPMGuidance, DDIMGuidance

        self.model.eval()
        context = context.to(self.device)
        B = context.shape[0]

        guidance_cls = DDIMGuidance if self._guidance == "ddim" else DDPMGuidance

        # Build a minimal GluonTS-like batch dict.
        obs_mask = torch.ones_like(context)
        batch = {
            "past_target":          context,
            "past_observed_values": obs_mask,
        }

        all_samples: list[np.ndarray] = []
        with torch.no_grad():
            for _ in range(n_samples):
                sampler = guidance_cls(
                    model=self.model,
                    prediction_length=self.n_future,
                    num_samples=1,
                    guidance_scale=self._guidance_scale,
                )
                pred = sampler(batch)   # (1, B, n_future) or (B, 1, n_future)
                pred_np = pred.cpu().numpy()
                if pred_np.ndim == 3 and pred_np.shape[0] == 1:
                    pred_np = pred_np[0]   # (B, n_future)
                all_samples.append(pred_np)

        self.model.train()
        # stack: (n_samples, B, n_future) → transpose to (B, n_future, n_samples)
        result = np.stack(all_samples, axis=0)          # (n_samples, B, n_future)
        return result.transpose(1, 2, 0).astype(np.float32)

    # -----------------------------------------------------------------------
    # Checkpoint I/O (delegate to PL)
    # -----------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: Path) -> None:
        state = torch.load(Path(path), map_location=self.device)
        self.model.load_state_dict(state)
