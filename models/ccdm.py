"""
models/ccdm.py
==============
Wrapper around CCDM (Contrastive Conditional Diffusion Model) from
tsf_models-adam/models/CCDM/.

Imports the CCDM Denoiser and DDPM classes from the upstream repo via
sys.path insertion.  No source files are modified.

⚠️  KNOWN ISSUE — model files not yet committed
-------------------------------------------------
tsf_models-adam/models/CCDM/ is currently empty.  The required Python
files (network.py, diffusion.py) have not yet been committed to that
directory.  Until they land, instantiating CcdmModel will raise an
ImportError with a descriptive message.

All other pipeline infrastructure (config, registry entry, wrapper) is
complete and will work without further changes once the files are committed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from models.base import BaseModel

_CCDM_ROOT = (
    Path(__file__).resolve().parents[1].parent
    / "tsf_models-adam"
    / "models"
    / "CCDM"
).as_posix()
if _CCDM_ROOT not in sys.path:
    sys.path.insert(0, _CCDM_ROOT)


class CcdmModel(BaseModel):
    """CCDM (Contrastive Conditional Diffusion Model) wrapper."""

    def __init__(self, config: dict) -> None:
        mc = config["model"]
        dc = config["data"]
        tc = config["training"]

        self._name    = "ccdm"
        self.n_past   = int(dc["n_past"])
        self.n_future = int(dc["n_future"])

        device_str = tc.get("device", "auto")
        if device_str == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)

        # Lazily import from tsf_models-adam/models/CCDM/.
        # Raises ImportError with a clear message if model files are absent.
        try:
            from network   import Denoiser  # noqa: F401
            from diffusion import DDPM      # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "CCDM model files (network.py, diffusion.py) not found in "
                f"{_CCDM_ROOT}.  "
                "The tsf_models-adam/models/CCDM/ directory is currently empty — "
                "these files must be committed before CcdmModel can be used."
            ) from exc

        configs = self._build_configs(mc)

        self.denoiser  = Denoiser(configs).to(self.device)
        self.diffusion = DDPM(self.denoiser, configs).to(self.device)

    # ------------------------------------------------------------------
    # Config builder — mirrors build_configs() in forecast_ccdm.py
    # All CCDM settings are nested under model: in ccdm.yaml, consistent
    # with how other wrappers (ratd.py, csdi.py) are structured.
    # ------------------------------------------------------------------

    def _build_configs(self, mc: dict):
        class Configs:
            pass

        c = Configs()
        c.cont_len         = self.n_past
        c.pred_len         = self.n_future
        c.num_feat         = 1  # 1-D trajectories throughout this pipeline

        c.n_emb            = int(mc.get("n_emb",            2))
        c.cont_hidden_dim  = int(mc.get("cont_hidden_dim",  128))
        c.pred_hidden_dim  = int(mc.get("pred_hidden_dim",  128))
        c.step_hidden_dim  = int(mc.get("step_hidden_dim",  128))
        c.time_hidden_dim  = int(mc.get("time_hidden_dim",  128))
        c.n_depth          = int(mc.get("n_depth",          2))
        c.n_heads          = int(mc.get("n_heads",          8))
        c.attn_dropout     = float(mc.get("attn_dropout",   0.1))
        c.mlp_ratio        = int(mc.get("mlp_ratio",        1))
        c.non_attn         = bool(mc.get("non_attn",        False))

        c.n_steps          = int(mc.get("diffusion_steps",    100))
        c.beta_start       = float(mc.get("beta_start",       0.0001))
        c.beta_end         = float(mc.get("beta_end",         0.2))
        c.beta_schedule    = mc.get("beta_schedule",          "quad")
        c.parameterization = mc.get("parameterization",       "noise")

        c.device = self.device
        return c

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        samples = self.predict(x, n_samples=50)
        return torch.tensor(np.median(samples, axis=-1), dtype=torch.float32)

    def training_step(self, batch: tuple) -> torch.Tensor:
        context, target = batch
        context = context.unsqueeze(-1).to(self.device)   # (B, L, 1)
        target  = target.unsqueeze(-1).to(self.device)    # (B, H, 1)
        loss = self.diffusion.get_loss(context, target)
        return loss

    def validation_step(self, batch: tuple) -> torch.Tensor:
        self.denoiser.eval()
        self.diffusion.eval()
        with torch.no_grad():
            loss = self.training_step(batch)
        self.denoiser.train()
        self.diffusion.train()
        return loss

    def predict(
        self,
        context: torch.Tensor,
        n_samples: int = 500,
    ) -> np.ndarray:
        """Returns (B, n_future, n_samples) float32 ndarray."""
        self.denoiser.eval()
        self.diffusion.eval()
        context = context.unsqueeze(-1).to(self.device)   # (B, L, 1)

        all_samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                # diffusion.sampling returns (B, H, 1)
                s = self.diffusion.sampling(1, context, x_mark=None, y0_mark=None)
                all_samples.append(s.squeeze(-1))           # (B, H)

        self.denoiser.train()
        self.diffusion.train()

        # Stack → (B, H, n_samples)
        result = torch.stack(all_samples, dim=-1).cpu().numpy()
        return result.astype(np.float32)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "denoiser":  self.denoiser.state_dict(),
                "diffusion": self.diffusion.state_dict(),
            },
            path,
        )

    def load(self, path: Path) -> None:
        ckpt = torch.load(Path(path), map_location=self.device)
        self.denoiser.load_state_dict(ckpt["denoiser"])
        self.diffusion.load_state_dict(ckpt["diffusion"])
