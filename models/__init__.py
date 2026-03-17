"""
models/
=======
Thin wrappers around each model implementation, all conforming to the
``BaseModel`` interface defined in ``models.base``.

Every wrapper delegates actual computation to the upstream source repo
(NFTSF_ssh or tsf_models-adam) via ``sys.path`` insertion; neither source
repo is modified.

Available models
----------------
nftsf      Conditional Normalizing Flow (NFTSF_ssh/architecture.py)
tsdiff_q   TSDiff-Q  — DDPM guidance         (tsf_models-adam)
tsdiff_ms  TSDiff-MS — DDIM multi-step       (tsf_models-adam)
tsdiff_cond TSDiff-Cond — conditional/imputation (tsf_models-adam)
csdi       CSDI — score-based diffusion      (tsf_models-adam)
ratd       RATD — reference-aided diffusion  (tsf_models-adam)
nsdiff     NsDiff — neural SDE diffusion     (tsf_models-adam)
arima      Auto ARIMA baseline               (pmdarima)
"""

from models.base import BaseModel
from models.registry import get_model

__all__ = ["BaseModel", "get_model"]
