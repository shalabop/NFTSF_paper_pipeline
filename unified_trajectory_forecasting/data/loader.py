"""
data/loader.py
==============
Unified data-loading entry point for the NFTSF comparison pipeline.

``get_dataloader(config, canonical_npz)`` inspects the model name in the
merged config and dispatches to the correct adapter.  All callers get back
a simple namespace with ``train``, ``val``, ``test`` attributes; the type
of each attribute depends on the model family:

  NFTSF        → torch.utils.data.DataLoader  (context, target tensors)
  GluonTS      → gluonts.dataset.common.ListDataset
  ARIMA        → numpy arrays (context, target)
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Any

from data.canonical import load_canonical


# Models that use the NFTSF PyTorch data adapter.
_PYTORCH_MODELS = {"nftsf"}

# Models that use the GluonTS adapter.
_GLUONTS_MODELS = {"tsdiff_q", "tsdiff_ms", "tsdiff_cond", "csdi", "ratd", "nsdiff"}

# Models with no gradient training (fit at prediction time).
_ARIMA_MODELS = {"arima"}


@dataclass
class DataBundle:
    """Container returned by get_dataloader."""
    train: Any          # DataLoader | ListDataset | ndarray
    val: Any            # DataLoader | ListDataset | None
    test: Any           # (context, target) tensors or numpy arrays
    metadata: dict      # n_past, n_future, landscape, norm_mean, norm_std, …


def get_dataloader(config: dict, canonical_npz: Path) -> DataBundle:
    """
    Build a DataBundle for the given model and canonical dataset.

    Parameters
    ----------
    config        : Merged config dict (base.yaml + model config).  Must
                    contain ``model.name``, ``data.n_past``, ``data.n_future``,
                    ``data.val_fraction``, ``data.stride``,
                    ``training.batch_size``, ``training.seed``.
    canonical_npz : Path to the canonical ``.npz`` produced by
                    ``data.canonical``.

    Returns
    -------
    DataBundle with train / val / test and metadata.
    """
    canonical_npz = Path(canonical_npz)
    canon_meta    = load_canonical(canonical_npz)

    model_name   = config["model"]["name"].lower()
    n_past       = int(config["data"]["n_past"])
    n_future     = int(config["data"]["n_future"])
    val_fraction = float(config["data"]["val_fraction"])
    stride       = int(config["data"].get("stride", 1))
    batch_size   = int(config["training"].get("batch_size", 4096))
    seed         = int(config["data"]["seed"])
    device       = config["training"].get("device", "cpu")
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    metadata = {
        "n_past":     n_past,
        "n_future":   n_future,
        "landscape":  canon_meta.get("landscape", "unknown"),
        "norm_mean":  float(canon_meta["norm_mean"]),
        "norm_std":   float(canon_meta["norm_std"]),
    }

    if model_name in _PYTORCH_MODELS:
        from data.adapters.nftsf_adapter import (
            get_nftsf_loaders,
            get_nftsf_test_tensors,
        )
        train, val = get_nftsf_loaders(
            canonical_npz,
            n_past=n_past,
            n_future=n_future,
            stride=stride,
            val_fraction=val_fraction,
            batch_size=batch_size,
            seed=seed,
            device=device,
        )
        test = get_nftsf_test_tensors(canonical_npz, n_past, n_future)

    elif model_name in _GLUONTS_MODELS:
        from data.adapters.gluonts_adapter import (
            get_gluonts_train_val_datasets,
            get_gluonts_test_dataset,
        )
        train, val = get_gluonts_train_val_datasets(
            canonical_npz, n_past=n_past, n_future=n_future
        )
        test = get_gluonts_test_dataset(canonical_npz, n_past, n_future)

    elif model_name in _ARIMA_MODELS:
        import numpy as np
        from data.splitter import build_test_windows
        positions = canon_meta["positions"]
        train_idx = canon_meta["train_indices"]
        test_idx  = canon_meta["test_indices"]
        # For ARIMA: train is a raw (N_train, T) array.
        train = positions[train_idx]
        val   = None
        ctx_np, tgt_np = build_test_windows(
            positions[test_idx], n_past, n_future
        )
        test = (ctx_np, tgt_np)

    else:
        raise ValueError(
            f"Unknown model '{model_name}'.  "
            f"Expected one of {sorted(_PYTORCH_MODELS | _GLUONTS_MODELS | _ARIMA_MODELS)}."
        )

    return DataBundle(train=train, val=val, test=test, metadata=metadata)
