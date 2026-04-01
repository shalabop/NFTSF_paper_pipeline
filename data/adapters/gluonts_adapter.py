"""
data/adapters/gluonts_adapter.py
================================
Converts a canonical ``.npz`` dataset into GluonTS ``ListDataset`` objects
so that the diffusion-model training engines (TSDiff, CSDI, RATD, NsDiff)
receive identically split, identically normalised data.

Data flow
---------
canonical .npz
  → train trajectories (Z-score normalised, 80 % split)
  → each trajectory → one GluonTS time-series entry
  → OffsetSplitter removes last n_future steps for validation
  → GluonTS train / val / test ListDatasets

Notes on normalisation
----------------------
The canonical .npz already contains Z-score normalised data (``positions``).
Each diffusion model config must therefore set ``normalization: none`` (or
equivalent) to prevent double-normalisation.  The internal GluonTS scalers
(MeanScaler, NOPScaler) are bypassed by pre-normalising here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from data.canonical import load_canonical
from data.splitter import build_test_windows

# GluonTS imports are optional at module import time so the rest of the
# unified pipeline remains importable on machines without GluonTS.
try:
    from gluonts.dataset.common import ListDataset
    from gluonts.dataset.split import OffsetSplitter
    _GLUONTS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _GLUONTS_AVAILABLE = False
    ListDataset = None          # type: ignore[assignment,misc]
    OffsetSplitter = None       # type: ignore[assignment,misc]


_FAKE_START = pd.Timestamp("2000-01-01")   # GluonTS requires a start timestamp
_FREQ       = "1T"                          # 1 step = 1 "minute"; purely nominal


def _require_gluonts() -> None:
    if not _GLUONTS_AVAILABLE:
        raise ImportError(
            "GluonTS is required for the diffusion-model adapters.  "
            "Install it with: pip install gluonts"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_gluonts_train_val_datasets(
    canonical_npz: Path,
    n_past: int,
    n_future: int,
) -> tuple["ListDataset", "ListDataset"]:
    """
    Build GluonTS train and validation ``ListDataset`` objects.

    Training set
        Each of the 80 % train trajectories becomes one time-series entry
        with ``target = full normalised trajectory``.  GluonTS InstanceSplitter
        extracts random sliding windows of length ``n_past + n_future`` during
        training.

    Validation set
        The same trajectories are split via ``OffsetSplitter``, removing the
        last ``n_future`` steps so they can be held out for validation CRPS.

    Parameters
    ----------
    canonical_npz : Path to the canonical ``.npz``.
    n_past        : GluonTS ``context_length``.
    n_future      : GluonTS ``prediction_length``.

    Returns
    -------
    train_dataset : ListDataset (full trajectories).
    val_dataset   : ListDataset (trajectories truncated by n_future).
    """
    _require_gluonts()
    canon = load_canonical(canonical_npz)
    train_idx: np.ndarray = canon["train_indices"]
    positions: np.ndarray = canon["positions"][train_idx]   # (N_train, T)

    train_entries = _trajectories_to_entries(positions)

    splitter  = OffsetSplitter(offset=-n_future)
    train_ds_full = ListDataset(train_entries, freq=_FREQ)

    # OffsetSplitter.split returns (train, test-gen); we take train portion.
    train_ds, val_gen = splitter.split(train_ds_full)
    val_ds = ListDataset(
        [inp for inp, _ in val_gen.generate_instances(n_future)], freq=_FREQ
    )

    return train_ds, val_ds


def get_gluonts_test_dataset(
    canonical_npz: Path,
    n_past: int,
    n_future: int,
) -> "ListDataset":
    """
    Build a GluonTS test ``ListDataset`` from the 20 % held-out trajectories.

    Each entry has ``target = context + future`` (length ``n_past + n_future``),
    matching the GluonTS evaluation convention where the model sees the context
    portion and must predict the future portion.

    Returns
    -------
    test_dataset : ListDataset with N_test entries.
    """
    _require_gluonts()
    canon = load_canonical(canonical_npz)
    test_idx: np.ndarray = canon["test_indices"]
    positions: np.ndarray = canon["positions"][test_idx]   # (N_test, T)

    context_np, target_np = build_test_windows(positions, n_past, n_future)
    # Concatenate to form the full window expected by GluonTS evaluator.
    window = np.concatenate([context_np, target_np], axis=1)   # (N_test, n_past+n_future)

    entries = _trajectories_to_entries(window)
    return ListDataset(entries, freq=_FREQ)


def get_ground_truth_array(
    canonical_npz: Path,
    n_past: int,
    n_future: int,
) -> np.ndarray:
    """
    Return the (N_test, n_future) ground-truth array for the test set.
    Stored in physical (Z-score normalised) units matching ``positions``.
    """
    canon = load_canonical(canonical_npz)
    test_idx: np.ndarray = canon["test_indices"]
    positions: np.ndarray = canon["positions"][test_idx]
    _, target = build_test_windows(positions, n_past, n_future)
    return target


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _trajectories_to_entries(
    positions: np.ndarray,
) -> list[dict]:
    """Convert (N, T) array to a list of GluonTS time-series dicts."""
    entries: list[dict] = []
    for i, traj in enumerate(positions):
        entries.append({
            "start":   _FAKE_START,
            "target":  traj.astype(np.float32),
            "item_id": str(i),
        })
    return entries
