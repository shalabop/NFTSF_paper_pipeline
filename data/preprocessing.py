"""
data/preprocessing.py
=====================
Z-score normalisation and sliding-window segmentation, mirroring the logic
in NFTSF_ssh/train_model.py (normalize_data, extract_segments) so that
every model in the comparison pipeline receives identically prepared data.

All functions operate on NumPy arrays or PyTorch tensors and are
side-effect-free (no global state).
"""

from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def compute_norm_stats(data: np.ndarray) -> tuple[float, float]:
    """
    Compute Z-score statistics (mean, std) from a subset of trajectories.

    Parameters
    ----------
    data : (N, T) float32 array — training trajectories only.

    Returns
    -------
    mean : float
    std  : float  (guaranteed >= 1e-8)
    """
    mean = float(data.mean())
    std  = float(data.std()) + 1e-8
    return mean, std


def normalize(
    data: np.ndarray,
    mean: float,
    std: float,
) -> np.ndarray:
    """Apply Z-score normalisation: x_norm = (x - mean) / std."""
    return (data - mean) / std


def denormalize(
    data: np.ndarray,
    mean: float,
    std: float,
) -> np.ndarray:
    """Invert Z-score normalisation: x = x_norm * std + mean."""
    return data * std + mean


# ---------------------------------------------------------------------------
# Sliding-window segmentation (mirrors NFTSF_ssh/train_model.py:extract_segments)
# ---------------------------------------------------------------------------

def extract_segments(
    tracks: torch.Tensor,
    n_past: int,
    n_future: int,
    stride: int = 1,
) -> torch.Tensor:
    """
    Extract overlapping fixed-length segments from trajectory data using a
    sliding window, producing the training set for the conditional model.

    Mirrors NFTSF_ssh/train_model.py::extract_segments exactly.

    Parameters
    ----------
    tracks   : (B, T) tensor — B trajectories of length T.
    n_past   : Context window length.
    n_future : Forecast horizon length.
    stride   : Step between consecutive window starts (default 1).

    Returns
    -------
    segments : (N_seg, n_past + n_future) tensor.
    """
    n_extrp = n_past + n_future
    segs: list[torch.Tensor] = []

    num_tracks, length_track = tracks.shape
    for i in range(num_tracks):
        for start in range(0, length_track - n_extrp + 1, stride):
            segs.append(tracks[i, start : start + n_extrp])

    if not segs:
        raise ValueError(
            f"No segments extracted — data length {length_track} "
            f"is shorter than n_past + n_future = {n_extrp}."
        )
    return torch.stack(segs)


# ---------------------------------------------------------------------------
# Context / target split helper
# ---------------------------------------------------------------------------

def split_context_target(
    segments: torch.Tensor,
    n_past: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Split (N, n_past+n_future) segments into context and target tensors.

    Returns
    -------
    context : (N, n_past)
    target  : (N, n_future)
    """
    context = segments[:, :n_past]
    target  = segments[:, n_past:]
    return context, target
