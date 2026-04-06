"""
data/splitter.py
================
Trajectory-level and segment-level train / val / test splits.

The trajectory-level split mirrors NFTSF_ssh/splice_alanine.py (line 89):
a seeded random permutation of all N trajectory indices is divided 80 / 20
into train and test subsets.  The same seed is used throughout the pipeline
so that every model sees identical trajectory assignments.

The segment-level split mirrors NFTSF_ssh/train_model.py (lines 380-389):
a seeded random permutation of all segment indices is divided (1-val_fraction)
/ val_fraction into train and validation subsets.
"""

from __future__ import annotations

import numpy as np
import torch


def trajectory_split(
    n_total: int,
    train_fraction: float = 0.80,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split N trajectory indices into train and test sets.

    Mirrors the permutation logic in NFTSF_ssh/splice_alanine.py:89–92.

    Parameters
    ----------
    n_total        : Total number of trajectories.
    train_fraction : Fraction assigned to training (default 0.80).
    seed           : NumPy RNG seed for reproducibility.

    Returns
    -------
    train_indices : (N_train,) int64 array.
    test_indices  : (N_test,)  int64 array.
    """
    rng   = np.random.default_rng(seed)
    idx   = rng.permutation(n_total)
    n_tr  = int(n_total * train_fraction)
    return idx[:n_tr].astype(np.int64), idx[n_tr:].astype(np.int64)


def segment_split(
    n_segments: int,
    val_fraction: float = 0.10,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split N segment indices into train and validation sets.

    Mirrors NFTSF_ssh/train_model.py:380–389.

    Parameters
    ----------
    n_segments   : Total number of sliding-window segments.
    val_fraction : Fraction held out for validation (default 0.10).
    seed         : Torch manual seed used for randperm.

    Returns
    -------
    train_idx : (N_train,) int64 ndarray.
    val_idx   : (N_val,)   int64 ndarray.
    """
    # Use torch.randperm to exactly replicate NFTSF_ssh behaviour.
    gen = torch.Generator()
    gen.manual_seed(seed)
    perm   = torch.randperm(n_segments, generator=gen).numpy()
    n_val  = max(1, int(n_segments * val_fraction))
    return perm[n_val:].astype(np.int64), perm[:n_val].astype(np.int64)


def build_test_windows(
    trajectories: np.ndarray,
    n_past: int,
    n_future: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build one evaluation window per trajectory.

    The context is the ``n_past`` steps immediately before the final
    ``n_future`` steps; the target is those final ``n_future`` steps.

    Parameters
    ----------
    trajectories : (N, T) float32 array.
    n_past       : Context length.
    n_future     : Forecast horizon.

    Returns
    -------
    context : (N, n_past)   float32 array.
    target  : (N, n_future) float32 array.

    Raises
    ------
    ValueError if T < n_past + n_future.
    """
    T = trajectories.shape[1]
    if T < n_past + n_future:
        raise ValueError(
            f"Trajectory length {T} < n_past + n_future = {n_past + n_future}."
        )
    context = trajectories[:, T - n_past - n_future : T - n_future]
    target  = trajectories[:, T - n_future :]
    return context, target
