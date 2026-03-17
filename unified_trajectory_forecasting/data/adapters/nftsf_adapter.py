"""
data/adapters/nftsf_adapter.py
==============================
Converts a canonical ``.npz`` dataset into PyTorch DataLoaders whose
batches are ``(context, target)`` tensor pairs — exactly the format
consumed by the NFTSF normalizing-flow training engine.

Data flow
---------
canonical .npz
  → load train trajectories (already Z-score normalised)
  → sliding-window segmentation  (mirrors NFTSF_ssh/train_model.py:extract_segments)
  → random train / val segment split  (mirrors train_model.py:380-389)
  → DataLoader( (context, target) )
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data.canonical import load_canonical
from data.preprocessing import extract_segments, split_context_target
from data.splitter import segment_split, build_test_windows


def get_nftsf_loaders(
    canonical_npz: Path,
    n_past: int,
    n_future: int,
    stride: int = 1,
    val_fraction: float = 0.10,
    batch_size: int = 4096,
    seed: int = 42,
    device: str = "cpu",
) -> tuple[DataLoader, DataLoader]:
    """
    Build PyTorch train and validation DataLoaders from a canonical ``.npz``.

    Only the 80 % train-trajectory subset is used.  Sliding-window segments
    are extracted from those trajectories and then randomly split into train
    and validation sets.

    Parameters
    ----------
    canonical_npz : Path to the canonical ``.npz`` file.
    n_past        : Context window length.
    n_future      : Forecast horizon.
    stride        : Sliding-window stride (default 1, maximum augmentation).
    val_fraction  : Fraction of segments held out for validation.
    batch_size    : DataLoader batch size (0 = full-batch single tensor).
    seed          : Seed for the segment-level split.
    device        : Torch device string; tensors are moved here.

    Returns
    -------
    train_loader : DataLoader yielding ``(context, target)`` pairs.
    val_loader   : DataLoader yielding ``(context, target)`` pairs.
    """
    canon = load_canonical(canonical_npz)
    train_idx: np.ndarray = canon["train_indices"]

    # Normalised positions for train trajectories only.
    positions: np.ndarray = canon["positions"][train_idx]   # (N_train, T)
    tracks = torch.tensor(positions, dtype=torch.float32)

    # Sliding-window segmentation.
    segments = extract_segments(tracks, n_past, n_future, stride)   # (S, n_past+n_future)
    n_seg = segments.shape[0]

    # Train / val segment split.
    tr_idx, vl_idx = segment_split(n_seg, val_fraction, seed)
    train_segs = segments[tr_idx]
    val_segs   = segments[vl_idx]

    tr_ctx, tr_tgt = split_context_target(train_segs, n_past)
    vl_ctx, vl_tgt = split_context_target(val_segs,   n_past)

    def _make_loader(ctx: torch.Tensor, tgt: torch.Tensor) -> DataLoader:
        ds = TensorDataset(ctx, tgt)
        bs = len(ds) if batch_size <= 0 else batch_size
        return DataLoader(ds, batch_size=bs, shuffle=True,
                          pin_memory=(device != "cpu"))

    return _make_loader(tr_ctx, tr_tgt), _make_loader(vl_ctx, vl_tgt)


def get_nftsf_test_tensors(
    canonical_npz: Path,
    n_past: int,
    n_future: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build test (context, target) tensors from the 20 % held-out trajectories.

    One window per test trajectory: context = steps [T-n_past-n_future : T-n_future],
    target = steps [T-n_future : T].

    Returns
    -------
    context : (N_test, n_past)   float32 tensor.
    target  : (N_test, n_future) float32 tensor.
    """
    canon = load_canonical(canonical_npz)
    test_idx: np.ndarray = canon["test_indices"]
    positions: np.ndarray = canon["positions"][test_idx]   # (N_test, T)

    ctx_np, tgt_np = build_test_windows(positions, n_past, n_future)
    context = torch.tensor(ctx_np, dtype=torch.float32)
    target  = torch.tensor(tgt_np, dtype=torch.float32)
    return context, target
