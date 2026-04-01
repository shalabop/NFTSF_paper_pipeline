"""
data/canonical.py
=================
Build and load the canonical ``.npz`` dataset format consumed by every
data adapter in the unified pipeline.

Canonical schema
----------------
positions         (N, T)      float32  Z-score normalised trajectories
positions_raw     (N, T)      float32  original un-normalised values
train_indices     (N_train,)  int64    80 % trajectory indices
test_indices      (N_test,)   int64    20 % trajectory indices
time              (T,)        float32  time axis (uniform 0 … T-1)
landscape         str         e.g. "alanine_phi"
seed              int         42
n_past            int         100
n_future          int         100
norm_mean         float       Z-score μ (train set only)
norm_std          float       Z-score σ (train set only)

CLI usage
---------
# From existing NFTSF_ssh .npy files:
conda run -n unified_tsf python data/canonical.py \\
    --source nftsf \\
    --train /home/user/NFTSF_ssh/alanine_phi_train.npy \\
    --test  /home/user/NFTSF_ssh/alanine_phi_test.npy \\
    --landscape alanine_phi \\
    --out outputs/canonical/alanine_phi.npz

# From a raw (N, T) .npz produced by a generator:
conda run -n unified_tsf python data/canonical.py \\
    --source npz \\
    --input /path/to/double_well.npz \\
    --landscape double_well \\
    --out outputs/canonical/double_well.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Locate the NFTSF_ssh repo for its data loading helpers.
_NFTSF_ROOT = Path(__file__).resolve().parents[1].parent / "NFTSF_ssh"

from data.splitter import trajectory_split
from data.preprocessing import compute_norm_stats, normalize


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _nftsf_multisim_to_trajectories(arr: np.ndarray) -> np.ndarray:
    """
    Convert NFTSF multi_sim format (T, 1+NumSims) → (NumSims, T) float32.
    Column 0 of the .npy is the time index and is dropped.
    """
    # arr shape: (T, 1+NumSims)
    positions = arr[:, 1:].T.astype(np.float32)   # (NumSims, T)
    return positions


def _make_time_axis(T: int) -> np.ndarray:
    return np.arange(T, dtype=np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def from_nftsf_npy(
    train_npy: Path,
    test_npy: Path,
    landscape: str,
    n_past: int = 100,
    n_future: int = 100,
    seed: int = 42,
    out_path: Optional[Path] = None,
) -> Path:
    """
    Build a canonical ``.npz`` from NFTSF-format train and test ``.npy`` files.

    The NFTSF pipeline already performs the trajectory-level 80 / 20 split
    (via splice_alanine.py) before saving the ``.npy`` files.  This function
    reassembles both halves, records the original train / test index
    assignments, normalises using train-set statistics, and writes the
    canonical ``.npz``.

    Parameters
    ----------
    train_npy  : Path to ``*_train.npy`` in multi_sim format (T, 1+NumSims).
    test_npy   : Path to ``*_test.npy``  in multi_sim format (T, 1+NumSims).
    landscape  : String identifier, e.g. ``"alanine_phi"``.
    n_past     : Context window length (stored as metadata).
    n_future   : Forecast horizon (stored as metadata).
    seed       : Seed used during the original split (stored as metadata).
    out_path   : Destination ``.npz`` path.  Defaults to
                 ``outputs/canonical/{landscape}.npz`` relative to the
                 project root.

    Returns
    -------
    Path to the written ``.npz`` file.
    """
    train_raw = np.load(train_npy)   # (T, 1+N_train)
    test_raw  = np.load(test_npy)    # (T, 1+N_test)

    train_traj = _nftsf_multisim_to_trajectories(train_raw)   # (N_train, T)
    test_traj  = _nftsf_multisim_to_trajectories(test_raw)    # (N_test,  T)

    N_train, T = train_traj.shape
    N_test      = test_traj.shape[0]
    N           = N_train + N_test

    # Concatenate; first N_train rows are train, remaining are test.
    positions_raw = np.concatenate([train_traj, test_traj], axis=0)  # (N, T)

    train_indices = np.arange(N_train, dtype=np.int64)
    test_indices  = np.arange(N_train, N, dtype=np.int64)

    # Normalise using train-set statistics only.
    norm_mean, norm_std = compute_norm_stats(positions_raw[train_indices])
    positions_norm = normalize(positions_raw, norm_mean, norm_std)

    time = _make_time_axis(T)

    if out_path is None:
        out_path = Path("outputs") / "canonical" / f"{landscape}.npz"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_path,
        positions=positions_norm.astype(np.float32),
        positions_raw=positions_raw.astype(np.float32),
        train_indices=train_indices,
        test_indices=test_indices,
        time=time,
        landscape=np.array(landscape),
        seed=np.array(seed),
        n_past=np.array(n_past),
        n_future=np.array(n_future),
        norm_mean=np.array(norm_mean, dtype=np.float32),
        norm_std=np.array(norm_std, dtype=np.float32),
    )
    print(f"[canonical] Wrote {N} trajectories ({N_train} train / {N_test} test), "
          f"T={T} to {out_path}")
    return out_path


def from_npz_trajectories(
    npz_path: Path,
    landscape: str,
    positions_key: str = "positions",
    n_past: int = 100,
    n_future: int = 100,
    train_fraction: float = 0.80,
    seed: int = 42,
    out_path: Optional[Path] = None,
) -> Path:
    """
    Build a canonical ``.npz`` from a raw ``(N, T)`` ``.npz`` file
    (e.g. produced by tsf_models-adam's generator.py).

    A fresh trajectory-level 80 / 20 split is performed with the
    canonical seed, so the split is reproducible and identical to what
    the NFTSF adapter would use if given the same data.

    Parameters
    ----------
    npz_path       : Input ``.npz`` containing the trajectories array.
    landscape      : String identifier, e.g. ``"double_well"``.
    positions_key  : Key inside the ``.npz`` that holds the (N, T) array.
    n_past         : Context window length.
    n_future       : Forecast horizon.
    train_fraction : Fraction of trajectories for training (default 0.80).
    seed           : RNG seed for the trajectory split.
    out_path       : Destination ``.npz``.

    Returns
    -------
    Path to the written ``.npz`` file.
    """
    raw = np.load(npz_path)
    positions_raw = raw[positions_key].astype(np.float32)   # (N, T)
    N, T = positions_raw.shape

    train_indices, test_indices = trajectory_split(N, train_fraction, seed)

    norm_mean, norm_std = compute_norm_stats(positions_raw[train_indices])
    positions_norm = normalize(positions_raw, norm_mean, norm_std)

    time = _make_time_axis(T)

    if out_path is None:
        out_path = Path("outputs") / "canonical" / f"{landscape}.npz"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_path,
        positions=positions_norm.astype(np.float32),
        positions_raw=positions_raw.astype(np.float32),
        train_indices=train_indices,
        test_indices=test_indices,
        time=time,
        landscape=np.array(landscape),
        seed=np.array(seed),
        n_past=np.array(n_past),
        n_future=np.array(n_future),
        norm_mean=np.array(norm_mean, dtype=np.float32),
        norm_std=np.array(norm_std, dtype=np.float32),
    )
    N_train = len(train_indices)
    N_test  = len(test_indices)
    print(f"[canonical] Wrote {N} trajectories ({N_train} train / {N_test} test), "
          f"T={T} to {out_path}")
    return out_path


def load_canonical(npz_path: Path) -> dict:
    """
    Load a canonical ``.npz`` and return its contents as a plain dict.

    String scalars stored by NumPy (landscape, etc.) are decoded to ``str``.
    Scalar arrays (seed, n_past, …) are cast to Python ints / floats.
    """
    data = np.load(npz_path, allow_pickle=True)
    out: dict = {}
    for k in data.files:
        v = data[k]
        if v.ndim == 0:
            item = v.item()
            if isinstance(item, bytes):
                item = item.decode()
            out[k] = item
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build canonical .npz dataset.")
    p.add_argument("--source", required=True, choices=["nftsf", "npz"],
                   help="'nftsf': pair of NFTSF .npy files; 'npz': raw (N,T) .npz")
    # nftsf source
    p.add_argument("--train",     help="Path to *_train.npy  (for --source nftsf)")
    p.add_argument("--test",      help="Path to *_test.npy   (for --source nftsf)")
    # npz source
    p.add_argument("--input",     help="Path to raw .npz      (for --source npz)")
    p.add_argument("--positions_key", default="positions")
    # shared
    p.add_argument("--landscape", required=True)
    p.add_argument("--out",       required=True, help="Output .npz path")
    p.add_argument("--n_past",    type=int, default=100)
    p.add_argument("--n_future",  type=int, default=100)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--train_fraction", type=float, default=0.80)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.source == "nftsf":
        if not args.train or not args.test:
            sys.exit("--train and --test are required for --source nftsf")
        from_nftsf_npy(
            train_npy=Path(args.train),
            test_npy=Path(args.test),
            landscape=args.landscape,
            n_past=args.n_past,
            n_future=args.n_future,
            seed=args.seed,
            out_path=Path(args.out),
        )
    else:
        if not args.input:
            sys.exit("--input is required for --source npz")
        from_npz_trajectories(
            npz_path=Path(args.input),
            landscape=args.landscape,
            positions_key=args.positions_key,
            n_past=args.n_past,
            n_future=args.n_future,
            train_fraction=args.train_fraction,
            seed=args.seed,
            out_path=Path(args.out),
        )
