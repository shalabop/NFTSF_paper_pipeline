"""
sde_loader.py
=============
Data loading and splitting utilities for SDE trajectory datasets.

Pipeline context
----------------
This module is called by the four main pipeline entry points
(run_full_pipeline.sh, train_model.py, test_model.py, compare_models.py)
whenever ``data_format == 'sde'`` is specified.

SDE datasets consist of two paired .npy files:
    <model_key>_<test_id>_<skip>_x.npy  — positions, shape (num_traj, num_timesteps)
    <model_key>_<test_id>_<skip>_v.npy  — velocities, same shape

The canonical data source is ID=1, skip=10.  Files are generated offline
by sde/sde_data_gen.py and live in ../Data/Trajectories/ relative to that
script (default).

Extensibility
-------------
To add a new SDE model (e.g. double-well), append its key to
VALID_SDE_MODEL_KEYS and ensure the corresponding config CSV and integrator
function exist in the sde/ directory.
"""

from __future__ import annotations

import os
from typing import Optional

import torch  # must precede numpy: torch._C calls import_array() eagerly, which
              # primes numpy's C-API so that sklearn's later import_array() is a
              # no-op rather than a conflicting double-claim under NumPy 2.x.
import numpy as np
# sklearn is NOT imported at module level — its C-extensions call import_array()
# which conflicts with torch's own import_array() call under NumPy 2.x, causing
# "_ARRAY_API not found".  Importing inside the function avoids the double-claim.

# ---------------------------------------------------------------------------
# Registry of valid SDE model keys.
# To add double-well support, append the new key here and ensure
# the corresponding config CSV and integrator function exist.
# ---------------------------------------------------------------------------
VALID_SDE_MODEL_KEYS = [
    'sw_sle_em',       # single-well, no memory (Langevin / SLE)
    'sw_gle_oe_em',    # single-well, with memory (GLE / coloured noise)
    'dw_sle_em',       # double-well, no memory
    'dw_gle_oe_em',    # double-well, with memory
]


def load_sde_dataset(
    model_key: str,
    data_dir: str,
    test_id: int = 1,
    skip: int = 10,
) -> dict:
    """
    Load x and v trajectory arrays for a given SDE model.

    Supports two formats:
    1. Raw format: <model_key>_<test_id>_<skip>_x.npy, <model_key>_<test_id>_<skip>_v.npy
    2. Split format: <model_key>_x_train.npy, <model_key>_x_val.npy, <model_key>_x_test.npy, etc.

    Parameters
    ----------
    model_key : str
        One of VALID_SDE_MODEL_KEYS.
    data_dir : str
        Directory containing the .npy trajectory files (raw or split format).
    test_id : int
        Parameter-set ID used when generating the data (default: 1).
        Only used for raw format; ignored for split format.
    skip : int
        Skip value used when generating the data (default: 10).
        Only used for raw format; ignored for split format.

    Returns
    -------
    dict with either raw or split format (check 'is_split' key):

    Raw format (is_split=False):
        'x'         : np.ndarray, shape (num_traj, num_timesteps) — positions
        'v'         : np.ndarray OR None — velocities (None if file not found)
        'model_key' : str — echoed back for downstream labelling
        'has_memory': bool — True for GLE variants (False if v is None)
        'is_split'  : False

    Split format (is_split=True):
        'x_train'   : np.ndarray — training positions
        'x_val'     : np.ndarray OR None — validation positions (None if missing)
        'x_test'    : np.ndarray — test positions
        'v_train'   : np.ndarray OR None — training velocities (None if missing)
        'v_val'     : np.ndarray OR None — validation velocities (None if missing)
        'v_test'    : np.ndarray OR None — test velocities (None if missing)
        'model_key' : str — echoed back for downstream labelling
        'has_memory': bool — True for GLE variants (False if any v is None)
        'is_split'  : True
    """
    if model_key not in VALID_SDE_MODEL_KEYS:
        valid = ', '.join(repr(k) for k in VALID_SDE_MODEL_KEYS)
        raise ValueError(
            f"Unknown SDE model key {model_key!r}.  "
            f"Valid keys: [{valid}].  "
            f"To add a new key, append it to VALID_SDE_MODEL_KEYS in sde_loader.py "
            f"and ensure the corresponding integrator and config exist."
        )

    # Check if split format exists
    split_x_train_path = os.path.join(data_dir, f"{model_key}_x_train.npy")
    if os.path.exists(split_x_train_path):
        # Load split format
        print(f"[INFO] Loading split-format SDE dataset for {model_key}")

        split_x_test_path = os.path.join(data_dir, f"{model_key}_x_test.npy")
        split_x_val_path = os.path.join(data_dir, f"{model_key}_x_val.npy")
        split_v_train_path = os.path.join(data_dir, f"{model_key}_v_train.npy")
        split_v_test_path = os.path.join(data_dir, f"{model_key}_v_test.npy")
        split_v_val_path = os.path.join(data_dir, f"{model_key}_v_val.npy")

        if not os.path.exists(split_x_test_path):
            raise FileNotFoundError(
                f"SDE split test file not found: {split_x_test_path}\n"
                f"  Expected: {model_key}_x_test.npy in {data_dir}\n"
                f"  Generate with: python sde_preprocess.py --model_key {model_key} --data_dir <raw_dir> --output_dir {data_dir}"
            )

        x_train = np.load(split_x_train_path).astype(np.float32)
        x_test = np.load(split_x_test_path).astype(np.float32)
        x_val = np.load(split_x_val_path).astype(np.float32) if os.path.exists(split_x_val_path) else None

        v_train = np.load(split_v_train_path).astype(np.float32) if os.path.exists(split_v_train_path) else None
        v_test = np.load(split_v_test_path).astype(np.float32) if os.path.exists(split_v_test_path) else None
        v_val = np.load(split_v_val_path).astype(np.float32) if os.path.exists(split_v_val_path) else None

        has_memory = 'gle' in model_key

        return {
            'x_train':    x_train,
            'x_val':      x_val,
            'x_test':     x_test,
            'v_train':    v_train,
            'v_val':      v_val,
            'v_test':     v_test,
            'model_key':  model_key,
            'has_memory': has_memory,
            'is_split':   True,
        }

    # Load raw format
    print(f"[INFO] Loading raw-format SDE dataset for {model_key}")

    x_fname = f"{model_key}_{test_id}_{skip}_x.npy"
    v_fname = f"{model_key}_{test_id}_{skip}_v.npy"
    x_path  = os.path.join(data_dir, x_fname)
    v_path  = os.path.join(data_dir, v_fname)

    if not os.path.exists(x_path):
        raise FileNotFoundError(
            f"SDE position file not found: {x_path}\n"
            f"  Expected: {x_fname} in {data_dir}\n"
            f"  Generate it with: python sde/sde_data_gen.py {model_key} {test_id} <num_traj> {skip}"
        )

    x = np.load(x_path).astype(np.float32)

    v = None
    if os.path.exists(v_path):
        v = np.load(v_path).astype(np.float32)
    else:
        print(f"[WARNING] Velocity file not found — proceeding with x-only dataset")

    has_memory = 'gle' in model_key

    return {
        'x':          x,
        'v':          v,
        'model_key':  model_key,
        'has_memory': has_memory,
        'is_split':   False,
    }


def split_sde_dataset(
    dataset: dict,
    outer_test_size: float = 0.2,
    val_size: float = 0.1,
    random_seed: int = 42,
) -> dict:
    """
    Split trajectory data into train, validation, and test sets at the
    trajectory level (i.e. split rows of x and v, not timestep columns).

    Split strategy
    --------------
    1. Outer split : 80% train+val  /  20% test
    2. Inner split : 90% train      /  10% validation  (of the 80% pool)

    Final proportions of the full dataset:
        train :  72%   (0.80 * 0.90)
        val   :   8%   (0.80 * 0.10)
        test  :  20%

    Parameters
    ----------
    dataset : dict
        Output of load_sde_dataset().
    outer_test_size : float
        Fraction held out as test set (default: 0.2).
    val_size : float
        Fraction of the train+val pool used for validation (default: 0.1).
    random_seed : int
        Seed for reproducibility (default: 42).

    Returns
    -------
    dict with keys:
        'x_train', 'v_train'  : np.ndarray or None — 72% of trajectories
        'x_val',   'v_val'    : np.ndarray or None —  8% of trajectories
        'x_test',  'v_test'   : np.ndarray or None — 20% of trajectories
        'model_key'           : str
        'has_memory'          : bool
    """
    # Lazy import: avoids loading sklearn C-extensions at module import time,
    # which would double-claim numpy's C-API and crash torch under NumPy 2.x.
    from sklearn.model_selection import train_test_split  # noqa: PLC0415

    x = dataset['x']
    v = dataset['v']
    n_total = x.shape[0]

    # Build a shared index array so x and v receive identical shuffles.
    indices = np.arange(n_total)

    # 1. Outer split: 80% train+val / 20% test
    idx_trainval, idx_test = train_test_split(
        indices,
        test_size=outer_test_size,
        random_state=random_seed,
        shuffle=True,
    )

    # 2. Inner split: 90% train / 10% val  (applied to the 80% pool)
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=val_size,
        random_state=random_seed,
        shuffle=True,
    )

    return {
        'x_train':    x[idx_train],
        'v_train':    v[idx_train] if v is not None else None,
        'x_val':      x[idx_val],
        'v_val':      v[idx_val] if v is not None else None,
        'x_test':     x[idx_test],
        'v_test':     v[idx_test] if v is not None else None,
        'model_key':  dataset['model_key'],
        'has_memory': dataset['has_memory'],
    }


def save_sde_split(
    split: dict,
    output_dir: str,
    include_val: bool = True,
) -> dict:
    """
    Save a split dict (output of split_sde_dataset) to .npy files on disk.

    Parameters
    ----------
    split : dict
        Output of split_sde_dataset().
    output_dir : str
        Directory where files are written (created if it does not exist).
    include_val : bool
        If True (default), also save x_val.npy and v_val.npy.
        Set False if you only need train + test for export.

    Returns
    -------
    dict mapping split name → saved file path, e.g.:
        {'x_train': '/out/sw_sle_em_x_train.npy', ...}

    File naming convention
    ----------------------
    <model_key>_x_train.npy
    <model_key>_v_train.npy    (only if v_train is not None)
    <model_key>_x_val.npy      (only when include_val=True)
    <model_key>_v_val.npy      (only when include_val=True and v_val is not None)
    <model_key>_x_test.npy
    <model_key>_v_test.npy     (only if v_test is not None)
    """
    os.makedirs(output_dir, exist_ok=True)
    key = split['model_key']

    saves = {
        'x_train': split['x_train'],
        'x_test':  split['x_test'],
    }
    # Only include v_* if they exist (are not None)
    if split['v_train'] is not None:
        saves['v_train'] = split['v_train']
    if split['v_test'] is not None:
        saves['v_test'] = split['v_test']
    if include_val:
        saves['x_val'] = split['x_val']
        if split['v_val'] is not None:
            saves['v_val'] = split['v_val']

    paths = {}
    for name, arr in saves.items():
        fname = f"{key}_{name}.npy"
        fpath = os.path.join(output_dir, fname)
        np.save(fpath, arr)
        paths[name] = fpath
        print(f"  Saved {name:8s} {str(arr.shape):>14}  →  {fpath}")

    return paths
