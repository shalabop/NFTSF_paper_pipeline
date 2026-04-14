"""
data/sde_loader.py
==================
SDE trajectory data loader and train/val/test splitter for the NFTSF pipeline.

Loaded datasets contain paired position (x) and velocity (v) trajectory arrays
produced by the offline SDE integrators (see sde/sde_integrators.py).

Each dataset is stored as a pair of .npy files:
    <model_key>_<test_id>_<skip>_x.npy  —  shape (num_traj, num_timesteps)
    <model_key>_<test_id>_<skip>_v.npy  —  shape (num_traj, num_timesteps)

The canonical data source is test_id=1, skip=10, yielding files such as:
    sw_sle_em_1_10_x.npy   (3000, 1000)
    sw_sle_em_1_10_v.npy   (3000, 1000)
    sw_gle_oe_em_1_10_x.npy
    sw_gle_oe_em_1_10_v.npy

Extensibility
-------------
To add a double-well variant or any new SDE dataset:
    1. Append the new key to VALID_SDE_MODEL_KEYS below.
    2. Ensure the corresponding integrator function exists in
       sde/sde_integrators.py and the config CSV is present at
       ../Data/Configs/<key>.csv relative to the sde/ directory.
    3. No other changes are required in this module.

Pipeline integration summary
-----------------------------
Model-name dispatcher sets: _SDE_MODELS in train/train.py and eval/evaluate.py.
Entry points that import from this module:
    - run_full_pipeline.py  (run_full_pipeline)
    - train/train.py        (train_model)
    - eval/evaluate.py      (test_model)
    - compare_models.py     (compare_models)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Registry of valid SDE model keys.
# To add double-well support, append the new key here and ensure
# the corresponding config CSV and integrator function exist.
# ---------------------------------------------------------------------------
VALID_SDE_MODEL_KEYS: list[str] = [
    'sw_sle_em',       # single-well, no memory (Langevin)
    'sw_gle_oe_em',    # single-well, with memory (GLE)
    # 'dw_sle_em',     # double-well, no memory  — PENDING
    # 'dw_gle_oe_em',  # double-well, with memory — PENDING
]


def load_sde_dataset(
    model_key: str,
    data_dir: str,
    test_id: int = 1,
    skip: int = 10,
) -> dict:
    """
    Loads x and v trajectory arrays for a given SDE model.

    Parameters
    ----------
    model_key : str
        One of the keys in VALID_SDE_MODEL_KEYS.
        (Extensible — double-well keys will be added later by appending to
        VALID_SDE_MODEL_KEYS and providing the corresponding integrator.)
    data_dir : str
        Path to the directory containing the .npy trajectory files.
    test_id : int
        The parameter set ID used when generating the data (default: 1).
    skip : int
        The skip value used when generating the data (default: 10).

    Returns
    -------
    dict with keys:
        'x'         : np.ndarray, shape (num_traj, num_timesteps) — positions
        'v'         : np.ndarray, shape (num_traj, num_timesteps) — velocities
        'model_key' : str — echoed back for downstream labeling
        'has_memory': bool — True for GLE variants ('gle' in model_key),
                      False otherwise

    Raises
    ------
    ValueError
        If model_key is not in VALID_SDE_MODEL_KEYS.
    FileNotFoundError
        If either .npy file is missing; the full attempted path is included
        in the error message.
    """
    if model_key not in VALID_SDE_MODEL_KEYS:
        raise ValueError(
            f"Unknown SDE model key '{model_key}'.  "
            f"Valid keys: {VALID_SDE_MODEL_KEYS}.  "
            "To add a new variant, append it to VALID_SDE_MODEL_KEYS in "
            "data/sde_loader.py and ensure the corresponding integrator "
            "function and config CSV exist."
        )

    data_dir = Path(data_dir)
    x_name = f"{model_key}_{test_id}_{skip}_x.npy"
    v_name = f"{model_key}_{test_id}_{skip}_v.npy"
    x_path = data_dir / x_name
    v_path = data_dir / v_name

    if not x_path.exists():
        raise FileNotFoundError(
            f"Position trajectory file not found: {x_path.resolve()}\n"
            f"Expected '{x_name}' in directory: {data_dir.resolve()}"
        )
    if not v_path.exists():
        raise FileNotFoundError(
            f"Velocity trajectory file not found: {v_path.resolve()}\n"
            f"Expected '{v_name}' in directory: {data_dir.resolve()}"
        )

    x = np.load(x_path)
    v = np.load(v_path)

    return {
        'x':          x,
        'v':          v,
        'model_key':  model_key,
        'has_memory': 'gle' in model_key,
    }


def split_sde_dataset(
    dataset: dict,
    outer_test_size: float = 0.2,
    val_size: float = 0.1,
    random_seed: int = 42,
) -> dict:
    """
    Splits trajectory data into train, validation, and test sets at the
    trajectory (row) level.

    Split strategy:
        1. Outer split  : (1 - outer_test_size) train+val  /  outer_test_size test
        2. Inner split  : (1 - val_size) train  /  val_size validation
                          (applied only to the train+val portion from step 1)

    Final proportions with defaults (outer_test_size=0.2, val_size=0.1):
        - train : 72 %   (0.8 × 0.9 of total)
        - val   :  8 %   (0.8 × 0.1 of total)
        - test  : 20 %

    For the canonical 3 000-trajectory dataset:
        - train : 2 160 trajectories
        - val   :   240 trajectories
        - test  :   600 trajectories

    Both x and v arrays are split with identical index assignments (the same
    index array is used for both, guaranteeing alignment).

    Parameters
    ----------
    dataset : dict
        Output of load_sde_dataset().
    outer_test_size : float
        Fraction of all trajectories held out as the test set.
    val_size : float
        Fraction of the train+val portion reserved for validation.
    random_seed : int
        Seed for sklearn.model_selection.train_test_split (reproducibility).

    Returns
    -------
    dict with keys:
        'x_train', 'v_train'  : np.ndarray — 72 % of trajectories
        'x_val',   'v_val'    : np.ndarray —  8 % of trajectories
        'x_test',  'v_test'   : np.ndarray — 20 % of trajectories
        'model_key'           : str
        'has_memory'          : bool
    """
    x = dataset['x']
    v = dataset['v']
    n_traj = x.shape[0]

    # Single index array — both x and v receive identical index assignments.
    indices = np.arange(n_traj)

    # Step 1: outer split  (80 % train+val  /  20 % test)
    idx_trainval, idx_test = train_test_split(
        indices,
        test_size=outer_test_size,
        random_state=random_seed,
        shuffle=True,
    )

    # Step 2: inner split  (90 % train  /  10 % val, within the 80 % portion)
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=val_size,
        random_state=random_seed,
        shuffle=True,
    )

    return {
        'x_train':    x[idx_train],
        'v_train':    v[idx_train],
        'x_val':      x[idx_val],
        'v_val':      v[idx_val],
        'x_test':     x[idx_test],
        'v_test':     v[idx_test],
        'model_key':  dataset['model_key'],
        'has_memory': dataset['has_memory'],
    }
