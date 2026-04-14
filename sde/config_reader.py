"""
sde/config_reader.py
====================
Reads parameter CSV files for SDE data generation.

This file is an **offline data-generation tool only** — it is NOT imported at
pipeline runtime.

CSV format
----------
Config files live at ``../Data/Configs/<model_key>.csv`` relative to this
script (i.e. two directories up from sde/).

sw_sle_em.csv example
~~~~~~~~~~~~~~~~~~~~~~
    k,gamma,kT,m,dt,n_steps,seed
    1.0,1.0,1.0,1.0,0.01,100000,42
    2.0,0.5,0.8,1.0,0.01,100000,123

sw_gle_oe_em.csv example
~~~~~~~~~~~~~~~~~~~~~~~~~
    k,gamma,kT,tau,m,dt,n_steps,seed
    1.0,1.0,1.0,0.5,1.0,0.01,100000,42
    1.0,2.0,1.0,1.0,1.0,0.01,100000,99

Each row is a distinct parameter set identified by its 1-based row index
(``test_id``).  Row 1 is the canonical data source.

Double-well extension
---------------------
To add double-well configs, create ``dw_sle_em.csv`` and ``dw_gle_oe_em.csv``
at ``../Data/Configs/`` with the appropriate parameter columns and append
the corresponding keys to ``SUPPORTED_MODEL_KEYS`` below.
"""

from __future__ import annotations

import csv
from pathlib import Path


# Config directory: ../Data/Configs/ relative to this script's parent (sde/).
_CONFIGS_DIR = Path(__file__).resolve().parents[1].parent / "Data" / "Configs"

# Supported model keys — extend this list to add new SDE variants.
SUPPORTED_MODEL_KEYS = [
    "sw_sle_em",
    "sw_gle_oe_em",
    # "dw_sle_em",     # double-well, no memory  — PENDING
    # "dw_gle_oe_em",  # double-well, with memory — PENDING
]

# Expected CSV columns per model key.
_REQUIRED_COLUMNS: dict[str, list[str]] = {
    "sw_sle_em":    ["k", "gamma", "kT", "m", "dt", "n_steps", "seed"],
    "sw_gle_oe_em": ["k", "gamma", "kT", "tau", "m", "dt", "n_steps", "seed"],
    # "dw_sle_em":    ["k", "gamma", "kT", "m", "dt", "n_steps", "seed"],
    # "dw_gle_oe_em": ["k", "gamma", "kT", "tau", "m", "dt", "n_steps", "seed"],
}


def load_config(
    model_key: str,
    test_id: int,
    configs_dir: str | Path | None = None,
) -> dict:
    """
    Load one row from the model-specific config CSV.

    Parameters
    ----------
    model_key   : One of SUPPORTED_MODEL_KEYS.
    test_id     : 1-based row index into the CSV (row 1 = canonical config).
    configs_dir : Override the default config directory.  Useful for testing.

    Returns
    -------
    dict mapping column names (str) to values (float or int as appropriate).

    Raises
    ------
    ValueError
        If model_key is not in SUPPORTED_MODEL_KEYS.
    FileNotFoundError
        If the config CSV does not exist.
    IndexError
        If test_id exceeds the number of data rows in the CSV.
    """
    if model_key not in SUPPORTED_MODEL_KEYS:
        raise ValueError(
            f"Unknown model key '{model_key}'.  "
            f"Supported: {SUPPORTED_MODEL_KEYS}"
        )

    cfg_dir  = Path(configs_dir) if configs_dir else _CONFIGS_DIR
    csv_path = cfg_dir / f"{model_key}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Config CSV not found: {csv_path.resolve()}\n"
            f"Create '{model_key}.csv' at {cfg_dir.resolve()} with columns: "
            f"{_REQUIRED_COLUMNS[model_key]}"
        )

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows   = list(reader)

    if test_id < 1 or test_id > len(rows):
        raise IndexError(
            f"test_id={test_id} out of range for '{csv_path}' "
            f"(has {len(rows)} data row(s))."
        )

    row = rows[test_id - 1]   # 1-based → 0-based

    # Cast values: n_steps and seed are int; everything else is float.
    int_keys = {"n_steps", "seed"}
    return {
        k: int(float(v)) if k in int_keys else float(v)
        for k, v in row.items()
        if k and k.strip()
    }


def list_configs(
    model_key: str,
    configs_dir: str | Path | None = None,
) -> list[dict]:
    """
    Return all parameter rows for *model_key* as a list of dicts.

    Useful for batch job submission (e.g. SLURM array jobs).
    """
    if model_key not in SUPPORTED_MODEL_KEYS:
        raise ValueError(f"Unknown model key '{model_key}'.  "
                         f"Supported: {SUPPORTED_MODEL_KEYS}")

    cfg_dir  = Path(configs_dir) if configs_dir else _CONFIGS_DIR
    csv_path = cfg_dir / f"{model_key}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Config CSV not found: {csv_path.resolve()}")

    int_keys = {"n_steps", "seed"}
    results  = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                k: int(float(v)) if k in int_keys else float(v)
                for k, v in row.items()
                if k and k.strip()
            })
    return results
