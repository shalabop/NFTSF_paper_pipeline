"""
config_reader.py
================
Read parameter-set CSV files for SDE data generation.

CSV format (one row per parameter set, header required):
    id,<param1>,<param2>,...

Example (sw_gle_oe_em.csv):
    id,total_time,time_step,x0_mean,x0_std,v0_mean,v0_std,m,zeta,k,x_mu,T,boltzmann,tau_mem,sigma_mem
    1,100.0,0.001,0.0,0.5,0.0,0.5,1.0,1.0,1.0,0.0,1.0,1.0,1.0,1.0

Config files must live at ../Data/Configs/ relative to this script.
"""

from __future__ import annotations

import csv
import os
from typing import Optional


def _configs_dir() -> str:
    """Return the canonical config directory: ../Data/Configs/ relative to this file."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, '..', 'Data', 'Configs')


def load_config(model_key: str, test_id: int, configs_dir: Optional[str] = None) -> dict:
    """
    Load a single parameter row from the CSV config file for *model_key*.

    Parameters
    ----------
    model_key : str
        E.g. 'sw_sle_em' or 'sw_gle_oe_em'.  The CSV file is expected at
        ``{configs_dir}/{model_key}.csv``.
    test_id : int
        Row ID to load (matched against the 'id' column).
    configs_dir : str, optional
        Override the default config directory.

    Returns
    -------
    dict mapping column name → value (auto-cast to int/float where possible).

    Raises
    ------
    FileNotFoundError  if the CSV does not exist.
    KeyError           if no row with the given id is found.
    """
    if configs_dir is None:
        configs_dir = _configs_dir()

    csv_path = os.path.join(configs_dir, f"{model_key}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Config CSV not found: {csv_path}\n"
            f"  Create {model_key}.csv in {configs_dir} with columns matching the "
            f"integrator parameters."
        )

    with open(csv_path, newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            row_id = _cast(row.get('id', ''))
            if int(row_id) == int(test_id):
                return {k: _cast(v) for k, v in row.items()}

    raise KeyError(
        f"No row with id={test_id} found in {csv_path}.  "
        f"Available rows: check the 'id' column."
    )


def _cast(value: str):
    """Try int → float → str conversion."""
    try:
        return int(value)
    except (ValueError, TypeError):
        pass
    try:
        return float(value)
    except (ValueError, TypeError):
        pass
    return value
