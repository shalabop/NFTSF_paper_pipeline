"""
sde_data_gen.py
===============
CLI script for offline generation of SDE trajectory datasets.

This script reads physical parameters from a CSV config file, runs the
appropriate SDE integrator, and writes two .npy files:

    <model_key>_<test_id>_<skip>_x.npy   — positions,  shape (num_traj, T)
    <model_key>_<test_id>_<skip>_v.npy   — velocities, shape (num_traj, T)

Output landing directory:  ../Data/Trajectories/  (relative to this script)
Config CSV directory:       ../Data/Configs/       (relative to this script)

Usage
-----
    python sde_data_gen.py <model_type> <test_id> <num_traj> <skip>

    model_type : one of  sw_sle_em | sw_gle_oe_em
    test_id    : integer — selects the parameter row in the CSV config file
    num_traj   : number of independent trajectories to simulate
    skip       : record one state every this many integration steps
                 (skip=10 → 10x subsampling of the raw integration grid)

Examples
--------
    python sde_data_gen.py sw_sle_em   1 3000 10
    python sde_data_gen.py sw_gle_oe_em 1 3000 10

Config CSV location
-------------------
The config file for model_type must be present at:
    ../Data/Configs/<model_type>.csv

Each row corresponds to one parameter set identified by the 'id' column.
See config_reader.py for the expected CSV format.

Output files
------------
Written to ../Data/Trajectories/ relative to this script:
    sw_sle_em_1_10_x.npy    positions
    sw_sle_em_1_10_v.npy    velocities

These files are the canonical input for the NFTSF pipeline when
--data_format sde is specified.

IMPORTANT: Do NOT call this script at pipeline runtime.
           It is an offline data-generation tool only.
"""

from __future__ import annotations

import os
import sys

import numpy as np


# Integrator registry — add new model keys here when integrators are ready.
_INTEGRATOR_REGISTRY = {
    'sw_sle_em':    'sw_sle_em',
    'sw_gle_oe_em': 'sw_gle_oe_em',
    # 'dw_sle_em':    'dw_sle_em',    # PENDING
    # 'dw_gle_oe_em': 'dw_gle_oe_em', # PENDING
}

# Derived path helpers (relative to this script's location)
_HERE        = os.path.dirname(os.path.abspath(__file__))
_CONFIGS_DIR = os.path.join(_HERE, '..', 'Data', 'Configs')
_OUTPUT_DIR  = os.path.join(_HERE, '..', 'Data', 'Trajectories')


def _usage_and_exit():
    print(__doc__)
    sys.exit(1)


def generate(model_key: str, test_id: int, num_traj: int, skip: int) -> None:
    """Run the integrator and save output files."""
    if model_key not in _INTEGRATOR_REGISTRY:
        valid = ', '.join(_INTEGRATOR_REGISTRY.keys())
        print(f"ERROR: Unknown model_type '{model_key}'.  Valid options: {valid}")
        sys.exit(1)

    from config_reader import load_config
    from sde_integrators import sw_sle_em, sw_gle_oe_em

    print(f"\n{'='*60}")
    print(f"SDE Data Generation")
    print(f"  model_key : {model_key}")
    print(f"  test_id   : {test_id}")
    print(f"  num_traj  : {num_traj}")
    print(f"  skip      : {skip}")
    print(f"{'='*60}")

    # Load parameters from CSV
    cfg = load_config(model_key, test_id, configs_dir=_CONFIGS_DIR)
    print(f"\nLoaded config (id={test_id}):")
    for k, v in cfg.items():
        print(f"  {k} = {v}")

    # Dispatch to the correct integrator
    if model_key == 'sw_sle_em':
        positions, velocities = sw_sle_em(
            total_time = float(cfg['total_time']),
            time_step  = float(cfg['time_step']),
            x0_mean    = float(cfg['x0_mean']),
            v0_mean    = float(cfg['v0_mean']),
            zeta       = float(cfg['zeta']),
            m          = float(cfg['m']),
            k          = float(cfg['k']),
            x_mu       = float(cfg['x_mu']),
            T          = float(cfg['T']),
            boltzmann  = float(cfg['boltzmann']),
            num_traj   = num_traj,
            x0_std     = float(cfg.get('x0_std', 0.0)),
            v0_std     = float(cfg.get('v0_std', 0.0)),
            skip       = skip,
            seed       = int(cfg.get('seed', test_id)),
        )
    elif model_key == 'sw_gle_oe_em':
        positions, velocities = sw_gle_oe_em(
            total_time = float(cfg['total_time']),
            time_step  = float(cfg['time_step']),
            x0_mean    = float(cfg['x0_mean']),
            v0_mean    = float(cfg['v0_mean']),
            zeta       = float(cfg['zeta']),
            m          = float(cfg['m']),
            k          = float(cfg['k']),
            x_mu       = float(cfg['x_mu']),
            T          = float(cfg['T']),
            boltzmann  = float(cfg['boltzmann']),
            tau_mem    = float(cfg['tau_mem']),
            sigma_mem  = float(cfg['sigma_mem']),
            num_traj   = num_traj,
            x0_std     = float(cfg.get('x0_std', 0.0)),
            v0_std     = float(cfg.get('v0_std', 0.0)),
            skip       = skip,
            seed       = int(cfg.get('seed', test_id)),
        )
    else:
        raise NotImplementedError(f"No dispatch branch for model_key={model_key!r}")

    print(f"\nGenerated trajectories:")
    print(f"  positions  shape: {positions.shape}")
    print(f"  velocities shape: {velocities.shape}")

    # Save outputs
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    x_fname = f"{model_key}_{test_id}_{skip}_x.npy"
    v_fname = f"{model_key}_{test_id}_{skip}_v.npy"
    x_path  = os.path.join(_OUTPUT_DIR, x_fname)
    v_path  = os.path.join(_OUTPUT_DIR, v_fname)

    np.save(x_path, positions)
    np.save(v_path, velocities)

    print(f"\nSaved:")
    print(f"  {x_path}")
    print(f"  {v_path}")
    print(f"\nDone.")


def main():
    if len(sys.argv) != 5:
        print("ERROR: Expected exactly 4 positional arguments.")
        _usage_and_exit()

    _, model_key, test_id_str, num_traj_str, skip_str = sys.argv

    try:
        test_id  = int(test_id_str)
        num_traj = int(num_traj_str)
        skip     = int(skip_str)
    except ValueError as exc:
        print(f"ERROR: test_id, num_traj, and skip must be integers. ({exc})")
        sys.exit(1)

    generate(model_key, test_id, num_traj, skip)


if __name__ == '__main__':
    main()
