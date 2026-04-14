"""
sde/sde_data_gen.py
===================
Offline CLI script for generating SDE trajectory datasets.

This script is an **offline data-generation tool only** — it is NOT called at
pipeline runtime.  Run it manually once to produce the .npy files, then use
the pipeline (train/train.py, eval/evaluate.py, run_full_pipeline.py) to
train and evaluate models on those files.

Config files
------------
Parameter CSVs must be present at ``../Data/Configs/<model_type>.csv``
relative to this script (i.e. at ``<project_root>/../Data/Configs/``).

See ``config_reader.py`` for the required CSV column format.

Output
------
Generated files are written to ``../Data/Trajectories/`` relative to this
script (i.e. ``<project_root>/../Data/Trajectories/``):

    <model_type>_<test_id>_<skip>_x.npy   — positions,  shape (num_traj, T//skip)
    <model_type>_<test_id>_<skip>_v.npy   — velocities, shape (num_traj, T//skip)

Usage
-----
    python sde_data_gen.py <model_type> <test_id> <num_traj> <skip>

Arguments
---------
    model_type : str   One of sw_sle_em | sw_gle_oe_em
                        (extend by adding rows to sde_integrators.py and
                        config_reader.SUPPORTED_MODEL_KEYS).
    test_id    : int   1-based row index in the config CSV (row 1 = canonical).
    num_traj   : int   Number of independent trajectories to generate.
    skip       : int   Subsampling factor: keep 1 frame every *skip* steps.

Examples
--------
Generate the canonical single-well, no-memory dataset (3000 trajectories):

    python sde_data_gen.py sw_sle_em 1 3000 10

Generate the canonical single-well, with-memory (GLE) dataset:

    python sde_data_gen.py sw_gle_oe_em 1 3000 10

The output files will be:
    ../Data/Trajectories/sw_sle_em_1_10_x.npy    (3000, T//10)
    ../Data/Trajectories/sw_sle_em_1_10_v.npy    (3000, T//10)
    ../Data/Trajectories/sw_gle_oe_em_1_10_x.npy (3000, T//10)
    ../Data/Trajectories/sw_gle_oe_em_1_10_v.npy (3000, T//10)

These are then loaded by the pipeline via:
    data/sde_loader.load_sde_dataset('sw_sle_em', '../Data/Trajectories/')
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Ensure sde/ is on the path for sibling imports.
_SDE_DIR = Path(__file__).resolve().parent
if str(_SDE_DIR) not in sys.path:
    sys.path.insert(0, str(_SDE_DIR))

from config_reader import load_config, SUPPORTED_MODEL_KEYS
import sde_integrators as _integ

# Default output directory.
_OUTPUT_DIR = _SDE_DIR.parent.parent / "Data" / "Trajectories"

# Registry mapping model key → integrator function.
# To add double-well support, add the new key and function here.
_INTEGRATORS: dict[str, callable] = {
    "sw_sle_em":    _integ.sw_sle_em,
    "sw_gle_oe_em": _integ.sw_gle_oe_em,
    # "dw_sle_em":    _integ.dw_sle_em,     # PENDING
    # "dw_gle_oe_em": _integ.dw_gle_oe_em,  # PENDING
}


def generate(
    model_type: str,
    test_id: int,
    num_traj: int,
    skip: int,
    output_dir: str | Path | None = None,
    configs_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """
    Generate SDE trajectories and save them as .npy files.

    Parameters
    ----------
    model_type  : SDE model key (e.g. 'sw_sle_em').
    test_id     : 1-based row index in the config CSV.
    num_traj    : Number of trajectories to generate.
    skip        : Subsampling factor.
    output_dir  : Override default output directory.
    configs_dir : Override default config directory.

    Returns
    -------
    (x_path, v_path) — Paths to the written .npy files.
    """
    if model_type not in _INTEGRATORS:
        raise ValueError(
            f"Unknown model_type '{model_type}'.  "
            f"Supported: {sorted(_INTEGRATORS.keys())}"
        )

    out_dir = Path(output_dir) if output_dir else _OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load parameters from config CSV.
    params = load_config(model_type, test_id, configs_dir=configs_dir)
    print(f"[sde_data_gen] Config for {model_type} / test_id={test_id}: {params}")

    integrator = _INTEGRATORS[model_type]
    n_steps    = int(params.pop("n_steps"))
    seed       = int(params.pop("seed"))

    print(f"[sde_data_gen] Generating {num_traj} trajectories × {n_steps} steps "
          f"(skip={skip}) with {model_type} …")
    t0 = time.time()

    x, v = integrator(
        n_traj=num_traj,
        n_steps=n_steps,
        skip=skip,
        seed=seed,
        **params,
    )

    elapsed = time.time() - t0
    print(f"[sde_data_gen] Done in {elapsed:.1f}s.  "
          f"Output shape: x={x.shape}, v={v.shape}")

    x_name = f"{model_type}_{test_id}_{skip}_x.npy"
    v_name = f"{model_type}_{test_id}_{skip}_v.npy"
    x_path = out_dir / x_name
    v_path = out_dir / v_name

    np.save(x_path, x)
    np.save(v_path, v)
    print(f"[sde_data_gen] Saved:\n  {x_path}\n  {v_path}")

    return x_path, v_path


def main() -> None:
    if len(sys.argv) != 5:
        print(
            "Usage: python sde_data_gen.py <model_type> <test_id> <num_traj> <skip>\n"
            "\n"
            "  model_type : one of " + str(sorted(_INTEGRATORS.keys())) + "\n"
            "  test_id    : 1-based row in the config CSV (1 = canonical)\n"
            "  num_traj   : number of trajectories (e.g. 3000)\n"
            "  skip       : subsampling factor     (e.g. 10)\n"
            "\n"
            "Example (canonical datasets):\n"
            "  python sde_data_gen.py sw_sle_em    1 3000 10\n"
            "  python sde_data_gen.py sw_gle_oe_em 1 3000 10\n"
            "\n"
            "Config CSV must be at ../Data/Configs/<model_type>.csv.\n"
            "Output lands at ../Data/Trajectories/<model_type>_<test_id>_<skip>_[xv].npy",
            file=sys.stderr,
        )
        sys.exit(1)

    _, model_type, test_id_s, num_traj_s, skip_s = sys.argv
    try:
        test_id  = int(test_id_s)
        num_traj = int(num_traj_s)
        skip     = int(skip_s)
    except ValueError:
        print("ERROR: test_id, num_traj, and skip must be integers.", file=sys.stderr)
        sys.exit(1)

    generate(model_type, test_id, num_traj, skip)


if __name__ == "__main__":
    main()
