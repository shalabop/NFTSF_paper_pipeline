#!/usr/bin/env python3
"""
scripts/run_full_pipeline_paper_clean.py
==========================================
Clean orchestration entry point for paper experiments.

Wires together: configure → test_model_paper_clean (per method/dataset) →
compare_models_paper_clean (aggregate).

Configuration is fully explicit: either a JSON config file (--config_file)
or CLI flags.  No hidden globals.  Seed is set and logged before anything runs.

Use --dry_run to validate wiring without executing any inference.

Usage — full run
----------------
python scripts/run_full_pipeline_paper_clean.py \\
    --config_file configs/paper_pipeline_clean.json \\
    --output_dir  outputs/paper_clean/

Usage — dry run (validates config only)
----------------------------------------
python scripts/run_full_pipeline_paper_clean.py \\
    --config_file configs/paper_pipeline_clean.json \\
    --dry_run

Usage — single method inline
-----------------------------
python scripts/run_full_pipeline_paper_clean.py \\
    --methods   '[{"method":"nftsf","dataset":"single_well","checkpoint":"outputs/nftsf/single_well/run_001","data":"outputs/canonical/single_well.npz"}]' \\
    --n_samples 500 \\
    --seed      42 \\
    --output_dir outputs/paper_clean/

Config file format (JSON)
--------------------------
{
  "seed": 42,
  "n_samples": 500,
  "chunk_size": 32,
  "output_dir": "outputs/paper_clean",
  "methods": [
    {
      "method":     "nftsf",
      "dataset":    "single_well",
      "system":     "sw_sle_em",
      "checkpoint": "outputs/nftsf/single_well/run_001",
      "data":       "outputs/canonical/single_well.npz",
      "notes":      null
    }
  ]
}
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Seeding (mirrors test_model_paper_clean)
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

def _resolve_config(args: argparse.Namespace) -> dict:
    """Merge config file (if given) with CLI overrides."""
    cfg: dict = {}

    if args.config_file:
        with open(args.config_file) as f:
            cfg = json.load(f)

    if args.methods_json:
        cfg["methods"] = json.loads(args.methods_json)

    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.n_samples is not None:
        cfg["n_samples"] = args.n_samples
    if args.chunk_size is not None:
        cfg["chunk_size"] = args.chunk_size
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir

    # Defaults.
    cfg.setdefault("seed",       42)
    cfg.setdefault("n_samples",  500)
    cfg.setdefault("chunk_size", 32)
    cfg.setdefault("output_dir", "outputs/paper_clean")

    if "methods" not in cfg or not cfg["methods"]:
        sys.exit("[run_full_pipeline_paper_clean] No methods configured. "
                 "Provide --config_file or --methods.")

    return cfg


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _run_step(cmd: list[str], dry_run: bool) -> None:
    print(f"  CMD: {' '.join(cmd)}")
    if not dry_run:
        result = subprocess.run(cmd, check=True)


def run_pipeline(cfg: dict, dry_run: bool = False) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = cfg.get("run_id", ts)
    output_dir = Path(cfg["output_dir"]) / ts
    result_dir = output_dir / "results"

    _set_seed(cfg["seed"])

    print("\n" + "=" * 60)
    print("[run_full_pipeline_paper_clean] Resolved configuration:")
    print(json.dumps({k: v for k, v in cfg.items() if k != "methods"}, indent=2))
    print(f"  methods    : {len(cfg['methods'])} method/dataset pair(s)")
    print(f"  run_id     : {run_id}")
    print(f"  output_dir : {output_dir}")
    print(f"  dry_run    : {dry_run}")
    print("=" * 60 + "\n")

    timing_log = output_dir / "timing.jsonl"
    test_script    = str(_PROJECT_ROOT / "scripts" / "test_model_paper_clean.py")
    compare_script = str(_PROJECT_ROOT / "scripts" / "compare_models_paper_clean.py")

    # ---- Per-method evaluation ----
    for entry in cfg["methods"]:
        method  = entry["method"]
        dataset = entry["dataset"]
        ckpt    = entry["checkpoint"]
        data    = entry["data"]
        system  = entry.get("system", "null")
        notes   = entry.get("notes", "")

        print(f"[run_full_pipeline_paper_clean] === {method} / {dataset} ===")

        cmd = [
            sys.executable, test_script,
            "--checkpoint",  str(ckpt),
            "--data",        str(data),
            "--method",      method,
            "--dataset",     dataset,
            "--n_samples",   str(cfg["n_samples"]),
            "--chunk_size",  str(cfg["chunk_size"]),
            "--seed",        str(cfg["seed"]),
            "--output_dir",  str(result_dir),
            "--run_id",      run_id,
            "--timing_log",  str(timing_log),
        ]
        if system and system != "null":
            cmd += ["--system", system]
        if notes:
            cmd += ["--notes", notes]

        _run_step(cmd, dry_run)

    # ---- Aggregation ----
    print("\n[run_full_pipeline_paper_clean] === compare_models_paper_clean ===")
    compare_cmd = [
        sys.executable, compare_script,
        "--result_dir", str(result_dir),
        "--output_dir", str(output_dir),
    ]
    _run_step(compare_cmd, dry_run)

    if dry_run:
        print("\n[run_full_pipeline_paper_clean] DRY RUN complete — no inference run.")
    else:
        print(f"\n[run_full_pipeline_paper_clean] Pipeline complete."
              f"  Outputs: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Paper-clean full pipeline orchestrator."
    )
    p.add_argument("--config_file", default=None,
                   help="JSON config file (see module docstring for format).")
    p.add_argument("--methods", dest="methods_json", default=None,
                   help="JSON array of method specs (overrides config_file.methods).")
    p.add_argument("--seed",       type=int, default=None)
    p.add_argument("--n_samples",  type=int, default=None)
    p.add_argument("--chunk_size", type=int, default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--run_id",     default=None)
    p.add_argument("--dry_run",    action="store_true", default=False,
                   help="Validate config and print commands without running them.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg  = _resolve_config(args)
    if args.run_id:
        cfg["run_id"] = args.run_id
    run_pipeline(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
