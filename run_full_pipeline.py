#!/usr/bin/env python3
"""
run_full_pipeline.py
====================
Orchestrates the full **train → evaluate** pipeline for both SDE trajectory
datasets and the existing canonical (.npz) dataset format.

Dataset types
-------------
SDE types (--dataset_type sw_sle_em | sw_gle_oe_em):
    * Position and velocity trajectories are loaded from --sde_data_dir.
    * The NFTSF normalizing-flow architecture is used for all SDE types.
    * Data is split 72 / 8 / 20 % (train / val / test) at the trajectory
      level, then sliding-window segments are extracted for training.
    * The test split is evaluated immediately after training.

Canonical type (--dataset_type canonical):
    * Delegates directly to train/train.py and eval/evaluate.py using the
      standard canonical .npz format (--data must point to the .npz file).

Extensibility
-------------
To add double-well SDE support:
    1. Append 'dw_sle_em' (and/or 'dw_gle_oe_em') to _SDE_MODELS below.
    2. Do the same in train/train.py:_SDE_MODELS,
       eval/evaluate.py:_SDE_MODELS, and data/sde_loader.VALID_SDE_MODEL_KEYS.
    No other changes are required.

Usage — SDE
-----------
python run_full_pipeline.py \\
    --dataset_type sw_sle_em \\
    --sde_data_dir ../Data/Trajectories/ \\
    --config       configs/nftsf.yaml \\
    --run_id       run_001

python run_full_pipeline.py \\
    --dataset_type sw_gle_oe_em \\
    --sde_data_dir ../Data/Trajectories/ \\
    --config       configs/nftsf.yaml \\
    --n_samples    200

Usage — canonical
-----------------
python run_full_pipeline.py \\
    --dataset_type canonical \\
    --data         outputs/canonical/alanine_phi.npz \\
    --model        nftsf \\
    --config       configs/nftsf.yaml \\
    --run_id       run_001

Required arguments
------------------
--dataset_type   One of: sw_sle_em | sw_gle_oe_em | canonical
--config         Path to model-specific YAML config (e.g. configs/nftsf.yaml)

SDE-specific
------------
--sde_data_dir   Directory containing the .npy trajectory files.
                 Default: ../Data/Trajectories/ (relative to project root).

Canonical-specific
------------------
--data           Path to canonical .npz dataset file.
--model          Model name (e.g. nftsf, arima).  Default: nftsf.

Shared optional
---------------
--run_id         Unique run identifier (default: timestamp).
--base_config    Path to base.yaml (default: configs/base.yaml).
--epochs         Override training epochs.
--device         Override device: auto | cuda | cpu.
--output_dir     Root output directory (default: outputs/).
--n_samples      Forecast samples for evaluation (default: 500).
--skip_eval      Skip the evaluation step after training.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

# Ensure the project root is importable.
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# SDE dataset types supported by this script.
# To add double-well support, append the new key here AND to the _SDE_MODELS
# sets in train/train.py and eval/evaluate.py, and to
# data/sde_loader.VALID_SDE_MODEL_KEYS.
_SDE_MODELS = {"sw_sle_em", "sw_gle_oe_em"}

# Default SDE data directory (relative to the project root).
_DEFAULT_SDE_DATA_DIR = "../Data/Trajectories/"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    """Set all global random seeds for reproducibility."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _resolve_device(device_str: str) -> str:
    if device_str == "auto":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


# ---------------------------------------------------------------------------
# SDE pipeline (train + evaluate in-process)
# ---------------------------------------------------------------------------

def _run_sde_pipeline(args: argparse.Namespace) -> None:
    """
    Full SDE pipeline: load → split → train → evaluate.

    All four pipeline entry points are exercised:
        run_full_pipeline  (this function)
        train_model        (via train/train.py internals, imported below)
        test_model         (via eval/evaluate.py:evaluate)
        compare_models     (not called here; use compare_models.py separately)
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from data.sde_loader import load_sde_dataset, split_sde_dataset
    from data.preprocessing import (
        compute_norm_stats,
        normalize,
        extract_segments,
        split_context_target,
    )
    from data.loader import DataBundle
    from models.nftsf import NFTSFModel
    from train.engine_pytorch import run as pytorch_run
    from eval.evaluate import evaluate

    dataset_type = args.dataset_type
    sde_data_dir = Path(
        args.sde_data_dir if args.sde_data_dir else _DEFAULT_SDE_DATA_DIR
    )

    # ---- Config ----
    base_cfg_path  = Path(args.base_config) if args.base_config else (
        _PROJECT_ROOT / "configs" / "base.yaml"
    )
    model_cfg_path = Path(args.config)

    for p, name in [(base_cfg_path, "base config"), (model_cfg_path, "model config")]:
        if not p.exists():
            sys.exit(f"[run_full_pipeline] {name} not found: {p}")

    base_cfg  = _load_yaml(base_cfg_path)
    model_cfg = _load_yaml(model_cfg_path)
    merged    = _deep_merge(base_cfg, model_cfg)
    merged["data"] = base_cfg["data"]   # lock data section

    if args.epochs:
        merged["training"]["epochs"] = args.epochs
    if args.device:
        merged["training"]["device"] = args.device

    merged.setdefault("model", {})["name"] = dataset_type
    merged["training"]["device"] = _resolve_device(
        merged["training"].get("device", "auto")
    )

    seed   = int(merged["data"]["seed"])
    n_past = int(merged["data"]["n_past"])
    n_future = int(merged["data"]["n_future"])

    _set_seed(seed)

    # ---- Output directory ----
    run_id     = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out   = Path(args.output_dir) if args.output_dir else (
        _PROJECT_ROOT / "outputs"
    )
    output_dir = base_out / dataset_type / dataset_type / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run_full_pipeline] Dataset type : {dataset_type}")
    print(f"[run_full_pipeline] SDE data dir : {sde_data_dir}")
    print(f"[run_full_pipeline] Output       : {output_dir}")

    # ---- Load and split SDE data ----
    print("\n[run_full_pipeline] === Loading SDE data ===")
    raw_dataset = load_sde_dataset(dataset_type, sde_data_dir)
    splits      = split_sde_dataset(raw_dataset, random_seed=seed)

    memory_label = "(memory)" if splits["has_memory"] else "(no memory)"
    print(f"[run_full_pipeline] {dataset_type} {memory_label}")
    print(f"  x_train : {splits['x_train'].shape}")
    print(f"  x_val   : {splits['x_val'].shape}")
    print(f"  x_test  : {splits['x_test'].shape}")

    # ---- Normalisation (train-set statistics) ----
    x_train = splits["x_train"].astype(np.float32)
    x_val   = splits["x_val"].astype(np.float32)
    norm_mean, norm_std = compute_norm_stats(x_train)
    x_train_norm = normalize(x_train, norm_mean, norm_std)
    x_val_norm   = normalize(x_val,   norm_mean, norm_std)

    # ---- Build DataLoaders ----
    stride   = int(merged["data"].get("stride", 1))
    batch_sz = int(merged["training"].get("batch_size", 4096))
    device   = merged["training"]["device"]

    tracks_tr  = torch.tensor(x_train_norm)
    tracks_val = torch.tensor(x_val_norm)
    segs_tr    = extract_segments(tracks_tr,  n_past, n_future, stride)
    segs_val   = extract_segments(tracks_val, n_past, n_future, stride)
    tr_ctx, tr_tgt = split_context_target(segs_tr,  n_past)
    vl_ctx, vl_tgt = split_context_target(segs_val, n_past)

    def _loader(ctx: torch.Tensor, tgt: torch.Tensor) -> DataLoader:
        from torch.utils.data import TensorDataset, DataLoader as DL
        ds = TensorDataset(ctx, tgt)
        bs = len(ds) if batch_sz <= 0 else batch_sz
        return DL(ds, batch_size=bs, shuffle=True, pin_memory=(device != "cpu"))

    # ---- Attach metadata to config ----
    merged["sde"] = {
        "model_key":  dataset_type,
        "has_memory": splits["has_memory"],
        "norm_mean":  float(norm_mean),
        "norm_std":   float(norm_std),
    }
    merged["landscape"] = dataset_type
    merged["run_id"]    = run_id
    merged["data_path"] = str(sde_data_dir)

    with open(output_dir / "config.json", "w") as f:
        json.dump(merged, f, indent=2)

    # Save norm stats for compare_models.py.
    np.savez(
        output_dir / "norm_stats.npz",
        mean=np.array(norm_mean, dtype=np.float32),
        std=np.array(norm_std,  dtype=np.float32),
    )

    # ---- train_model ----
    print("\n[run_full_pipeline] === train_model ===")
    model = NFTSFModel(merged)
    results = pytorch_run(
        model=model,
        train_loader=_loader(tr_ctx, tr_tgt),
        val_loader=_loader(vl_ctx, vl_tgt),
        config=merged,
        output_dir=output_dir,
    )
    print(f"[run_full_pipeline] Training done. "
          f"Best val loss: {results.get('best_val_loss', float('nan')):.6f}")

    # ---- test_model ----
    if not args.skip_eval:
        print("\n[run_full_pipeline] === test_model ===")
        n_samples = int(args.n_samples) if args.n_samples else 500
        evaluate(
            run_dir=output_dir,
            data_path=sde_data_dir,
            n_samples=n_samples,
        )
    else:
        print("\n[run_full_pipeline] Skipping evaluation (--skip_eval).")

    print(f"\n[run_full_pipeline] Pipeline complete. Outputs at: {output_dir}")


# ---------------------------------------------------------------------------
# Canonical pipeline (delegates to subprocess calls for full isolation)
# ---------------------------------------------------------------------------

def _run_canonical_pipeline(args: argparse.Namespace) -> None:
    """
    Canonical pipeline: train then evaluate via train.py + evaluate.py.

    Delegates to the existing scripts as subprocesses so that the full
    canonical config-loading, model-registry, and engine dispatch logic is
    exercised unchanged.
    """
    if not args.data:
        sys.exit("[run_full_pipeline] --data (canonical .npz path) is required "
                 "for --dataset_type canonical")
    if not args.model:
        sys.exit("[run_full_pipeline] --model (e.g. nftsf) is required "
                 "for --dataset_type canonical")

    train_cmd = [
        sys.executable, str(_PROJECT_ROOT / "train" / "train.py"),
        "--model",  args.model,
        "--config", args.config,
        "--data",   args.data,
    ]
    if args.run_id:
        train_cmd += ["--run_id", args.run_id]
    if args.base_config:
        train_cmd += ["--base_config", args.base_config]
    if args.epochs:
        train_cmd += ["--epochs", str(args.epochs)]
    if args.device:
        train_cmd += ["--device", args.device]
    if args.output_dir:
        train_cmd += ["--output_dir", args.output_dir]

    print(f"[run_full_pipeline] Running: {' '.join(train_cmd)}")
    result = subprocess.run(train_cmd, check=True)

    if args.skip_eval:
        print("[run_full_pipeline] Skipping evaluation (--skip_eval).")
        return

    # Discover the run directory created by train.py.
    model_name = args.model.lower()
    base_out   = Path(args.output_dir) if args.output_dir else (
        _PROJECT_ROOT / "outputs"
    )

    # If run_id was specified, reconstruct the path; otherwise take the
    # most recently modified subdirectory (handles auto-timestamp run_id).
    import data.canonical as _canon_mod
    canon = _canon_mod.load_canonical(Path(args.data))
    landscape = str(canon.get("landscape", "unknown"))

    run_root = base_out / model_name / landscape
    if args.run_id:
        run_dir = run_root / args.run_id
    else:
        subdirs = sorted(run_root.iterdir(), key=lambda p: p.stat().st_mtime)
        if not subdirs:
            sys.exit(f"[run_full_pipeline] No run directories found under {run_root}")
        run_dir = subdirs[-1]

    eval_cmd = [
        sys.executable, str(_PROJECT_ROOT / "eval" / "evaluate.py"),
        "--checkpoint", str(run_dir),
        "--data",       args.data,
    ]
    if args.n_samples:
        eval_cmd += ["--n_samples", str(args.n_samples)]

    print(f"[run_full_pipeline] Running: {' '.join(eval_cmd)}")
    subprocess.run(eval_cmd, check=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Full train→evaluate pipeline for SDE and canonical datasets.\n\n"
            "Recognised --dataset_type values:\n"
            "  sw_sle_em    — single-well SDE, no memory\n"
            "  sw_gle_oe_em — single-well SDE, with memory (GLE)\n"
            "  canonical    — existing canonical .npz format\n\n"
            "To add double-well support later, append 'dw_sle_em' / "
            "'dw_gle_oe_em' to _SDE_MODELS in this file and the other "
            "three entry points."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--dataset_type", required=True,
        choices=sorted(_SDE_MODELS) + ["canonical"],
        help="Dataset/physical-system type.  SDE types use --sde_data_dir; "
             "canonical uses --data + --model.",
    )
    p.add_argument(
        "--config", required=True,
        help="Model-specific YAML config (e.g. configs/nftsf.yaml).",
    )

    # SDE-specific
    p.add_argument(
        "--sde_data_dir", default=None,
        help=f"Directory containing the .npy SDE trajectory files.  "
             f"Default: '{_DEFAULT_SDE_DATA_DIR}' relative to the project root.",
    )

    # Canonical-specific
    p.add_argument(
        "--data", default=None,
        help="Canonical .npz dataset path (required for --dataset_type canonical).",
    )
    p.add_argument(
        "--model", default="nftsf",
        help="ML model name for canonical datasets (default: nftsf).",
    )

    # Shared optional
    p.add_argument("--run_id",      default=None,
                   help="Run identifier (default: timestamp).")
    p.add_argument("--base_config", default=None,
                   help="Path to base.yaml (default: configs/base.yaml).")
    p.add_argument("--epochs",      type=int, default=None,
                   help="Override training epochs.")
    p.add_argument("--device",      default=None,
                   help="Override device: auto | cuda | cpu.")
    p.add_argument("--output_dir",  default=None,
                   help="Root output directory (default: outputs/).")
    p.add_argument("--n_samples",   type=int, default=500,
                   help="Forecast samples for evaluation (default: 500).")
    p.add_argument("--skip_eval",   action="store_true", default=False,
                   help="Skip the evaluation step after training.")

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.dataset_type in _SDE_MODELS:
        _run_sde_pipeline(args)
    else:
        _run_canonical_pipeline(args)


if __name__ == "__main__":
    main()
