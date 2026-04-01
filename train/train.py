"""
train/train.py
==============
Unified training entry point for the NFTSF comparison pipeline.

Usage
-----
conda run -n unified_tsf python train/train.py \\
    --model  nftsf \\
    --config configs/nftsf.yaml \\
    --data   outputs/canonical/alanine_phi.npz \\
    --run_id run_001

Required arguments
------------------
--model   Model name: nftsf | tsdiff_q | tsdiff_ms | tsdiff_cond |
                      csdi | ratd | nsdiff | arima
--config  Path to a model-specific YAML config (overrides base.yaml).
--data    Path to the canonical .npz dataset file.

Optional arguments
------------------
--run_id       Unique run identifier (default: timestamp).
--base_config  Path to base.yaml (default: configs/base.yaml next to this script).
--epochs       Override training epochs.
--device       Override device (auto | cuda | cpu).
--output_dir   Root output directory (default: outputs/ in project root).

Config merge order
------------------
base.yaml is loaded first; per-model config overrides non-data keys.
``data.*`` keys from base.yaml are *always* canonical and cannot be
overridden by model configs.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

# Ensure the project root is on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    """Set all global random seeds for reproducibility."""
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
    """Recursively merge *override* into *base* (returns new dict)."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _resolve_device(device_str: str) -> str:
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


# ---------------------------------------------------------------------------
# Model family routing
# ---------------------------------------------------------------------------

_PYTORCH_MODELS   = {"nftsf"}
_GLUONTS_MODELS   = {"tsdiff_q", "tsdiff_ms", "tsdiff_cond", "csdi", "ratd", "nsdiff"}
_ARIMA_MODELS     = {"arima"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Unified model training script.")
    parser.add_argument("--model",       required=True, help="Model name.")
    parser.add_argument("--config",      required=True, help="Model-specific YAML config.")
    parser.add_argument("--data",        required=True, help="Canonical .npz dataset path.")
    parser.add_argument("--run_id",      default=None,  help="Run identifier (default: timestamp).")
    parser.add_argument("--base_config", default=None,  help="Path to base.yaml.")
    parser.add_argument("--epochs",      type=int, default=None)
    parser.add_argument("--device",      default=None)
    parser.add_argument("--output_dir",  default=None)
    args = parser.parse_args()

    # ---- Resolve paths ----
    project_root = _PROJECT_ROOT
    base_cfg_path = Path(args.base_config) if args.base_config else (
        project_root / "configs" / "base.yaml"
    )
    model_cfg_path = Path(args.config)
    data_path      = Path(args.data)

    for p, name in [(base_cfg_path, "base config"), (model_cfg_path, "model config"),
                    (data_path, "dataset")]:
        if not p.exists():
            sys.exit(f"[train] {name} not found: {p}")

    # ---- Load and merge configs ----
    base_cfg  = _load_yaml(base_cfg_path)
    model_cfg = _load_yaml(model_cfg_path)

    # Merge: model config overrides base, EXCEPT data section.
    merged = _deep_merge(base_cfg, model_cfg)

    # Lock data section to base values — models cannot change split/seed/horizon.
    merged["data"] = base_cfg["data"]

    # CLI overrides.
    if args.epochs:
        merged["training"]["epochs"] = args.epochs
    if args.device:
        merged["training"]["device"] = args.device

    # Resolve model name.
    model_name = args.model.lower()
    merged.setdefault("model", {})["name"] = model_name

    # Device.
    merged["training"]["device"] = _resolve_device(
        merged["training"].get("device", "auto")
    )

    # ---- Run ID and output directory ----
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    from data.canonical import load_canonical
    meta     = load_canonical(data_path)
    landscape = meta.get("landscape", "unknown")

    base_out = Path(args.output_dir) if args.output_dir else (
        project_root / "outputs"
    )
    output_dir = base_out / model_name / landscape / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Seed ----
    seed = int(merged["data"]["seed"])
    _set_seed(seed)
    print(f"[train] Seed: {seed}")
    print(f"[train] Model: {model_name}  |  Landscape: {landscape}  |  Run: {run_id}")
    print(f"[train] Output: {output_dir}")

    # Save merged config for reproducibility.
    merged["run_id"]   = run_id
    merged["data_path"] = str(data_path)
    with open(output_dir / "config.json", "w") as f:
        json.dump(merged, f, indent=2)

    # ---- Load data ----
    from data.loader import get_dataloader
    bundle = get_dataloader(merged, data_path)
    print(f"[train] Data loaded — landscape: {bundle.metadata['landscape']}, "
          f"n_past={bundle.metadata['n_past']}, n_future={bundle.metadata['n_future']}")

    # ---- Instantiate model ----
    from models.registry import get_model
    ModelClass = get_model(model_name)
    model = ModelClass(merged)
    print(f"[train] Model instantiated: {model.__class__.__name__}")

    # ---- Train ----
    if model_name in _PYTORCH_MODELS:
        from train.engine_pytorch import run as pytorch_run
        results = pytorch_run(
            model=model,
            train_loader=bundle.train,
            val_loader=bundle.val,
            config=merged,
            output_dir=output_dir,
        )

    elif model_name in _GLUONTS_MODELS:
        from train.engine_lightning import run as lightning_run
        results = lightning_run(
            model=model,
            train_dataset=bundle.train,
            val_dataset=bundle.val,
            config=merged,
            output_dir=output_dir,
        )

    elif model_name in _ARIMA_MODELS:
        # ARIMA fits at prediction time; nothing to train.
        print("[train] ARIMA has no gradient training — saving placeholder checkpoint.")
        model.save(output_dir / "model_final.pth")
        results = {"loss_history": [], "val_loss_history": [],
                   "best_epoch": 0, "best_val_loss": float("nan")}
        with open(output_dir / "train_results.json", "w") as f:
            json.dump(results, f, indent=2)

    else:
        sys.exit(f"[train] No training engine for model '{model_name}'.")

    print(f"\n[train] Training complete.  Results saved to {output_dir}")
    if results.get("best_val_loss") is not None:
        print(f"[train] Best val loss: {results['best_val_loss']:.6f} "
              f"at epoch {results.get('best_epoch', '?')}")


if __name__ == "__main__":
    main()
