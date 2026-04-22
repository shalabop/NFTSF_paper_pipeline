#!/usr/bin/env python3
"""
forecast_nf.py
==============
NF-TSF forecast script with batched inference and timing that matches
CSDI and TSDiff-Cond measurement conventions.

Uses SampleTimer (defined inline) for consistent instrumentation.
"""

import os
import sys
import json
import argparse
import warnings
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, "..", ".."))

sys.path.insert(0, os.path.join(PROJECT_ROOT, "architectures", "NF"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "architectures"))
sys.path.insert(0, PROJECT_ROOT)

from architecture import create_nfm

# ---------------------------------------------------------------------------
# Helpers for SampleTimer
# ---------------------------------------------------------------------------
def _cuda_sync() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass

def _torch_version() -> str | None:
    try:
        import torch
        return torch.__version__
    except Exception:
        return None

def _device_name(device: str) -> str | None:
    try:
        import torch
        if device.startswith("cuda"):
            idx = int(device.split(":")[-1]) if ":" in device else 0
            return torch.cuda.get_device_name(idx)
    except Exception:
        pass
    return None

# ---------------------------------------------------------------------------
# SampleTimer context manager
# ---------------------------------------------------------------------------
class SampleTimer:
    def __init__(
        self,
        *,
        method: str,
        dataset: str,
        horizon: int,
        n_samples: int,
        n_items: int,
        system: str | None = None,
        device: str = "cpu",
        notes: str | None = None,
        warmup_fn: Callable[[], Any] | None = None,
    ) -> None:
        self.method = method
        self.dataset = dataset
        self.horizon = horizon
        self.n_samples = n_samples
        self.n_items = n_items
        self.system = system
        self.device = device
        self.notes = notes
        self.warmup_fn = warmup_fn
        self.record: dict[str, Any] = {}
        self._start: float = 0.0

    def __enter__(self) -> "SampleTimer":
        if self.warmup_fn is not None:
            self.warmup_fn()
        _cuda_sync()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        _cuda_sync()
        elapsed = time.perf_counter() - self._start
        denom = max(1, self.n_items * self.n_samples)
        self.record = {
            "method": self.method,
            "dataset": self.dataset,
            "system": self.system,
            "horizon": self.horizon,
            "n_samples": self.n_samples,
            "n_items": self.n_items,
            "elapsed_sec": round(elapsed, 6),
            "sec_per_sample": round(elapsed / denom, 9),
            "device": self.device,
            "device_name": _device_name(self.device),
            "torch_version": _torch_version(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "notes": self.notes,
        }

def append_timing_record(record: dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")

# ---------------------------------------------------------------------------
# Argument parsing – matches the requested interface
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="NF-TSF forecast script")
    p.add_argument("--config", required=True, help="Path to YAML config (architecture)")
    p.add_argument("--input", required=True, help="Path to test .npz file")
    p.add_argument("--ckpt", required=True, help="Path to model .pth checkpoint")
    p.add_argument("--out", "-o", required=True, help="Output .npz path")
    p.add_argument("--device", default="cuda", help="Device, e.g. cuda or cpu")
    p.add_argument("--test_size", type=int, required=False, help="Number of trajectories to use")
    p.add_argument("--prediction_length", type=int, required=True)
    p.add_argument("--context_length", type=int, required=True)
    p.add_argument("--num_of_samples", type=int, default=500, help="Number of ensemble samples")
    p.add_argument("--train_test_split", type=int, required=True, help="Split index")
    p.add_argument("--batch_size", type=int, default=64, help="Batch size for inference")
    p.add_argument("--timing_log", type=str, default="timing.jsonl", help="JSONL file for timing records")
    p.add_argument("--notes", type=str, default=None, help="Optional notes")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def setup_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    print(f"Device: {device}")
    return device

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def load_model(config_path, model_path, context_length, prediction_length, device):
    import yaml
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    # Architecture defaults
    flow_blocks = cfg.get("flow_blocks", 6)
    hidden_units = cfg.get("hidden_units", 64)
    hidden_layers = cfg.get("hidden_layers", "1,2")
    hidden_layers_list = tuple(int(x) for x in str(hidden_layers).split(","))
    print(f"Architecture: flow_blocks={flow_blocks}, hidden_units={hidden_units}, hidden_layers={hidden_layers_list}")
    model = create_nfm(
        device,
        latent_size=prediction_length,
        context_size=context_length,
        K=flow_blocks,
        hidden_units=hidden_units,
        hidden_layers_list=hidden_layers_list,
    )
    state = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    print(f"Loaded weights: {model_path}")
    return model

def load_test_data(data_path):
    data = np.load(data_path, allow_pickle=True)
    positions = data["positions"].astype(np.float32)
    time_arr = data["time"] if "time" in data else None
    print(f"Test data: {positions.shape}  (N_traj, T)")
    return positions, time_arr

# ---------------------------------------------------------------------------
# Batched inference with timing
# ---------------------------------------------------------------------------
def run_forecast_batched(
    model,
    positions: np.ndarray,
    context_length: int,
    prediction_length: int,
    n_samples: int,
    train_test_split: int,
    test_size: int,
    batch_size: int,
    device: torch.device,
    timing_log_path: str,
    notes: str | None,
    dataset_name: str,   # <-- added parameter
):
    # Data preparation (not timed)
    if test_size is None:
        test_size = positions.shape[0]
    test_size = min(test_size, positions.shape[0])
    positions = positions[:test_size]
    N = positions.shape[0]
    tts = train_test_split
    ctx_np = positions[:, tts - context_length : tts]          # (N, L)
    gt_np  = positions[:, tts : tts + prediction_length]       # (N, H)
    ctx_tensor = torch.tensor(ctx_np, dtype=torch.float32, device=device)
    samples_out = np.zeros((N, prediction_length, n_samples), dtype=np.float32)

    # Warm-up
    def warmup():
        dummy_ctx = ctx_tensor[:min(2, N)]
        dummy_tiled = dummy_ctx.repeat_interleave(max(2, min(4, n_samples)), dim=0)
        with torch.no_grad():
            try:
                _ = model.sample(dummy_tiled.shape[0], dummy_tiled)
            except Exception:
                pass

    # Timed inference
    with SampleTimer(
        method="nftsf",
        dataset=dataset_name,   # <-- use passed dataset name
        horizon=prediction_length,
        n_samples=n_samples,
        n_items=N,
        device=str(device),
        notes=notes,
        warmup_fn=warmup,
    ) as timer:
        n_batches = (N + batch_size - 1) // batch_size
        for b in range(n_batches):
            b_start = b * batch_size
            b_end = min(b_start + batch_size, N)
            B = b_end - b_start
            ctx_b = ctx_tensor[b_start:b_end]
            ctx_b_tiled = ctx_b.repeat_interleave(n_samples, dim=0)
            with torch.no_grad():
                try:
                    samp, _ = model.sample(B * n_samples, ctx_b_tiled)
                except Exception as e:
                    warnings.warn(f"Batch {b} failed: {e}, using zeros")
                    samp = torch.zeros(B * n_samples, prediction_length, device=device)
            samp_np = samp.cpu().numpy().reshape(B, n_samples, prediction_length)
            samples_out[b_start:b_end] = samp_np.transpose(0, 2, 1)  # (B, H, S)

    time_elapsed = timer.record["elapsed_sec"]
    print(f"Inference time : {time_elapsed:.2f}s  "
          f"({time_elapsed / N * 1000:.1f} ms/trajectory, "
          f"{time_elapsed / (N * n_samples) * 1000:.3f} ms/sample)")
    append_timing_record(timer.record, Path(timing_log_path))
    return samples_out, gt_np, ctx_np, time_elapsed, timer.record

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(42)
    device = setup_device(args.device)
    positions, time_arr = load_test_data(args.input)
    N, T = positions.shape

    # Use provided train_test_split
    train_test_split = args.train_test_split
    print(f"train_test_split : {train_test_split}")

    assert train_test_split >= args.context_length, \
        f"train_test_split ({train_test_split}) < context_length ({args.context_length})"
    assert train_test_split + args.prediction_length <= T, \
        f"train_test_split + prediction_length > T"

    model = load_model(args.config, args.ckpt, args.context_length, args.prediction_length, device)

    dataset_name = os.path.basename(args.input).replace(".npz", "")
    samples, ground_truth, contexts, time_elapsed, _ = run_forecast_batched(
        model=model,
        positions=positions,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        n_samples=args.num_of_samples,
        train_test_split=train_test_split,
        test_size=args.test_size,
        batch_size=args.batch_size,
        device=device,
        timing_log_path=args.timing_log,
        notes=args.notes,
        dataset_name=dataset_name,
    )

    # CI bands
    ci90_lower = np.percentile(samples,  5, axis=2)
    ci90_upper = np.percentile(samples, 95, axis=2)
    ci50_lower = np.percentile(samples, 25, axis=2)
    ci50_upper = np.percentile(samples, 75, axis=2)

    out_path = args.out
    if not out_path.endswith(".npz"):
        out_path += ".npz"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    save_dict = {
        "samples": samples,
        "ground_truth": ground_truth,
        "contexts": contexts,
        "ci90_lower": ci90_lower,
        "ci90_upper": ci90_upper,
        "ci50_lower": ci50_lower,
        "ci50_upper": ci50_upper,
        "full_trajectories": positions[:samples.shape[0]],
        "train_test_split": train_test_split,
        "prediction_length": args.prediction_length,
        "context_length": args.context_length,
        "num_of_samples": args.num_of_samples,
        "time_elapsed": time_elapsed,
    }
    if time_arr is not None:
        save_dict["time"] = time_arr
        save_dict["time_train"] = time_arr[:train_test_split]
        save_dict["time_test"] = time_arr[train_test_split:train_test_split+args.prediction_length]

    np.savez_compressed(out_path, **save_dict)
    print(f"\nSaved: {out_path}")
    print(f"  samples      : {samples.shape}  (N, H, S)")
    print(f"  ground_truth : {ground_truth.shape}  (N, H)")
    print(f"  contexts     : {contexts.shape}  (N, L)")
    print(f"  time_elapsed : {time_elapsed:.2f}s")

    # Sanity checks
    assert not np.isnan(samples).any(), "NaN in samples"
    assert not np.isinf(samples).any(), "Inf in samples"
    assert np.all(ci90_lower <= ci90_upper), "CI90 ordering violated"
    print("Sanity checks passed ✓")

if __name__ == "__main__":
    main()