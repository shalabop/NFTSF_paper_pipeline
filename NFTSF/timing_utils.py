"""
timing_utils.py
===============
Shared timing wrapper for the future-sampling / inference step.

Call site contract
------------------
Wrap *only* the model.predict() call (or equivalent sampling call).
Data loading, normalization, and result post-processing must be timed
separately (or not at all) and must never be folded into the timed region.

If preprocessing is unavoidably entangled with sampling in a given model,
time the composite block and record it as a separate "preprocess+sample"
entry with ``includes_preprocessing=True``.

Output schema
-------------
Per run: <output_dir>/timing_<method>_<dataset>_<timestamp>.json
Registry: <output_dir>/timing_registry.csv  (one row appended per run)
"""

from __future__ import annotations

import contextlib
import csv
import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def _device_string(device_str: str) -> str:
    """Return a human-readable device label; never raises."""
    try:
        import torch
        dev = torch.device(device_str)
        if dev.type == "cuda":
            idx = dev.index if dev.index is not None else 0
            return f"cuda:{idx} ({torch.cuda.get_device_name(idx)})"
    except Exception:
        pass
    return device_str


def _cuda_sync(device_str: str) -> None:
    """Synchronize CUDA if the device is GPU; no-op on CPU."""
    try:
        import torch
        if torch.device(device_str).type == "cuda":
            torch.cuda.synchronize()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _git_commit() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TimingRecord:
    """Structured result of one timed inference run (potentially repeated)."""

    method_name: str
    dataset_name: str
    n_samples: int
    horizon: int
    device: str                       # human-readable string
    batch_size: Optional[int]         # trajectory count per forward pass; None if N/A
    torch_sync_used: bool             # whether cuda.synchronize() bracketed the call
    includes_preprocessing: bool      # True if timing unavoidably includes non-sampling work
    n_repeats: int
    repeat_times_seconds: list[float] # wall-clock per repeat
    median_seconds: float
    iqr_seconds: float                # interquartile range (Q75 - Q25)
    timestamp_utc: str                # ISO-8601
    git_commit: str

    # derived convenience
    wall_clock_seconds: float = field(init=False)   # alias for median

    def __post_init__(self) -> None:
        self.wall_clock_seconds = self.median_seconds

    def to_dict(self) -> dict:
        d = asdict(self)
        d["wall_clock_seconds"] = self.wall_clock_seconds
        return d

    # CSV row (flat, no nested lists)
    def to_csv_row(self) -> dict:
        d = self.to_dict()
        d["repeat_times_seconds"] = json.dumps(self.repeat_times_seconds)
        return d


# ---------------------------------------------------------------------------
# Context manager (single-shot timing)
# ---------------------------------------------------------------------------

class _TimerHandle:
    """Mutable handle returned by timed_sampling().__enter__."""
    elapsed: float = 0.0


@contextlib.contextmanager
def timed_sampling(
    method_name: str,
    dataset_name: str,
    n_samples: int,
    horizon: int,
    *,
    device: str = "cpu",
    batch_size: Optional[int] = None,
    sync_cuda: bool = True,
    includes_preprocessing: bool = False,
) -> Iterator[_TimerHandle]:
    """
    Context manager that times the enclosed block (the sampling step only).

    Usage
    -----
    handle = _TimerHandle()
    with timed_sampling("nftsf", "double_well", n_samples=200, horizon=100,
                        device="cuda:0") as handle:
        samples = model.predict(context, n_samples=200)
    print(f"Elapsed: {handle.elapsed:.3f}s")

    Notes
    -----
    - On GPU, call torch.cuda.synchronize() before and after if sync_cuda=True.
    - The returned handle is populated *after* the ``with`` block exits.
    """
    handle = _TimerHandle()
    if sync_cuda:
        _cuda_sync(device)
    t0 = time.perf_counter()
    try:
        yield handle
    finally:
        if sync_cuda:
            _cuda_sync(device)
        handle.elapsed = time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Multi-repeat timing
# ---------------------------------------------------------------------------

def run_timed_inference(
    fn: Callable[[], Any],
    method_name: str,
    dataset_name: str,
    n_samples: int,
    horizon: int,
    *,
    device: str = "cpu",
    batch_size: Optional[int] = None,
    n_repeats: int = 5,
    sync_cuda: bool = True,
    includes_preprocessing: bool = False,
) -> TimingRecord:
    """
    Call *fn* ``n_repeats`` times, record wall-clock per repeat, return
    a :class:`TimingRecord`.

    The first repeat is treated as a warm-up and is included in the recorded
    times (this is intentional; if warm-up effects are large, they are
    scientifically relevant).  Use n_repeats=1 to get a single-shot time.

    Parameters
    ----------
    fn : Zero-argument callable wrapping *only* the sampling/predict call.
    method_name : Model identifier string (e.g. "nftsf", "arima").
    dataset_name : Dataset / landscape name (e.g. "double_well").
    n_samples : Ensemble size passed to the model.
    horizon : Forecast horizon (n_future).
    device : Torch device string used during inference.
    batch_size : Number of test trajectories processed per forward pass.
    n_repeats : Number of independent timed calls.
    sync_cuda : Whether to synchronize GPU before/after each repeat.
    includes_preprocessing : Flag if timing unavoidably includes non-sampling steps.

    Returns
    -------
    TimingRecord
    """
    times: list[float] = []
    for _ in range(n_repeats):
        if sync_cuda:
            _cuda_sync(device)
        t0 = time.perf_counter()
        fn()
        if sync_cuda:
            _cuda_sync(device)
        times.append(time.perf_counter() - t0)

    arr = np.array(times)
    median = float(np.median(arr))
    iqr = float(np.percentile(arr, 75) - np.percentile(arr, 25))

    return TimingRecord(
        method_name=method_name,
        dataset_name=dataset_name,
        n_samples=n_samples,
        horizon=horizon,
        device=_device_string(device),
        batch_size=batch_size,
        torch_sync_used=sync_cuda,
        includes_preprocessing=includes_preprocessing,
        n_repeats=n_repeats,
        repeat_times_seconds=times,
        median_seconds=median,
        iqr_seconds=iqr,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=_git_commit(),
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

_CSV_FIELDNAMES = [
    "method_name", "dataset_name", "n_samples", "horizon",
    "device", "batch_size", "torch_sync_used", "includes_preprocessing",
    "n_repeats", "median_seconds", "iqr_seconds", "wall_clock_seconds",
    "repeat_times_seconds", "timestamp_utc", "git_commit",
]


def save_timing_result(
    record: TimingRecord,
    output_dir: Path,
    csv_registry: Optional[Path] = None,
) -> Path:
    """
    Save *record* as a JSON file in *output_dir* and append a row to the
    CSV registry (default: ``output_dir/timing_registry.csv``).

    Returns the path to the written JSON file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = record.timestamp_utc.replace(":", "").replace("-", "").split(".")[0]
    json_path = output_dir / f"timing_{record.method_name}_{record.dataset_name}_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(record.to_dict(), f, indent=2)

    if csv_registry is None:
        csv_registry = output_dir / "timing_registry.csv"
    csv_registry = Path(csv_registry)

    write_header = not csv_registry.exists()
    with open(csv_registry, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(record.to_csv_row())

    return json_path


def load_timing_registry(csv_registry: Path) -> list[dict]:
    """Load all timing records from *csv_registry*, parsing numeric fields."""
    csv_registry = Path(csv_registry)
    if not csv_registry.exists():
        return []
    records: list[dict] = []
    with open(csv_registry, newline="") as f:
        for row in csv.DictReader(f):
            for key in ("n_samples", "horizon", "n_repeats"):
                if row.get(key) not in (None, ""):
                    row[key] = int(row[key])
            for key in ("median_seconds", "iqr_seconds", "wall_clock_seconds"):
                if row.get(key) not in (None, ""):
                    row[key] = float(row[key])
            row["torch_sync_used"] = row.get("torch_sync_used", "").lower() == "true"
            row["includes_preprocessing"] = row.get("includes_preprocessing", "").lower() == "true"
            if row.get("batch_size") not in (None, "", "None"):
                row["batch_size"] = int(row["batch_size"])
            else:
                row["batch_size"] = None
            if row.get("repeat_times_seconds"):
                try:
                    row["repeat_times_seconds"] = json.loads(row["repeat_times_seconds"])
                except (json.JSONDecodeError, TypeError):
                    row["repeat_times_seconds"] = []
            records.append(row)
    return records
