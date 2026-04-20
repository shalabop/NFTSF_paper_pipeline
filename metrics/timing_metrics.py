"""
metrics/timing_metrics.py
==========================
Consistent future-sampling timing instrumentation for the paper path.

Only the inference/sampling stage that generates future trajectories is timed.
Data loading, model construction, normalisation, and metric computation are
explicitly excluded.  This makes the elapsed time directly comparable across
methods, including expensive diffusion-style baselines.

CUDA synchronization policy
----------------------------
If torch.cuda is available AND the model appears to be on a CUDA device,
``torch.cuda.synchronize()`` is called immediately before both the start and
stop timestamps to ensure GPU work is fully completed before the clock reads.
This prevents artificially short measurements due to asynchronous GPU dispatch.
The synchronization is guarded by try/except so CPU-only environments work
without modification.

Warm-up
-------
Pass ``warmup_fn`` to SampleTimer to run one warm-up call before timing begins.
This avoids measuring JIT compilation or CUDA kernel launch overhead.

Usage example
-------------
    from metrics.timing_metrics import SampleTimer, append_timing_record

    with SampleTimer(
        method="nftsf",
        dataset="single_well",
        horizon=50,
        n_samples=500,
        n_items=200,
        device="cuda:0",
        notes="run_001",
    ) as t:
        samples = model.predict(context, n_samples=500)

    append_timing_record(t.record, Path("outputs/paper_clean/timing.jsonl"))
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cuda_sync() -> None:
    """Call torch.cuda.synchronize() if available; silently skip otherwise."""
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
    """Return GPU name string if device is CUDA, else None."""
    try:
        import torch
        if device.startswith("cuda"):
            idx = int(device.split(":")[-1]) if ":" in device else 0
            return torch.cuda.get_device_name(idx)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

class SampleTimer:
    """
    Context manager that times only the inference/sampling stage.

    Attributes
    ----------
    record : dict  — structured timing record, populated after __exit__.

    Parameters
    ----------
    method    : model/method identifier string (e.g. "nftsf", "tsdiff_q").
    dataset   : dataset identifier (e.g. "single_well").
    horizon   : forecast horizon in time steps.
    n_samples : number of ensemble members drawn per item.
    n_items   : number of windows/trajectories forecasted in this call.
    system    : optional physical-system label.
    device    : device string, e.g. "cuda:0" or "cpu".
    notes     : free-form annotation for the paper record.
    warmup_fn : callable to invoke once before timing starts (optional).
                The warm-up is not included in elapsed_sec.
    """

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
        self.method    = method
        self.dataset   = dataset
        self.horizon   = horizon
        self.n_samples = n_samples
        self.n_items   = n_items
        self.system    = system
        self.device    = device
        self.notes     = notes
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
            "method":        self.method,
            "dataset":       self.dataset,
            "system":        self.system,
            "horizon":       self.horizon,
            "n_samples":     self.n_samples,
            "n_items":       self.n_items,
            "elapsed_sec":   round(elapsed, 6),
            "sec_per_sample": round(elapsed / denom, 9),
            "device":        self.device,
            "device_name":   _device_name(self.device),
            "torch_version": _torch_version(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "notes":         self.notes,
        }


# ---------------------------------------------------------------------------
# JSONL persistence
# ---------------------------------------------------------------------------

def append_timing_record(record: dict[str, Any], path: Path) -> None:
    """
    Append *record* as one JSON line to *path* (creating the file if needed).

    Multiple runs accumulate cleanly; each line is a self-contained record.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def load_timing_records(path: Path) -> list[dict[str, Any]]:
    """Load all records from a JSONL timing file."""
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
