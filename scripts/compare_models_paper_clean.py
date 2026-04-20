#!/usr/bin/env python3
"""
scripts/compare_models_paper_clean.py
=======================================
Clean multi-method aggregation/comparison script for paper use.

Reads per-method result JSONs produced by test_model_paper_clean.py and
aggregates them into comparison.csv and comparison.json in a timestamped
output directory.  Does NOT mix legacy outputs.

Stable paper-column names (used as CSV/JSON keys throughout the paper path):
  method, dataset, system, horizon, n_samples, n_items
  crps, crps_reliability, crps_resolution, crps_uncertainty
  calibration_pit_mean, calibration_pit_std, rank_hist_chi2
  elapsed_sec, sec_per_sample, device, device_name
  run_id, timestamp_utc

See docs/paper_clean_path.md for scientific framing.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# Stable column order for CSV output.
_COLUMNS = [
    "method", "dataset", "system", "horizon", "n_samples", "n_items",
    "crps",
    "crps_reliability", "crps_resolution", "crps_uncertainty",
    "calibration_pit_mean", "calibration_pit_std", "rank_hist_chi2",
    "elapsed_sec", "sec_per_sample", "device", "device_name",
    "run_id", "timestamp_utc",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_result(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def discover_results(result_dir: Path) -> list[dict[str, Any]]:
    """
    Recursively find all result_*.json files under result_dir.

    Ignores comparison.json to avoid self-ingestion on re-runs.
    """
    files = sorted(result_dir.rglob("result_*.json"))
    records = []
    for f in files:
        try:
            records.append(_load_result(f))
        except Exception as e:
            print(f"[compare_models_paper_clean] Warning: could not load {f}: {e}")
    return records


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(v: Any, precision: int = 4) -> str:
    if v is None:
        return "null"
    if isinstance(v, float):
        return f"{v:.{precision}f}"
    return str(v)


def _print_summary(records: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 72)
    print(f"{'Method':<20} {'Dataset':<16} {'CRPS':>8} {'Reliab':>8} "
          f"{'Resolut':>8} {'Elapsed':>10} {'s/sample':>10}")
    print("-" * 72)
    for r in sorted(records, key=lambda x: (x.get("dataset",""), x.get("method",""))):
        print(
            f"{r.get('method','?'):<20} "
            f"{r.get('dataset','?'):<16} "
            f"{_fmt(r.get('crps')):>8} "
            f"{_fmt(r.get('crps_reliability')):>8} "
            f"{_fmt(r.get('crps_resolution')):>8} "
            f"{_fmt(r.get('elapsed_sec'), 2):>10} "
            f"{_fmt(r.get('sec_per_sample'), 6):>10}"
        )
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------

def emit_comparison(
    records: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[Path, Path]:
    """
    Write comparison.csv and comparison.json to output_dir.

    Returns (csv_path, json_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalise: fill missing columns with None.
    rows = []
    for r in records:
        row = {col: r.get(col, None) for col in _COLUMNS}
        rows.append(row)

    # CSV
    csv_path = output_dir / "comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # JSON
    json_path = output_dir / "comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=str)

    return csv_path, json_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        sys.exit(f"[compare_models_paper_clean] result_dir not found: {result_dir}")

    records = discover_results(result_dir)
    if not records:
        sys.exit(f"[compare_models_paper_clean] No result_*.json found under {result_dir}")

    print(f"[compare_models_paper_clean] Loaded {len(records)} result(s) from {result_dir}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.output_dir) / ts
    csv_path, json_path = emit_comparison(records, out_root)

    _print_summary(records)

    print(f"[compare_models_paper_clean] CSV  : {csv_path}")
    print(f"[compare_models_paper_clean] JSON : {json_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate per-method result JSONs into paper comparison tables."
    )
    p.add_argument("--result_dir", required=True,
                   help="Directory containing result_*.json files from "
                        "test_model_paper_clean.py.")
    p.add_argument("--output_dir", default="outputs/paper_clean",
                   help="Root output directory; a timestamped subdirectory is created.")
    return p.parse_args()


if __name__ == "__main__":
    main()
