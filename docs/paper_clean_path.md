# Paper-Clean Pipeline: Design and Scientific Framing

This document is the single authoritative location for the motivation behind
the clean paper path.  Individual modules contain only brief function-level
docs; the "why" lives here.

---

## Scientific Framing

The paper argues for **efficient, well-calibrated probabilistic forecasting**
against expensive diffusion-style baselines such as CSDI, RATD, TSDiff, NsDiff.

Raw CRPS is insufficient for this argument: a method can achieve low CRPS by
being sharp but miscalibrated, or by being well-calibrated but unsharp.  The
paper needs to separate these effects.

**CRPS decomposition** (Hersbach 2000) provides:
- **Reliability** — miscalibration of the forecast CDF.  Lower is better.
  A method with low Reliability has a well-calibrated predictive distribution.
- **Resolution** — forecast's ability to discriminate between outcomes beyond
  climatology.  Higher is better.  A method with high Resolution is sharp and
  useful.
- **Uncertainty** — a property of the observations alone (climatological
  spread).  Does not depend on the forecast; sets the irreducible floor.

Budget identity (holds for the Hersbach integral estimator):
  `CRPS = Reliability − Resolution + Uncertainty`

**Timing** is a first-class comparison metric because diffusion-style baselines
require many denoising steps per sample, making their wall-clock cost
substantially higher than NFTSF's single normalizing-flow forward pass.
The paper claims: *NFTSF achieves competitive or superior Reliability at a
fraction of the inference cost.*

---

## File Inventory (new clean-path files only; legacy files are untouched)

| File | Role |
|---|---|
| `metrics/crps_decomposition.py` | Hersbach (2000) CRPS decomposition (Phases B, C) |
| `metrics/metrics_probabilistic.py` | Thin façade: CRPS + decomp + PIT/rank-hist scaffolding (Phase B) |
| `metrics/timing_metrics.py` | `SampleTimer` context manager + JSONL record (Phase D) |
| `scripts/test_model_paper_clean.py` | Single-model evaluation entry point (Phase B) |
| `scripts/compare_models_paper_clean.py` | Multi-method aggregation → CSV/JSON (Phase E) |
| `scripts/run_full_pipeline_paper_clean.py` | Full orchestration (Phase F) |
| `scripts/run_full_pipeline_paper_clean.sh` | Shell wrapper for the orchestrator |
| `tests/test_crps_decomposition.py` | Unit tests for CRPS decomposition (Phase G) |
| `docs/paper_clean_path.md` | This document |

---

## Stable Output Schema

The following field names are locked for paper figures and tables.  Do not
rename them without updating both the comparison script and downstream
figure/table code.

```
method, dataset, system, horizon, n_samples, n_items
crps
crps_reliability, crps_resolution, crps_uncertainty
calibration_pit_mean, calibration_pit_std, rank_hist_chi2
elapsed_sec, sec_per_sample, device, device_name
run_id, timestamp_utc
```

---

## Assumptions and Caveats Affecting Reproducibility

1. **CRPS estimators differ between legacy and clean paths.**
   `eval/evaluate.py` calls `properscoring.crps_ensemble` (a scalar Python
   loop).  The clean path uses a vectorized energy-score estimator
   (`crps_ensemble` in `metrics/crps_decomposition.py`).  Both are unbiased,
   but numerical values will differ by ≤ floating-point noise for the same
   inputs.  Use only one estimator within any single paper comparison.

2. **CRPS_decomp vs CRPS (fair estimator).**
   `crps_decomp` in the output dict is the Hersbach integral estimator (used
   for the budget identity).  `crps` is the independent fair energy-score
   estimator.  For large M and N these agree within ~1 %; for small ensembles
   (M < 20) they may differ by several percent.  The paper should report the
   fair estimator (`crps`) in tables; `crps_decomp` is for internal budget
   verification only.

3. **Reliability clamped to ≥ 0.**
   The raw `reli_inner + tails` can be slightly negative due to floating-point
   cancellation when the ensemble is very well calibrated.  It is clamped to 0.
   This does NOT affect the budget check (which uses `crps_decomp` directly).

4. **Binning effects for small M.**
   For M < 10 ensemble members, the M−1 inner bins are too few to resolve the
   forecast CDF accurately.  Reliability and Resolution become coarse-grained.
   Use M ≥ 50 for publication-quality decomposition.

5. **Seeding.**
   All clean-path scripts set seed 42 by default (configurable via `--seed`).
   Seed is applied to Python `random`, NumPy, and PyTorch before any random
   operation.  The seed is logged in every result JSON.

6. **Timing hardware comparability.**
   `sec_per_sample` values are only comparable across methods run on the same
   hardware (same device, same GPU if applicable).  `device_name` is recorded
   in every timing record; filter by this field before comparing runtimes.

7. **CUDA warm-up.**
   `SampleTimer` performs one warm-up forward pass before starting the clock.
   This amortizes CUDA kernel compilation overhead.  The warm-up uses a batch
   of ≤ 4 samples and is excluded from `elapsed_sec`.

8. **Calibration scaffolds (`calibration_pit_mean`, `calibration_pit_std`,
   `rank_hist_chi2`).**
   These are computed if `compute_calibration=True` (default).  For very large
   datasets or models that are slow on CPU, setting `compute_calibration=False`
   in `metrics_probabilistic.compute_all()` skips them; they appear as `null`
   in the output.

9. **Legacy scripts are not run via the clean path.**
   `eval/evaluate.py`, `compare_models.py`, `compare_methods.py`, and
   `run_full_pipeline.py` remain unchanged and fully runnable.  Their outputs
   must NOT be mixed with clean-path outputs in paper figures.

---

## Example Commands

### Full pipeline (clean path)
```bash
# With a JSON config file (recommended):
python scripts/run_full_pipeline_paper_clean.py \
    --config_file configs/paper_pipeline_clean.json \
    --output_dir  outputs/paper_clean/

# Dry run (validate wiring only):
python scripts/run_full_pipeline_paper_clean.py \
    --config_file configs/paper_pipeline_clean.json \
    --dry_run

# Via the shell wrapper:
bash scripts/run_full_pipeline_paper_clean.sh
```

### Single-model evaluation
```bash
python scripts/test_model_paper_clean.py \
    --checkpoint outputs/nftsf/single_well/run_001 \
    --data       outputs/canonical/single_well.npz \
    --method     nftsf \
    --dataset    single_well \
    --n_samples  500 \
    --seed       42 \
    --output_dir outputs/paper_clean/results/ \
    --timing_log outputs/paper_clean/timing.jsonl
```

### Comparison aggregation only (from pre-computed result JSONs)
```bash
python scripts/compare_models_paper_clean.py \
    --result_dir outputs/paper_clean/results/ \
    --output_dir outputs/paper_clean/
```

### Run unit tests
```bash
python -m pytest tests/test_crps_decomposition.py -v
```
