# Audit Report: NFTSF_ssh × tsf_models-adam Pipeline Divergences

_Generated as part of Phase 1 of the unified integration plan._

---

## 1. Repository Overviews

### NFTSF_ssh (Primary)
| Item | Detail |
|------|--------|
| **Purpose** | Conditional Normalizing Flow for trajectory forecasting on molecular systems |
| **Key files** | `train_model.py`, `test_model.py`, `architecture.py`, `splice_alanine.py`, `generate_trajectories.py` |
| **Model** | Conditional NF — K blocks of Autoregressive RQS + LU Linear Permute (normflows library) |
| **Training framework** | Custom PyTorch loop |
| **Data** | Alanine Dipeptide dihedral angles (real MD, 750 chunks × 1000 steps) + synthetic Langevin dynamics |

### tsf_models-adam (Secondary — Mahmoud's baselines)
| Item | Detail |
|------|--------|
| **Purpose** | Diffusion-based time series forecasting baselines |
| **Models** | TSDiff-Q, TSDiff-MS, TSDiff-Cond, CSDI, RATD, NsDiff, ARIMA |
| **Key files** | `train_tsfdiff.py`, `train_cond_tsfdiff_model.py`, `train_csdi.py`, `train_ratd.py`, `train_nsdiff.py`, `run_auto_arima.py`, `plot_metrics.py`, `evaluate.py`, `metrics.py` |
| **Training framework** | GluonTS + PyTorch Lightning |
| **Data** | Synthetic Langevin dynamics (9000 trajectories × 500 steps) |

---

## 2. Divergence Analysis

### Divergence 1 — Data Format

| | NFTSF_ssh | tsf_models-adam |
|-|-----------|-----------------|
| **Format** | `.npy` files | `.npz` files |
| **Shape** | `(T, 1+NumSims)` — col 0 is time index, cols 1+ are trajectories | `positions: (N, T)` — rows are trajectories |
| **Source files** | `alanine_phi_train.npy`, `alanine_phi_test.npy` (shape `(1000, 1+601)` and `(1000, 1+149)`) | `data/double_well.npz`, `data/single_well.npz`, etc. |
| **Load path** | `np.load(path)` → `tensor.T` to get `(Batch, Time)` | `np.load(path)["positions"]` → `(N, T)` |

**Impact:** Direct data interchange is impossible without format conversion.

**Resolution:** Canonical `.npz` schema (`positions: (N, T)` + metadata) used throughout the new pipeline. Both repos' data are converted once to this format.

---

### Divergence 2 — Normalization

| | NFTSF_ssh | tsf_models-adam |
|-|-----------|-----------------|
| **Method** | Z-score: `(x − μ) / σ` — global statistics across all train data | Mean-absolute scaling (MeanScaler): `x / mean(|x|)` per series |
| **Source** | `normalize_data()` in `train_model.py:250–283` | `MeanScaler` in `models/unconditional_time_series_diffusion/src/uncond_ts_diff/model/linear/_scaler.py` |
| **Storage** | `norm_stats_{ts}.npz` (mean + std saved) | Embedded in model checkpoint (`std.npy` for CSDI) |
| **Config key** | `--normalize` flag (default True) | `normalization: "mean"` in YAML config |

**Impact:** Models trained with different normalizations are not comparable; unit recovery at evaluation time uses wrong formula.

**Resolution:** Z-score is canonical. The unified `data/preprocessing.py` applies Z-score to all models. GluonTS adapter pre-normalises before constructing the GluonTS dataset, disabling the model's internal scaling.

---

### Divergence 3 — Random Seed

| | NFTSF_ssh | tsf_models-adam |
|-|-----------|-----------------|
| **Training seed** | 42 (CLI `--seed`, default) | **12 — hardcoded** in `generator.py:206` |
| **Test seed** | 29182 (CLI `--seed` in `test_model.py:68`) | Not separately configurable |
| **Scope set** | `torch`, `numpy`, `cuda` | `torch`, `numpy`, `cuda` |
| **Python `random`** | Not set | Not set |

**Impact:** Data splits and weight initialisation differ between any two model runs, making comparisons non-deterministic.

**Resolution:** Canonical seed = **42**, set in `configs/base.yaml`. Applied via unified `set_seed()` function at start of every training run. Python `random` module is also seeded.

---

### Divergence 4 — Train / Test Split

| | NFTSF_ssh | tsf_models-adam |
|-|-----------|-----------------|
| **Strategy** | **Trajectory-level split** — 80 % of trajectory chunks assigned to train, 20 % to test; chunks never share data across splits | **Time-based split** — all trajectories used; train = first 400 / 500 steps; test = last 100 steps |
| **Validation** | 10 % of all sliding-window segments from train trajectories, randomly permuted | `OffsetSplitter` removes last `prediction_length` steps from each train trajectory |
| **Source** | `splice_alanine.py:89–92` (trajectory shuffle + split), `train_model.py:380–389` (segment val split) | `data/configs/double_well.yaml: train_test_split: 400`, `train_tsfdiff.py:97–122` |

**Impact:** Different models "see" fundamentally different distributions of training and test samples.

**Resolution (agreed with user):** NFTSF's trajectory-level 80 / 20 split is canonical. `data/splitter.py::trajectory_split()` implements it with seed=42. GluonTS adapter constructs per-trajectory windows from the **train trajectories only**, ensuring both data adapters use identical trajectory assignments.

---

### Divergence 5 — Training Framework

| | NFTSF_ssh | tsf_models-adam |
|-|-----------|-----------------|
| **Framework** | Custom PyTorch training loop | GluonTS + PyTorch Lightning |
| **Batching** | In-memory tensor slicing, configurable `batch_size` | `TrainDataLoader` with GluonTS transforms |
| **Callbacks** | Manual early stopping + `ReduceLROnPlateau` scheduler | `ModelCheckpoint`, `RichProgressBar`, `EvaluateCallback` |
| **Checkpoints** | `checkpoint_epoch_{N}.pth` dict with model + optimizer + scheduler + loss history | PyTorch Lightning `save_top_k=3` by train loss |

**Impact:** No unified way to train all models without two separate execution paths.

**Resolution:** `train/train.py` is the single entry point; it dispatches to `engine_pytorch.py` (for NFTSF) or `engine_lightning.py` (for diffusion models). Both engines share the same seed-setting, config-loading, and output-directory conventions.

---

### Divergence 6 — Prediction Horizon Implementation

| | NFTSF_ssh | tsf_models-adam |
|-|-----------|-----------------|
| **n_past / context_length** | 100 (CLI default) | 100 (YAML config default) |
| **n_future / prediction_length** | 100 (CLI default) | 100 (YAML config default) |
| **Window construction** | Sliding windows over train trajectories: `extract_segments(tracks, 100, 100, stride=1)` | GluonTS `InstanceSplitter` with `ExpectedNumInstanceSampler` (random windows during train) / `TestSplitSampler` (fixed end-aligned window at test) |
| **Test evaluation** | Separate test `.npy` files loaded; all windows from test trajectories | Last 100 steps of each test entry in GluonTS dataset |

**Impact:** At test time, NFTSF evaluates on many windows per trajectory while GluonTS models evaluate on a single end-aligned window.

**Resolution:** Unified test protocol: for each held-out test trajectory, take **one** evaluation window — context = steps `[T-200 : T-100]`, target = steps `[T-100 : T]` (last `n_past` + `n_future` steps). `eval/evaluate.py` constructs this window identically for all models.

---

### Divergence 7 — Output & Metrics Format

| | NFTSF_ssh | tsf_models-adam |
|-|-----------|-----------------|
| **Results storage** | `test_trajectories_{ts}.npz` (ground truth + sample arrays), `results_summary_{ts}.json` | `results/{model}/{dataset}.npz` with `samples`, `ground_truth`, `ci90_lower/upper`, `ci50_lower/upper`, `time_test`, etc. |
| **Metrics computed** | CRPS, MAE, per-step MAE, empirical quantile coverage | CRPS per step, MAE per step, CI90/CI50 coverage per step |
| **Metrics library** | Custom implementation in `test_model.py` | `properscoring` via `metrics.py::crps()`, `mae()`, `ci_coverage()` |
| **Visualization** | Per-trajectory confidence bands, reliability diagrams, 2D histograms, error-vs-horizon plots | 2×2 metric grid (CRPS, MAE, CI90, CI50 vs time), auto-colored by model name |

**Impact:** Results from the two repos cannot be directly compared or plotted together.

**Resolution:** `eval/evaluate.py` defines one canonical `results.npz` schema (see below). `viz/comparison.py` reads only this schema.

---

## 3. Canonical `results.npz` Schema

All models must produce a `results.npz` containing:

```
samples           : (N_test, n_future, S)  float32 — S forecast samples per trajectory
ground_truth      : (N_test, n_future)     float32 — observed future values
ci90_lower        : (N_test, n_future)     float32 — 5th percentile
ci90_upper        : (N_test, n_future)     float32 — 95th percentile
ci50_lower        : (N_test, n_future)     float32 — 25th percentile
ci50_upper        : (N_test, n_future)     float32 — 75th percentile
crps_t            : (n_future,)            float32 — CRPS per forecast step
mae_t             : (n_future,)            float32 — MAE per forecast step
rmse_t            : (n_future,)            float32 — RMSE per forecast step
ci90_t            : (n_future,)            float32 — CI90 coverage per step
ci50_t            : (n_future,)            float32 — CI50 coverage per step
model_name        : str
landscape         : str
run_id            : str
```

---

## 4. Canonical Data `.npz` Schema

Produced by `data/canonical.py`, consumed by all data adapters:

```
positions         : (N, T)      float32 — all trajectories, normalised (Z-score)
positions_raw     : (N, T)      float32 — un-normalised original values
train_indices     : (N_train,)  int64   — 80 % trajectory indices (seeded permutation)
test_indices      : (N_test,)   int64   — 20 % trajectory indices
time              : (T,)        float32 — time axis
landscape         : str
seed              : int         — 42
n_past            : int         — 100
n_future          : int         — 100
norm_mean         : float       — Z-score μ (computed on train set only)
norm_std          : float       — Z-score σ
```

---

## 5. Minimal Change Set for Unification

| Priority | Change | Where | Scope |
|----------|--------|--------|-------|
| 1 | Produce canonical `.npz` from existing NFTSF `.npy` files | `data/canonical.py` | New file |
| 2 | Implement trajectory-level split with seed=42 | `data/splitter.py` | New file |
| 3 | Z-score normalization on train-set statistics only | `data/preprocessing.py` | New file (mirrors NFTSF logic) |
| 4 | NFTSF adapter: canonical → segment DataLoaders | `data/adapters/nftsf_adapter.py` | New file |
| 5 | GluonTS adapter: canonical → InMemoryDataset | `data/adapters/gluonts_adapter.py` | New file |
| 6 | Model wrappers implementing shared `BaseModel` | `models/` | New files (no source modification) |
| 7 | Unified training entry point + seed propagation | `train/train.py` | New file |
| 8 | Canonical evaluation metrics + results writer | `eval/evaluate.py` | New file |
| 9 | Auto-discovery comparison visualization | `viz/comparison.py` | New file |

**Zero lines of code in NFTSF_ssh or tsf_models-adam are modified.**

---

## 6. Proposed Module Structure

```
unified_trajectory_forecasting/
├── audit_report.md          ← this file
├── README.md
├── configs/
│   ├── base.yaml            ← canonical shared params (seed, split, horizon)
│   ├── nftsf.yaml
│   ├── tsdiff_q.yaml
│   ├── tsdiff_ms.yaml
│   ├── tsdiff_cond.yaml
│   ├── csdi.yaml
│   ├── ratd.yaml
│   ├── nsdiff.yaml
│   └── arima.yaml
├── data/
│   ├── __init__.py
│   ├── canonical.py         ← build/load canonical .npz
│   ├── preprocessing.py     ← Z-score normalization, sliding-window extraction
│   ├── splitter.py          ← trajectory-level + segment-level splits
│   ├── loader.py            ← get_dataloader(config) unified entry point
│   └── adapters/
│       ├── __init__.py
│       ├── nftsf_adapter.py ← canonical → NFTSF DataLoaders
│       └── gluonts_adapter.py ← canonical → GluonTS InMemoryDataset
├── models/
│   ├── __init__.py
│   ├── base.py              ← AbstractBaseModel interface
│   ├── nftsf.py             ← wraps NFTSF_ssh/architecture.py (unmodified)
│   ├── tsdiff.py            ← wraps tsf_models-adam TSDiff variants
│   ├── csdi.py
│   ├── ratd.py
│   ├── nsdiff.py
│   └── arima.py
├── train/
│   ├── __init__.py
│   ├── train.py             ← unified CLI entry point
│   ├── engine_pytorch.py    ← custom loop for NFTSF
│   └── engine_lightning.py  ← PL dispatch for diffusion models
├── eval/
│   ├── __init__.py
│   └── evaluate.py          ← canonical metrics + results.npz writer
├── viz/
│   ├── __init__.py
│   └── comparison.py        ← auto-discovery + publication-quality plots
└── outputs/
    └── {model}/{landscape}/{run_id}/
        ├── model.pth
        ├── config.json
        ├── results.npz
        └── plots/
```
