# Unified Trajectory Forecasting Pipeline

A unified comparison harness for **NFTSF** (Conditional Normalizing Flow) and
Mahmoud's **diffusion baselines** (TSDiff-Q/MS/Cond, CSDI, RATD, NsDiff, ARIMA).

All models are trained and evaluated on **identical data** — same trajectories,
same random seed (42), same 80/20 train/test split, same prediction horizon
(n_past=100, n_future=100) — enabling apples-to-apples comparison.

---

## Repository layout

```
unified_trajectory_forecasting/
├── audit_report.md          Full divergence analysis between the two repos
├── configs/
│   ├── base.yaml            Canonical shared params (seed, split, horizon)
│   └── {model}.yaml         Per-model hyperparameter overrides
├── data/
│   ├── canonical.py         Build canonical .npz from NFTSF .npy or raw .npz
│   ├── preprocessing.py     Z-score normalisation + sliding-window segmentation
│   ├── splitter.py          Trajectory-level + segment-level splits
│   ├── loader.py            get_dataloader(config) unified entry point
│   └── adapters/
│       ├── nftsf_adapter.py    canonical → PyTorch DataLoaders
│       └── gluonts_adapter.py  canonical → GluonTS ListDataset
├── models/
│   ├── base.py              AbstractBaseModel interface
│   ├── nftsf.py             NFTSF Normalizing Flow wrapper
│   ├── tsdiff.py            TSDiff-Q / MS / Cond wrapper
│   ├── csdi.py              CSDI wrapper
│   ├── ratd.py              RATD wrapper
│   ├── nsdiff.py            NsDiff wrapper
│   └── arima.py             Auto-ARIMA wrapper
├── train/
│   ├── train.py             Unified CLI entry point
│   ├── engine_pytorch.py    Custom loop for NFTSF
│   └── engine_lightning.py  PL dispatch for diffusion models
├── eval/
│   └── evaluate.py          Canonical metrics + results.npz writer
├── viz/
│   └── comparison.py        Auto-discover + publication plots
└── outputs/
    └── {model}/{landscape}/{run_id}/
        ├── config.json
        ├── model_best.pth
        └── results.npz
```

---

## Prerequisites

Install all dependencies from both source repos:

```bash
# From NFTSF_ssh
pip install normflows torch numpy matplotlib tqdm

# From tsf_models-adam (for diffusion baselines)
pip install gluonts pytorch-lightning properscoring

# ARIMA baseline
pip install pmdarima

# Optional: tueplots for publication-style figures
pip install tueplots
```

Both source repos must live as siblings of this directory:
```
/home/user/NFTSF_ssh/
/home/user/tsf_models-adam/
/home/user/unified_trajectory_forecasting/   ← this repo
```

### Python runner configuration (HPC-safe)

Shell launchers in this repo source `runner.sh`, which centralizes Python
execution via:

- `ENV_NAME` (default: `unified_tsf`)
- `conda run -n "$ENV_NAME" python`

To switch environments without editing scripts:

```bash
ENV_NAME=my_env bash train_sol.sh
```

---

## Step 0 — Build a canonical dataset

### From existing NFTSF Alanine Dipeptide .npy files

```bash
cd /home/user/unified_trajectory_forecasting

conda run -n unified_tsf python data/canonical.py \
  --source    nftsf \
  --train     /home/user/NFTSF_ssh/alanine_phi_train.npy \
  --test      /home/user/NFTSF_ssh/alanine_phi_test.npy \
  --landscape alanine_phi \
  --out       outputs/canonical/alanine_phi.npz
```

### From a raw (N, T) .npz (e.g. Mahmoud's generator output)

```bash
conda run -n unified_tsf python data/canonical.py \
  --source    npz \
  --input     /home/user/tsf_models-adam/data/double_well.npz \
  --landscape double_well \
  --out       outputs/canonical/double_well.npz
```

The canonical .npz encodes the 80/20 trajectory split and Z-score
normalisation statistics so every downstream model uses identical data.

---

## Step 1 — Train any model

```bash
# NFTSF Normalizing Flow
conda run -n unified_tsf python train/train.py \
  --model  nftsf \
  --config configs/nftsf.yaml \
  --data   outputs/canonical/alanine_phi.npz \
  --run_id run_001

# TSDiff-Q
conda run -n unified_tsf python train/train.py \
  --model  tsdiff_q \
  --config configs/tsdiff_q.yaml \
  --data   outputs/canonical/alanine_phi.npz \
  --run_id run_001

# CSDI
conda run -n unified_tsf python train/train.py \
  --model  csdi \
  --config configs/csdi.yaml \
  --data   outputs/canonical/alanine_phi.npz

# ARIMA (no gradient training; runs instantly)
conda run -n unified_tsf python train/train.py \
  --model  arima \
  --config configs/arima.yaml \
  --data   outputs/canonical/alanine_phi.npz
```

Checkpoints and configs are saved to:
`outputs/{model}/{landscape}/{run_id}/`

---

## Step 2 — Evaluate a trained model

```bash
conda run -n unified_tsf python eval/evaluate.py \
  --checkpoint outputs/nftsf/alanine_phi/run_001/ \
  --data        outputs/canonical/alanine_phi.npz \
  --n_samples   500
```

This writes `results.npz` (canonical schema) to the checkpoint directory.

Repeat for every model you want to compare.

---

## Step 3 — Generate comparison plots

```bash
conda run -n unified_tsf python viz/comparison.py \
  --results_dir outputs/ \
  --landscape   alanine_phi \
  --trajectory_id 42 \
  --pdf
```

The script **auto-discovers** all models that have a `results.npz` for the
requested landscape.  No model names need to be specified.

Output files (saved to `outputs/alanine_phi/plots/`):
- `comparison_traj42.png` / `.pdf`  — per-trajectory plot
- `aggregate_metrics.png` / `.pdf`  — 2×2 metric grid
- `aggregate_bar.png` / `.pdf`      — mean CRPS / MAE bar chart

---

## Canonical data schema (`results.npz`)

| Key            | Shape                      | Description                      |
|----------------|----------------------------|----------------------------------|
| `samples`      | (N_test, n_future, S)      | S forecast sample paths          |
| `ground_truth` | (N_test, n_future)         | Observed future values           |
| `ci90_lower/upper` | (N_test, n_future)     | 5th / 95th percentiles           |
| `ci50_lower/upper` | (N_test, n_future)     | 25th / 75th percentiles          |
| `crps_t`       | (n_future,)                | CRPS per forecast step           |
| `mae_t`        | (n_future,)                | MAE per forecast step            |
| `rmse_t`       | (n_future,)                | RMSE per forecast step           |
| `ci90_t`       | (n_future,)                | CI90 coverage per step           |
| `ci50_t`       | (n_future,)                | CI50 coverage per step           |
| `model_name`   | str                        | Model identifier                 |
| `landscape`    | str                        | Landscape identifier             |
| `run_id`       | str                        | Training run identifier          |

---

## Extending the pipeline

### Add a new model

1. Create `models/my_model.py` implementing `BaseModel`.
2. Add `"my_model"` to `models/registry.py::get_model()`.
3. Add `configs/my_model.yaml`.
4. If the model uses a custom training loop, add it to `_PYTORCH_MODELS`
   or `_GLUONTS_MODELS` in `train/train.py`.

### Add a new landscape / coordinate label

Add an entry to `COORDINATE_LABEL_MAP` in `viz/comparison.py`:
```python
COORDINATE_LABEL_MAP["my_landscape"] = r"My Quantity [units]"
```

---

## Key design constraints

- **NFTSF_ssh is never modified** — `models/nftsf.py` imports from it via
  `sys.path` insertion.
- **tsf_models-adam is never modified** — all model wrappers import from it
  the same way.
- **Canonical seed = 42** — set in `configs/base.yaml`; cannot be overridden
  by per-model configs.
- **Canonical split = 80/20 trajectory-level** — same NFTSF logic
  (`data/splitter.py::trajectory_split`).
- **Canonical normalisation = Z-score** — computed on train trajectories only
  and stored in the canonical `.npz`.
