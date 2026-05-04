markdown
# NFTSF: Neural Forecasting for Time Series

NFTSF is a framework for training and evaluating advanced time series forecasting models. It supports multiple state-of-the-art methods including **CSDI**, **TSDiff-Cond**, **TSDiff**, and **ARIMA**. This repository provides training pipelines, evaluation scripts, and example configurations for different datasets.

---

## Features
- Train and evaluate **NFTSF** (Normalizing-Flow Time Series Forecasting)
- Train and evaluate **CSDI** (Conditional Score-based Diffusion Imputation)
- Train and evaluate **TSDiff-Cond** (Conditional Time Series Diffusion)
- Train and evaluate **TSDiff** (Unconditional Time Series Diffusion)
- Run **ARIMA** for time series forecasting


## Installation

Create a Python environment and install required dependencies:

```bash
# Create Python 3.10 environment
conda create -n venv310 python=3.10
conda init
conda activate venv310
```

# Install pip and required packages
```
conda install pip
conda install --file requirements.txt
````

---

## Usage

### 1. NFTSF

```bash
python eval/NF/forecast_nf.py \
    --model_path   checkpoints_25_25/nf/alanine_phi_25_25.pth     \
    --config configs/nf/alanine_phi_25_25_forecast.yaml \
    --data_path    data/alanine_phi_test.npz    \
    --out          results/nf/alanine_phi_25_25.npz \
    --context_length     25    \
    --prediction_length     25   \
    --n_samples    1000 \
    --train_test_split 900 \
    --test_size 3000
```
### 2. CSDI

**Train CSDI:**

```bash
python train/CSDI/train_csdi.py \
    --config configs/csdi_train/alanine_phi_25_25.yaml \
    --input  data/alanine_phi_train.npz \
    --out    checkpoints_25_25_dummy/csdi/alanine_phi \
    --device cuda:0
```

**Evaluate CSDI:**

```bash
python eval/CSDI/forecast_csdi.py \
    --config configs/csdi_train/alanine_phi_25_25.yaml \
    --input  DATA/alanine_phi_test.npz \
    --ckpt   checkpoints_25_25_dummy/csdi_with_early_stopping/alanine_phi/model.pth \
    --out    results/csdi_with_early_stopping/alanine_phi_25_25.npz \
    --device cuda:0
```

---

### 3. TSDiff-Cond

**Install TSDiff-Cond:**

```bash
cd architectures/unconditional_time_series_diffusion
pip install -e .
```

**Train TSDiff-Cond:**

```bash
python train/TSDiff/train_cond_tsdiff.py \
    --dataset_path gluonts_datasets/alanine_phi \
    --config configs/tsdiff_cond_train/alanine_phi_25_25.yaml \
    --out_dir checkpoints_25_25_dummy/tsdiff_cond/alanine_phi
```

**Evaluate TSDiff-Cond:**

```bash
python eval/TSDiff/forecast_tsdiff_cond.py \
    --config configs/tsdiff_forecast/alanine_phi_cond_25_25.yaml \
    --dataset_path gluonts_datasets/alanine_phi \
    --out results/tsdiff_cond/alanine_phi_25_25.npz
```

---

### 4. TSDiff

**Train TSDiff:**

```bash
python train/TSDiff/train_tsdiff.py \
    --dataset_path gluonts_datasets/alanine_phi \
    --config configs/tsdiff_train/alanine_phi_25_25.yaml \
    --out_dir checkpoints_25_25_dummy/tsdiff/alanine_phi
```

**Evaluate TSDiff:**

```bash
python eval/TSDiff/forecast_tsdiff.py \
    --config configs/tsdiff_forecast/alanine_phi_mse_03_25_25.yaml \
    --dataset_path gluonts_datasets/alanine_phi \
    --out results/tsdiff_mse/alanine_phi_mse_03_25_25.npz

python eval/TSDiff/forecast_tsdiff.py \
    --config configs/tsdiff_forecast/alanine_phi_q_4_25_25.yaml \
    --dataset_path gluonts_datasets/alanine_phi \
    --out results/tsdiff_q/alanine_phi_q_4_25_25.npz
```

---

### 4. ARIMA

**Run ARIMA forecasting:**

```bash
python eval/ARIMA/run_auto_arima.py \
    -c configs/arima/25_25.yaml \
    --input data/double_well_test.npz \
    --out results/arima/double_well_25_25.npz \
    --n_jobs 8
```

---

## Configuration

All training and evaluation scripts rely on YAML configuration files located in `configs/`. Examples include:

* `csdi_train/alanine_phi_25_25.yaml` for CSDI
* `tsdiff_cond_train/alanine_phi_25_25.yaml` for TSDiff-Cond
* `tsdiff_train/alanine_phi_25_25.yaml` for TSDiff
* `arima/25_25.yaml` for ARIMA

Modify these configs to adjust dataset paths, hyperparameters, and training options.

---

## Notes

* Ensure you have CUDA enabled if using GPU acceleration.
* Paths to datasets, checkpoints, and outputs may need to be adjusted based on your local setup.
* TSDiff-Cond requires installation in editable mode (`pip install -e .`) before running training or evaluation.

---

## Results

All evaluation results will be saved in the `results/` folder, organized by model type and dataset.