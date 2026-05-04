# NFTSF Paper Pipeline – Training & Evaluation Guide

This repository provides scripts and configuration files to train and evaluate several time‑series forecasting methods: **NFTSF**, **CSDI**, **TSDiff‑Cond** (conditional diffusion), **TSDiff** (unconditional diffusion), and **ARIMA**. The code supports multiple landscapes (single/double well, alanine dihedral angles) and forecast horizons (e.g., 25, 50 steps).

project_root/
├── configs/
│   ├── csdi_train/
│   ├── tsdiff_cond_train/
│   ├── tsdiff_train/
│   ├── tsdiff_forecast/
│   └── arima/
├── train/
│   ├── CSDI/
│   ├── TSDiff/
│   └── NFTSF/          (if applicable)
├── eval/
│   ├── CSDI/
│   ├── TSDiff/
│   └── ARIMA/
├── data/               (raw and normalised .npz files)
├── gluonts_datasets/   (GluonTS format for TSDiff)
├── checkpoints_*/      (model checkpoints)
└── results/            (forecast bundles)

## 📦 Environment Setup

Create and activate a dedicated Conda environment with Python 3.10:

```bash
conda create -n venv310 python=3.10
conda init
conda activate venv310
conda install pip
conda install --file requirements.txt