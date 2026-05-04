# NF-TSF: Normalizing Flow Time Series Forecasting

> Author and funding information omitted for anonymous review.

<!-- TODO: fill from paper — replace the title above and add a 2–3 sentence abstract snippet here. -->

NF-TSF is a conditional normalizing flow model for probabilistic trajectory
forecasting. Given a context window of past observations, the model learns a
flow-based distribution over future trajectories and supports exact
likelihood evaluation, calibrated quantile estimation, and fast sampling.

---

## Requirements

Python 3.10+ and a CUDA-capable GPU are recommended for training.

```bash
# 1. Create and activate a conda environment
conda create -n nf_tsf python=3.10
conda activate nf_tsf

# 2. Install PyTorch with CUDA (adjust cuda version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 \
      -c pytorch -c nvidia

# 3. Install remaining dependencies
pip install -r requirements.txt
```

`requirements.txt` pins the versions used during the experiments reported in
the paper.

### Environment setup helper (SLURM / HPC)

```bash
bash setup_environment.sh
```

---

## Training

### Quick start — single landscape

```bash
python train_model.py \
    --data_path alanine_phi_train.npy \
    --data_format multi_sim \
    --n_past 100 --n_future 100 \
    --flow_blocks 6 --hidden_units 64 --hidden_layers 1,2 \
    --tail_bound 30 \
    --epochs 1000 --learning_rate 1e-3 \
    --batch_size 4096 --stride 5 \
    --weight_decay 1e-5 --seed 42 \
    --normalize --use_scheduler \
    --output_dir results/alanine_phi \
    --model_name nftsf_alanine_phi
```

### Model variants

| Variant | Flag | Description |
|---------|------|-------------|
| `ar` (default) | `--model_variant ar` | A-RQS autoregressive flow; raw standardized context |

### Full pipeline (all landscapes, SLURM)

```bash
# Default variant (ar), all landscapes
sbatch run_full_pipeline.sh

# Override landscape subset via env vars
RUN_LANDSCAPES="single_well double_well" sbatch run_full_pipeline.sh
```

Key hyperparameters can be overridden without editing the script:

```bash
EPOCHS=500 FLOW_BLOCKS=4 HIDDEN_UNITS=32 sbatch run_full_pipeline.sh
```

### Hyperparameter sweep (SLURM array job)

```bash
sbatch run_sweep.sh
```

---

## Evaluation

### Single model evaluation

```bash
python test_model.py \
    --model_path results/alanine_phi/nftsf_alanine_phi.pth \
    --norm_stats_path results/alanine_phi/norm_stats_<timestamp>.npz \
    --data_path alanine_phi_test.npy \
    --data_format multi_sim \
    --n_past 100 --n_future 100 \
    --n_samples 500 \
    --output_dir results/alanine_phi_eval
```

Pass `--config_path results/alanine_phi/config_<timestamp>.json` to
automatically restore the correct model architecture.

### Full test suite (all landscapes)

```bash
bash run_all_tests.sh
```

### Reproduce paper figures

```bash
# Trajectory comparison figure (requires test output .npz)
python scripts/make_paper_trajectory_figure.py \
    --config configs/paper_trajectory_figure_example.json

# Multi-model comparison plots
bash run_compare_models.sh
```

### Generating synthetic data (if needed before training)

```bash
python generate_trajectories.py \
    --landscape single_well --num_sims 9000 \
    --output_path data/single_well_train.npy --seed 42
```

See `data/README.md` for the full dataset acquisition guide.

---

## Pre-trained Models

See `checkpoints/README.md` for the expected directory layout and instructions
for pointing `test_model.py` at a downloaded checkpoint.

---

## Results


**Exact commands used to produce these numbers:**

```bash
# Train (ar variant, single well, paper hyperparameters)
MODEL_VARIANT=ar RUN_LANDSCAPES="single_well" sbatch run_full_pipeline.sh

# Evaluate
bash run_all_tests.sh

# Aggregate comparison
bash run_compare_models.sh
```

---

## Repository layout

```
.
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup_environment.sh         # HPC environment bootstrap
│
├── train_model.py               # Training entry point
├── test_model.py                # Evaluation entry point
├── generate_trajectories.py     # Synthetic data generation
│
├── architecture.py              # A-RQS normalizing flow definition
├── architecture_encoder.py      # GRU encoder variants
├── generators.py                # Synthetic Langevin dynamics generators
├── timing_utils.py              # Inference timing helpers
│
├── sde/                         # SDE integrators and data generation
│   ├── config_reader.py
│   ├── sde_data_gen.py
│   └── sde_integrators.py
│
├── sde_loader.py                # SDE data loading utilities
├── sde_preprocess.py            # SDE data preprocessing
├── splice_alanine.py            # Alanine-dipeptide data splitter
├── splice_alanine_circular.py   # Circular-aware alanine splitter (paper version)
│
├── compare.py                   # Figure generation (SVG/PNG/CSV)
├── compare_models.py            # Multi-model comparison figures
├── calibration_diagnostics.py   # Quantile calibration diagnostics
├── metrics_crps_decomp.py       # CRPS + Hersbach decomposition
├── plot_loss.py                 # Training loss curves
├── plot_random_training_trajectories.py  # Training trajectory visualization
├── replot_test_results.py       # Replot from saved .npz results
│
├── configs/                     # Paper figure configurations
│   └── paper_trajectory_figure_example.json
│
├── scripts/                     # Figure scripts
│   └── make_paper_trajectory_figure.py
│
├── run_training.sh              # SLURM: single training job
├── run_testing.sh               # SLURM: single evaluation job
├── run_sweep.sh                 # SLURM: hyperparameter sweep array
├── run_all_tests.sh             # Local: evaluate all landscapes
├── run_full_pipeline.sh         # SLURM: end-to-end pipeline
├── run_compare_models.sh        # SLURM: multi-model comparison
├── run_plot_trajectories.sh     # SLURM: trajectory plotting
│
├── alanine_phi_train.npy        # Preprocessed MD data — training split
├── alanine_phi_test.npy         # Preprocessed MD data — test split
│
├── data/
│   └── README.md                # Dataset acquisition instructions
└── checkpoints/
    └── README.md                # Pre-trained model download instructions
```

---

## License

<!-- TODO: add license before camera-ready submission. -->

---

## Contributing

This repository is released for reproducibility purposes accompanying an
anonymous NeurIPS submission. Issues and pull requests are welcome after
the review period.
