# Data

This directory describes how to obtain all datasets used in the paper.
No large dataset files are committed to this repository.

---

## Preprocessed files included in this repo

Two small preprocessed files are committed directly and require no download:

| File | Size | Contents |
|------|------|----------|
| `alanine_phi_train.npy` | 2.3 MB | Alanine-dipeptide φ dihedral angle — training split (multi_sim format) |
| `alanine_phi_test.npy`  | 592 KB | Alanine-dipeptide φ dihedral angle — test split (multi_sim format) |

These were produced from the raw MD trajectory via `splice_alanine_circular.py`
(see below). You can reproduce them from the raw source.

---

## Synthetic datasets (generated locally)

Synthetic landscapes are generated on-the-fly by `generate_trajectories.py`.
No download required.

```bash
# Linear Gaussian
python generate_trajectories.py --landscape linear_gaussian \
    --num_sims 9000 --output_path data/linear_gaussian_train.npy --seed 42
python generate_trajectories.py --landscape linear_gaussian \
    --num_sims 3000 --output_path data/linear_gaussian_test.npy  --seed 1

# Single Well (overdamped Langevin)
python generate_trajectories.py --landscape single_well \
    --num_sims 9000 --output_path data/single_well_train.npy --seed 42
python generate_trajectories.py --landscape single_well \
    --num_sims 3000 --output_path data/single_well_test.npy  --seed 1

# Double Well
python generate_trajectories.py --landscape double_well \
    --num_sims 9000 --output_path data/double_well_train.npy --seed 42
python generate_trajectories.py --landscape double_well \
    --num_sims 3000 --output_path data/double_well_test.npy  --seed 1
```

SDE-based landscapes (sw_sle_em, sw_gle_oe_em, dw_sle_em, dw_gle_oe_em) are
generated internally by `sde/sde_data_gen.py` when `--data_format sde` is
passed to `train_model.py` / `test_model.py`.

---

## Alanine-dipeptide MD dataset (raw source)

The raw molecular dynamics data is the publicly available
**alanine-dipeptide-3x250ns-backbone-dihedrals** dataset from the
[MDShark / HTMD project](https://github.com/Acellera/htmd):

```
alanine-dipeptide-3x250ns-backbone-dihedrals.npz
```

**TODO:** Add an anonymous permanent download link (e.g. Zenodo or anonymous
figshare) once one is created for the submission.

### Preprocessing steps

Once you have the raw `.npz`, run:

```bash
# Produce the circular-aware (sin/cos) multi_sim .npy splits used in the paper
python splice_alanine_circular.py \
    --input /path/to/alanine-dipeptide-3x250ns-backbone-dihedrals.npz \
    --output_train alanine_phi_train.npy \
    --output_test  alanine_phi_test.npy
```

This reproduces `alanine_phi_train.npy` and `alanine_phi_test.npy` already
committed to the repo.

---

## Data format reference

All `.npy` files use the **multi_sim** format unless otherwise noted:

```
shape: (T, 1 + N)
  column 0   : time index
  columns 1–N: simulation trajectories
```

Pass `--data_format multi_sim` (the default) to `train_model.py` and
`test_model.py` when using these files.
