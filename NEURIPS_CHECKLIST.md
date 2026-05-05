# NeurIPS Reproducibility Checklist

Branch: `claude/anonymize-plots-neurips-Nb5PF`  
Generated: 2026-05-05  
Submission type: Double-blind (NeurIPS)

---

## Checklist Table

| Item | Status | Location / Notes |
|---|---|---|
| 1. Specification of dependencies | ✅ | `requirements.txt` (root) — fully pinned. Sub-module deps: `NFTSF/requirements.txt`, `architectures/CSDI/requirements.txt`, `architectures/unconditional_time_series_diffusion/requirements.txt` + `pyproject.toml`. |
| 2. Training code | ✅ | `train/CSDI/train_csdi_with_early_stopping.py`, `train/NFTSF/train_nftsf.py`, `train/TSDiff/train_cond_tsdiff_with_early_stopping.py`, `train/TSDiff/train_tsdiff.py`. Per-dataset/split SLURM scripts in `scripts/`. |
| 3. Evaluation code | ✅ | `eval/ARIMA/run_auto_arima.py`, `eval/CSDI/forecast_csdi.py`, `eval/NF/forecast_nf.py`, `eval/TSDiff/forecast_tsdiff.py`, `eval/TSDiff/forecast_tsdiff_cond.py`. |
| 4. Pre-trained models (link) | ✅ | Zenodo anonymous preview link in `README.md` § *Downloading the Zenodo Data Archive*. Checkpoints in `checkpoints_25_25/` and `checkpoints_50_50/` (4 models × 4 datasets × 2 splits). |
| 5. README with results + commands | ⚠️ | `README.md` — training and eval commands are present and verified against code. **Results table metric values are TODO** (must be filled from paper draft or by running eval on Zenodo checkpoints). |
| 6. Figure reproduction | ✅ | `compare.py` (this branch) reads `results/*.npz` and writes SVG/PNG. Usage documented in `README.md` § *Reproducing the Plots*. Exact per-figure CLI args are marked TODO. |
| 7. Anonymization — content | ✅ | Scan clean: no author emails, names (`mahmuod`, `mea`, `@gmail`), institution keywords (`arizona state`, `asu`), or credential patterns found in any tracked text/code/config file. SLURM `--partition`, `--qos`, `--gres` headers replaced with `# TODO` placeholders. |
| 8. Anonymization — commit history | ⚠️ | Pre-existing commits on `plots` are authored by `mea <mahmuodimad@gmail.com>`. History was **not rewritten** (per task constraint). Anyone who clones and runs `git log` will see this identity. **Action required before sharing the repo publicly:** squash or filter history, or distribute only as a ZIP/tarball. |
| 9. Zenodo folders excluded from git | ✅ | All 7 folders removed from index (`git rm --cached`). `.gitignore` updated to permanently exclude them. `git ls-files` returns 0 matches for these paths. |
| 10. Zenodo download instructions | ✅ | `README.md` § *Downloading the Zenodo Data Archive* — step-by-step with anonymous preview link, unzip commands, and wrapper-directory fallback. |
| 11. License | ⚠️ | No root `LICENSE` file present. Vendored baselines carry their own (MIT for CSDI, Apache-2.0 for TSDiff). **TODO: add a LICENSE file for the original NFTSF code.** |
| 12. Paper title | ⚠️ | README uses `Neural Flow Time Series Forecasting (NFTSF)` as a placeholder title. **TODO: confirm exact paper title from the submission.** |

---

## Status Key

| Symbol | Meaning |
|---|---|
| ✅ | Fully addressed |
| ⚠️ | Partially addressed — action required before camera-ready / public release |
| ❌ | Not addressed |

---

## Action Items Before Public Release / Camera-Ready

1. **Results table** (`README.md` lines 331–340): fill metric names and values from paper or by running `eval/` scripts on Zenodo checkpoints.
2. **Exact plot commands** (`README.md` line 315): record the precise `compare.py` invocation(s) used to generate each paper figure.
3. **Zenodo archive filename** (`README.md` line 79): replace `<ARCHIVE_NAME>.zip` with the actual filename from the Zenodo record page.
4. **GPU spec** (`README.md` line 135): confirm GPU model/count used for experiments.
5. **Paper title** (`README.md` line 1): replace placeholder with the exact submission title.
6. **LICENSE file**: add a root `LICENSE` file for the original NFTSF code.
7. **Commit history**: before sharing the repo URL publicly, rewrite/squash history to remove author identity (`mea <mahmuodimad@gmail.com>`) from `git log`, or distribute as a ZIP without `.git/`.
8. **De-anonymize**: replace all `Anonymous` placeholders in README and scripts with real author/institution details once the review period ends.
