# Where we left off — resume guide

**Last update:** 2026-11-11 · **Branch:** `PIA_devl` · **Current milestone:** M3 (baseline model)

## 1. Status at a glance

| Milestone | Status |
|---|---|
| M0 Setup & governance | ✅ Done (docs, hook, Option B, package scaffold) |
| M1 Data | 🟡 Synthetic generator (S2) done · public MSK data (S1) and hospital data (S3) pending |
| M2 Data pipeline | 🟡 Manifest + patient-level split done · de-identification, EDA pending |
| **M3 Baseline model** | 🟡 **Code done and tested; first real GPU training not completed** (see §3) |
| M4–M8 | ⏳ Not started |

Code that exists and passes tests (61 tests):
`grading.py`, `metrics.py`, `data/manifest.py`, `data/split.py`, `data/synthetic.py`,
`data/dataset.py`, `models/net.py`, `train.py`, `evaluate.py`.

## 2. Set up the machine again (needs internet once)

```bash
git config core.hooksPath .githooks
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/python -m pip install "timm>=1.0" "pyyaml>=6.0"
.venv/Scripts/python -m pytest
```
If `.venv` still exists, skip the installs. Pretrained EfficientNet-B0 weights are
downloaded from the Hugging Face hub on first training run (needs internet; cached afterwards).
Offline alternative: `--set model.pretrained=false`.

Regenerate synthetic data (the previous copy was in a temporary folder):
```bash
.venv/Scripts/rausnet-synth --out D:/rausnet_data/synth --patients 200 --seed 0
```
Then set `RAUSNET_DATA_DIR=D:/rausnet_data` (or pass `--set data.root=D:/rausnet_data/synth`).

## 3. Exactly where it stopped

First GPU training of `configs/train/baseline.yaml` failed twice **before epoch 1**:

1. `OpenBLAS error: Memory allocation still failed` with `num_workers: 4`.
   → Fix committed: `train.py` sets `OPENBLAS/OMP/MKL_NUM_THREADS=1` for workers. **Not yet verified.**
2. `OSError: [WinError 1455] page file too small` loading `cufft64_11.dll` in each worker
   (every Windows worker process loads the CUDA DLLs).
   → `baseline.yaml` now uses `num_workers: 0` (safe). **Not yet verified.**

The 1 200-image set is small, so `num_workers: 0` should be fast enough.
Optional later: try `--set train.num_workers=2`, or enlarge the Windows page file
(System → Advanced → Performance → Virtual memory) — a system setting for you to change.

## 4. Next steps, in order

1. [ ] Smoke run: `python -m rausnet.train --config configs/train/smoke.yaml --set data.root=D:/rausnet_data/synth`
2. [ ] Baseline run: `python -m rausnet.train --config configs/train/baseline.yaml` (30 epochs, early stopping).
3. [ ] `python -m rausnet.evaluate --run outputs/<run_dir>`; copy numbers into a new logbook entry (M3-T3.2).
4. [ ] Generate confusion-matrix / training-curve figures into `docs/paper/figures/`.
5. [ ] M5 retraining command (`init_checkpoint` already supported in `train.py`).
6. [ ] M6: ONNX export → FastAPI service → PySide6 desktop client.
7. [ ] M7: Colab notebook + self-hosted runner workflow (`docs/03_colab_and_runners.md`).
8. [ ] S1: download selected UMUD datasets, log licences.

## 5. Reminders
- Commits: `[Type] [scope] : description`, body ≤ 25 words, English, **no co-author trailers** (hook enforces).
- Never commit data, `outputs/` or logs (git-ignored).
- Each step above closes with a logbook entry (`docs/logbook/TEMPLATE.md`).
