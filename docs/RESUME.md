# Where we left off — resume guide

**Last update:** 2026-11-13 · **Branch:** `PIA_devl` · **Next milestone:** M5 (retraining loop)

## 1. Status at a glance

| Milestone | Status |
|---|---|
| M0 Setup & governance | ✅ Done (docs, hook, Option B, package scaffold) |
| M1 Data | 🟡 Synthetic generator (S2) done · public MSK data (S1) and hospital data (S3) pending |
| M2 Data pipeline | 🟡 Manifest + patient-level split done · de-identification, EDA pending |
| M3 Baseline model | ✅ Synthetic baseline trained: test combined QWK 0.908 (logbook M3-T3.2) · MLflow pending |
| M4–M8 | ⏳ Not started |

Code that exists and passes tests (61 tests):
`grading.py`, `metrics.py`, `data/manifest.py`, `data/split.py`, `data/synthetic.py`,
`data/dataset.py`, `models/net.py`, `train.py`, `evaluate.py`, plus `scripts/make_figures.py`.

## 2. Set up the machine again (needs internet once)

```bash
git config core.hooksPath .githooks
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/python -m pip install "timm>=1.0" "pyyaml>=6.0" "matplotlib>=3.8"
.venv/Scripts/python -m pytest
```
If `.venv` still exists, skip the installs. Pretrained EfficientNet-B0 weights are
downloaded from the Hugging Face hub on first training run (needs internet; cached afterwards).
Offline alternative: `--set model.pretrained=false`.

Synthetic data lives at `J:/rausnet_data/synth` (D: is only 50 MB). If missing, regenerate
(reproducible, manifest SHA-256 `f6bb9768…`):
```bash
.venv/Scripts/rausnet-synth --out J:/rausnet_data/synth --patients 200 --seed 0
```
Then set `RAUSNET_DATA_DIR=J:/rausnet_data` before training.

## 3. Exactly where it stopped

- The Windows data-loader blocker (M3-T3.1) is solved with `num_workers: 0`.
- Baseline run `outputs/baseline_s2_20261113-131223` (git-ignored, local only):
  test combined QWK 0.908, balanced accuracy 0.945, AUC(≥2) 0.993.
- Figures: `docs/paper/figures/baseline_s2_confusion.png`, `baseline_s2_curves.png`
  (`python scripts/make_figures.py outputs/<run_dir>`).

## 4. Next steps, in order

1. [ ] **M5** retraining command: merge `data/incoming` manifest, fine-tune from current model
   (`train.init_checkpoint` already supported), promote only if frozen-test metrics do not degrade,
   keep `models/registry.json`.
2. [ ] **M6** ONNX export with JSON sidecar → FastAPI inference service → PySide6 desktop client
   (with "not clinically validated" banner and "submit correction" button).
3. [ ] **M4** Grad-CAM on PD false positives; US-style augmentations; backbone comparison.
4. [ ] **M7** Colab notebook + self-hosted runner workflow (`docs/03_colab_and_runners.md`).
5. [ ] **S1** download selected UMUD datasets, log licences.
6. [ ] T3.4 MLflow tracking (optional; `history.csv` + `model_card.json` cover the basics).

## 5. Reminders
- Commits: `[Type] [scope] : description`, body ≤ 25 words, English, **no co-author trailers** (hook enforces).
- Never commit data, `outputs/` or logs (git-ignored).
- Each step above closes with a logbook entry (`docs/logbook/TEMPLATE.md`).
