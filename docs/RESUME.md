# Where we left off — resume guide

**Last update:** 2026-11-13 · **Branch:** `PIA_devl` · **Next milestone:** M4 (model improvement) / M7 (runtimes)

## 1. Status at a glance

| Milestone | Status |
|---|---|
| M0 Setup & governance | ✅ Done (docs, hook, Option B, package scaffold) |
| M1 Data | 🟡 Synthetic generator (S2) done · public MSK data (S1) and hospital data (S3) pending |
| M2 Data pipeline | 🟡 Manifest + patient-level split done · de-identification, EDA pending |
| M3 Baseline model | ✅ Synthetic baseline trained: test combined QWK 0.908 (logbook M3-T3.2) · MLflow pending |
| M4 | ⏳ Not started |
| M5 Retraining loop | ✅ `rausnet.retrain` + registry + paired-bootstrap promotion (logbook M5) |
| M6 Product | ✅ ONNX export, FastAPI, PySide6 app (logbook M6) · PDF report, installer, usability test pending |
| M7–M8 | ⏳ Not started |

Code that exists and passes tests (94 tests):
`grading.py`, `metrics.py`, `data/manifest.py`, `data/split.py`, `data/synthetic.py`,
`data/dataset.py`, `data/merge.py`, `models/net.py`, `train.py`, `evaluate.py`, `retrain.py`,
`registry.py`, `export.py`, `serve/inference.py`, `serve/api.py`, `ui/client.py`, `ui/app.py`,
plus `scripts/make_figures.py`, `scripts/gui_screenshot.py`.

## 2. Set up the machine again (needs internet once)

```bash
git config core.hooksPath .githooks
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/python -m pip install "timm>=1.0" "pyyaml>=6.0" "matplotlib>=3.8"
.venv/Scripts/python -m pip install -e ".[serve,ui]" onnx onnxscript
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

- M3 baseline done (test combined QWK 0.908). M5 retraining loop done and demonstrated.
- Local registry `models/registry.json` (git-ignored): `v0001` archived, `v0002` rejected
  (old tolerance-0 rule), **`v0003` current** (synthetic + batch_001; paired bootstrap diff −0.004,
  95 % CI [−0.015, +0.004] → promoted).
- Data root `J:/rausnet_data`: `synth/`, `incoming/batch_001/`, `datasets/v0001`–`v0003`.
- Promotion rule decided: paired bootstrap (logbook M5-T5.3). Metrics now carry 95 % CIs.

How to retrain with new data (e.g. hospital batch already de-identified, with a manifest):
```bash
export RAUSNET_DATA_DIR=J:/rausnet_data
.venv/Scripts/python -m rausnet.retrain --config configs/retrain/default.yaml --add hospital/batch1/manifest.csv
.venv/Scripts/python -m rausnet.registry list
```

## 4. Next steps, in order

Use the app (current model must be exported once after each promotion):
```bash
.venv/Scripts/python -m rausnet.export          # models/<current>/model.onnx + model.json
.venv/Scripts/rausnet-desktop                   # local API + window (Spanish); --lang en
.venv/Scripts/rausnet-api --port 8000           # API only (http://127.0.0.1:8000/docs)
```

1. [ ] **M4** Grad-CAM (PD false positives + overlay in the app), confidence calibration
   (temperature scaling — synthetic model shows ~100 % confidence), US-style augmentations,
   backbone comparison.
2. [ ] **M7** Colab notebook + self-hosted runner workflow (`docs/03_colab_and_runners.md`).
3. [ ] **M6 leftovers** PDF report, PyInstaller installer, Docker image for the API, SUS usability test.
4. [ ] Auto-export ONNX after a promotion in `rausnet.retrain`.
5. [ ] **S1** download selected UMUD datasets, log licences.
6. [ ] T3.4 MLflow tracking (optional).

## 5. Reminders
- Commits: `[Type] [scope] : description`, body ≤ 25 words, English, **no co-author trailers** (hook enforces).
- Never commit data, `outputs/` or logs (git-ignored).
- Each step above closes with a logbook entry (`docs/logbook/TEMPLATE.md`).
