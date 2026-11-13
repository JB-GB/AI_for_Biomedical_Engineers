# Where we left off — resume guide

**Last update:** 2026-11-13 · **Branch:** `PIA_devl` · **Next milestone:** M6 (ONNX + API + desktop client)

## 1. Status at a glance

| Milestone | Status |
|---|---|
| M0 Setup & governance | ✅ Done (docs, hook, Option B, package scaffold) |
| M1 Data | 🟡 Synthetic generator (S2) done · public MSK data (S1) and hospital data (S3) pending |
| M2 Data pipeline | 🟡 Manifest + patient-level split done · de-identification, EDA pending |
| M3 Baseline model | ✅ Synthetic baseline trained: test combined QWK 0.908 (logbook M3-T3.2) · MLflow pending |
| M4 | ⏳ Not started |
| M5 Retraining loop | ✅ `rausnet.retrain` + registry + paired-bootstrap promotion (logbook M5) |
| M6–M8 | ⏳ Not started |

Code that exists and passes tests (76 tests):
`grading.py`, `metrics.py`, `data/manifest.py`, `data/split.py`, `data/synthetic.py`,
`data/dataset.py`, `data/merge.py`, `models/net.py`, `train.py`, `evaluate.py`, `retrain.py`,
`registry.py`, plus `scripts/make_figures.py`.

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

1. [ ] **M6** ONNX export of the registry `current` model with JSON sidecar → FastAPI inference
   service → PySide6 desktop client ("not clinically validated" banner, "submit correction"
   button writing to `incoming/`).
2. [ ] **M4** Grad-CAM on PD false positives; US-style augmentations; backbone comparison.
3. [ ] **M7** Colab notebook + self-hosted runner workflow (`docs/03_colab_and_runners.md`).
4. [ ] **S1** download selected UMUD datasets, log licences.
5. [ ] T3.4 MLflow tracking (optional).

## 5. Reminders
- Commits: `[Type] [scope] : description`, body ≤ 25 words, English, **no co-author trailers** (hook enforces).
- Never commit data, `outputs/` or logs (git-ignored).
- Each step above closes with a logbook entry (`docs/logbook/TEMPLATE.md`).
