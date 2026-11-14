# Where we left off — resume guide

**Last update:** 2026-11-13 · **Branch:** `PIA_devl` · **Next milestone:** first Colab/runner runs (M7) · M6 leftovers · real data (S1)

## 1. Status at a glance

| Milestone | Status |
|---|---|
| M0 Setup & governance | ✅ Done (docs, hook, Option B, package scaffold) |
| M1 Data | 🟡 Synthetic generator (S2) done · public MSK data (S1) and hospital data (S3) pending |
| M2 Data pipeline | 🟡 Manifest + patient-level split done · de-identification, EDA pending |
| M3 Baseline model | ✅ Synthetic baseline trained: test combined QWK 0.908 (logbook M3-T3.2) · MLflow pending |
| M4 Model improvement | ✅ on synthetic data: Grad-CAM, gated calibration, GS0 → PD0 rule, augmentation fix, 5-fold CV backbone comparison, referral threshold, v0005 (logbook M4) · repeat on real data |
| M5 Retraining loop | ✅ `rausnet.retrain` + registry + paired-bootstrap promotion (logbook M5) |
| M6 Product | ✅ ONNX export, FastAPI, PySide6 app (logbook M6) · PDF report, installer, usability test pending |
| M7 Runtimes | 🟡 setup script, resumable training, Colab notebook, CI + self-hosted workflow, CPU≡GPU check (logbook M7) · first Colab and runner runs pending |
| M8 Paper | ⏳ Not started |

Code that exists and passes tests (125 tests):
`grading.py`, `metrics.py`, `data/manifest.py`, `data/split.py`, `data/synthetic.py`,
`data/dataset.py`, `data/merge.py`, `models/net.py`, `train.py`, `evaluate.py`, `retrain.py`,
`registry.py`, `explain.py`, `heatmap.py`, `calibrate.py`, `cv.py`, `export.py`, `serve/inference.py`, `serve/api.py`, `ui/client.py`, `ui/app.py`,
plus `scripts/make_figures.py`, `scripts/gui_screenshot.py`, `scripts/analyze_cam.py`, `scripts/setup_env.py`,
`scripts/compare_runs.py`, `notebooks/colab_train.ipynb`, `.github/workflows/{ci,train-self-hosted}.yml`.

## 2. Set up the machine again (needs internet once)

```bash
py -3.11 scripts/setup_env.py          # venv + CUDA torch + all extras + commit hook (--cpu without GPU)
.venv/Scripts/python -m pip install onnxscript   # only for ONNX export
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
- Local registry `models/registry.json` (git-ignored): `v0001`, `v0003` archived, `v0002` rejected
  (old tolerance-0 rule), `v0004` archived, **`v0005` current** (B0, fixed basic augmentation, from
  scratch; test combined QWK 0.993, +0.047 [+0.015, +0.098] vs v0004). Calibrated (T_GS 1, T_PD 1.30,
  referral threshold 0.505 at 98 %) and exported (`model.onnx`).
- Data root `J:/rausnet_data`: `synth/`, `incoming/batch_001/`, `incoming/batch_002/`, `datasets/v0001`–`v0005`. CV outputs in `outputs/cv/`.
- M7 built but not yet run outside this laptop: Colab notebook and self-hosted runner (setup steps in
  `docs/03_colab_and_runners.md` §2–3). CI (`ci.yml`) runs on the next push.
- Promotion rule decided: paired bootstrap (logbook M5-T5.3). Metrics now carry 95 % CIs.

How to retrain with new data (e.g. hospital batch already de-identified, with a manifest):
```bash
export RAUSNET_DATA_DIR=J:/rausnet_data
.venv/Scripts/python -m rausnet.retrain --config configs/retrain/default.yaml --add hospital/batch1/manifest.csv
.venv/Scripts/python -m rausnet.registry list
```

## 4. Next steps, in order

Use the app (after each promotion: calibrate, then export):
```bash
.venv/Scripts/python -m rausnet.calibrate --data-root J:/rausnet_data   # temperatures -> calibration.json
.venv/Scripts/python -m rausnet.export          # models/<current>/model.onnx (+ CAM outputs) + model.json
.venv/Scripts/rausnet-desktop                   # local API + window (Spanish); --lang en
.venv/Scripts/rausnet-api --port 8000           # API only (http://127.0.0.1:8000/docs)
```

1. [ ] **M7 first runs** Colab (free T4) and self-hosted runner; compare with `scripts/compare_runs.py`; check first CI run.
2. [x] **M4 rest** done on synthetic data (logbook M4-T4.2). Repeat `python -m rausnet.cv --config
   configs/cv/compare.yaml` on real data; choose the referral target with clinicians.
3. [ ] **M6 leftovers** PDF report, PyInstaller installer, Docker image for the API, SUS usability test.
4. [ ] Auto calibrate + export ONNX after a promotion in `rausnet.retrain` (the runner workflow already does it).
5. [ ] **S1** download selected UMUD datasets, log licences.
6. [ ] T3.4 MLflow tracking (optional).

## 5. Reminders
- Commits: `[Type] [scope] : description`, body ≤ 25 words, English, **no co-author trailers** (hook enforces).
- Never commit data, `outputs/` or logs (git-ignored).
- Each step above closes with a logbook entry (`docs/logbook/TEMPLATE.md`).
