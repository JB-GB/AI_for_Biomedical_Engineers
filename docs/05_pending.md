# Pending work (status 2026-11-13)

What is left after M0–M7 were built on synthetic data. Grouped by **who can unblock it**, because almost
everything remaining depends on people, accounts or real images rather than on code.

## 1. Critical path: clinical data (blocks clinical results and the paper)
| Task | What is needed | Owner |
|---|---|---|
| T1.1 Data-sharing agreement | Contact the rheumatology service | Team + clinic |
| T1.2 Ethics protocol | Submit anonymisation/consent/storage protocol (deid procedure: `rausnet-deid`) | Team + ethics committee |
| T1.3 Acquisition protocol | Joints, probe ≥ 12 MHz, views, GS + PD settings | Rheumatologists |
| T1.4 Labelling guide | EULAR-OMERACT GS/PD/combined, examples per grade | Rheumatologists |
| T1.5 Labelling tool + double rating | Label Studio/CVAT, ≥ 20 % double-rated, inter-rater kappa | Team + raters |
| T1.6 Datasets from authors | Request and log replies | Team |
| T1.8 / S1 public MSK US data | Download selected UMUD datasets, log licences | Team (needs approval to download) |

As soon as real images arrive (each step already has code):
1. `rausnet-deid` with a device preset tuned on real screenshots (`configs/deid/devices.yaml`).
2. Manifest → `scripts/eda.py` (class balance, per-device distribution).
3. `python -m rausnet.retrain --add <manifest>` (fine-tune + promotion + calibration + export).
4. `python -m rausnet.cv --config configs/cv/compare.yaml` on real data (backbone and `us` augmentation
   must be re-ranked: synthetic results saturate at ≈ 0.99).
5. Fix the refer-to-specialist target accuracy with clinicians (currently 98 %).

## 2. Needs an account or a person, code ready
| Task | Action | Status |
|---|---|---|
| T7.2 Colab | Run `notebooks/colab_train.ipynb` once on a free T4, compare with `scripts/compare_runs.py` | Built, never run |
| T7.3 Self-hosted runner | Register the laptop (labels `self-hosted, gpu, rausnet`), set `RAUSNET_PYTHON`, `RAUSNET_DATA_DIR`, `RAUSNET_MODELS_DIR`, dispatch `train` | Built, never run |
| T7.4 Equivalence | Log Colab/runner vs local paired bootstrap | Local CPU ≡ GPU done |
| T6.4 Usability test | ≥ 3 non-technical users, SUS questionnaire, task times | Not started |
| T0.2 Hook on every clone | Each contributor runs `git config core.hooksPath .githooks` | Per clone |
| T0.6 Work packages | Assign tasks to the 3 contributors | Team decision |

## 3. Paper (M8)
- T8.1 Final evaluation on the clinical test split, comparison with the literature table.
- T8.2 Reader study (optional): model vs non-expert vs expert.
- T8.3 Draft v0.1 exists (`docs/paper/manuscript.md`); clinical results, abstract, conclusions,
  ethics id and author contributions are `[TBD]`. References in `docs/paper/references.md` are **not yet
  verified**. Missing figures: architecture diagram, CONSORT-style data flow, real examples + Grad-CAM,
  clinical confusion matrix/ROC, retraining curve.
- T8.4 Tag `v1.0`, freeze environment and data manifest.

## 4. Optional / nice to have
- T3.4 MLflow tracking (history.csv + model cards cover the paper needs today).
- Automatic device preset from DICOM manufacturer/model in `deid`; DICOM path tested only by unit tests
  (pydicom not installed: `pip install -e .[deid]`).
- Windows installer wizard (Inno Setup) around the PyInstaller folder; frozen-app fallbacks could move from
  `packaging/rausnet_desktop_entry.py` into `ui/app.py`.
- PDF report: downscale very large inputs before embedding.
- US foundation-model backbone (USF-MAE / OpenUS) once real data exist.

## 5. Done on synthetic data (for reference)
Generator, manifests and patient-level split, two-head model, PD scoring rule, gated calibration,
Grad-CAM in ONNX, refer-to-specialist threshold, 5-fold CV recipe selection, retraining with bootstrap
promotion + automatic calibration/export, FastAPI + desktop app + PDF report, Docker + Windows bundle,
de-identification and EDA tools, multi-runtime setup, CI. Current model **v0005** (test combined QWK 0.993,
synthetic). 148 tests.
