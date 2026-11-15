# Pending work (status 2026-11-14, release v0.2.0)

Everything that can be built without people, accounts or real images is built and tested (203 tests,
CI green). What is left is grouped by **who can unblock it**.

## 0. Summary
| Area | State | Blocked by |
|---|---|---|
| Clinical data (M1) | **not started**: the only critical path left | clinic agreement, ethics, raters |
| Model on real images (M3–M5 rerun) | code ready, runs on synthetic data | real images |
| Desktop app + release (M6) | **v0.2.0 released** (Windows bundle; one-command install from source) | — |
| Usability study (T6.4 / F5) | kit ready, automated audit 0 errors | ≥ 3 participants, NVDA walkthrough |
| Runtimes (M7) | Colab notebooks, self-hosted workflow and release workflow built | first Colab run, runner registration |
| Paper (M8) | draft v0.1 with synthetic results | clinical results, verified references |

## 1. Critical path: clinical data (blocks clinical results and the paper)
| Task | What is needed | Owner |
|---|---|---|
| T1.1 Data-sharing agreement | Contact the rheumatology service | Team + clinic |
| T1.2 Ethics protocol | Submit anonymisation/consent/storage protocol (deid procedure: `rausnet-deid`); ask also about the staff usability study | Team + ethics committee |
| T1.3 Acquisition protocol | Joints, probe ≥ 12 MHz, views, GS + PD settings | Rheumatologists |
| T1.4 Labelling guide | EULAR-OMERACT GS/PD/combined, examples per grade | Rheumatologists |
| T1.5 Labelling tool + double rating | Label Studio/CVAT, ≥ 20 % double-rated, inter-rater kappa | Team + raters |
| T1.6 Datasets from authors | Request and log replies | Team |
| T1.8 / S1 public MSK US data | Download selected UMUD datasets, log licences | Team (needs approval to download) |

As soon as real images arrive (each step already has code):
1. `rausnet-deid` with a device preset tuned on real screenshots (`configs/deid/devices.yaml`).
2. Manifest → `scripts/eda.py` (class balance, per-device distribution).
3. Corrections from the app or `python -m rausnet.retrain --add <manifest>` (fine-tune + promotion +
   calibration + export); the global model continues from v0005 (now v0006 and later).
4. `python -m rausnet.cv --config configs/cv/compare.yaml` on real data (backbone and `us` augmentation
   must be re-ranked: synthetic results saturate at ≈ 0.99).
5. Fix the refer-to-specialist target accuracy with clinicians (currently 98 %).
6. Rheumatologist review of the grade words (leve/moderada/grave; mild/moderate/severe).

## 2. Needs an account or a person, code ready
| Task | Action | Status |
|---|---|---|
| T6.4 Usability study (F5) | NVDA + Accessibility Insights walkthrough, then ≥ 3 sessions per `docs/07_usability_protocol.md`; `scripts/usability_report.py` | Kit ready; sessions not run |
| T7.2 Colab | Run `notebooks/colab_train.ipynb` and `colab_improve.ipynb` once on a free T4 | Built, never run |
| T7.3 Self-hosted runner | Register the laptop (labels `self-hosted, gpu, rausnet`), set `RAUSNET_PYTHON`, `RAUSNET_DATA_DIR`, `RAUSNET_MODELS_DIR`, dispatch `train` | Built, never run |
| T7.4 Equivalence | Log Colab/runner vs local paired bootstrap | Local CPU ≡ GPU done |
| Release checks by a person | Open the v0.2.0 zip on a clean Windows PC (SmartScreen, first start, Drive folder); full-size local improvement from the app (EfficientNet-B0, 10 rounds) | Self-test, seed retraining (v0005 → next) and installer parse checks done |
| T0.2 Hook on every clone | Each contributor runs `git config core.hooksPath .githooks` (the installer does it) | Per clone |
| T0.6 Work packages | Assign tasks to the 3 contributors | Team decision |

## 3. Paper (M8)
- T8.1 Final evaluation on the clinical test split, comparison with the literature table.
- T8.2 Reader study (optional): model vs non-expert vs expert.
- T8.3 Draft v0.1 (`docs/paper/manuscript.md`):
  - `[TBD]`: clinical results, SUS (§ 3.3), abstract, conclusions, ethics id, author contributions;
  - references in `docs/paper/references.md` are **not yet verified**, including the validated
    Spanish SUS;
  - missing figures: architecture diagram, CONSORT-style data flow, real examples + Grad-CAM, clinical
    confusion matrix/ROC, retraining curve.
- T8.4 Tag `v1.0` with clinical results; freeze environment and data manifest.

## 4. Optional / nice to have
- Code signing of the Windows bundle (removes the SmartScreen warning) and an installer wizard (Inno Setup).
- Joint and side read from file names (decide with users in the usability study).
- T3.4 MLflow tracking (history.csv + model cards cover the paper needs today).
- Automatic device preset from DICOM manufacturer/model in `deid`; DICOM path tested only by unit tests
  (pydicom not installed: `pip install -e .[deid]`).
- PDF report: downscale very large inputs before embedding.
- US foundation-model backbone (USF-MAE / OpenUS) once real data exist.

## 5. Done (for reference)
- **Model and data, on synthetic data:**
  - generator, manifests and patient-level split;
  - two-head model, PD scoring rule, gated calibration, Grad-CAM in ONNX;
  - refer-to-specialist threshold, 5-fold CV recipe selection;
  - retraining with bootstrap promotion + automatic calibration/export.
- **Serving:** FastAPI, Docker, de-identification and EDA tools, multi-runtime setup, CI.
- **Desktop app, interface phases F1–F5 (logbooks M6-T6.6 to T6.10):**
  - shared workspace and improvement panel;
  - accessible analysis flow;
  - 3-click corrections with undo;
  - local improvement with progress and cancel;
  - version history and roll back;
  - accessibility audit and fixes, usability-study tools.
- **Release v0.2.0:**
  - Windows bundle with a workspace seed (first start works offline; empty shared folders set up
    from the app);
  - release workflow on tags;
  - one-command installers (`scripts/install.ps1`, `scripts/install.sh`).
- **Current model v0005:** test combined QWK 0.993, synthetic.
