# Manuscript draft (v0.1, 2026-11-13)

> Working draft assembled from the logbook "Paper notes". Every number comes from a logbook entry and a
> committed script. `[TBD]` = needs clinical data or a run not done yet. `[REF:key]` = citation from
> `references.md`, to verify before submission. Synthetic results verify the pipeline only; they are not
> clinical performance and must be presented as such.

**Working title:** Retrainable Deep Learning Tool for EULAR-OMERACT Synovitis Grading on Joint Ultrasound in
Rheumatoid Arthritis

## Abstract (write last)
`[TBD]` Background · Methods · Results (primary metric: patient-level combined-grade QWK vs expert, 95 % CI)
· Conclusions.

## 1. Introduction
- Rheumatoid arthritis (RA) needs early detection of synovitis; conventional radiography detects late,
  structural damage. Ultrasound (US) is more sensitive to early inflammatory change but is operator-dependent
  and shows inter-observer variability, which motivates automated grading [REF:EULAR-OMERACT Part 1/2].
- EULAR-OMERACT scores grey-scale synovial hypertrophy (GS 0–3) and power Doppler (PD 0–3) per joint, with
  a combined score; PD is only scored where synovial hypertrophy is present.
- Prior deep-learning work grades synovitis from US [REF:Andersen 2019; Rheumatology 2023; DIANA;
  medRxiv 2024 synthetic pre-training], but tools are rarely retrainable by the clinical team, rarely open,
  and rarely designed for non-technical users.
- **Aim:** an open, retrainable pre-diagnosis tool for MCP2–3 and wrist (GS + PD, 4-class) that flags
  low-confidence cases for a specialist, intended to reduce unnecessary referrals to radiography.
- **Contributions:** (1) reproducible pipeline that runs locally, on Colab and on a self-hosted runner;
  (2) clinician-in-the-loop retraining with a statistical promotion rule on a frozen test split;
  (3) scoring-rule-consistent two-head model with explainability and a refer-to-specialist threshold;
  (4) desktop GUI for non-technical staff, evaluated with SUS `[TBD]`.

## 2. Materials and methods

### 2.1 Data
- **Staged strategy** (no public synovitis-labelled US dataset): S0 generic US pre-training weights, S1
  public musculoskeletal US `[TBD]`, S2 synthetic synovitis images, S3 hospital images `[TBD: ethics id,
  centre, devices, probe ≥ 12 MHz, raters, double-rating ≥ 20 %, inter-rater kappa]`.
- **Synthetic generator (S2):** procedural renderer of dorsal longitudinal MCP2–3 and wrist (skin,
  fibrillar tendon, cortical bone with acoustic shadow, cartilage, hypoechoic synovium, Rayleigh speckle,
  depth attenuation, log compression). GS controls synovial thickness/shape; PD spots inside a colour box
  follow the EULAR-OMERACT coverage definitions (PD1 ≤ 3 spots, PD2 < 50 %, PD3 > 50 % of synovium,
  verified per image). 15 % of images contain a vessel outside the synovium as a distractor. Six joints per
  patient with patient-level disease activity; per-image seeds give exact reproducibility.
- **Development dataset (v0005):** 1 920 images, 320 synthetic patients (initial 200 + two incoming
  batches). **Split:** patient-level hash split 70/15/15 with a fixed salt, so the test split stays frozen
  across all model versions (test: 282 images, 47 patients).
- **De-identification (S3):** `rausnet.data.deid` — metadata stripped by pixel re-encoding (DICOM → PNG),
  device-specific cropping of burned-in text, HMAC-SHA256 pseudonyms with a salt kept outside the repository
  `[TBD: confirm with ethics protocol]`.

### 2.2 Model
- Input: 4-channel tensor (GS grey + PD RGB), 224 px. Shared ImageNet-pretrained backbone (timm) with two
  linear 4-class heads (GS, PD); the combined grade is *derived* (0 if GS = 0, else max(GS, PD)), not
  learned. Combined probabilities: P(c) from independent head probabilities under that rule.
- **Scoring-rule consistency:** P(PD = d) = P(GS = 0)·[d = 0] + P(GS ≥ 1)·p_PD(d); PD loss masked on GS-0
  joints (no effect, see Results).
- Training: class-weighted cross-entropy, AdamW (lr 3·10⁻⁴, ConvNeXt 10⁻⁴), mixed precision, early
  stopping on validation combined QWK. Augmentation: flip, random crop, gain, Gaussian noise with a
  **random** level σ ~ U(0, 0.03) and independent fields for GS and PD. An optional US-specific augmentation
  (rotation, zoom, time-gain ramp, gamma, multiplicative speckle, Doppler gain) was evaluated.
- **Recipe selection:** patient-level 5-fold CV on train + val only (test untouched); for fold i, train on
  3 folds, early-stop on fold i+1, predict fold i; out-of-fold (OOF) predictions compared with a paired
  patient-level bootstrap.

### 2.3 Calibration, explainability, referral
- Per-head temperature scaling fitted on validation, applied only if (a) a 2-fold patient cross-fit lowers
  NLL versus T = 1 and (b) the validation predictions are not perfectly separated (no NLL minimum).
- Grad-CAM: with global average pooling followed by linear heads it reduces exactly to CAM and is computed
  inside the exported ONNX graph. Localisation measured by the pointing game against the synthetic
  synovium mask (chance = mask area fraction).
- **Refer to specialist:** the lowest top-label confidence threshold for which accepted validation cases
  reach a target accuracy (98 %, to be fixed with clinicians `[TBD]`); reported with coverage and accuracy
  of accepted and referred cases. Referral never changes the displayed grade.

### 2.4 Retraining loop
- New labelled data (including clinician corrections from the GUI) enter as immutable, hashed manifests;
  each candidate (fine-tuned or trained from scratch with a new recipe) is compared with the deployed model
  on the identical frozen test split by a paired patient-level bootstrap (2 000 resamples) of the
  combined-grade QWK difference, and rejected only if the whole 95 % CI lies below 0. Promoted models are
  calibrated and exported automatically and stored in a versioned registry.

### 2.5 Software and deployment
- ONNX export (max |Δ| vs PyTorch ≈ 2·10⁻⁶); FastAPI service on ONNX Runtime (~35 ms per image pair on
  CPU); PySide6 desktop client (Spanish/English) with heat maps, refer warning, correction submission and
  PDF report; Docker image for research servers and Windows installer.
- Runtimes: local CPU/GPU, Google Colab (free tier, resumable training), self-hosted GitHub runner
  (manual dispatch). Equivalence judged by paired bootstrap, not identical weights.

### 2.6 Statistics
- Primary: combined-grade quadratic weighted kappa (QWK). Secondary: GS and PD QWK, accuracy, balanced
  accuracy, per-class recall, AUC (grade ≥ 2), ECE, NLL, coverage. All 95 % CIs by patient-level
  (cluster) bootstrap, 2 000 resamples. Metrics fixed before clinical data (see `docs/paper/README.md`).

## 3. Results

### 3.1 Synthetic data (pipeline verification)
| Item | Result |
|---|---|
| Baseline B0, 200 patients (v0001) | test combined QWK 0.908 |
| PD scoring rule (same model, v0003) | PD QWK 0.807 → 0.950; GS-0 PD false positives 5 → 0 |
| PD loss masking (v0004 vs ablation) | ΔQWK +0.003 (−0.003, +0.011): no effect |
| Augmentation bug (fixed noise level) | clean-image QWK 0.10 vs 0.91 with noise; fixed by random level |
| **Final model v0005** (B0, fixed aug.) | test combined QWK **0.993 (0.984–0.998)**, GS 1.000, PD 0.992 |
| v0005 vs v0004 (promotion) | Δ +0.047 (+0.015, +0.098) |
| Grad-CAM pointing game (v0005) | GS 100 %, PD 100 % (chance 8 %) |
| Calibration (v0005) | T_GS rejected (separable), T_PD 1.30; test ECE 0.013 → 0.018 (≈ 4 errors / 282) |
| Referral, 98 % target (v0004 / v0005) | v0004: 97 % coverage, 96 % accepted accuracy on test (val 98 %); v0005: 100 % coverage |
| CPU vs GPU training | Δ +0.009 (−0.018, +0.036): equivalent |

**Table 2 – Recipe comparison, 5-fold CV, OOF combined QWK (1 638 images / 273 patients)**
| Recipe | OOF QWK (95 % CI) | Δ vs B0 basic (paired 95 % CI) |
|---|---|---|
| EfficientNet-B0, basic aug. | 0.991 (0.987–0.994) | — |
| ConvNeXt-nano, US aug. | 0.991 (0.987–0.994) | +0.000 (−0.004, +0.004) |
| B0, no PD masking | 0.990 (0.986–0.993) | −0.001 (−0.003, +0.001) |
| ResNet-34, US aug. | 0.985 (0.981–0.989) | −0.006 (−0.010, −0.001) |
| DeiT-small, US aug. | 0.984 (0.979–0.988) | −0.007 (−0.012, −0.003) |
| B0, US aug. | 0.979 (0.974–0.984) | −0.012 (−0.017, −0.007) |

Figures: `synthetic_examples.png`, `cv_comparison.png`, `v0005_confusion.png`, `v0005_curves.png`,
`gradcam_examples.png`, `calibration_v0005.png`, `referral_v0004.png`/`referral_v0005.png`, `gui_desktop_gradcam.png`.

### 3.2 Clinical data `[TBD]`
- Inter-rater agreement; test QWK/AUC vs expert with CIs; comparison with literature table; confusion matrix
  and ROC; referral coverage at the agreed target; retraining curve (performance vs added data).

### 3.3 Usability `[TBD]`
- SUS score (≥ 3 non-technical users), task completion time, qualitative feedback.

### 3.4 Runtimes `[TBD]`
- Colab and self-hosted runner runs vs local, paired bootstrap.

## 4. Discussion
- **Main message (provisional):** the full retrainable pipeline works end-to-end and encodes the
  EULAR-OMERACT rules; clinical validity is not yet established.
- Encoding the scoring rule (PD conditioned on GS) gave a larger gain than any architecture change.
- Synthetic data saturate (≈ 0.99): architecture rankings and the negative effect of US augmentation reflect
  the generator's low acquisition variability and must be repeated on real images.
- A subtle augmentation bug (fixed noise level) degraded every model for several versions without failing
  training; out-of-fold CV and from-scratch retraining exposed it. Lesson: evaluate on clean inputs and
  re-validate recipes from scratch.
- Calibration and referral thresholds fitted on a few hundred validation images were optimistic or
  degenerate; we gate them and report coverage/accepted accuracy rather than bare thresholds.

### Limitations
- Idealised synthetic anatomy and acoustics (no reverberation, anisotropy, device artefacts); large expected
  synthetic-to-real gap. Single centre `[TBD]`. Joint-level, not patient-level diagnosis. Training is not
  bit-reproducible across devices.

## 5. Conclusions
`[TBD]`

## Ethics, data and code availability
- Ethics approval `[TBD]`; anonymisation procedure (§2.1). Code: GitHub repository, release tag `v1.0`
  `[TBD]`; model card and dataset manifests (hashes) in the registry.

## Author contributions
`[TBD]` (maximum three contributors).
