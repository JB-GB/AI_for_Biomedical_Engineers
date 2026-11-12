# 00 — Feasibility Assessment

**Project:** AI-assisted ultrasound pre-screening for Rheumatoid Arthritis (RA)
**Branch:** `PIA_devl` · **Date:** 2026-11-11 · **Status:** Feasible with conditions

---

## 1. Original idea (summary)

Ultrasound (US) can support an RA pre-diagnosis, but it is underused because the
human eye struggles to identify relevant structures inside tendons/joints, not
because the image quality is poor. Train an AI on US images to give a faster,
more consistent pre-diagnosis and reduce the need to refer patients to X-ray (RX).

## 2. Technical corrections to the premise (important for the paper)

| Original statement | Evidence-based reframing |
|---|---|
| "X-ray is the easiest/fastest way to detect RA" | X-ray detects **erosions and joint-space narrowing**, which are *late* findings. RA diagnosis is **clinical** (2010 ACR/EULAR criteria: joint count, RF/anti-CCP serology, CRP/ESR, symptom duration). Imaging supports it. |
| "US is not used because the eye can't see structures in tendons" | Partly true: US is **operator-dependent** and scoring has inter-observer variability. But US is actually **more sensitive than X-ray for early inflammation** (synovitis, tenosynovitis, Power Doppler signal). |
| "Avoid sending to RX" | Better framing: **triage / decision support** — flag patients with likely active synovitis for rheumatology referral, and standardise US scoring. X-ray remains part of baseline work-up for damage. |

**Reframed research question:**
> Can a deep-learning model grade synovitis on joint ultrasound images using the
> EULAR-OMERACT scale (0–3) with agreement comparable to expert sonographers,
> and can it be packaged as a retrainable, non-technical tool?

This reframing makes the project measurable, aligned with a validated clinical
standard, and defensible in a paper. The tool must be described as **research /
decision support only, not a diagnostic device**.

## 3. Prior work (verify each reference before citing)

| Work | What it shows |
|---|---|
| EULAR-OMERACT US taskforce, *Part 1 & 2* (D'Agostino/Terslev et al., RMD Open 2017) | Standard 0–3 grading of Grey-Scale (GS) synovial hypertrophy + Power Doppler (PD), and a combined score. **This is our labelling standard.** |
| Andersen et al., *RMD Open* 2019, "Neural networks for automatic scoring of arthritis disease activity on US images" | CNNs scoring PD synovitis with ~86 % accuracy (reported in reviews). |
| *Rheumatology* 2023, doi:10.1093/rheumatology/kead366 | ResNet models on static + dynamic GS/PD images; compared with 12 radiologists. |
| medRxiv 2024.08.30.24312848 | Pre-training on **synthetic** US images, fine-tuning on 156 real images (74 patients); 4-class acc ≈ 0.84. Shows that **small datasets are workable**. |
| DIANA model (PMC13292146) | AI vs multiple experts for synovitis/osteophytes on EULAR-OMERACT scales. |

**Conclusion:** the problem is solvable in principle and published baselines exist
(acc 0.80–0.87). Our contribution can be: open pipeline, easy retraining loop,
non-technical GUI, and reproducible multi-runtime execution (local / Colab / self-hosted runner).

## 4. Feasibility by dimension

| Dimension | Rating | Notes |
|---|---|---|
| Algorithmic | High | Image classification/segmentation with transfer learning is mature. |
| **Data** | **LOW — critical risk** | No well-known public, downloadable RA-synovitis US dataset was found. Data must come from a clinical partner or be built. |
| Labels | Medium | Requires at least one trained rheumatologist/sonographer applying EULAR-OMERACT. Ideally 2 raters for inter-rater kappa. |
| Compute | High | Small models (ResNet18 / EfficientNet-B0 / ViT-S) train on a free Colab T4 or a consumer GPU; inference runs on CPU. |
| GUI | High | Gradio/Streamlit or a desktop app is straightforward. |
| Retraining | High | Achievable with a versioned data folder + config-driven training script. |
| Ethics / legal | Medium | Patient images are sensitive data (Mexico: LFPDPPP; hospital ethics committee). Anonymisation is mandatory. **No patient data in GitHub.** |
| Academic value | High | Clear clinical standard, measurable metrics, comparison to literature. |

**Verdict:** **Feasible**, provided a data source is secured by Milestone M1.
If not, the fallback (Section 5) still produces a valid paper (methods +
synthetic/pilot dataset + pipeline), but with weaker clinical claims.

## 5. Data strategy (ordered by preference)

1. **Clinical partnership** (UANL Hospital Universitario / rheumatology service):
   anonymised DICOM/PNG of MCP2–3, PIP, wrist joints, with EULAR-OMERACT labels.
2. **Request data from authors** of the papers above (data-availability statements).
3. **Self-acquired pilot set** with a portable probe on volunteers (healthy = grade 0)
   + clinical partner for positives. Requires ethics approval.
4. **Synthetic pre-training** (as in medRxiv 2024) + generic US pre-training
   (other public US datasets) → fine-tune on small real set.

### 5.1 Adopted strategy: staged transfer learning (decided 2026-11-11)

No direct clinical access yet, so the base model is built without patient data
and the pipeline stays open for hospital data later.

| Stage | Data | Purpose | Clinically valid? |
|---|---|---|---|
| S0 | ImageNet or open US foundation weights (USF-MAE, OpenUS) | Generic visual features | No |
| S1 | Public MSK US, unlabelled (UMUD index: muscle/tendon datasets) | Domain-adaptive self-supervised pre-training | No |
| S2 | Synthetic MCP/wrist images with controlled GS/PD grades 0–3 | Train grading head, test the full pipeline end-to-end | No |
| S3 | Hospital data labelled by specialists (via retraining loop) | Fine-tune and validate | **Yes, after validation** |

Rules: every model records its provenance (`stage`, data hashes); the UI shows
a "not clinically validated" banner until an S3 model passes the frozen test set.
Synthetic-data results are reported in the paper only as pipeline verification.

## 6. Open questions — answered 2026-11-11 (see logbook M0-T0.3)

| # | Answer |
|---|---|
| 1 | No direct access; possible hospital data later → strategy 5.1 |
| 2 | MCP 2–3 + wrist |
| 3 | GS **and** PD |
| 4 | 4-class (0–3) |
| 5 | No fixed date; internal review, then course forum |
| 6 | RTX 4070 Laptop (8 GB), i7-13700HX, 32 GB RAM |
| 7–8 | Pending |

### Original questions

1. Do you have access to a rheumatologist / hospital that can provide and label images? (Decides M1.)
2. Which joints? Recommended: **MCP 2–3 dorsal longitudinal + wrist** (most studied).
3. Grey-scale only, or also Power Doppler? (PD adds the "activity" signal.)
4. Task granularity: binary (synovitis yes/no), 4-class (0–3), or segmentation + grading?
5. Deadline of the PIA / paper submission and target venue (course, congress, journal)?
6. Hardware available: local GPU model (NVIDIA? VRAM?), or CPU only?
7. Who are the other 2 contributors and which work packages do they own?
8. Is a self-hosted GitHub runner a machine you control 24/7, or occasional?
