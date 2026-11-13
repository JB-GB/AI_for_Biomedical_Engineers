# Logbook

One file per entry: `YYYY-MM-DD_<milestone>-<task>_<slug>.md` (e.g.
`2026-11-11_M0-T0.1_project-kickoff.md`). Copy `TEMPLATE.md`.

**When to write an entry (mandatory):**
- Closing any task or milestone.
- Any blocker, decision (ADR), unexpected finding, or failed experiment.
- Every training run that is kept or discarded (link MLflow run id).

Entries are append-only: correct mistakes with a new entry that references the old one.
These entries feed the *Methods*, *Results* and *Limitations* sections of the paper.

| Date | Entry | Milestone | Type |
|---|---|---|---|
| 2026-11-11 | [Project kickoff & feasibility](2026-11-11_M0-T0.1_project-kickoff.md) | M0 | Decision |
| 2026-11-11 | [Team decisions & architecture choice](2026-11-11_M0-T0.3_team-decisions.md) | M0 | Decision |
| 2026-11-11 | [Synthetic joint US generator](2026-11-11_M1-T1.7_synthetic-generator.md) | M1 | Result |
| 2026-11-11 | [Training pipeline & first GPU attempt](2026-11-11_M3-T3.1_training-pipeline.md) | M3 | Blocker |
| 2026-11-13 | [Baseline results on synthetic data](2026-11-13_M3-T3.2_baseline-results.md) | M3 | Result |
| 2026-11-13 | [Retraining loop with promotion rule](2026-11-13_M5-T5.2_retraining-loop.md) | M5 | Result |
| 2026-11-13 | [Paired bootstrap promotion & CIs](2026-11-13_M5-T5.3_bootstrap-promotion.md) | M5 | Decision |
| 2026-11-13 | [ONNX export, API and desktop client](2026-11-13_M6-T6.2_onnx-api-desktop.md) | M6 | Result |
| 2026-11-13 | [Grad-CAM explanations & calibration](2026-11-13_M4-T4.4_gradcam-calibration.md) | M4 | Result / Correction |
| 2026-11-13 | [PD scoring rule & loss masking (v0004)](2026-11-13_M4-T4.1_pd-gs0-rule.md) | M4 | Result / Decision |
| 2026-11-13 | [Multi-runtime: local, Colab, runner](2026-11-13_M7-T7.1_multi-runtime.md) | M7 | Result |
