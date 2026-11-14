# Plan — accessible interface for single-image checks and continuous model improvement

**Status:** plan (2026-11-14) · **Milestone:** M6 (new tasks T6.6–T6.10) · **Builds on:** desktop app
(`ui/app.py`, PDF report), API (`serve/api.py`), shared workspace + `rausnet.improve` (M7-T7.5).

## 1. Goal and users
One desktop application, usable without technical knowledge, that does two jobs:
1. **Check one case:** load the GS + PD images of a joint and get the grade, confidence, heat map,
   refer warning and PDF; correct the grade if it is wrong.
2. **Improve the model continuously:** see how many corrections are waiting, start an improvement
   (Colab or this PC) with one click, and adopt the new global model when it is ready.

| User | Needs | Sees |
|---|---|---|
| Clinician / sonographer | Fast check, clear result, correct a grade | "Analizar" |
| Study lead / data steward | Know whether the model improves, launch improvements, roll back | "Analizar" + "Mejorar el modelo" |
| Administrator (IT) | Workspace folder, language, referral target | "Ajustes" |

## 2. Principles (accessibility and manageability)
Target: WCAG 2.2 AA principles applied to a desktop app, verified with a screen reader (NVDA) and
Accessibility Insights for Windows.
- **Keyboard first:** logical tab order; shortcuts `Ctrl+O` open images, `Ctrl+Enter` analyse,
  `Ctrl+P` PDF, `Ctrl+K` correct, `F1` help; visible focus.
- **Screen readers:** every control has `accessibleName`/`accessibleDescription`; the result is
  announced as text ("Grado combinado 2, confianza alta, no requiere derivación").
- **Never colour alone:** grade as number + words + icon; refer warning with icon and text; contrast
  ≥ 4.5:1; colour-blind-safe heat-map palette option (cividis); high-contrast theme.
- **Readable:** follows Windows scaling plus in-app text size 100–200 %; targets ≥ 44 px; plain Spanish
  (English available), no ML jargon ("calidad del modelo" instead of "QWK").
- **Forgiving:** drag and drop; GS/PD pairing by file name; errors say what to do; corrections can be
  undone during the session; consequential actions (roll back, change referral target) ask to confirm.
- **Minimum clicks** (budget, measured in the usability test):

| Task | Steps | Budget |
|---|---|---|
| Check one case | drag both images → `Analizar` | 2 |
| Export PDF | `Exportar PDF` → save | 2 |
| Correct a grade | `Corregir` → choose grades → `Enviar` | 3 |
| Improve on Colab | `Mejorar en Colab` → (Colab) `Run all` → allow Drive | 3 (+1 Google sign-in first time) |
| Adopt new model | banner `Usar el nuevo modelo` | 1 |

## 3. Screens
### A. "Analizar" (single image check)
- Drop zone for GS + PD (or `Abrir`), joint/side remembered from the last case, optional pseudonym.
- Result card: large combined grade with words, GS and PD below, confidence as text and bar, refer
  banner when `refer` is true, heat-map toggle (overlay opacity slider), `Exportar PDF`, `Corregir`.
- Correction dialog pre-filled with the predicted grades; one click per grade; sends to the workspace.
- "Casos de hoy" list (pseudonym, joint, grade, corrected yes/no), kept locally, cleared daily.

### B. "Mejorar el modelo" (continuous improvement)
Data from `rausnet.improve status` (via a new API endpoint).
- Status card: global model version and date, quality in words ("muy alta" + number on hover),
  **progress bar "18 de 20 casos nuevos para la próxima mejora"**, incomplete batches (still syncing).
- Buttons: `Mejorar en Colab` (opens `notebooks/colab_improve.ipynb` in the browser; enabled when the
  threshold is met or with a confirmation below it) and `Mejorar en este equipo` (only if a GPU is
  detected; background run with progress and cancel).
- "En curso" indicator from the workspace lock (who: Colab / this PC / runner, since when).
- Version history: date, new cases, result (promoted / kept previous) with the reason in plain words
  ("el modelo nuevo no fue mejor"), from `improve_log.jsonl` and `registry.json`.
- New-model banner when `registry.json` changes: `Usar el nuevo modelo` reloads it between analyses.
- Roll back (lead/admin): choose an archived version → confirm → registry switch, logged.

### C. "Ajustes"
Workspace folder (validated: has `models/registry.json` or can be initialised), language, text size,
theme, heat-map palette; referral target accuracy (admin, read-only otherwise). The "admin" flag only
prevents accidental changes; real access control is the permission on the shared Drive folder.

## 4. Technical design
- Stay on **PySide6** (offline, already packaged in the Windows bundle). New modules
  `ui/analyze_view.py`, `ui/improve_view.py`, `ui/settings_view.py`; `app.py` becomes a tab container.
- API additions: `GET /workspace/status` (wraps `improve.status`), `POST /model/reload` (atomic swap of
  the ONNX session when the registry's current version changes), `POST /model/rollback` (admin).
- `QFileSystemWatcher` on `registry.json` for the new-model banner; Colab URL built from repository and
  branch in the settings; local improvement in a `QThread` running `rausnet.improve.improve`.
- Settings stored per user with `QSettings`; workspace also accepted from `RAUSNET_WORKSPACE`.
- Tests (offscreen, like `tests/test_ui.py`): every interactive widget has an accessible name; tab order
  covers all controls; shortcuts trigger their actions; status panel renders a fake workspace; reload
  swaps the model; rollback writes the registry and the log.

## 5. Phases and acceptance criteria
| Phase | Tasks | Done when |
|---|---|---|
| **F1** Workspace + improvement panel (T6.6) | settings → workspace; status endpoint and panel; `Mejorar en Colab`; new-model banner + reload | lead sees pending count and launches Colab in ≤ 3 clicks; new model adopted without restarting |
| **F2** Accessible single-image flow (T6.7) | drag & drop pairing, shortcuts, accessible names, spoken result, text size, palettes | keyboard-only and NVDA walkthrough of "check one case" passes; automated accessible-name test green |
| **F3** Correction workflow (T6.8) | pre-filled dialog, undo, "Casos de hoy", input checks (size, modality) | correction in ≤ 3 clicks; undo works; nothing identifying stored |
| **F4** Local improvement + history (T6.9) | QThread runner with progress/cancel, version history, roll back | improvement runs on a GPU PC from the app; roll back restores the previous model and logs it |
| **F5** Evaluation (T6.10 = T6.4) | Accessibility Insights audit, SUS with ≥ 3 non-technical users, fixes | SUS ≥ 68; click budgets met; findings logged for the paper |

Order: F1 → F2 → F3 → F4 → F5 (F2 and F3 can run in parallel; they touch different views).

## 6. Risks
| Risk | Mitigation |
|---|---|
| Drive sync delay: corrections or a new model arrive late | incomplete batches reported, not used; version folder published before the registry |
| Two improvements at once | workspace lock (stale after 6 h) |
| Clinical images on a cloud drive | only de-identified data; only with ethics approval; otherwise use the self-hosted runner or a local GPU on the same workspace |
| Too many retrains on few cases | `min_new_images` threshold; promotion rule protects the served model |
| Model changes in the middle of a case | reload only between analyses; PDF records the model version |
