# Usability and accessibility evaluation protocol (interface phase F5, T6.10 = T6.4)

**Status:** ready to run (2026-11-14) · **Acceptance** (`06_interface_plan.md` F5): mean SUS ≥ 68, every click
budget met (median), findings logged for the paper · **Tools:** `rausnet-desktop --study`, `scripts/usability_report.py`,
`scripts/a11y_audit.py`.

## 1. What is evaluated
1. **Usability** with people who will use the app: task success, actions per task against the click
   budgets, time per task, System Usability Scale (SUS), comments.
2. **Accessibility:**
   - **automatic:** `scripts/a11y_audit.py` (names, keyboard reach, target size, contrast, 200 % text,
     fits a 1366 × 768 screen). Run before every session; must report 0 errors;
   - **manual:** NVDA walkthrough and Accessibility Insights for Windows (§ 7).

## 2. Participants
- **≥ 3, target 5** non-technical staff who would use the app: rheumatology or radiology nurses,
  sonographers, residents. None involved in building the app.
- At least one participant who uses Windows text scaling ≥ 150 % or a screen reader, if available.
- Participation is voluntary. Each person gets a code (`P01`, `P02`…); the list linking codes to names
  stays with the facilitator, on paper, and is destroyed after the analysis.
- **No patient data:** the session uses synthetic images only. Ask the ethics committee whether a staff
  usability study with synthetic images needs review; keep their answer with the study file.

## 3. Setting and materials
- A Windows laptop (ideally 1366 × 768 or 1920 × 1080 at 125 %), mouse, the full installation.
- A shared workspace prepared on that laptop with two model versions, so the "adopt" task can be shown:
  ```bash
  # once: a workspace whose v0002 exists but v0001 is served
  python -m rausnet.improve versions --workspace D:/usab_ws          # check: v0002 current, v0001 archived
  python -m rausnet.improve rollback --workspace D:/usab_ws --version v0001 --reason "usability setup"
  ```
- A folder `casos/` with 3 synthetic GS + PD pairs named `caso1_gs.png`, `caso1_pd.png`…
  (`rausnet-synth --out casos_src --patients 2 --seed 11`, then copy and rename 3 pairs).
- Printed: consent sheet, task cards (§ 4, one per card), SUS questionnaire (§ 5).
- Start the app for each participant:
  ```bash
  rausnet-desktop --workspace D:/usab_ws --study P01
  ```
  The window title says the actions are being logged. The log, `P01_<date>_<time>.jsonl`, is in
  `%LOCALAPPDATA%/RAUSNet/desktop/study/`. It holds only control names and times: no file names,
  no typed text, no patient data.

## 4. Procedure (about 30 min)
1. **Welcome (3 min):** read the introduction:
   > Vamos a probar la aplicación, no a usted. Piense en voz alta mientras trabaja. No puedo ayudarle
   > durante las tareas; si se atasca, dígalo y pasamos a la siguiente.

   Sign consent.
2. **Tasks (15 min):** before each task, the facilitator presses **Ctrl+Shift+N** (task mark) and hands
   over the card. Note success, problems and quotes on the observation sheet. Stop a task after 3 min
   without progress: it counts as not completed.

| # | Card (Spanish, as given) | Done when (goal logged) | Budget |
|---|---|---|---|
| T1 | "Analice el caso 1 (imágenes en la carpeta *casos*) y diga en voz alta el grado y si hay que derivar." | `analyzed`, and the participant states grade and referral | 2 |
| T2 | "Guarde el informe en PDF en el escritorio." | `pdf_exported` | 2 |
| T3 | "La reumatóloga dice que el caso es GS 3 y PD 2. Corrija el resultado. Seudónimo: HOSP-0001." | `correction_sent` with GS 3, PD 2 | 3 |
| T4 | "Se equivocó de caso: deshaga la corrección." | `undone` | 1 |
| T5 | "Inicie una mejora del modelo en Colab." (stop when the browser opens) | `colab_opened` | 1 |
| T6 | "Hay un modelo nuevo. Empiece a usarlo." (see the note below) | `model_adopted` | 1 |

   **T6 preparation:** while the participant does T5, run
   `python -m rausnet.improve rollback --workspace D:/usab_ws --version v0002` in another window. Wait
   until the blue "modelo nuevo" bar appears (the app checks every 2 min), then mark the task and hand
   over the card.

   Typing the pseudonym does not count as an action. The budgets are the in-app steps from
   `06_interface_plan.md` §2.
3. **SUS (3 min):** the participant fills in § 5 alone, right after the tasks.
4. **Interview (5 min):**
   - ¿Qué fue lo más difícil?
   - ¿Qué cambiaría?
   - ¿Usaría el mapa de calor? ¿Por qué?
   - ¿Le pareció clara la advertencia de derivación?
5. Close the app (the log is complete). Copy the log and type the SUS answers into `sus.csv`.
6. **Reset for the next participant:** roll back to v0001 again (§ 3) and delete the corrections of the
   session (`D:/usab_ws/data/incoming/corrections/<date>/`).

## 5. SUS questionnaire (Spanish)
Answer each statement from 1 (*totalmente en desacuerdo*) to 5 (*totalmente de acuerdo*).

| # | Afirmación |
|---|---|
| 1 | Creo que me gustaría usar esta aplicación con frecuencia. |
| 2 | La aplicación me pareció innecesariamente compleja. |
| 3 | La aplicación me pareció fácil de usar. |
| 4 | Creo que necesitaría la ayuda de una persona con conocimientos técnicos para poder usarla. |
| 5 | Las distintas funciones de la aplicación están bien integradas. |
| 6 | Me pareció que la aplicación tenía demasiadas incoherencias. |
| 7 | Creo que la mayoría de las personas aprendería a usar esta aplicación muy rápido. |
| 8 | La aplicación me pareció muy difícil de manejar. |
| 9 | Me sentí muy seguro/a usando la aplicación. |
| 10 | Necesité aprender muchas cosas antes de poder empezar a usar la aplicación. |

- **Scoring** (Brooke, 1996): odd items give answer − 1, even items 5 − answer; the sum × 2.5 gives
  0–100. 68 is the usual average.
- **Translation:**
  - This wording follows the 10 original items, with "sistema" changed to "aplicación".
  - For the paper, prefer a validated Spanish version and cite it. A candidate is the Spanish SUS
    validated by Sevilla-Gonzalez et al. (JMIR Human Factors, 2020); check the reference and wording
    before citing.

`sus.csv`:
```
participant,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10
P01,4,2,5,1,4,2,5,1,4,2
```

## 6. Analysis
```bash
python scripts/usability_report.py --sus sus.csv --logs "study_logs/*.jsonl" --out docs/paper/usability_results.md
```
- **SUS:** mean with 95 % t confidence interval, and each participant's score.
- **Tasks:** completion n/N, median actions against the budget, median time (from the task mark to the
  goal). Exit code 0 only when the acceptance is met.
- Log the problems found (§ 4 notes) in a logbook entry with severity:
  - 1 cosmetic;
  - 2 slows down;
  - 3 blocks a task;
  - 4 could cause a clinical error.
- Fix severity ≥ 3 before the next participant if possible, and record the fix.

## 7. Manual accessibility walkthrough (one person, before the sessions)

| Step | Tool | Pass when |
|---|---|---|
| T1–T4 with keyboard only (Tab, arrows, Space, Ctrl+O / Ctrl+Intro / Ctrl+P / Ctrl+K / Ctrl+Z) | — | every task done without the mouse; the focus frame is always visible |
| T1–T4 with NVDA on | NVDA 2024+ | every control is read with its name and role; after analysing, the result sentence is read; the correction radios are read with number and word; input warnings are read |
| T5–T6 and the version table with NVDA | NVDA | progress stages and the result of a local run are read; table cells are read with their column |
| FastPass on each tab | Accessibility Insights for Windows | 0 automated failures; tab stops visit every control in order |
| Text size 200 % + high contrast | app settings | nothing cut off; the window fits the screen (scroll if needed, no horizontal scroll) |

Record findings:

| Date | Tool | Where | What happened | Severity | Fix / decision |
|---|---|---|---|---|---|
| | | | | | |
