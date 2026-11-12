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
