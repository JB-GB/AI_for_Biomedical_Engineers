# Contributing

## Setup (once per clone)
```bash
git config core.hooksPath .githooks
```
The `commit-msg` hook enforces the rules below.

## Commit rules
- **Title:** `[Type] [scope] : short description` — e.g. `[Add] [docs] : feasibility assessment`
  - Types: `Add`, `Fix`, `Update`, `Remove`, `Refactor`, `Docs`, `Test`, `Config`, `Data`, `Model`.
- **Body:** bullet list explaining *what* and *why*, **max 25 words**.
- **Language:** English (code, comments, commits).
- **Atomic:** one logical change per commit.
- **Authorship:** only the human author. No `Co-Authored-By` or other co-author trailers
  (the course grading script counts contributors; max 3).

## Data rules
- Never commit patient images, DICOM, or any identifiable data. `data/` is git-ignored.
- Never commit secrets or `.env` files.
- Strip notebook outputs before committing.
