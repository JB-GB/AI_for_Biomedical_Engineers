# 04 — Execution Runtimes: Local, Colab, Self-hosted Runner

The same command must work everywhere:
`python -m rausnet.train --config configs/train/baseline.yaml`

## 1. Local host
- `python -m venv .venv` → `pip install -e .[gpu]` (or `[cpu]`).
- Data lives **outside** the repo; path set via env var `RAUSNET_DATA_DIR` or `.env` (git-ignored).
- GUI: `python -m app.gradio_app` → opens `http://127.0.0.1:7860` (bind to localhost only).

## 2. Google Colab — safe configuration

**Cost safety**
- Use the **free tier** only; never buy compute units on the shared account. Each member uses their own Google account.
- No paid GPU types in notebook metadata; keep `accelerator: GPU` (T4) generic.
- Save checkpoints to Drive every epoch → sessions can be disconnected without losing work.
- Add an auto-stop cell at the end: `from google.colab import runtime; runtime.unassign()`.
- Do not use Colab Enterprise / GCP projects with billing enabled.

**Data & secrets safety**
- Patient data only in a **private Drive folder**, mounted read-only by convention; never in the repo, never in notebook outputs.
- Tokens (W&B, GitHub) via **Colab Secrets** (`google.colab.userdata`), never hard-coded.
- Clone the repo read-only (`git clone --depth 1 -b PIA_devl`); push from local machines, not from Colab, so authorship stays correct.
- Do not use Gradio `share=True` with real patient images (public link).

**Notebook hygiene in Git**
- Strip outputs before commit (`nbstripout`), otherwise notebooks leak images and produce huge, non-atomic diffs.
- Notebooks only orchestrate; logic lives in `rausnet/`.

## 3. Self-hosted GitHub Actions runner

**Security rules (repository is public → critical)**
- GitHub warns that self-hosted runners on **public** repos can be abused by fork PRs.
  Either make the repo private, or ensure training workflows trigger **only** on
  `workflow_dispatch` (manual) and never on `pull_request` from forks.
- Settings → Actions → "Require approval for all outside collaborators".
- Runner labels: `self-hosted, gpu, rausnet`; workflow uses `runs-on: [self-hosted, gpu, rausnet]`.
- Run the runner as an unprivileged user; data folder mounted read-only.
- `permissions: contents: read` in workflow; no secrets exposed to PRs.
- Upload only metrics/model artefacts (`actions/upload-artifact`), never data.

**Workflow sketch (`.github/workflows/train-self-hosted.yml`, M7)**
```yaml
on:
  workflow_dispatch:
    inputs:
      config: {default: configs/train/baseline.yaml}
permissions: {contents: read}
jobs:
  train:
    runs-on: [self-hosted, gpu, rausnet]
    timeout-minutes: 360
    steps:
      - uses: actions/checkout@v4
      - run: pip install -e .[gpu]
      - run: python -m rausnet.train --config ${{ inputs.config }}
        env: {RAUSNET_DATA_DIR: /srv/rausnet/data}
      - uses: actions/upload-artifact@v4
        with: {name: model, path: outputs/}
```
GitHub-hosted runners (free minutes) are used only for lint + unit tests on synthetic data.
