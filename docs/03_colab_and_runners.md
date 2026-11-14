# 04 — Execution Runtimes: Local, Colab, Self-hosted Runner

The same commands work everywhere (M7):

```bash
python -m rausnet.train   --config configs/train/baseline.yaml [--set key=value ...]
python -m rausnet.train   --resume outputs/<run_dir>          # continue an interrupted run
python -m rausnet.retrain --config configs/retrain/default.yaml --add <manifest> [--add ...]
```

A promoted retrain candidate is calibrated (`calibration.json`) and exported to ONNX
(`model.onnx` + `model.json`) in the same command, so it is ready to serve. `--no-postprocess`
skips both; the `postprocess:` section of `configs/retrain/default.yaml` controls them. If ONNX
packages are missing the export is skipped with a warning (the promotion stands); run
`python -m rausnet.export` later. Rejected candidates are left untouched.

| Runtime | Entry point | Data | Results |
|---|---|---|---|
| Local host | `python scripts/setup_env.py [--cpu]` | `RAUSNET_DATA_DIR` (outside the repo) | `outputs/`, `models/` |
| Google Colab | `notebooks/colab_train.ipynb` | generated, or private Drive copied to `/content/data` | private Drive folder |
| Self-hosted runner | Actions → *train-self-hosted* → Run workflow | `RAUSNET_DATA_DIR` on the runner | registry on the runner; metrics as artefact |
| GitHub-hosted (free) | `.github/workflows/ci.yml` on every push/PR | synthetic, generated in tests | lint + test results |

## 1. Local host (T7.1)
- `python scripts/setup_env.py` creates `.venv`, installs PyTorch from the official index
  (CUDA 12.6 wheels, or `--cpu`), then `pip install -e .[train,serve,ui,dev,paper]` and
  activates the commit-msg hook. Re-running it is safe.
- Data lives **outside** the repo; set `RAUSNET_DATA_DIR` (e.g. `J:/rausnet_data`).
- Windows: keep `train.num_workers: 0` (page-file error WinError 1455 with workers > 0).
- GUI: `rausnet-desktop` (starts a local API on 127.0.0.1 if none is running).

## 2. Google Colab — safe configuration (T7.2)

Open `notebooks/colab_train.ipynb` in Colab (*File → Open notebook → GitHub*, branch `PIA_devl`),
edit the settings cell, then *Runtime → Run all*.

**Cost safety**
- Use the **free tier** only; never buy compute units. Each member uses their own Google account.
- Notebook metadata asks for a generic GPU (T4 on the free tier), never a paid type.
- `last.pt` is written to Drive every epoch (atomic write). After a disconnect, *Run all* again:
  the notebook finds the unfinished run and calls `rausnet.train --resume`. At most one epoch is lost.
- The last cell releases the runtime (`runtime.unassign()`), so no idle GPU time is used.
- Do not use Colab Enterprise / GCP projects with billing enabled.

**Data & secrets safety**
- Patient data only in a **private Drive folder** (`DRIVE_DIR/data`). It is copied to the session's
  local disk for speed and disappears with the session. The notebook never displays images.
- No tokens are needed (public repo, read-only clone). If one is ever needed, use **Colab Secrets**
  (`google.colab.userdata`), never hard-coded.
- Clone read-only (`git clone --depth 1 -b PIA_devl`); push from local machines only, so
  authorship rules stay correct.

**Notebook hygiene in Git**
- Commit notebooks **without outputs** (`nbstripout`, in the `dev` extra).
  `tests/test_notebooks.py` fails if outputs, execution counts, tokens or `share=True` appear.
- Notebooks only orchestrate; all logic lives in `src/rausnet/`.

## 3. Self-hosted GitHub Actions runner (T7.3)

Workflow: `.github/workflows/train-self-hosted.yml`.

**Security rules (the repository is public → critical)**
- Trigger is **`workflow_dispatch` only** (manual, by people with write access). Never add
  `push` or `pull_request` triggers: fork PRs could then run code on the GPU machine.
- Settings → Actions → General → *Require approval for all outside collaborators*.
- `permissions: contents: read`; `persist-credentials: false`; no secrets used.
- Inputs reach the scripts through environment variables, never `${{ }}` inside `run:`
  (prevents script injection).
- Only metrics leave the machine (model card, test metrics, history, config, retrain report,
  calibration). Weights and data stay on the runner.
- Run the runner service as an unprivileged user; one GPU job at a time (`concurrency`).

**One-time setup (on the GPU machine)**
1. Repo → Settings → Actions → Runners → *New self-hosted runner*; follow the steps and add the
   labels `gpu` and `rausnet` (the runner also has `self-hosted`).
2. Create a persistent environment outside the runner workspace, e.g.
   `python scripts/setup_env.py --venv D:/rausnet-runner/venv --extras train,serve`.
3. Repo → Settings → Secrets and variables → Actions → **Variables** (not secrets):
   | Variable | Example (Windows) |
   |---|---|
   | `RAUSNET_PYTHON` | `D:/rausnet-runner/venv/Scripts/python.exe` |
   | `RAUSNET_DATA_DIR` | `J:/rausnet_data` |
   | `RAUSNET_MODELS_DIR` | `J:/rausnet_models` (persistent registry) |
   Use forward slashes. On Windows the workflow uses Git Bash (`shell: bash`).

**Running:** Actions → *train-self-hosted* → *Run workflow* → choose `train` or `retrain`. A
`retrain` calibrates and exports ONNX when the candidate is promoted (done by `rausnet.retrain`);
unchecking *export* passes `--no-postprocess`.

**GitHub-hosted runners** (free minutes) only run `ci.yml`: ruff + the full test suite with CPU
PyTorch and offscreen Qt, on synthetic data generated by the tests.

## 4. Equivalence across runtimes (T7.4)
The same config and seed produce the same split and dataset hash everywhere (deterministic
synthetic generator, hash-based patient split). Weights are **not** bit-identical across GPUs,
CUDA versions or after `--resume` (non-deterministic kernels, data-order RNG). Compare runs with
`python scripts/compare_runs.py <run_a> <run_b>`: same `manifest_sha256` and split sizes are
required; test QWK differences are judged against the bootstrap CI. Log each comparison in the
logbook.
