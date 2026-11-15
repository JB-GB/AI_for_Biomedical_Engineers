# 04 - Deployment

RAUSNet ships in two forms, both torch-free (ONNX Runtime only):

| Target | What | Files |
|---|---|---|
| Research server (Linux) | Docker image with the HTTP API (`rausnet-api`) | `Dockerfile`, `.dockerignore`, `docker-compose.yml` |
| Clinician workstation (Windows) | One-folder desktop bundle (`RAUSNet.exe`) that starts a private local API | `packaging/rausnet_desktop.spec`, `packaging/rausnet_desktop_entry.py`, `scripts/build_installer.py` |

In both cases the **model registry is never baked in**. It is the folder produced by
training/export:

```
models/
  registry.json          # exactly one entry with "status": "current"
  v0005/model.onnx       # exported with: python -m rausnet.export
  v0005/model.json       # sidecar: preprocessing, temperatures, referral threshold
```

Only the current version's `model.onnx` + `model.json` are read; `.pt` checkpoints and other
versions may stay in the folder but are ignored. To serve a specific version instead of the
current one, start the API with `--model-dir models/v0004`.

Research use only. The model is not clinically validated and is not a medical device.

## 1. Docker (research servers)

The image is `python:3.11-slim` + `pip install ".[serve]"` (FastAPI, uvicorn, ONNX Runtime CPU,
python-multipart). No PyTorch, no CUDA. Uncompressed size: about 460 MB.

```bash
docker build -t rausnet-api .

# plain docker: registry mounted read-only, port published on localhost only
docker run -d --name rausnet-api --restart unless-stopped \
  -p 127.0.0.1:8000:8000 \
  -v /srv/rausnet/models:/models:ro \
  --read-only --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges:true \
  rausnet-api

curl http://127.0.0.1:8000/health      # {"status":"ok","model_version":"v0005"}
# interactive API docs: http://127.0.0.1:8000/docs
curl -F gs=@gs.png -F pd=@pd.png http://127.0.0.1:8000/predict
```

With Compose (same settings):

```bash
docker compose up -d --build                                # uses ./models
RAUSNET_MODELS_DIR=/srv/rausnet/models RAUSNET_PORT=8000 docker compose up -d
docker compose ps                                           # STATUS shows (healthy)
docker compose down
```

On Windows hosts (Docker Desktop) use a full path, e.g.
`-v "J:/path/to/models:/models:ro"`; in Git Bash prefix the command with `MSYS_NO_PATHCONV=1`.

Details:

* The container runs as the unprivileged user `rausnet` (uid 10001) with a read-only root
  filesystem; `/models` is mounted read-only.
* `HEALTHCHECK` polls `GET /health` every 30 s.
* Default command: `rausnet-api --models-dir /models --host 0.0.0.0 --port 8000`. Arguments after
  the image name replace it, e.g. `... rausnet-api --model-dir /models/v0004 --host 0.0.0.0`.
* **Updating the model**: export the new version, promote it in `registry.json`, then
  `docker compose restart api`. The model is loaded once at start-up.
* **Clinician corrections** (`POST /corrections`) are disabled by default (HTTP 503). To enable
  them, mount a writable data folder and pass `--data-dir`, e.g. add
  `-v /srv/rausnet/data:/data` and the command
  `--models-dir /models --data-dir /data --host 0.0.0.0`. The folder must be writable by uid 10001.

## 2. Windows desktop bundle (PyInstaller) and releases

**Releases** are built by `.github/workflows/release.yml` when a tag `vX.Y.Z` is pushed:
1. it checks that the tag equals `rausnet.__version__` and that `docs/releases/<tag>.md` exists;
2. it builds on `windows-latest`, runs the self-test, and publishes
   `RAUSNet-<version>-windows-x64.zip` plus its `.sha256` as a GitHub pre-release.

```bash
git tag -a v0.2.0 -m "RAUSNet 0.2.0" && git push origin v0.2.0
```

**Local build** (from the repository root, in the project venv with the `serve` and `ui` extras):

```powershell
.venv\Scripts\python.exe -m pip install -e ".[serve,ui]" pyinstaller
.venv\Scripts\python.exe scripts\build_installer.py --zip      # windowed build + release zip
.venv\Scripts\python.exe scripts\build_installer.py --console  # with console, for debugging
```

The script runs `PyInstaller packaging/rausnet_desktop.spec` into `build/` (cache) and
`dist/RAUSNet/` (output). It then copies the **workspace seed** (`release/workspace` →
`workspace_seed/`) and the Spanish guide `release/LEEME.txt` next to the exe, and runs
`RAUSNet.exe --self-test`. The self-test:
- imports the GUI, ONNX Runtime and API stacks;
- makes a temporary workspace from the seed, starts the local API, and reads the status and versions;
- exits 0 on success.

`build/` and `dist/` are git-ignored. The training stack (torch, timm, mlflow, onnx) is excluded, so
improvement runs on Colab (or in a source installation with a GPU).

### The workspace seed

`release/workspace` is a complete starting workspace, small enough for git (about 32 MB):
- `models/registry.json` with the current version only (v0005), and `v0005/` with `model.onnx`,
  `model.json`, `calibration.json` and `model.pt` (the checkpoint is needed to fine-tune the next
  version);
- `data/datasets/v0005/manifest.csv`, whose SHA-256 matches the registry;
- `workspace.json`: the deterministic synthetic sets the manifest uses (seed images + two batches).
  Images are regenerated on demand during an improvement, never stored.

The seed is used in two ways:
1. **First start:** with no shared folder chosen and no `--models-dir`, the app creates the local
   workspace `%LOCALAPPDATA%\RAUSNet\desktop\workspace` from the seed. Analysing, corrections, the
   improvement panel and the version table work offline.
2. **New shared folder:** in *Ajustes*, choosing an empty folder (e.g. on Google Drive) offers to set
   it up from the seed. Every computer then chooses the same folder.

A real improvement on a seeded workspace was checked: regenerated data, 24 corrections, fine-tune,
promotion, calibration and ONNX export.

Layout of the release folder:

```
RAUSNet\
  RAUSNet.exe
  LEEME.txt
  _internal\
  workspace_seed\  (models\, data\datasets\, workspace.json)
```

Other options:

```powershell
RAUSNet.exe --workspace G:\MyDrive\RAUSNet    # shared folder (also remembered from Ajustes)
RAUSNet.exe --models-dir D:\rausnet\models      # serve a registry instead of a workspace
RAUSNet.exe --server http://127.0.0.1:8000      # use a running API (e.g. via SSH tunnel to the server)
RAUSNet.exe --lang en                           # English interface
```

An old-style `models\` folder in the working directory or next to the exe is still served as
before. If neither a seed nor a registry is found, the app shows an error dialog.

## 3. Security notes

* **Localhost binding.** The API has no authentication. `rausnet-api` defaults to
  `--host 127.0.0.1`; the desktop app binds its private API to `127.0.0.1` on a random free port;
  Compose publishes the container port on `127.0.0.1` only. Inside the container the server
  listens on `0.0.0.0`, which is reachable only through that host-side mapping. To use the API from
  other machines, put it behind a reverse proxy with TLS and authentication, or use an SSH tunnel
  (`ssh -L 8000:127.0.0.1:8000 server`). Do not change the mapping to `0.0.0.0:8000`.
* **No PHI in logs.** The API logs only uvicorn access lines (method, path, status, client IP);
  image contents, file names and grades are not logged. Do not add request bodies or file names
  to logs, and do not upload images with patient identifiers burned into the pixels. Corrections
  accept pseudonymous `patient_id`s only and images are re-encoded as PNG, which drops embedded
  metadata (EXIF/text chunks). DICOM is not accepted; export anonymised PNGs.
* **Read-only models.** The registry is mounted read-only in Docker and is only read by the
  desktop app, so a compromised or buggy server cannot alter the model. Keep write access to the
  registry limited to the retraining pipeline, and record `registry.json` changes.
* **Small attack surface.** The image contains no training code dependencies, no compilers and no
  data; the build context excludes `models/`, `data/`, `outputs/`, `.venv/`, `.git/` and notebooks
  (`.dockerignore`). Uploads are limited to 20 MB per image.
* **Images are never pushed** to a public registry; build them on the research server or push to an
  institution-internal registry.
