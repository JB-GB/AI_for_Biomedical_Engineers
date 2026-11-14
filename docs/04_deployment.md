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

## 2. Windows desktop bundle (PyInstaller)

Build (from the repository root, in the project venv with the `serve` and `ui` extras):

```powershell
.venv\Scripts\python.exe -m pip install -e ".[serve,ui]" pyinstaller
.venv\Scripts\python.exe scripts\build_installer.py            # windowed build
.venv\Scripts\python.exe scripts\build_installer.py --console  # with console, for debugging
```

The script runs `PyInstaller packaging/rausnet_desktop.spec` into `build/` (cache) and
`dist/RAUSNet/` (output), then runs `RAUSNet.exe --self-test`, which imports the GUI, ONNX
Runtime and API stacks, loads the current model from `./models` if present, starts the local API,
and exits with code 0 on success. `build/` and `dist/` are git-ignored; never commit binaries.

The bundle is a one-folder build: `dist/RAUSNet/RAUSNet.exe` plus `dist/RAUSNet/_internal/`.
Zip the whole `RAUSNet` folder to distribute it (an optional Inno Setup/MSI wrapper can be added
around that folder later). The training stack (torch, timm, mlflow, onnx) is excluded.

### Where the models folder goes

The model registry is **not** bundled. When neither `--models-dir` nor `--server` is given the
app looks for `registry.json` in:

1. `models\` in the current working directory, then
2. `models\` next to `RAUSNet.exe`.

So the normal layout for a workstation is:

```
RAUSNet\
  RAUSNet.exe
  _internal\
  models\
    registry.json
    v0005\model.onnx
    v0005\model.json
```

Other options:

```powershell
RAUSNet.exe --models-dir D:\rausnet\models      # registry elsewhere
RAUSNet.exe --server http://127.0.0.1:8000      # use a running API (e.g. via SSH tunnel to the server)
RAUSNet.exe --lang en                           # English interface
```

If no registry is found the app shows an error dialog naming the expected location.

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
