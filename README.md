# AI_for_Biomedical_Engineers
A semestral course to take a deep look at AI. In person course at UANL FIME

## PIA project: RAUSNet

Retrainable EULAR-OMERACT synovitis grading (0–3, grey-scale + Power Doppler)
on MCP 2–3 and wrist ultrasound, with an accessible desktop app for clinicians.
**Research use only — not a diagnostic device.** The bundled model is trained on synthetic images.

Docs: [feasibility](docs/00_feasibility.md) · [roadmap](docs/01_roadmap.md) ·
[architecture](docs/02_architecture.md) · [runtimes](docs/03_colab_and_runners.md) ·
[deployment](docs/04_deployment.md) · [**pending work**](docs/05_pending.md) · [interface](docs/06_interface_plan.md) ·
[usability protocol](docs/07_usability_protocol.md) · [**resume guide**](docs/RESUME.md) ·
[logbook](docs/logbook/README.md) · [paper](docs/paper/README.md) · [contributing](CONTRIBUTING.md)

### Install everything with one command
Needs Python 3.11+ and Git. The script clones the repository (branch `PIA_devl`) into `~/RAUSNet`,
creates `.venv`, installs PyTorch (CUDA if an NVIDIA GPU is found, otherwise CPU) and all extras,
checks that the bundled model loads and, on Windows, adds a desktop shortcut.

**Windows (PowerShell):**
```powershell
irm https://raw.githubusercontent.com/JB-GB/AI_for_Biomedical_Engineers/PIA_devl/scripts/install.ps1 | iex
```
**Linux / macOS:**
```bash
curl -fsSL https://raw.githubusercontent.com/JB-GB/AI_for_Biomedical_Engineers/PIA_devl/scripts/install.sh | bash
```
- **Options** (environment variables, set before the command):
  - `RAUSNET_APP_ONLY=1`: desktop app only, without PyTorch;
  - `RAUSNET_CPU=1`: CPU PyTorch;
  - `RAUSNET_DIR`: install folder.

  Example in PowerShell: `$env:RAUSNET_APP_ONLY=1; irm … | iex`.
- **Already cloned:** run `powershell -ExecutionPolicy Bypass -File scripts\install.ps1` or
  `bash scripts/install.sh` from the repository.
- **Start the app:** use the RAUSNet shortcut, `~/RAUSNet/.venv/Scripts/rausnet-desktop` (Windows) or
  `~/RAUSNet/.venv/bin/rausnet-desktop`.

### Desktop release (Windows, no installation)
[Releases](https://github.com/JB-GB/AI_for_Biomedical_Engineers/releases):
1. Download `RAUSNet-<version>-windows-x64.zip`.
2. Unzip it.
3. Open `RAUSNet.exe` (guide in `LEEME.txt`).

| Need | In the app |
|---|---|
| Predict on an image | *Analizar*: drop GS + PD → *Analizar* (grade, confidence, referral warning, heat map, PDF) |
| Feed the model | correct the grade → *Enviar corrección*; *Mejorar el modelo* → *Mejorar en Colab* / *en este equipo* |
| Model state and version | header (version in use) and *Mejorar el modelo* (global version, quality, pending images, history, versions, roll back) |

Each release is built and published by `.github/workflows/release.yml` when a `vX.Y.Z` tag is pushed
(notes in `docs/releases/`). To build it locally: `python scripts/build_installer.py --zip`.

**Improve the global model on Colab (free GPU):**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JB-GB/AI_for_Biomedical_Engineers/blob/PIA_devl/notebooks/colab_improve.ipynb)
→ `Runtime > Run all` → allow Drive ([how it works](docs/03_colab_and_runners.md)).

### Quick start (development, manual)
```bash
git config core.hooksPath .githooks
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"   # Linux/macOS: .venv/bin/python
.venv/Scripts/python -m pytest
```
- **Extras:** `train` (PyTorch), `serve` (FastAPI + ONNX Runtime), `ui` (PySide6 desktop).
- **`python scripts/setup_env.py`** (`--cpu`, `--app-only`) does the same as the installer inside a clone.

### Use the desktop app from source
```bash
.venv/Scripts/rausnet-desktop               # Spanish UI; --lang en for English
```
- **First start:** a local workspace is created from the bundled seed (`release/workspace`) in the user's
  local app data. *Ajustes* → *Elegir carpeta…* switches to a shared folder (e.g. on Google Drive); an
  empty folder is set up with the bundled model.
- **Your own registry:** `--models-dir models` serves it instead.
