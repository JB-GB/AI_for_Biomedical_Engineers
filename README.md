# AI_for_Biomedical_Engineers
A semestral course to take a deep look at AI. In person course at UANL FIME

## PIA project: RAUSNet

Retrainable EULAR-OMERACT synovitis grading (0–3, grey-scale + Power Doppler)
on MCP 2–3 and wrist ultrasound. **Research use only — not a diagnostic device.**

Docs: [feasibility](docs/00_feasibility.md) · [roadmap](docs/01_roadmap.md) ·
[architecture](docs/02_architecture.md) · [runtimes](docs/03_colab_and_runners.md) ·
[**resume guide**](docs/RESUME.md) · [logbook](docs/logbook/README.md) · [paper](docs/paper/README.md) · [contributing](CONTRIBUTING.md)

### Quick start (development)
```bash
git config core.hooksPath .githooks
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"   # Linux/macOS: .venv/bin/python
.venv/Scripts/python -m pytest
```
Extras: `train` (PyTorch), `serve` (FastAPI + ONNX Runtime), `ui` (PySide6 desktop).

### Use the desktop app
```bash
.venv/Scripts/python -m pip install -e ".[serve,ui]"
.venv/Scripts/python -m rausnet.export      # once per promoted model
.venv/Scripts/rausnet-desktop               # Spanish UI; --lang en for English
```
