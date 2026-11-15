#!/usr/bin/env bash
# RAUSNet: install everything with one command (Linux / macOS).
#
#   curl -fsSL https://raw.githubusercontent.com/JB-GB/AI_for_Biomedical_Engineers/PIA_devl/scripts/install.sh | bash
#
# Inside a clone: bash scripts/install.sh. Options as environment variables:
#   RAUSNET_CPU=1      PyTorch without CUDA          RAUSNET_APP_ONLY=1  desktop app only (no PyTorch)
#   RAUSNET_DIR=path   where to clone (default ~/RAUSNet)   RAUSNET_BRANCH=name (default PIA_devl)
#
# Steps: find Python 3.11+ and Git -> clone or update the repository -> scripts/setup_env.py
# (virtual environment, PyTorch for the GPU if nvidia-smi is found, all extras) -> tests a model load.
set -euo pipefail
REPO="https://github.com/JB-GB/AI_for_Biomedical_Engineers.git"
DIR="${RAUSNET_DIR:-$HOME/RAUSNet}"
BRANCH="${RAUSNET_BRANCH:-PIA_devl}"
step() { printf '\n==> %s\n' "$1"; }
fail() { printf '\nERROR: %s\n' "$1" >&2; exit 1; }

step "Looking for Python 3.11 or newer"
PY=""
for c in python3.11 python3.12 python3.13 python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
    PY="$c"; break
  fi
done
[ -n "$PY" ] || fail "Python 3.11+ not found (Ubuntu: sudo apt install python3.11 python3.11-venv; macOS: brew install python@3.11)"
echo "Python: $PY ($("$PY" -V 2>&1))"

step "Getting the code"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || true)"
if [ -n "$HERE" ] && [ -f "$HERE/../pyproject.toml" ]; then
  ROOT="$(cd "$HERE/.." && pwd)"; echo "Using this clone: $ROOT"
else
  command -v git >/dev/null 2>&1 || fail "Git not found; install it and run this command again"
  if [ -d "$DIR/.git" ]; then
    git -C "$DIR" fetch --quiet origin "$BRANCH"
    git -C "$DIR" checkout --quiet "$BRANCH"
    git -C "$DIR" pull --quiet --ff-only origin "$BRANCH" || fail "git pull failed (local changes?)"
    echo "Updated $DIR"
  else
    git clone --quiet --branch "$BRANCH" "$REPO" "$DIR"; echo "Cloned into $DIR"
  fi
  ROOT="$(cd "$DIR" && pwd)"
fi

ARGS=()
if [ "${RAUSNET_APP_ONLY:-0}" = "1" ]; then
  ARGS+=(--app-only); step "Installing the desktop app (no PyTorch)"
elif [ "${RAUSNET_CPU:-0}" = "1" ] || ! command -v nvidia-smi >/dev/null 2>&1; then
  ARGS+=(--cpu); step "Installing everything (PyTorch for CPU: no NVIDIA GPU found)"
else
  step "Installing everything (PyTorch with CUDA for the NVIDIA GPU)"
fi
cd "$ROOT"
"$PY" scripts/setup_env.py "${ARGS[@]}" || fail "setup failed (see the messages above)"

step "Checking the bundled model"
.venv/bin/python -c "from rausnet.serve.api import resolve_model_dir; from rausnet.serve.inference import Predictor; p = Predictor(resolve_model_dir('release/workspace/models')); print('model', p.info['version'], 'ready')" \
  || fail "the model could not be loaded"

printf '\nRAUSNet is installed in %s\nStart the app:  %s/.venv/bin/rausnet-desktop   (--lang en for English)\n' "$ROOT" "$ROOT"
echo "Research use only: the bundled model is trained on synthetic images."
