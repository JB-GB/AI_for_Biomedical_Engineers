"""One-command local setup: virtual environment + PyTorch (GPU or CPU) + RAUSNet.

Usage (from the repository root, any OS, Python >= 3.10):
    python scripts/setup_env.py            # NVIDIA GPU (CUDA 12.6 wheels)
    python scripts/setup_env.py --cpu      # no GPU
    python scripts/setup_env.py --extras train,serve,ui   # fewer extras (default: all)

Creates ``.venv`` (or ``--venv``), installs torch from the official PyTorch index, then
``pip install -e .[extras]``, activates the commit-msg hook and prints a device check.
Re-running is safe: existing packages are kept when they already satisfy the pins.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TORCH_INDEX = {"gpu": "https://download.pytorch.org/whl/cu126", "cpu": "https://download.pytorch.org/whl/cpu"}


def venv_python(path: Path) -> Path:
    return path / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def run(*cmd) -> None:
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cpu", action="store_true", help="CPU-only PyTorch")
    parser.add_argument("--venv", default=".venv")
    parser.add_argument("--extras", default="train,serve,ui,dev,paper")
    args = parser.parse_args(argv)
    if sys.version_info < (3, 10):
        sys.exit("Python >= 3.10 required")

    env_dir = ROOT / args.venv
    if not venv_python(env_dir).exists():
        print(f"creating {env_dir}")
        venv.create(env_dir, with_pip=True)
    py = venv_python(env_dir)
    run(py, "-m", "pip", "install", "-q", "--upgrade", "pip")
    index = TORCH_INDEX["cpu" if args.cpu else "gpu"]
    run(py, "-m", "pip", "install", "-q", "torch", "torchvision", "--index-url", index)
    run(py, "-m", "pip", "install", "-q", "-e", f".[{args.extras}]")
    run("git", "config", "core.hooksPath", ".githooks")
    run(py, "-c", "import torch, rausnet; print('rausnet', rausnet.__version__, '| torch', torch.__version__, "
                  "'| CUDA', torch.cuda.is_available())")
    print("\nnext: set RAUSNET_DATA_DIR to a folder outside the repo, then see docs/RESUME.md")


if __name__ == "__main__":
    main()
