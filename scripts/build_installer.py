"""Build the Windows desktop bundle of RAUSNet with PyInstaller (one-folder).

    python scripts/build_installer.py [--console] [--no-clean] [--skip-test]

Requires the ``ui`` and ``serve`` extras plus ``pyinstaller`` in the active environment.
Result: ``dist/RAUSNet/RAUSNet.exe``. The model registry is not bundled; copy a
``models`` folder next to the exe (see docs/04_deployment.md). After building, the
exe is run with ``--self-test`` (imports + local API start if ./models exists).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "packaging" / "rausnet_desktop.spec"
DIST = ROOT / "dist"
WORK = ROOT / "build"


def build(console: bool = False, clean: bool = True) -> Path:
    env = dict(os.environ, RAUSNET_CONSOLE="1" if console else "0")
    cmd = [sys.executable, "-m", "PyInstaller", str(SPEC), "--noconfirm",
           "--distpath", str(DIST), "--workpath", str(WORK)]
    if clean:
        cmd.append("--clean")
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)
    exe = DIST / "RAUSNet" / "RAUSNet.exe"
    if not exe.exists():
        raise SystemExit(f"build finished but {exe} is missing")
    return exe


def self_test(exe: Path, models_dir: Path | None) -> int:
    cmd = [str(exe), "--self-test"]
    if models_dir is not None:
        cmd += ["--models-dir", str(models_dir)]
    print("+", " ".join(cmd), flush=True)
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    return subprocess.run(cmd, cwd=exe.parent, env=env, timeout=120).returncode


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--console", action="store_true", help="build with a console window (debugging)")
    parser.add_argument("--no-clean", action="store_true", help="reuse the PyInstaller cache in build/")
    parser.add_argument("--skip-test", action="store_true", help="do not run the exe after building")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models",
                        help="registry used by the post-build self-test (not bundled)")
    args = parser.parse_args(argv)
    exe = build(console=args.console, clean=not args.no_clean)
    size_mb = sum(f.stat().st_size for f in exe.parent.rglob("*") if f.is_file()) / 2**20
    print(f"built {exe} ({size_mb:.0f} MB folder)")
    if not args.skip_test:
        models = args.models_dir if (args.models_dir / "registry.json").exists() else None
        code = self_test(exe, models)
        print("self-test", "passed" if code == 0 else f"FAILED (exit code {code})")
        sys.exit(code)


if __name__ == "__main__":
    main()
