"""Build the Windows desktop bundle of RAUSNet with PyInstaller (one-folder).

    python scripts/build_installer.py [--console] [--no-clean] [--skip-test] [--zip]

Requires the ``ui`` and ``serve`` extras plus ``pyinstaller`` in the active environment.
Result: ``dist/RAUSNet/RAUSNet.exe``. The model is not compiled into the exe: the workspace seed
(``release/workspace``: current model, its dataset manifest, synthetic spec) is copied next to it as
``workspace_seed/``, with a short Spanish guide ``LEEME.txt``. After building, the exe is run with
``--self-test`` (imports, seed -> local workspace, local API). ``--zip`` writes
``dist/RAUSNet-<version>-windows-x64.zip`` and its SHA-256 for a release.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "packaging" / "rausnet_desktop.spec"
DIST = ROOT / "dist"
WORK = ROOT / "build"
SEED = ROOT / "release" / "workspace"
GUIDE = ROOT / "release" / "LEEME.txt"


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


def add_seed(exe: Path, seed: Path = SEED) -> Path:
    """Copy the workspace seed and the guide next to the exe (replacing an older copy)."""
    target = exe.parent / "workspace_seed"
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(seed, target)
    if GUIDE.exists():
        shutil.copy2(GUIDE, exe.parent / GUIDE.name)
    return target


def make_zip(folder: Path, version: str) -> tuple[Path, str]:
    base = DIST / f"RAUSNet-{version}-windows-x64"
    archive = Path(shutil.make_archive(str(base), "zip", root_dir=folder.parent, base_dir=folder.name))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_name(archive.name + ".sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    return archive, digest


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
    parser.add_argument("--models-dir", type=Path, default=None,
                        help="test this registry instead of the bundled workspace seed")
    parser.add_argument("--zip", action="store_true", help="also write the release zip + SHA-256")
    args = parser.parse_args(argv)
    exe = build(console=args.console, clean=not args.no_clean)
    add_seed(exe)
    size_mb = sum(f.stat().st_size for f in exe.parent.rglob("*") if f.is_file()) / 2**20
    print(f"built {exe} ({size_mb:.0f} MB folder)")
    if not args.skip_test:
        models = args.models_dir if args.models_dir and (args.models_dir / "registry.json").exists() else None
        code = self_test(exe, models)
        print("self-test", "passed" if code == 0 else f"FAILED (exit code {code})")
        if code:
            sys.exit(code)
    if args.zip:
        from rausnet import __version__

        archive, digest = make_zip(exe.parent, __version__)
        print(f"release archive {archive} ({archive.stat().st_size / 2**20:.0f} MB) sha256 {digest}")


if __name__ == "__main__":
    main()
