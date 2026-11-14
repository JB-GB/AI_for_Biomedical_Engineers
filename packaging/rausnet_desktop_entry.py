"""PyInstaller entry point for the RAUSNet desktop app (wraps ``rausnet.ui.app.main``).

Extra behaviour, only relevant to the frozen Windows build:

* Windowed builds have ``sys.stdout``/``sys.stderr`` set to ``None``; uvicorn's log
  formatter calls ``isatty()`` on them, so they are redirected to ``os.devnull``.
* If neither ``--models-dir`` nor ``--server`` is given, the registry is looked up
  in ``./models`` (current directory) and then in ``models`` next to the executable.
* ``--self-test`` imports the GUI, ONNX Runtime and API stacks (and, if a registry is
  found, loads the current model and starts the local API), then exits with code 0
  on success or 1 on failure without opening a window. Used by the build script.
* Start-up errors are shown in a message box instead of disappearing silently.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path


def _exe_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path.cwd()


def _default_models_dir() -> Path:
    cwd_models = Path.cwd() / "models"
    if (cwd_models / "registry.json").exists():
        return cwd_models
    return _exe_dir() / "models"


def _fix_std_streams() -> None:
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))


def _with_default_models(argv: list[str]) -> list[str]:
    if any(a == "--server" or a.startswith("--server=") for a in argv):
        return argv
    if any(a == "--models-dir" or a.startswith("--models-dir=") for a in argv):
        return argv
    return [*argv, "--models-dir", str(_default_models_dir())]


def self_test(argv: list[str]) -> int:
    import onnxruntime
    import PySide6.QtWidgets  # noqa: F401
    import uvicorn  # noqa: F401

    import rausnet.serve.api  # noqa: F401
    from rausnet.ui import app

    print(f"imports ok: onnxruntime {onnxruntime.__version__}, providers {onnxruntime.get_available_providers()}")
    argv = _with_default_models(argv)
    models_dir = Path(argv[argv.index("--models-dir") + 1]) if "--models-dir" in argv else None
    if models_dir is not None and (models_dir / "registry.json").exists():
        url = app.start_local_server(str(models_dir))
        from rausnet.ui.client import ApiClient

        print(f"local API ok at {url}: {ApiClient(url).health()}")
    else:
        print(f"no registry found at {models_dir}; model loading not tested")
    return 0


def main() -> None:
    _fix_std_streams()
    argv = sys.argv[1:]
    if "--self-test" in argv:
        argv.remove("--self-test")
        try:
            code = self_test(argv)
        except Exception:
            traceback.print_exc()
            code = 1
        sys.stdout.flush()
        os._exit(code)  # skip waiting for the daemon server thread

    from rausnet.ui.app import main as app_main

    try:
        app_main(_with_default_models(argv))
    except SystemExit:
        raise
    except Exception as exc:
        traceback.print_exc()
        try:
            from PySide6.QtWidgets import QApplication, QMessageBox

            _qt = QApplication.instance() or QApplication(sys.argv[:1])
            QMessageBox.critical(None, "RAUSNet", f"RAUSNet could not start:\n\n{exc}\n\n"
                                 f"Expected a model registry (registry.json) in:\n{_default_models_dir()}")
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
