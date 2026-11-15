"""PyInstaller entry point for the RAUSNet desktop app (wraps ``rausnet.ui.app.main``).

Extra behaviour, only relevant to the frozen Windows build:

* Windowed builds have ``sys.stdout``/``sys.stderr`` set to ``None``; uvicorn's log
  formatter calls ``isatty()`` on them, so they are redirected to ``os.devnull``.
* The release ships ``workspace_seed/`` next to the executable: on first start the app makes a
  local workspace from it (``rausnet.ui.app.seed_dir``). Only if an old-style ``models`` folder
  exists (``./models`` or next to the executable) and no ``--models-dir``/``--server``/``--workspace``
  is given, that registry is served instead, as in earlier builds.
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
    for flag in ("--server", "--models-dir", "--workspace"):
        if any(a == flag or a.startswith(flag + "=") for a in argv):
            return argv
    legacy = _default_models_dir()
    if (legacy / "registry.json").exists():
        return [*argv, "--models-dir", str(legacy)]
    return argv  # the app uses the bundled workspace seed


def self_test(argv: list[str]) -> int:
    import onnxruntime
    import PySide6.QtWidgets  # noqa: F401
    import uvicorn  # noqa: F401

    import rausnet.serve.api  # noqa: F401
    from rausnet.ui import app

    print(f"imports ok: onnxruntime {onnxruntime.__version__}, providers {onnxruntime.get_available_providers()}")
    from rausnet.ui.client import ApiClient

    argv = _with_default_models(argv)
    models_dir = Path(argv[argv.index("--models-dir") + 1]) if "--models-dir" in argv else None
    seed = app.seed_dir()
    if models_dir is not None and (models_dir / "registry.json").exists():
        url = app.start_local_server(str(models_dir))
        print(f"local API ok at {url}: {ApiClient(url).health()}")
    elif seed is not None:
        import tempfile

        from rausnet import improve

        ws = Path(tempfile.mkdtemp(prefix="rausnet_selftest_")) / "workspace"
        improve.seed_workspace(ws, seed)
        url = app.start_local_server(str(ws / "models"), str(ws / "data"), workspace=str(ws))
        client = ApiClient(url)
        st = client.workspace_status()
        print(f"workspace seed ok ({seed.name}): {client.health()}, pending {st['pending_images']}, "
              f"versions {[v['version'] for v in client.model_versions()['versions']]}")
    else:
        print("no registry or workspace seed found; model loading not tested")
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
                                 f"Expected the folder 'workspace_seed' next to RAUSNet.exe "
                                 f"(or a model registry in {_default_models_dir()}).")
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
