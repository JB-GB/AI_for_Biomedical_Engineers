# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the RAUSNet desktop app (Windows, one-folder build).
#
#   python scripts/build_installer.py          # preferred
#   python -m PyInstaller packaging/rausnet_desktop.spec --distpath dist --workpath build
#
# Output: dist/RAUSNet/RAUSNet.exe plus its _internal/ folder.
# The model registry is NOT bundled: put a "models" folder (registry.json +
# <version>/model.onnx + model.json) next to RAUSNet.exe, or pass --models-dir.
# Set RAUSNET_CONSOLE=1 to build with a console window (useful for debugging).
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

ROOT = Path(SPECPATH).resolve().parent  # noqa: F821 (SPECPATH is injected by PyInstaller)
CONSOLE = os.environ.get("RAUSNET_CONSOLE", "0") == "1"

hiddenimports = (
    collect_submodules("rausnet.serve")
    + collect_submodules("rausnet.ui")
    + collect_submodules("uvicorn")  # uvicorn picks loops/protocols by string at runtime
    + collect_submodules("onnxruntime.capi")
    + ["rausnet.registry", "rausnet.heatmap", "rausnet.metrics", "rausnet.grading", "rausnet.data.manifest",
       "multipart", "python_multipart", "httpx", "anyio._backends._asyncio"]
)
binaries = collect_dynamic_libs("onnxruntime")
datas = collect_data_files("onnxruntime", includes=["**/*.dll", "**/*.pyd"])

# Training stack and unused heavy packages must never end up in the desktop build.
excludes = [
    "torch", "torchvision", "timm", "mlflow", "onnx", "matplotlib", "pandas", "scipy", "sklearn",
    "IPython", "jupyter", "notebook", "pytest", "tkinter",
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore", "PySide6.QtQuick",
    "PySide6.QtQml", "PySide6.QtMultimedia", "PySide6.QtCharts", "PySide6.QtDataVisualization",
]

a = Analysis(  # noqa: F821
    [str(ROOT / "packaging" / "rausnet_desktop_entry.py")],
    pathex=[str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="RAUSNet",
    debug=False,
    strip=False,
    upx=False,
    console=CONSOLE,
)
coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="RAUSNet",
)
