"""Desktop release: workspace seed (first start and empty shared folders), release files, version."""

import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import pytest

from rausnet import __version__
from rausnet.improve import Workspace, _ensure_synthetic, seed_workspace, status
from rausnet.registry import Registry

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "release" / "workspace"


def test_release_seed_is_complete_and_consistent():
    cur = Registry(SEED / "models").current()
    assert cur is not None and len(Registry(SEED / "models").entries) == 1
    folder = SEED / "models" / Path(cur.model_path).parent
    assert (folder / "model.onnx").exists() and (folder / "model.json").exists()
    assert (folder / "model.pt").exists()  # fine-tuning the next version needs the checkpoint
    manifest = SEED / "data" / cur.dataset_manifest
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == cur.dataset_sha256  # provenance kept
    spec = json.loads((SEED / "workspace.json").read_text())
    dirs = {s["dir"] for s in spec["synthetic"]}
    used = {"/".join(line.split(",")[4].split("/")[:-2]) for line in manifest.read_text().splitlines()[1:]}
    assert used <= dirs, "every image of the dataset must be regenerable from the spec"
    assert json.loads((folder / "model.json").read_text())["version"] == cur.version


def test_seed_workspace_once_and_status(tmp_path):
    ws = tmp_path / "ws"
    assert seed_workspace(ws, SEED)
    assert not seed_workspace(ws, SEED)  # never overwrites a global model
    st = status(ws)
    assert st["current_model"]["version"] == Registry(SEED / "models").current().version
    assert st["pending_images"] == 0 and not st["incomplete_batches"]
    assert (ws / "data/incoming/corrections").is_dir()
    with pytest.raises(FileNotFoundError):
        seed_workspace(tmp_path / "other", tmp_path / "not_a_seed")


def test_synthetic_spec_accepts_several_sets(tmp_path):
    spec = {"synthetic": [{"dir": "synth", "patients": 1, "seed": 0, "size": 64},
                          {"dir": "incoming/b1", "patients": 1, "seed": 1, "size": 64}]}
    assert _ensure_synthetic(spec, tmp_path) == "synth/manifest.csv"
    assert (tmp_path / "synth/manifest.csv").exists() and (tmp_path / "incoming/b1/manifest.csv").exists()
    single = {"synthetic": {"dir": "s", "patients": 1, "seed": 0, "size": 64}}
    assert _ensure_synthetic(single, tmp_path) == "s/manifest.csv"


def test_release_guide_and_version():
    guide = (ROOT / "release" / "LEEME.txt").read_text(encoding="utf-8")
    for section in ("PREDECIR", "ALIMENTAR AL MODELO", "ESTADO Y LA VERSIÓN", "SOLO PARA INVESTIGACIÓN"):
        assert section in guide
    assert __version__ in guide.splitlines()[0]
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert re.search(r'^version = "([^"]+)"', pyproject, re.M).group(1) == __version__
    assert (ROOT / "docs" / "releases" / f"v{__version__}.md").exists()


# ---- desktop: first start and empty shared folder
pytest.importorskip("PySide6")
pytest.importorskip("fastapi")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture()
def small_seed(tmp_path, onnx_model):
    """A seed like release/workspace but with the small test model."""
    from rausnet.data.manifest import write_manifest
    from rausnet.registry import ModelEntry

    seed = tmp_path / "seed"
    shutil.copytree(onnx_model[0], seed / "models/v0001")
    side = json.loads((seed / "models/v0001/model.json").read_text())
    (seed / "models/v0001/model.json").write_text(json.dumps({**side, "version": "v0001"}))
    Registry(seed / "models").add(ModelEntry("v0001", "2026-11-14T10:00:00", "v0001/model.pt",
                                             "datasets/v0001/manifest.csv", "x", "S2", None), promote=True)
    (seed / "data/datasets/v0001").mkdir(parents=True)
    write_manifest([], seed / "data/datasets/v0001/manifest.csv")
    (seed / "workspace.json").write_text(json.dumps({"synthetic": {"dir": "synth", "patients": 1, "seed": 0}}))
    return seed


def test_seed_dir_finds_the_source_seed_and_honours_env(monkeypatch, tmp_path, small_seed):
    from rausnet.ui.app import seed_dir, seed_version

    monkeypatch.delenv("RAUSNET_SEED", raising=False)
    assert seed_dir() == SEED
    monkeypatch.setenv("RAUSNET_SEED", str(small_seed))
    assert seed_dir() == small_seed and seed_version(small_seed) == "v0001"
    monkeypatch.setenv("RAUSNET_SEED", str(tmp_path / "missing"))
    assert seed_dir() is None


def test_empty_folder_is_offered_as_new_shared_folder(qapp, small_seed, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from PySide6.QtWidgets import QMessageBox

    from rausnet.serve.api import create_app, resolve_model_dir
    from rausnet.ui.app import MainWindow
    from rausnet.ui.client import ApiClient

    local = tmp_path / "local"
    seed_workspace(local, small_seed)

    def connect(path):
        ws = Path(path)
        return ApiClient(http=TestClient(create_app(resolve_model_dir(ws / "models"), ws / "data",
                                                    models_dir=ws / "models", workspace=ws)))

    w = MainWindow(connect(local), "es", connect=connect, workspace=str(local), seed=small_seed,
                   local_workspace=str(local))
    assert "carpeta local de este equipo" in w.ws_label.text()
    shared = tmp_path / "Drive" / "RAUSNet"
    shared.mkdir(parents=True)
    asked = []

    def answer(reply):
        def ask(_parent, _title, text, *a, **k):
            asked.append(text)
            return reply
        return ask

    monkeypatch.setattr(QMessageBox, "question", answer(QMessageBox.StandardButton.No))
    assert not w.choose_workspace(str(shared))
    assert "v0001" in asked[-1] and not (shared / "models").exists()
    monkeypatch.setattr(QMessageBox, "question", answer(QMessageBox.StandardButton.Yes))
    assert w.choose_workspace(str(shared))
    assert Workspace(shared).spec["synthetic"]["dir"] == "synth"
    assert w.ws_label.text() == str(shared) and w.improve_view.last["current_model"]["version"] == "v0001"


def test_build_script_adds_seed_and_zips(tmp_path, monkeypatch):
    import importlib.util
    import zipfile

    spec = importlib.util.spec_from_file_location("build_installer", ROOT / "scripts/build_installer.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    monkeypatch.setattr(build, "DIST", tmp_path)
    folder = tmp_path / "RAUSNet"
    folder.mkdir()
    (folder / "RAUSNet.exe").write_bytes(b"exe")
    seed = build.add_seed(folder / "RAUSNet.exe")
    assert (seed / "models/registry.json").exists() and (folder / "LEEME.txt").exists()
    archive, digest = build.make_zip(folder, "9.9.9")
    assert archive.name == "RAUSNet-9.9.9-windows-x64.zip"
    names = zipfile.ZipFile(archive).namelist()
    assert "RAUSNet/RAUSNet.exe" in names and "RAUSNet/workspace_seed/models/registry.json" in names
    assert archive.with_name(archive.name + ".sha256").read_text().split()[0] == digest
