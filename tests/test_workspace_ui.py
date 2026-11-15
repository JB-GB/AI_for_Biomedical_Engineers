"""Interface phase F1: workspace status, model reload, improvement panel and new-model banner."""

import json
import os
import shutil

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("fastapi")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from fastapi.testclient import TestClient
from PySide6.QtWidgets import QApplication, QMessageBox

from rausnet.data.manifest import read_manifest, write_manifest
from rausnet.data.synthetic import generate_dataset
from rausnet.improve import Workspace, colab_url
from rausnet.registry import ModelEntry, Registry
from rausnet.serve.api import create_app, resolve_model_dir
from rausnet.ui.app import MainWindow, settings, workspace_problem
from rausnet.ui.client import ApiClient, ApiError
from rausnet.ui.improve_view import history_line, quality_words


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def no_modal_dialogs(monkeypatch):
    def fail(_parent, title, text, *args, **kwargs):
        raise AssertionError(f"unexpected dialog: {title}: {text}")

    for name in ("critical", "information", "question"):
        monkeypatch.setattr(QMessageBox, name, fail)


def _add_version(ws, onnx_dir, version, promote=True, qwk=0.95):
    folder = ws.models / version
    shutil.copytree(onnx_dir, folder)
    side = json.loads((folder / "model.json").read_text())
    (folder / "model.json").write_text(json.dumps({**side, "version": version}))
    Registry(ws.models).add(ModelEntry(version, "2026-11-14T10:00:00", f"{version}/model.pt",
                                       "datasets/v0001/manifest.csv", "x", "S2", None,
                                       test_metrics={"qwk": qwk}), promote=promote)


@pytest.fixture()
def workspace(tmp_path, onnx_model):
    ws = Workspace(tmp_path / "ws").ensure()
    (ws.data / "datasets/v0001").mkdir(parents=True)
    write_manifest([], ws.data / "datasets/v0001/manifest.csv")
    _add_version(ws, onnx_model[0], "v0001")
    return ws


def _client(ws, min_new=20):
    app = create_app(resolve_model_dir(ws.models), ws.data, models_dir=ws.models, workspace=ws.root,
                     min_new_images=min_new)
    return ApiClient(http=TestClient(app))


def test_status_and_reload_endpoints(workspace, onnx_model):
    client = _client(workspace)
    st = client.workspace_status()
    assert st["served_version"] == "v0001" and not st["new_model"]
    assert st["pending_images"] == 0 and st["min_new_images"] == 20 and not st["ready"]
    assert client.reload_model() == {"changed": False, "version": "v0001"}

    _add_version(workspace, onnx_model[0], "v0002")  # a Colab run published a new global model
    assert client.workspace_status()["new_model"]
    assert client.health()["model_version"] == "v0001"  # nothing switches by itself
    assert client.reload_model() == {"changed": True, "version": "v0002"}
    assert client.model()["version"] == "v0002" and not client.workspace_status()["new_model"]


def test_reload_keeps_serving_when_new_model_is_broken(workspace, onnx_model):
    client = _client(workspace)
    _add_version(workspace, onnx_model[0], "v0002")
    (workspace.models / "v0002/model.onnx").write_bytes(b"not a model")
    with pytest.raises(ApiError, match="409"):
        client.reload_model()
    assert client.health()["model_version"] == "v0001"


def test_endpoints_disabled_without_registry_or_workspace(onnx_model):
    client = ApiClient(http=TestClient(create_app(onnx_model[0])))
    for call in (client.workspace_status, client.reload_model):
        with pytest.raises(ApiError, match="503"):
            call()


def test_improve_panel_counts_and_opens_colab(qapp, workspace, monkeypatch):
    generate_dataset(workspace.data / "incoming/b1", n_patients=1, seed=3, size=64)
    n = len(read_manifest(workspace.data / "incoming/b1/manifest.csv"))
    workspace.log({"time": "2026-11-14 09:00:00", "runtime": "colab", "action": "retrained",
                   "version": "v0001", "promoted": True})
    w = MainWindow(_client(workspace, min_new=n + 1), "es")
    v = w.improve_view
    assert v.progress_text.text() == f"{n} de {n + 1} imágenes nuevas para la próxima mejora"
    assert v.progress.value() == n and v.progress.maximum() == n + 1
    assert v.progress.accessibleName() == v.progress_text.text()
    assert v.quality.text() == "muy alta" and "0.950" in v.quality.toolTip()
    assert "Colab: nuevo modelo global v0001" in v.history.text()
    assert w.new_model_bar.isHidden()

    opened = []
    v.open_url = opened.append
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    assert v.open_colab() is None and not opened  # below the threshold: asks first, user declined
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    assert v.open_colab() == colab_url() and opened == [colab_url()]
    assert colab_url().endswith("/notebooks/colab_improve.ipynb")


def test_new_model_banner_and_adoption_without_restart(qapp, workspace, onnx_model, synth_small):
    w = MainWindow(_client(workspace), "en")
    r = read_manifest(synth_small / "manifest.csv")[0]
    w.gs_panel.set_path(synth_small / r.gs_path)
    assert w.analyze()["model_version"] == "v0001" and w.export_btn.isEnabled()

    _add_version(workspace, onnx_model[0], "v0002")
    w.improve_view.refresh()
    assert not w.new_model_bar.isHidden() and "v0002" in w.new_model_text.text()
    assert w.adopt_new_model()["version"] == "v0002"
    assert "v0002" in w.model_label.text() and w.new_model_bar.isHidden()
    assert not w.export_btn.isEnabled()  # old result no longer matches the served model
    assert w.analyze()["model_version"] == "v0002"


def test_panel_without_workspace_guides_to_settings(qapp, onnx_model):
    w = MainWindow(ApiClient(http=TestClient(create_app(onnx_model[0]))), "es")
    assert "Ajustes" in w.improve_view.notice.text() and not w.improve_view.colab_btn.isEnabled()
    assert not w.ws_btn.isEnabled()  # external server: folder configured there
    assert [w.tabs.tabText(i) for i in range(3)] == ["Analizar", "Mejorar el modelo", "Ajustes"]


def test_choose_workspace_validates_and_switches(qapp, workspace, onnx_model, tmp_path):
    first = ApiClient(http=TestClient(create_app(onnx_model[0])))
    w = MainWindow(first, "en", connect=lambda path: _client(Workspace(path)))
    assert workspace_problem(tmp_path / "empty") is not None
    assert not w.choose_workspace(str(tmp_path / "empty")) and "no ready global model" in w.ws_msg.text()
    assert workspace_problem(workspace.root) is None
    assert w.choose_workspace(str(workspace.root))
    assert settings().value("workspace") == str(workspace.root) and "v0001" in w.model_label.text()
    assert w.improve_view.last["served_version"] == "v0001" and w.improve_view.colab_btn.isEnabled()


def test_plain_words():
    t = {"q_very_high": "vh", "q_high": "h", "q_moderate": "m", "q_low": "l", "q_unknown": "?",
         "h_nothing": "no change ({reason})", "h_promoted": "new {version}", "h_kept": "kept",
         "h_few": "{n} new", "h_needed": ", {m} needed",
         "where_colab": "Colab", "where_runner": "runner", "where_local": "local"}
    assert [quality_words(q, t) for q in (0.95, 0.85, 0.7, 0.3, None)] == ["vh", "h", "m", "l", "?"]
    assert history_line({"time": "T", "runtime": "local-windows", "action": "retrained", "promoted": False},
                        t) == "T · local: kept"
    few = {"time": "T", "runtime": "github-runner", "action": "nothing to do", "new_images": 3, "min_new": 20,
           "reason": "3 new images (< 20)"}
    assert history_line(few, t) == "T · runner: no change (3 new, 20 needed)"
    assert history_line({"time": "T", "runtime": "colab", "action": "nothing to do", "reason": "no data"},
                        t) == "T · Colab: no change (no data)"
