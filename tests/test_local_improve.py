"""Interface phase F4: local improvement from the app, version history and roll back."""

import json
import os
import shutil
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("fastapi")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from fastapi.testclient import TestClient
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QInputDialog, QMessageBox

from rausnet.data.manifest import write_manifest
from rausnet.improve import PROGRESS_PREFIX, Lock, Workspace, WorkspaceLocked, rollback, versions
from rausnet.registry import ModelEntry, Registry
from rausnet.serve.api import create_app, resolve_model_dir
from rausnet.ui.app import MainWindow
from rausnet.ui.client import ApiClient, ApiError
from rausnet.ui.improve_view import STRINGS, history_line, run_message
from rausnet.ui.local_improve import LocalImprover, availability, release_lock_of

T = STRINGS["es"]


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def no_modal_dialogs(monkeypatch):
    def fail(_parent, title, text, *args, **kwargs):
        raise AssertionError(f"unexpected dialog: {title}: {text}")

    for name in ("critical", "information", "question"):
        monkeypatch.setattr(QMessageBox, name, fail)
    monkeypatch.setattr(QInputDialog, "getText", fail)


def _add_version(ws, onnx_dir, version, promote=True, qwk=0.95, onnx=True):
    folder = ws.models / version
    shutil.copytree(onnx_dir, folder)
    side = json.loads((folder / "model.json").read_text())
    (folder / "model.json").write_text(json.dumps({**side, "version": version}))
    if not onnx:
        (folder / "model.onnx").unlink()
    Registry(ws.models).add(ModelEntry(version, f"2026-11-1{version[-1]}T10:00:00", f"{version}/model.pt",
                                       "datasets/v0001/manifest.csv", "x", "S2", None,
                                       test_metrics={"qwk": qwk}), promote=promote)


@pytest.fixture()
def workspace(tmp_path, onnx_model):
    """v0001 archived, v0002 rejected (no ONNX), v0003 current."""
    ws = Workspace(tmp_path / "ws").ensure()
    (ws.data / "datasets/v0001").mkdir(parents=True)
    write_manifest([], ws.data / "datasets/v0001/manifest.csv")
    _add_version(ws, onnx_model[0], "v0001", qwk=0.91)
    _add_version(ws, onnx_model[0], "v0002", promote=False, qwk=0.70, onnx=False)
    _add_version(ws, onnx_model[0], "v0003", qwk=0.85)
    return ws


def _client(ws):
    app = create_app(resolve_model_dir(ws.models), ws.data, models_dir=ws.models, workspace=ws.root)
    return ApiClient(http=TestClient(app))


# ---- backend
def test_rollback_switches_registry_logs_and_respects_the_lock(workspace):
    assert [(v["version"], v["status"], v["servable"]) for v in versions(workspace.root)] == [
        ("v0003", "current", True), ("v0002", "rejected", False), ("v0001", "archived", True)]
    with pytest.raises(ValueError, match="rejected"):
        rollback(workspace.root, "v0002")
    with pytest.raises(KeyError):
        rollback(workspace.root, "v0009")
    with Lock(workspace), pytest.raises(WorkspaceLocked):
        rollback(workspace.root, "v0001")  # never while an improvement may publish
    assert Registry(workspace.models).current().version == "v0003"

    entry = rollback(workspace.root, "v0001", "new model misses mild cases")
    assert entry["action"] == "rolled back" and entry["previous"] == "v0003"
    reg = Registry(workspace.models)
    assert reg.current().version == "v0001" and reg.get("v0003").status == "archived"
    assert workspace.history()[-1]["reason"] == "new model misses mild cases"
    assert history_line(entry, T).endswith("se volvió a la versión v0001 (antes v0003)")
    assert not workspace.lock_path.exists()


def test_rollback_endpoint_swaps_the_served_model(workspace):
    client = _client(workspace)
    assert client.model_versions()["served_version"] == "v0003"
    for version, code in (("v0009", "404"), ("bad", "422"), ("v0002", "409")):
        with pytest.raises(ApiError, match=code):
            client.rollback(version)
    res = client.rollback("v0001", "test")
    assert res["served_version"] == "v0001" and client.health()["model_version"] == "v0001"
    assert not client.workspace_status()["new_model"]  # served and global agree

    (workspace.models / "v0003/model.onnx").write_bytes(b"broken")
    with pytest.raises(ApiError, match="409"):
        client.rollback("v0003")  # loaded first: a broken version changes nothing
    assert Registry(workspace.models).current().version == "v0001"
    assert client.health()["model_version"] == "v0001"


def test_rollback_endpoints_need_a_workspace(onnx_model):
    client = ApiClient(http=TestClient(create_app(onnx_model[0])))
    for call in (client.model_versions, lambda: client.rollback("v0001")):
        with pytest.raises(ApiError, match="503"):
            call()


def test_availability_explains_why_local_training_is_off(monkeypatch):
    assert availability({"torch": True, "cuda": True, "name": "RTX"}) == (False, "local_bundle")  # RAUSNET_GPU=off
    monkeypatch.setenv("RAUSNET_GPU", "")
    assert availability(None) == (False, "local_checking")
    assert availability({"torch": False, "cuda": False}) == (False, "local_no_torch")
    assert availability({"torch": True, "cuda": False}) == (False, "local_no_gpu")
    assert availability({"torch": True, "cuda": True, "name": "RTX"}) == (True, "local_gpu")
    monkeypatch.setenv("RAUSNET_GPU", "cpu")
    assert availability({"torch": True, "cuda": False}) == (True, "local_cpu")


def test_lock_release_only_for_its_own_run(tmp_path, monkeypatch):
    ws = Workspace(tmp_path).ensure()
    monkeypatch.setenv("RAUSNET_RUN_ID", "abc")
    Lock(ws).__enter__()
    assert not release_lock_of(ws.root, "other") and ws.lock_path.exists()
    assert not release_lock_of(ws.root, "") and ws.lock_path.exists()
    assert release_lock_of(ws.root, "abc") and not ws.lock_path.exists()


# ---- UI: versions and roll back
def test_versions_table_and_lead_only_rollback(qapp, workspace, monkeypatch):
    w = MainWindow(_client(workspace), "es")
    v = w.improve_view
    rows = [[v.versions.item(i, j).text() for j in (0, 2, 3)] for i in range(v.versions.rowCount())]
    assert rows == [["v0003", "en uso (este equipo)", "alta"], ["v0002", "no aprobada", "moderada"],
                    ["v0001", "anterior", "muy alta"]]
    assert v.rollback_btn.isHidden() and v.rollback("v0001", "x") is None  # not a lead: nothing happens

    w.admin_box.setChecked(True)  # Ajustes: lead options, remembered
    assert not v.rollback_btn.isHidden() and w.prefs.value("admin") == "true"
    v.select_version("v0002")
    assert not v.rollback_btn.isEnabled()  # rejected versions cannot be served
    v.select_version("v0001")
    assert v.rollback_btn.isEnabled()

    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("", False))
    assert v.rollback() is None and Registry(workspace.models).current().version == "v0003"  # cancelled
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("missed mild cases", True))
    res = v.rollback()
    assert res["served_version"] == "v0001"
    assert "v0001" in w.model_label.text() and not w.export_btn.isEnabled()
    assert v.run_text.text() == "Ahora se usa la versión v0001 (antes v0003)."
    assert "se volvió a la versión v0001" in v.history.text()
    assert v.versions.item(0, 2).text() == "anterior" and v.versions.item(2, 2).text() == "en uso (este equipo)"
    assert w.new_model_bar.isHidden()


# ---- UI: local improvement with a fake training process
FAKE = """
import json, os, sys, time
ws, staging, mode = sys.argv[1], sys.argv[2], sys.argv[3]
P = {prefix!r}
def say(stage, **kw): print(P + json.dumps({{"stage": stage, **kw}}), flush=True)
say("checking"); say("staging")
if mode in ("hang", "crash"):
    with open(os.path.join(ws, ".improve.lock"), "w") as fh:
        json.dump({{"time": time.time(), "run_id": os.environ.get("RAUSNET_RUN_ID")}}, fh)
if mode == "crash":
    print("RuntimeError: CUDA out of memory", flush=True); sys.exit(1)
say("training", epochs=2)
if mode == "hang":
    time.sleep(60)
print("epoch=1 train_loss=0.5", flush=True); print("epoch=2 train_loss=0.4", flush=True)
say("publishing", version="v0004")
summary = {{"action": "retrained", "version": "v0004", "promoted": True}}
say("done", summary=summary); print(json.dumps(summary, indent=2))
"""


def _fake_runner(tmp_path, mode):
    script = tmp_path / "fake_improve.py"
    script.write_text(FAKE.format(prefix=PROGRESS_PREFIX))
    runner = LocalImprover(command=lambda ws, staging: (sys.executable, [str(script), ws, staging, mode]))
    runner.gpu = {"torch": True, "cuda": True, "name": "Test GPU"}
    return runner


def _wait(runner, seconds=30):
    end = time.time() + seconds
    while runner.running and time.time() < end:
        QTest.qWait(50)
    assert not runner.running


@pytest.fixture()
def local_window(qapp, workspace, monkeypatch):
    monkeypatch.setenv("RAUSNET_GPU", "")

    def make(tmp_path, mode):
        runner = _fake_runner(tmp_path, mode)
        w = MainWindow(_client(workspace), "es")
        v = w.improve_view
        v.runner = runner
        runner.progress.connect(v._on_progress)
        runner.finished.connect(v._on_finished)
        v.refresh()
        return w, v, runner

    return make


def test_local_run_shows_progress_and_result(local_window, tmp_path, monkeypatch):
    _w, v, runner = local_window(tmp_path, "ok")
    assert v.local_btn.isEnabled() and "Test GPU" in v.local_btn.toolTip()
    seen = []
    runner.progress.connect(lambda pct, stage, info: seen.append((pct, stage, info.get("epoch"))))
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)  # below threshold
    assert v.start_local()
    staging = Path(runner._staging)
    assert not v.cancel_btn.isHidden() and not v.local_btn.isEnabled()
    assert v.running.text() == "Mejora en curso en este equipo"
    _wait(runner)
    pcts = [p for p, _, _ in seen]
    assert pcts == sorted(pcts) and pcts[-1] == 100
    assert ("training" in [s for _, s, _ in seen]) and (52, "training", 1) in seen
    assert v.run_text.text() == "Mejora terminada: nuevo modelo global v0004."
    assert v.cancel_btn.isHidden() and v.run_bar.isHidden() and v.local_btn.isEnabled()
    assert not staging.exists()


def test_cancel_kills_the_run_and_frees_the_workspace(local_window, tmp_path, workspace, monkeypatch):
    _w, v, runner = local_window(tmp_path, "hang")
    assert v.start_local(confirm=False)
    end = time.time() + 30
    while not workspace.lock_path.exists() and time.time() < end:
        QTest.qWait(50)
    staging = runner._staging
    assert workspace.lock_path.exists() and Path(staging).is_dir()
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    v.cancel_local()
    assert runner.running  # the user changed their mind
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    v.cancel_local()
    _wait(runner)
    assert v.run_text.text() == T["run_cancelled"]
    assert not workspace.lock_path.exists() and not Path(staging).exists()
    assert Registry(workspace.models).current().version == "v0003"


def test_failed_run_explains_and_releases_its_lock(local_window, tmp_path, workspace):
    _w, v, runner = local_window(tmp_path, "crash")
    assert v.start_local(confirm=False)
    _wait(runner)
    assert "CUDA out of memory" in v.run_text.text() and v.run_text.text().startswith("La mejora falló")
    assert not workspace.lock_path.exists()


def test_local_run_refused_while_another_runs(local_window, tmp_path, workspace):
    _w, v, runner = local_window(tmp_path, "ok")
    workspace.lock_path.write_text(json.dumps({"time": time.time(), "started": "10:00", "runtime": "colab"}))
    v.refresh()
    assert not v.start_local(confirm=False) and not runner.running
    assert v.local_note.text() == "Ya hay una mejora en curso en Colab. Espera a que termine."


def test_run_messages():
    assert run_message({"action": "retrained", "promoted": False}, "", T) == T["run_kept"]
    nothing = {"action": "nothing to do", "new_images": 3, "min_new": 20}
    assert run_message(nothing, "", T) == "Mejora terminada sin cambios: 3 imágenes nuevas, se necesitan 20."
    assert run_message(None, "", T).endswith("Detalle: ?")
