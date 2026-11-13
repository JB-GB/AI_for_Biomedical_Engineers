import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("fastapi")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from fastapi.testclient import TestClient
from PySide6.QtWidgets import QApplication

from rausnet.data.manifest import read_manifest
from rausnet.data.merge import load_source
from rausnet.serve.api import create_app
from rausnet.ui.app import MainWindow
from rausnet.ui.client import ApiClient, ApiError


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def no_modal_dialogs(monkeypatch):
    """A modal QMessageBox would block the offscreen test run forever: fail instead."""
    from PySide6.QtWidgets import QMessageBox

    def fail(_parent, title, text, *args, **kwargs):
        raise AssertionError(f"unexpected dialog: {title}: {text}")

    monkeypatch.setattr(QMessageBox, "critical", fail)
    monkeypatch.setattr(QMessageBox, "information", fail)


@pytest.fixture()
def window(qapp, onnx_model, tmp_path):
    client = ApiClient(http=TestClient(create_app(onnx_model[0], data_dir=tmp_path)))
    return MainWindow(client, "en"), tmp_path


def test_banner_shows_unvalidated(window):
    w, _ = window
    assert "NOT CLINICALLY VALIDATED" in w.banner.text()


def test_analyze_and_submit_correction(window, synth_small):
    w, data = window
    r = read_manifest(synth_small / "manifest.csv")[0]
    w.gs_panel.set_path(synth_small / r.gs_path)
    w.pd_panel.set_path(synth_small / r.pd_path)
    res = w.analyze()
    assert w.combined.text() == str(res["combined"]["grade"])
    assert sum(b.value() for b in w.bars) in range(97, 104)

    w.patient.setText("HOSP-0042")
    w.gs_fix.setCurrentIndex(2)
    w.pd_fix.setCurrentIndex(1)
    sent = w.submit()
    assert sent and "HOSP" not in sent["image_id"]
    recs = load_source(data, sent["manifest"])  # corrections are directly usable by retrain
    assert recs[0].patient_id == "HOSP-0042" and recs[0].gs_grade == 2 and recs[0].pd_grade == 1


def test_client_raises_api_error(onnx_model):
    client = ApiClient(http=TestClient(create_app(onnx_model[0])))
    with pytest.raises(ApiError, match="503"):
        client.correct(__file__, None, patient_id="H1", joint="MCP2", side="L", gs_grade=1, pd_grade=0)


def test_warnings_are_localized(qapp, onnx_model, synth_small):
    client = ApiClient(http=TestClient(create_app(onnx_model[0])))
    w = MainWindow(client, "es")
    r = read_manifest(synth_small / "manifest.csv")[0]
    w.gs_panel.set_path(synth_small / r.gs_path)
    w.analyze()
    assert "no validado" in w.warnings.text() and "Sin imagen Power Doppler" in w.warnings.text()


def test_heatmap_toggle(window, synth_small):
    w, _ = window
    r = read_manifest(synth_small / "manifest.csv")[0]
    w.gs_panel.set_path(synth_small / r.gs_path)
    w.pd_panel.set_path(synth_small / r.pd_path)
    w.analyze()
    assert w.gs_panel.heatmap is not None and w.pd_panel.heatmap is not None
    original = w.pd_panel.view.pixmap().toImage()
    w.heatmap_box.setChecked(True)
    assert w.pd_panel.view.pixmap().toImage() != original
    w.heatmap_box.setChecked(False)
    assert w.pd_panel.view.pixmap().toImage() == original
