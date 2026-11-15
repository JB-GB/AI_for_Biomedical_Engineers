"""Interface phase F3: correction workflow (3 clicks, undo, today's cases, input checks)."""

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("fastapi")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from rausnet.data.manifest import read_manifest, write_manifest
from rausnet.serve.api import create_app
from rausnet.ui import corrections as corr
from rausnet.ui.app import STRINGS, MainWindow
from rausnet.ui.client import ApiClient


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def no_modal_dialogs(monkeypatch):
    def fail(_parent, title, text, *args, **kwargs):
        raise AssertionError(f"unexpected dialog: {title}: {text}")

    for name in ("critical", "information", "question"):
        monkeypatch.setattr(QMessageBox, name, fail)


@pytest.fixture()
def setup(qapp, onnx_model, tmp_path, synth_small):
    data = tmp_path / "data"
    w = MainWindow(ApiClient(http=TestClient(create_app(onnx_model[0], data_dir=data))), "es")
    r = max(read_manifest(synth_small / "manifest.csv"), key=lambda r: r.pd_grade)
    w.load_files([str(synth_small / r.gs_path), str(synth_small / r.pd_path)])
    w.patient.setText("HOSP-0007")
    return w, data


def _stored(data: Path) -> list:
    return [rec for m in sorted(data.glob("incoming/corrections/*/manifest.csv")) for rec in read_manifest(m)]


def test_correction_in_three_clicks_with_undo(setup):
    w, data = setup
    w.patient.setText("HOSP-0042")
    res = w.analyze()
    ai_gs, ai_pd = res["gs"]["grade"], res["pd"]["grade"]
    new_gs, new_pd = (ai_gs + 1) % 4, (ai_pd + 2) % 4 if (ai_gs + 1) % 4 else 0  # PD > 0 needs GS > 0
    w.resize(1180, 900)
    w.show()  # real clicks need laid-out widgets
    QApplication.processEvents()
    clicks = [w.gs_fix.group.button(new_gs), w.pd_fix.group.button(new_pd), w.submit_btn]
    for b in clicks:  # real mouse clicks: 3 in total
        QTest.mouseClick(b, Qt.MouseButton.LeftButton)
    recs = _stored(data)
    assert [(r.patient_id, r.gs_grade, r.pd_grade) for r in recs] == [("HOSP-0042", new_gs, new_pd)]
    assert w.status.text().startswith("Corrección guardada") and w.undo_btn.isEnabled()
    assert not w.submit_btn.isEnabled()  # no accidental duplicate
    assert w.cases.corrected() == 1 and "corregido" in w.today_list.item(0).text()

    QTest.mouseClick(w.undo_btn, Qt.MouseButton.LeftButton)
    assert _stored(data) == [] and w.status.text() == STRINGS["es"]["undone"]
    assert w.cases.corrected() == 0 and not w.undo_btn.isEnabled() and w.submit_btn.isEnabled()
    w.hide()


def test_missing_pseudonym_is_explained_not_an_error(setup):
    w, data = setup
    w.patient.setText("")
    assert w.submit() is None and w.status.text() == STRINGS["es"]["need_patient"] and _stored(data) == []
    w.patient.setText("Juan Perez")  # a name is refused before anything is sent
    assert w.submit() is None and w.focusWidget() is w.patient


def test_resend_replaces_and_confirmation_is_recorded(setup):
    w, data = setup
    res = w.analyze()
    w.submit()  # grades unchanged: a confirmation of the AI result
    assert w.status.text() == STRINGS["es"]["sent_confirm"] and "confirmado" in w.today_list.item(0).text()
    w.gs_fix.setCurrentIndex((res["gs"]["grade"] + 1) % 4)
    w.pd_fix.setCurrentIndex(0)
    w.submit_btn.setEnabled(True)
    w.submit()  # changed a grade and sent again: replaces, never duplicates
    recs = _stored(data)
    assert len(recs) == 1 and recs[0].gs_grade == (res["gs"]["grade"] + 1) % 4
    assert len(w.cases.cases) == 1


def test_undo_refused_once_a_model_used_it(setup):
    w, data = setup
    w.analyze()
    w.submit()
    (data / "datasets/v0002").mkdir(parents=True)
    write_manifest(_stored(data), data / "datasets/v0002/manifest.csv")
    errors = []
    QMessageBox.critical = lambda _p, _t, text: errors.append(text)
    assert w.undo() is None and "v0002" in errors[0] and len(_stored(data)) == 1


def test_today_cases_store_nothing_identifying_and_reset_daily(setup, tmp_path):
    w, _ = setup
    w.patient.setText("HOSP-0001")
    w.analyze()
    w.submit()
    raw = Path(os.environ["RAUSNET_CASES_FILE"]).read_text(encoding="utf-8")
    saved = json.loads(raw)
    assert set(saved["cases"][0]) == set(corr.TodayCases.FIELDS)
    assert ".png" not in raw and "synth_small" not in raw and "gs/" not in raw  # no file names or paths
    assert len(corr.TodayCases().cases) == 1  # survives a restart the same day
    assert corr.TodayCases(today="2099-01-01").cases == []  # a new day starts empty


def test_input_checks(qapp, tmp_path, setup):
    t = STRINGS["es"]
    grey = np.full((100, 100), 90, np.uint8)
    colour = np.stack([grey] * 3, -1)
    colour[40:50, 40:50] = (230, 40, 30)
    Image.fromarray(grey).save(tmp_path / "g.png")
    Image.fromarray(colour).save(tmp_path / "c.png")
    Image.fromarray(np.full((80, 120), 90, np.uint8)).save(tmp_path / "other.png")
    Image.fromarray(np.full((20, 20), 90, np.uint8)).save(tmp_path / "tiny.png")
    (tmp_path / "bad.png").write_bytes(b"not an image")

    def codes(gs, pd):
        return [(i["code"], i["block"]) for i in corr.check_inputs(gs and str(tmp_path / gs),
                                                                 pd and str(tmp_path / pd), t)]

    assert codes("g.png", "c.png") == []
    assert codes("c.png", "g.png") == [("gs_colour", False), ("pd_grey", False)]  # swapped slots
    assert codes("g.png", "other.png") == [("size", True), ("pd_grey", False)]
    assert codes("tiny.png", None) == [("small", True)]
    assert codes("bad.png", None) == [("unreadable", True)]

    w, data = setup
    w.pd_panel.set_path(tmp_path / "other.png")  # loading runs the checks
    assert "tamaños distintos" in w.input_note.text()
    assert w.analyze() is None and w.submit() is None and _stored(data) == []
    assert "tamaños distintos" in w.status.text()
