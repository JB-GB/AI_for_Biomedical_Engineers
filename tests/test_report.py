import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtWidgets import QApplication

from rausnet.ui.app import STRINGS
from rausnet.ui.report import report_html, write_pdf

RED_PNG = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIA"
           "X8jx0gAAAABJRU5ErkJggg==")


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def result(refer: bool) -> dict:
    codes = ["not_validated"] + (["refer_specialist"] if refer else [])
    return {
        "model_version": "v0005", "stage": "candidate",
        "gs": {"grade": 2, "probabilities": [0.05, 0.15, 0.7, 0.1]},
        "pd": {"grade": 1, "probabilities": [0.2, 0.6, 0.15, 0.05]},
        "combined": {"grade": 2, "probabilities": [0.05, 0.2, 0.55, 0.2], "confidence": 0.55},
        "warnings": ["x"] * len(codes), "warning_codes": codes,
        "refer": refer, "referral_threshold": 0.6 if refer else None,
        "heatmaps": {"gs": RED_PNG, "pd": RED_PNG},
    }


@pytest.mark.parametrize("lang", sorted(STRINGS))
def test_html_has_grades_and_disclaimer(lang):
    t = STRINGS[lang]
    html = report_html(result(False), t, fields={t["patient"]: "HOSP-0007"})
    assert t["combined"] in html and "<b>2</b>" in html and "55%" in html
    assert t["r_disclaimer"] in html and "v0005" in html and "HOSP-0007" in html
    assert t["r_refer"] not in html and t["w_not_validated"] in html


@pytest.mark.parametrize("lang", sorted(STRINGS))
def test_html_refer_warning(lang):
    t = STRINGS[lang]
    html = report_html(result(True), t)
    assert t["r_refer"] in html and "60%" in html
    assert t["w_refer_specialist"] not in html  # shown once, prominently


def test_html_embeds_images(tmp_path):
    gs = tmp_path / "gs.png"
    Image.new("L", (8, 8), 128).save(gs)
    html = report_html(result(False), STRINGS["en"], gs_path=gs)
    assert html.count("<img") == 2 and "Grad-CAM" in html  # GS + its overlay; no PD image given


def test_every_report_string_is_translated():
    keys = [set(table) for table in STRINGS.values()]
    assert all(k == keys[0] for k in keys)


def test_write_pdf(qapp, tmp_path):
    gs = tmp_path / "gs.png"
    Image.new("L", (64, 64), 90).save(gs)
    out = write_pdf(report_html(result(True), STRINGS["es"], gs_path=gs, pd_path=gs), tmp_path / "r.pdf")
    data = out.read_bytes()
    assert data.startswith(b"%PDF") and len(data) > 1000


def test_app_export_button(qapp, onnx_model, synth_small, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from rausnet.data.manifest import read_manifest
    from rausnet.serve.api import create_app
    from rausnet.ui.app import MainWindow
    from rausnet.ui.client import ApiClient

    w = MainWindow(ApiClient(http=TestClient(create_app(onnx_model[0]))), "en")
    assert not w.export_btn.isEnabled() and w.export_pdf(str(tmp_path / "none.pdf")) is None
    r = read_manifest(synth_small / "manifest.csv")[0]
    w.gs_panel.set_path(synth_small / r.gs_path)
    w.pd_panel.set_path(synth_small / r.pd_path)
    w.analyze()
    assert w.export_btn.isEnabled()
    out = w.export_pdf(str(tmp_path / "report.pdf"))
    assert out.read_bytes().startswith(b"%PDF")
