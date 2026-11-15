"""Interface phase F5: automated accessibility audit of the desktop app."""

import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("fastapi")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from fastapi.testclient import TestClient
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLineEdit, QPushButton, QTabWidget, QVBoxLayout, QWidget

from rausnet.serve.api import create_app
from rausnet.ui import accessibility as a11y
from rausnet.ui import audit
from rausnet.ui.app import MainWindow
from rausnet.ui.client import ApiClient


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


# ---- audit
def test_contrast_rules_of_both_themes():
    assert audit.contrast_ratio("#000000", "#ffffff") == pytest.approx(21.0)
    assert audit.contrast_ratio("#767676", "#ffffff") == pytest.approx(4.54, abs=0.01)
    rules = audit.stylesheet_rules(a11y.stylesheet(False))
    assert rules["#banner"] == {"color": "#ffffff", "background": "#b3261e"}
    assert audit.check_contrast(False) == [] and audit.check_contrast(True) == []


def test_contrast_check_flags_weak_colours(monkeypatch):
    monkeypatch.setattr(a11y, "stylesheet", lambda hc=False: "#x { color:#999999; background:#ffffff; }\n"
                                                              "QPushButton:focus { border:2px solid #eeeeee; }")
    found = {(f.widget, f.detail.split(" =")[0]) for f in audit.check_contrast(False)}
    assert ("#x", "text #999999 on #ffffff") in found
    assert ("QPushButton:focus", "focus frame #eeeeee on #f0f0f0") in found


def test_page_checks_find_unnamed_small_and_unreachable_controls(qapp):
    page = QWidget()
    lay = QVBoxLayout(page)
    tiny = QPushButton("")
    tiny.setFixedHeight(20)
    mid = QPushButton("Aceptar")
    mid.setFixedHeight(30)
    field = QLineEdit()
    field.setAccessibleName("Seudónimo")
    field.setFixedHeight(44)
    field.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
    for w in (tiny, mid, field):
        lay.addWidget(w)
    page.show()
    qapp.processEvents()
    got = {(f.check, f.severity, f.widget.split(" ")[0]) for f in audit.check_page(page, "test")}
    assert got == {("name", "error", "QPushButton"), ("target", "error", "QPushButton"),
                   ("target", "warning", "QPushButton"), ("keyboard", "error", "QLineEdit")}


def test_app_passes_the_automated_audit(qapp, onnx_model, tmp_path):
    w = MainWindow(ApiClient(http=TestClient(create_app(onnx_model[0], tmp_path / "data"))), "es")
    findings = audit.audit_window(w)
    portable = [f for f in findings if f.check in ("name", "keyboard", "target", "contrast")]
    assert portable == [], audit.report(portable)
    if QApplication.platformName() == "windows":  # size checks need the real fonts, not offscreen ones
        assert findings == [], audit.report(findings)
    assert w.text_size.currentIndex() == 0 and not w.contrast_box.isChecked()  # preferences restored
    assert "0 errors" in audit.report([])


def _host(*widths):
    host = QWidget()
    host.tabs = QTabWidget()
    QVBoxLayout(host).addWidget(host.tabs)
    for i, width in enumerate(widths):
        page = QWidget()
        page.setMinimumWidth(width)
        host.tabs.addTab(page, f"T{i}")
    return host


def test_fits_check_flags_wide_content_and_large_windows(qapp):
    found = audit.check_fits(_host(1500, 400), "test")
    assert [f.widget for f in found] == ["main window", "tab 'T0'"]
    assert audit.check_fits(_host(400, 900), "test") == []
