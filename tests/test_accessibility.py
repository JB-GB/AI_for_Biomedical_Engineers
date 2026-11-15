"""Interface phase F2: accessible single-image flow (pairing, words, keyboard, names, themes)."""

import os
import shutil
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("fastapi")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from fastapi.testclient import TestClient
from PySide6.QtCore import QMimeData, QPointF, Qt, QUrl
from PySide6.QtGui import QDropEvent, QShortcut
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QApplication,
    QComboBox,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QTabBar,
)

from rausnet.data.manifest import read_manifest
from rausnet.serve.api import create_app
from rausnet.ui import accessibility as a11y
from rausnet.ui.app import STRINGS, MainWindow, settings
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
def window(qapp, onnx_model, tmp_path):
    w = MainWindow(ApiClient(http=TestClient(create_app(onnx_model[0], data_dir=tmp_path))), "es")
    yield w
    a11y.apply_text_scale(100)  # the application font is shared by every test


@pytest.fixture()
def pair(synth_small):
    r = read_manifest(synth_small / "manifest.csv")[0]
    return str(synth_small / r.gs_path), str(synth_small / r.pd_path)


def test_pair_images_by_name_folder_and_colour(synth_small, tmp_path):
    r = max(read_manifest(synth_small / "manifest.csv"), key=lambda r: r.pd_grade)  # visible Doppler signal
    gs, pd = str(synth_small / r.gs_path), str(synth_small / r.pd_path)
    assert a11y.has_colour(pd) and not a11y.has_colour(gs)
    assert a11y.pair_images([pd, gs]) == {"gs": gs, "pd": pd, "guessed": False}  # gs/ and pd/ folders
    a, b = tmp_path / "a.png", tmp_path / "b.png"  # names say nothing: the coloured one is PD
    shutil.copy(pd, a)
    shutil.copy(gs, b)
    assert a11y.pair_images([a, b]) == {"gs": str(b), "pd": str(a), "guessed": False}
    named = tmp_path / "case1_PD.png", tmp_path / "case1.png"
    shutil.copy(gs, named[0])  # name wins over content
    shutil.copy(gs, named[1])
    assert a11y.pair_images(named) == {"gs": str(named[1]), "pd": str(named[0]), "guessed": False}
    c = tmp_path / "c.png"
    shutil.copy(gs, c)
    assert a11y.pair_images([b, c])["guessed"]  # two grey images: keep order, ask to check
    assert a11y.pair_images([a]) == {"gs": None, "pd": str(a), "guessed": False}
    assert a11y.pair_images([tmp_path / "notes.txt"]) == {"gs": None, "pd": None, "guessed": False}


def test_words_and_spoken_sentence():
    t = STRINGS["es"]
    assert a11y.grade_text(2, t) == "2 · moderada"
    assert [a11y.confidence_words(c, t) for c in (0.95, 0.8, 0.6, 0.3)] == ["muy alta", "alta", "media", "baja"]
    res = {"combined": {"grade": 3, "confidence": 0.42}, "refer": True}
    assert a11y.spoken_result(res, t) == "Grado combinado 3, grave. Confianza baja, 42 %. Derivar a un reumatólogo."
    assert a11y.spoken_result(res, STRINGS["en"]).endswith("Refer to a rheumatologist.")


def test_every_control_has_an_accessible_name(window):
    unnamed = []
    for wd in window.findChildren(QAbstractButton) + window.findChildren(QComboBox) + \
            window.findChildren(QLineEdit) + window.findChildren(QProgressBar):
        if isinstance(wd.parent(), QTabBar | QComboBox | QAbstractItemView):
            continue  # Qt's own tab scroll arrows / combo and table internals carry Qt's names
        text = wd.text() if isinstance(wd, QAbstractButton) else ""
        if not (wd.accessibleName() or text):
            unnamed.append(type(wd).__name__)
    assert not unnamed
    small = [wd.accessibleName() or wd.text() for wd in window.tab_chain
             if wd is not window.result_summary and wd.minimumHeight() < 44]
    assert not small  # targets of at least 44 px


def test_tab_order_follows_the_task(window):
    chain, seen, wd = window.tab_chain, [], window.tab_chain[0]
    for _ in range(200):
        if wd in chain and wd not in seen:
            seen.append(wd)
        wd = wd.nextInFocusChain()
    assert seen == chain


def test_keyboard_only_check_of_one_case(window, pair, monkeypatch):
    shortcuts = {s.key().toString(): s for s in window.findChildren(QShortcut)}
    assert {"Ctrl+Return", "Ctrl+Enter", "Ctrl+O", "Ctrl+P", "Ctrl+K", "F1"} <= set(shortcuts)
    window.load_files(list(reversed(pair)))  # what a drop or Ctrl+O delivers
    assert (window.gs_panel.path, window.pd_panel.path) == pair
    assert window.status.text().startswith("Imágenes cargadas: GS ")
    window.show()
    shortcuts["Ctrl+Return"].activated.emit()
    assert window.result and window.combined.text()[0] in "0123"
    summary = window.result_summary.text()
    assert summary.startswith("Grado combinado ") and window.result_summary.accessibleDescription() == summary
    assert window.focusWidget() is window.result_summary  # focus moves to the spoken result
    shortcuts["Ctrl+K"].activated.emit()
    assert window.focusWidget() is window.gs_fix.group.checkedButton()  # predicted GS grade
    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda _p, title, text: shown.append(title))
    shortcuts["F1"].activated.emit()
    assert shown == [STRINGS["es"]["help_title"]]
    window.hide()


def test_drop_on_window_and_on_panel(window, pair, tmp_path):
    gs, pd = pair

    def drop(target, files):
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(f) for f in files])
        target.dropEvent(QDropEvent(QPointF(5, 5), Qt.DropAction.CopyAction, mime, Qt.MouseButton.NoButton,
                                    Qt.KeyboardModifier.NoModifier))

    drop(window, [pd, gs])
    assert (Path(window.gs_panel.path), Path(window.pd_panel.path)) == (Path(gs), Path(pd))
    window.gs_panel.set_path(None)
    drop(window.gs_panel, [gs])  # one file on a panel goes to that panel
    assert Path(window.gs_panel.path) == Path(gs)
    (tmp_path / "x.txt").write_text("x")
    drop(window, [str(tmp_path / "x.txt")])
    assert window.status.text() == STRINGS["es"]["not_image"]


def test_text_size_contrast_and_palette_are_applied_and_remembered(window, pair, qapp, onnx_model):
    window.load_files(list(pair))
    warm = window.analyze()["heatmaps"]["gs"]
    base = qapp.font().pointSizeF()
    window.text_size.setCurrentIndex(a11y.TEXT_SCALES.index(200))
    assert qapp.font().pointSizeF() == pytest.approx(2 * base)
    assert window.combined.font().pointSizeF() == pytest.approx(2 * base * 2.8)
    # ordinary controls too: under a style sheet the application font alone does not reach them
    for w in (window.gs_panel.button, window.heatmap_box, window.gs_panel, window.improve_view.colab_btn):
        assert w.font().pointSizeF() >= 2 * base - 0.01, type(w).__name__
    window.contrast_box.setChecked(True)
    assert "#000000" in window.styleSheet()
    window.palette_box.setCurrentIndex(window.palette_box.findData("cividis"))
    assert window.analyze()["heatmaps"]["gs"] != warm
    assert settings().value("palette") == "cividis" and int(settings().value("text_scale")) == 200

    again = MainWindow(ApiClient(http=TestClient(create_app(onnx_model[0]))), "en")
    assert again.contrast_box.isChecked() and again.palette_name == "cividis"
    assert again.text_size.currentText() == "200 %"
    for key in ("palette", "text_scale", "high_contrast"):
        settings().remove(key)
