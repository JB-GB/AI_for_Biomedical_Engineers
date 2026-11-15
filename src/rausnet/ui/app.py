"""Desktop application for non-technical users.

    rausnet-desktop                 # starts a private local API with the current registry model
    rausnet-desktop --server URL    # uses an existing API (e.g. a lab server)
    rausnet-desktop --lang en       # English interface (default: Spanish)
    rausnet-desktop --workspace DIR # shared workspace: global model + corrections (remembered)

Workflow: load the grey-scale image (and the Power Doppler image) -> Analyze ->
read grades, probabilities and warnings -> optionally submit a corrected label,
which is stored for the next retraining round. The "Mejorar el modelo" tab shows the
shared workspace (images waiting, runs, history), opens the Colab improvement notebook and
adopts a new global model without restarting.
"""

from __future__ import annotations

import argparse
import itertools
import os
import socket
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QByteArray, QSettings, QStandardPaths, Qt, QTimer
from PySide6.QtGui import QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from rausnet.ui import accessibility as a11y
from rausnet.ui import corrections as corr
from rausnet.ui.client import ApiClient, ApiError
from rausnet.ui.improve_view import STRINGS as IMPROVE_STRINGS
from rausnet.ui.improve_view import ImproveView
from rausnet.ui.report import report_html, write_pdf
from rausnet.ui.study import StudyLog

STRINGS = {
    "es": {
        "title": "RAUSNet — Graduación de sinovitis por ultrasonido",
        "study_on": "estudio de usabilidad {code}: se registran las acciones",
        "banner_unvalidated": "MODELO NO VALIDADO CLÍNICAMENTE — solo investigación, no usar para diagnóstico",
        "banner_validated": "Modelo validado para investigación clínica",
        "model": "Modelo {version} · etapa {stage}",
        "gs_image": "Escala de grises (GS)",
        "pd_image": "Power Doppler (PD, opcional)",
        "load": "Cargar imagen…",
        "no_image": "Sin imagen",
        "analyze": "Analizar",
        "result": "Resultado (EULAR-OMERACT)",
        "combined": "Grado combinado",
        "confidence": "Confianza",
        "gs_grade": "Grado GS",
        "pd_grade": "Grado PD",
        "probabilities": "Probabilidad por grado combinado",
        "correction": "Corregir etiqueta (para reentrenamiento)",
        "patient": "ID de paciente (seudónimo)",
        "joint": "Articulación",
        "side": "Lado",
        "submit": "Enviar corrección",
        "sent": "Corrección guardada: {image_id}",
        "need_gs": "Primero carga la imagen en escala de grises.",
        "error": "Error",
        "images": "Imágenes (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        "w_not_validated": "Modelo no validado clínicamente: solo investigación, no para diagnóstico.",
        "w_no_pd": "Sin imagen Power Doppler: los grados PD y combinado no son confiables.",
        "w_refer_specialist": "Confianza baja: derivar a un especialista para la calificación.",
        "heatmap": "Mostrar mapa de calor",
        "heatmap_help": "Zonas que más influyeron en el grado (Grad-CAM).",
        "export_pdf": "Exportar informe PDF",
        "pdf_files": "Documentos PDF (*.pdf)",
        "pdf_saved": "Informe guardado: {path}",
        "r_title": "RAUSNet — Informe de prediagnóstico de sinovitis",
        "r_date": "Fecha y hora",
        "r_model": "Versión del modelo",
        "r_item": "Evaluación",
        "r_grade": "Grado",
        "r_images": "Imágenes de entrada y mapas de calor",
        "r_refer": "Derivar a un reumatólogo: la confianza del modelo es baja para este caso.",
        "r_disclaimer": ("Prediagnóstico de apoyo a la decisión generado por un modelo de IA; no es un "
                         "diagnóstico. Debe ser confirmado por un reumatólogo."),
    },
    "en": {
        "title": "RAUSNet — Ultrasound synovitis grading",
        "study_on": "usability study {code}: actions are logged",
        "banner_unvalidated": "MODEL NOT CLINICALLY VALIDATED — research use only, not for diagnosis",
        "banner_validated": "Model validated for clinical research",
        "model": "Model {version} · stage {stage}",
        "gs_image": "Grey-scale (GS)",
        "pd_image": "Power Doppler (PD, optional)",
        "load": "Load image…",
        "no_image": "No image",
        "analyze": "Analyze",
        "result": "Result (EULAR-OMERACT)",
        "combined": "Combined grade",
        "confidence": "Confidence",
        "gs_grade": "GS grade",
        "pd_grade": "PD grade",
        "probabilities": "Combined-grade probability",
        "correction": "Correct label (for retraining)",
        "patient": "Patient ID (pseudonym)",
        "joint": "Joint",
        "side": "Side",
        "submit": "Submit correction",
        "sent": "Correction saved: {image_id}",
        "need_gs": "Load the grey-scale image first.",
        "error": "Error",
        "images": "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        "w_not_validated": "Model not clinically validated: research use only, not for diagnosis.",
        "w_no_pd": "No Power Doppler image: PD and combined grades are unreliable.",
        "w_refer_specialist": "Low confidence: refer to a specialist for grading.",
        "heatmap": "Show heat map",
        "heatmap_help": "Regions that most influenced the grade (Grad-CAM).",
        "export_pdf": "Export PDF report",
        "pdf_files": "PDF documents (*.pdf)",
        "pdf_saved": "Report saved: {path}",
        "r_title": "RAUSNet — Synovitis pre-diagnosis report",
        "r_date": "Date and time",
        "r_model": "Model version",
        "r_item": "Assessment",
        "r_grade": "Grade",
        "r_images": "Input images and heat maps",
        "r_refer": "Refer to a rheumatologist: model confidence is low for this case.",
        "r_disclaimer": ("Decision-support pre-diagnosis generated by an AI model; this is not a "
                         "diagnosis. It must be confirmed by a rheumatologist."),
    },
}

SETTINGS_STRINGS = {
    "es": {
        "ws_folder": "Carpeta compartida (modelo global y correcciones)",
        "ws_none": "ninguna: se usa el modelo de este equipo",
        "choose_folder": "Elegir carpeta…",
        "ws_invalid": ("Esta carpeta no tiene un modelo global listo (models/registry.json con model.onnx). "
                       "Crea el primero con «Mejorar en Colab»."),
        "ws_remote": "La aplicación usa un servidor externo: la carpeta compartida se configura en el servidor.",
        "ws_applied": "Carpeta compartida en uso: {path}",
    },
    "en": {
        "ws_folder": "Shared folder (global model and corrections)",
        "ws_none": "none: this computer's model is used",
        "choose_folder": "Choose folder…",
        "ws_invalid": ("This folder has no ready global model (models/registry.json with model.onnx). "
                       "Create the first one with 'Improve on Colab'."),
        "ws_remote": "The app uses an external server: the shared folder is configured on the server.",
        "ws_applied": "Shared folder in use: {path}",
    },
}
for _lang, _table in STRINGS.items():
    _table.update(IMPROVE_STRINGS[_lang])
    _table.update(SETTINGS_STRINGS[_lang])
    _table.update(a11y.STRINGS[_lang])
    _table.update(corr.STRINGS[_lang])
    _table["no_image"] = _table["drop_hint"]

PREVIEW = 260
POLL_MS = 120_000  # workspace status refresh (new model banner, images waiting)


def settings() -> QSettings:
    """Per-user preferences (registry on Windows; tests redirect the default format to a file)."""
    return QSettings(QSettings.defaultFormat(), QSettings.Scope.UserScope, "RAUSNet", "desktop")


def _local_files(event) -> list[str]:
    mime = event.mimeData()
    return [u.toLocalFile() for u in mime.urls() if u.isLocalFile()] if mime.hasUrls() else []


def workspace_problem(path: str | Path) -> str | None:
    """None if ``path`` holds a servable global model, else the reason."""
    from rausnet.serve.api import resolve_model_dir

    try:
        resolve_model_dir(Path(path) / "models")
    except (FileNotFoundError, ValueError, OSError) as exc:
        return str(exc)
    return None


def _scrolling(page: QWidget) -> QScrollArea:
    area = QScrollArea()
    area.setWidget(page)
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    return area


class ImagePanel(QGroupBox):
    """One image slot. A single dropped image goes here; two dropped images go to ``on_files``
    (the window pairs them as GS + PD)."""

    def __init__(self, title: str, t: dict, on_files: Callable[[list[str]], object] | None = None):
        super().__init__(title)
        self.t, self.path, self.on_files = t, None, on_files
        self.on_change: Callable[[], object] | None = None
        self.original: QPixmap | None = None
        self.heatmap: QPixmap | None = None
        self.view = QLabel(t["no_image"])
        self.view.setObjectName("imageView")
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setFixedSize(PREVIEW, PREVIEW)
        self.view.setWordWrap(True)  # the drop hint must fit at 200 % text
        self.view.setAccessibleName(t["preview"].format(title=title))
        self.button = QPushButton(t["load"])
        self.button.setMinimumHeight(44)
        self.button.setAccessibleName(f"{t['load']} {title}")
        self.button.clicked.connect(self.choose)
        self.setAcceptDrops(True)
        lay = QVBoxLayout(self)
        lay.addWidget(self.view)
        lay.addWidget(self.button)

    def dragEnterEvent(self, event) -> None:
        if any(a11y.is_image(f) for f in _local_files(event)):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        files = [f for f in _local_files(event) if a11y.is_image(f)]
        if len(files) == 1:
            self.set_path(files[0])
        elif files and self.on_files:
            self.on_files(files)
        event.acceptProposedAction()

    def choose(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, self.title(), "", self.t["images"])
        if path:
            self.set_path(path)

    @staticmethod
    def _scaled(pix: QPixmap) -> QPixmap:
        return pix.scaled(PREVIEW, PREVIEW, Qt.AspectRatioMode.KeepAspectRatio,
                          Qt.TransformationMode.SmoothTransformation)

    def set_path(self, path: str | Path | None) -> None:
        self.path = str(path) if path else None
        self.heatmap = None
        self.original = self._scaled(QPixmap(self.path)) if self.path else None
        if self.original:
            self.view.setPixmap(self.original)
            self.view.setAccessibleDescription(Path(self.path).name)
        else:
            self.view.setText(self.t["no_image"])
            self.view.setAccessibleDescription("")
        if self.on_change:
            self.on_change()

    def set_heatmap(self, data_url: str | None) -> None:
        """Store a Grad-CAM overlay received from the API as a base64 PNG data URL."""
        self.heatmap = None
        if data_url and self.path:
            pix = QPixmap()
            pix.loadFromData(QByteArray.fromBase64(data_url.split(",", 1)[1].encode("ascii")), "PNG")
            self.heatmap = self._scaled(pix)

    def show_heatmap(self, on: bool) -> None:
        pix = self.heatmap if on and self.heatmap else self.original
        if pix:
            self.view.setPixmap(pix)


class MainWindow(QMainWindow):
    """``connect(workspace)`` returns a client for a (new) local API on that workspace; without it
    (external server) the workspace is configured on the server."""

    def __init__(self, client: ApiClient, lang: str = "es",
                 connect: Callable[[str], ApiClient] | None = None, workspace: str | None = None,
                 study: StudyLog | None = None):
        super().__init__()
        self.client, self.t, self.connect, self.workspace = client, STRINGS[lang], connect, workspace
        self.study = study  # usability study log (``--study``), None otherwise
        self.setWindowTitle(self.t["title"])
        info = client.model()

        self.prefs = settings()
        self.banner = QLabel()
        self.banner.setObjectName("banner")
        validated = info.get("clinically_validated", False)
        self.banner.setProperty("validated", "true" if validated else "false")
        self.banner.setText(self.t["banner_validated" if validated else "banner_unvalidated"])
        self.banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.model_label = QLabel()
        self.model_label.setObjectName("modelLabel")
        self._show_model(info)

        # new global model available in the workspace
        self.new_model_text = QLabel("")
        self.new_model_btn = QPushButton(self.t["use_new"])
        self.new_model_btn.setMinimumHeight(44)
        self.new_model_btn.setAccessibleName(self.t["use_new"])
        self.new_model_btn.clicked.connect(lambda: self.adopt_new_model())
        self.new_model_bar = QWidget()
        self.new_model_bar.setObjectName("newModelBar")
        self.new_model_bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        bar = QHBoxLayout(self.new_model_bar)
        bar.addWidget(self.new_model_text, 1)
        bar.addWidget(self.new_model_btn)
        self.new_model_bar.hide()

        self.gs_panel = ImagePanel(self.t["gs_image"], self.t, self.load_files)
        self.pd_panel = ImagePanel(self.t["pd_image"], self.t, self.load_files)
        self.setAcceptDrops(True)
        self.analyze_btn = QPushButton(f"{self.t['analyze']}  ({self.t['sc_analyze']})")
        self.analyze_btn.setAccessibleName(self.t["analyze"])
        self.analyze_btn.setMinimumHeight(48)
        self.analyze_btn.clicked.connect(self.analyze)
        self.heatmap_box = QCheckBox(self.t["heatmap"])
        self.heatmap_box.setMinimumHeight(44)
        # short label + wrapped explanation: a checkbox text cannot wrap at 200 % text size
        self.heatmap_box.setAccessibleDescription(self.t["heatmap_help"])
        self.heatmap_help = QLabel(self.t["heatmap_help"])
        self.heatmap_help.setWordWrap(True)
        self.heatmap_box.toggled.connect(self.toggle_heatmap)
        self.result: dict | None = None
        self.export_btn = QPushButton(f"{self.t['export_pdf']}  ({self.t['sc_pdf']})")
        self.export_btn.setAccessibleName(self.t["export_pdf"])
        self.export_btn.setMinimumHeight(44)
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(lambda: self.export_pdf())
        self.palette_name = str(self.prefs.value("palette", "warm"))

        images = QHBoxLayout()
        images.addWidget(self.gs_panel)
        images.addWidget(self.pd_panel)

        # Result panel
        self.result_summary = QLabel(self.t["no_result"])
        self.result_summary.setObjectName("resultSummary")
        self.result_summary.setWordWrap(True)
        self.result_summary.setFocusPolicy(Qt.FocusPolicy.StrongFocus)  # Tab reaches it; readers read it
        self.result_summary.setAccessibleName(self.t["result_summary"])
        self.result_summary.setAccessibleDescription(self.t["no_result"])
        self.refer_banner = QLabel("⚠ " + self.t["r_refer"])
        self.refer_banner.setObjectName("referBanner")
        self.refer_banner.setWordWrap(True)
        self.refer_banner.hide()
        self.combined = QLabel("-")
        self.confidence = QLabel("-")
        self.gs_grade = QLabel("-")
        self.pd_grade = QLabel("-")
        self.bars = []
        result = QGroupBox(self.t["result"])
        form = QFormLayout(result)
        form.addRow(self.result_summary)
        form.addRow(self.refer_banner)
        form.addRow(self.t["combined"], self.combined)
        form.addRow(self.t["confidence"], self.confidence)
        form.addRow(self.t["gs_grade"], self.gs_grade)
        form.addRow(self.t["pd_grade"], self.pd_grade)
        grid = QGridLayout()
        for g in range(4):
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setFormat("%p %")
            bar.setAccessibleName(self.t["bar"].format(g=g))
            self.bars.append(bar)
            grid.addWidget(QLabel(f"{g}"), g, 0)
            grid.addWidget(bar, g, 1)
        form.addRow(QLabel(self.t["probabilities"]))
        form.addRow(grid)
        self.warnings = QLabel("")
        self.warnings.setObjectName("warnings")
        self.warnings.setWordWrap(True)
        form.addRow(self.warnings)

        # Correction panel
        corr_box = QGroupBox(self.t["correction"])
        cform = QFormLayout(corr_box)
        self.patient = QLineEdit()
        self.patient.setPlaceholderText("HOSP-0001")
        self.joint = QComboBox()
        self.joint.addItems(info.get("joints", ["MCP2", "MCP3", "WRIST"]))
        self.side = QComboBox()
        self.side.addItems(["L", "R"])
        self.gs_fix = corr.GradeSelector(f"{self.t['correction']}: {self.t['gs_grade']}", self.t)
        self.pd_fix = corr.GradeSelector(f"{self.t['correction']}: {self.t['pd_grade']}", self.t)
        for sel in (self.gs_fix, self.pd_fix):
            sel.changed.connect(lambda _g: self.submit_btn.setEnabled(True))
        self.submit_btn = QPushButton(self.t["submit"])
        self.submit_btn.setMinimumHeight(44)
        self.submit_btn.clicked.connect(lambda: self.submit())
        self.undo_btn = QPushButton(f"{self.t['undo']}  ({self.t['sc_undo']})")
        self.undo_btn.setAccessibleName(self.t["undo"])
        self.undo_btn.setMinimumHeight(44)
        self.undo_btn.setEnabled(False)
        self.undo_btn.clicked.connect(lambda: self.undo())
        self.status = QLabel("")
        self.status.setWordWrap(True)
        for label, widget in ((self.t["patient"], self.patient), (self.t["joint"], self.joint),
                              (self.t["side"], self.side)):
            widget.setAccessibleName(label)
            widget.setMinimumHeight(44)
            cform.addRow(label, widget)
        cform.addRow(self.t["gs_grade"], self.gs_fix)
        cform.addRow(self.t["pd_grade"], self.pd_fix)
        send = QHBoxLayout()
        send.addWidget(self.submit_btn, 2)
        send.addWidget(self.undo_btn, 1)
        cform.addRow(send)
        cform.addRow(self.status)

        # today's cases (this computer only, cleared daily) and checks of the loaded images
        self.cases = corr.TodayCases()
        self.case_index: int | None = None
        self.last_correction: tuple[str, int] | None = None
        self.today_box = QGroupBox(self.t["today"])
        self.today_list = QListWidget()
        self.today_list.setAccessibleName(self.t["today"])
        self.today_list.setMinimumHeight(120)
        QVBoxLayout(self.today_box).addWidget(self.today_list)
        self.input_note = QLabel("")
        self.input_note.setObjectName("warnings")
        self.input_note.setWordWrap(True)
        self.input_issues: list[dict] = []
        self.gs_panel.on_change = self.check_inputs
        self.pd_panel.on_change = self.check_inputs
        self._show_cases()

        right = QVBoxLayout()
        right.addWidget(result)
        right.addWidget(corr_box)
        left = QVBoxLayout()
        left.addLayout(images)
        left.addWidget(self.input_note)
        left.addWidget(self.analyze_btn)
        left.addWidget(self.heatmap_box)
        left.addWidget(self.heatmap_help)
        left.addWidget(self.export_btn)
        left.addWidget(self.today_box, 1)
        body = QHBoxLayout()
        body.addLayout(left)
        body.addLayout(right)
        analyze_tab = QWidget()
        analyze_tab.setLayout(body)

        self.improve_view = ImproveView(client, self.t)
        self.improve_view.status_changed.connect(self._on_status)
        self.improve_view.model_switched.connect(lambda res: self._model_changed(res["served_version"]))
        self.improve_view.goal = self._goal
        self.tabs = QTabWidget()
        # each tab scrolls vertically: the window fits small laptop screens at every text size
        self.tabs.addTab(_scrolling(analyze_tab), self.t["tab_analyze"])
        self.tabs.addTab(_scrolling(self.improve_view), self.t["tab_improve"])
        self.tabs.addTab(_scrolling(self._settings_tab()), self.t["tab_settings"])
        self.tabs.currentChanged.connect(lambda i: self.improve_view.refresh() if i == 1 else None)

        root = QVBoxLayout()
        root.addWidget(self.banner)
        root.addWidget(self.model_label)
        root.addWidget(self.new_model_bar)
        root.addWidget(self.tabs)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

        hint = QLabel(self.t["help_hint"])
        self.statusBar().addPermanentWidget(hint)
        for keys, action in ((("Ctrl+Return", "Ctrl+Enter"), self.analyze), (("Ctrl+O",), self.open_images),
                             (("Ctrl+P",), self.export_pdf), (("Ctrl+K",), self.focus_correction),
                             (("Ctrl+Z",), self.undo),
                             (("F1",), self.show_help)):
            for key in keys:
                QShortcut(QKeySequence(key), self, activated=lambda a=action, k=key: self._shortcut(k, a))
        if self.study is not None:
            QShortcut(QKeySequence("Ctrl+Shift+N"), self, activated=self.study.mark)
            note = self.t["study_on"].format(code=self.study.participant)
            self.setWindowTitle(f"{self.windowTitle()} · {note}")
        chain = [self.gs_panel.button, self.pd_panel.button, self.analyze_btn, self.result_summary,
                 self.heatmap_box, self.export_btn, self.patient, self.joint, self.side,
                 *self.gs_fix.buttons(), *self.pd_fix.buttons(), self.submit_btn, self.undo_btn, self.today_list]
        for a, b in itertools.pairwise(chain):
            QWidget.setTabOrder(a, b)
        self.tab_chain = chain
        self.scaled_fonts = [(self.analyze_btn, 1.25, True), (self.combined, 2.8, True),
                             (self.result_summary, 1.15, False), (self.refer_banner, 1.1, True),
                             *self.improve_view.scaled_fonts]
        self.apply_preferences()

        self.improve_view.refresh()
        self.poll = QTimer(self)
        self.poll.setInterval(POLL_MS)
        self.poll.timeout.connect(self.improve_view.refresh)
        self.poll.start()

    def _settings_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self.ws_label = QLabel(self.workspace or self.t["ws_none"])
        self.ws_label.setWordWrap(True)
        self.ws_btn = QPushButton(self.t["choose_folder"])
        self.ws_btn.setMinimumHeight(44)
        self.ws_btn.setAccessibleName(self.t["choose_folder"])
        self.ws_btn.setAccessibleDescription(self.t["ws_folder"])
        self.ws_btn.clicked.connect(lambda: self.choose_workspace())
        self.ws_msg = QLabel("" if self.connect else self.t["ws_remote"])
        self.ws_msg.setWordWrap(True)
        self.ws_btn.setEnabled(self.connect is not None)
        form.addRow(self.t["ws_folder"], self.ws_label)
        form.addRow(self.ws_btn)
        form.addRow(self.ws_msg)

        self.text_size = QComboBox()
        self.text_size.addItems([f"{p} %" for p in a11y.TEXT_SCALES])
        self.contrast_box = QCheckBox(self.t["high_contrast"])
        self.palette_box = QComboBox()
        self.palette_box.addItem(self.t["pal_warm"], "warm")
        self.palette_box.addItem(self.t["pal_cividis"], "cividis")
        scale = int(self.prefs.value("text_scale", 100))
        self.text_size.setCurrentIndex(a11y.TEXT_SCALES.index(scale) if scale in a11y.TEXT_SCALES else 0)
        self.contrast_box.setChecked(str(self.prefs.value("high_contrast", "false")).lower() == "true")
        self.palette_box.setCurrentIndex(max(0, self.palette_box.findData(self.prefs.value("palette", "warm"))))
        for label, widget in ((self.t["text_size"], self.text_size), (self.t["palette"], self.palette_box)):
            widget.setAccessibleName(label)
            widget.setMinimumHeight(44)
        self.contrast_box.setMinimumHeight(44)
        form.addRow(self.t["text_size"], self.text_size)
        form.addRow(self.contrast_box)
        form.addRow(self.t["palette"], self.palette_box)
        self.admin_box = QCheckBox(self.t["admin"])
        self.admin_box.setMinimumHeight(44)
        self.admin_box.setChecked(str(self.prefs.value("admin", "false")).lower() == "true")
        form.addRow(self.admin_box)
        self.admin_box.toggled.connect(lambda _on: self.save_preferences())
        self.text_size.currentIndexChanged.connect(lambda _i: self.save_preferences())
        self.contrast_box.toggled.connect(lambda _on: self.save_preferences())
        self.palette_box.currentIndexChanged.connect(lambda _i: self.save_preferences())
        return tab

    def save_preferences(self) -> None:
        self.prefs.setValue("text_scale", a11y.TEXT_SCALES[self.text_size.currentIndex()])
        self.prefs.setValue("high_contrast", "true" if self.contrast_box.isChecked() else "false")
        self.prefs.setValue("palette", self.palette_box.currentData())
        self.prefs.setValue("admin", "true" if self.admin_box.isChecked() else "false")
        self.apply_preferences()

    def apply_preferences(self) -> None:
        """Theme, text size and heat-map colours; the latter apply from the next analysis."""
        self.setStyleSheet(a11y.stylesheet(self.contrast_box.isChecked()))
        a11y.apply_text_scale(a11y.TEXT_SCALES[self.text_size.currentIndex()], self.scaled_fonts, root=self)
        self.palette_name = self.palette_box.currentData()
        self.improve_view.set_admin(self.admin_box.isChecked())

    # ---- keyboard and drag-and-drop
    def dragEnterEvent(self, event) -> None:
        if any(a11y.is_image(f) for f in _local_files(event)):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        self.load_files(_local_files(event))
        event.acceptProposedAction()

    def load_files(self, paths: list[str]) -> dict | None:
        """Put one or two image files in the right slots (GS / PD) and say what happened."""
        if not any(a11y.is_image(p) for p in paths):
            self._tell(self.t["not_image"])
            return None
        pair = a11y.pair_images(paths)
        for panel, key in ((self.gs_panel, "gs"), (self.pd_panel, "pd")):
            if pair[key]:
                panel.set_path(pair[key])
        names = {k: Path(v).name if v else self.t["none"] for k, v in
                 (("gs", self.gs_panel.path), ("pd", self.pd_panel.path))}
        msg = self.t["paired"].format(**names)
        self._tell(msg + (" " + self.t["paired_guess"] if pair["guessed"] else ""))
        return pair

    def open_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, self.t["open_images"], "", self.t["images"])
        if paths:
            self.load_files(paths)

    def focus_correction(self) -> None:
        self.tabs.setCurrentIndex(0)
        self.gs_fix.focus()

    def show_help(self) -> None:
        QMessageBox.information(self, self.t["help_title"], self.t["help_text"])

    def _tell(self, text: str) -> None:
        self.status.setText(text)
        a11y.announce(self.status, text)

    def _show_model(self, info: dict) -> None:
        self.model_label.setText(self.t["model"].format(version=info["version"], stage=info["stage"]))

    def _on_status(self, st: dict | None) -> None:
        if st and st.get("new_model"):
            self.new_model_text.setText(self.t["new_model"].format(version=st["current_model"]["version"]))
            self.new_model_bar.show()
        else:
            self.new_model_bar.hide()

    def adopt_new_model(self) -> dict | None:
        """Switch the API to the workspace's current model; runs between analyses (UI thread)."""
        try:
            res = self.client.reload_model()
        except ApiError as exc:
            self._error(exc)
            return None
        self._model_changed(res["version"])
        self.improve_view.refresh()
        self._goal("model_adopted")
        return res

    def _goal(self, name: str) -> None:
        if self.study is not None:
            self.study.goal(name)

    def _shortcut(self, keys: str, action) -> None:
        if self.study is not None:
            self.study.shortcut(keys)
        action()

    def _model_changed(self, version: str) -> None:
        self._show_model(self.client.model())
        self.result = None  # a report must describe the model that produced it
        self.export_btn.setEnabled(False)
        self.status.setText(self.t["adopted"].format(version=version))

    def closeEvent(self, event) -> None:
        # a killed child would keep the workspace lock for hours: cancel cleanly instead
        self.improve_view.runner.cancel()
        super().closeEvent(event)

    def choose_workspace(self, path: str | None = None) -> bool:
        if self.connect is None:
            return False
        if path is None:
            path = QFileDialog.getExistingDirectory(self, self.t["ws_folder"], self.workspace or "")
            if not path:
                return False
        problem = workspace_problem(path)
        if problem:
            self.ws_msg.setText(f"{self.t['ws_invalid']}\n({problem})")
            return False
        try:
            client = self.connect(path)
        except Exception as exc:  # keep the current server on any start failure
            self._error(exc)
            return False
        self.client, self.workspace = client, path
        settings().setValue("workspace", path)
        self.ws_label.setText(path)
        self.ws_msg.setText(self.t["ws_applied"].format(path=path))
        self._show_model(client.model())
        self.result = None
        self.export_btn.setEnabled(False)
        self.improve_view.set_client(client)
        return True

    def _error(self, exc: Exception) -> None:
        QMessageBox.critical(self, self.t["error"], str(exc))

    def analyze(self) -> dict | None:
        if not self.gs_panel.path:
            QMessageBox.information(self, self.t["title"], self.t["need_gs"])
            return None
        if self._blocked():
            return None
        try:
            res = self.client.predict(self.gs_panel.path, self.pd_panel.path, explain=True,
                                      palette=self.palette_name)
        except (ApiError, OSError) as exc:
            self._error(exc)
            return None
        # the report must match the images it describes, so keep the paths with the result
        self.result = {"res": res, "gs": self.gs_panel.path, "pd": self.pd_panel.path}
        self.export_btn.setEnabled(True)
        c = res["combined"]
        self.combined.setText(a11y.grade_text(c["grade"], self.t))
        self.confidence.setText(f"{a11y.confidence_words(c['confidence'], self.t)} ({a11y.percent(c['confidence'])})")
        self.gs_grade.setText(a11y.grade_text(res["gs"]["grade"], self.t))
        self.pd_grade.setText(a11y.grade_text(res["pd"]["grade"], self.t))
        self.refer_banner.setVisible(bool(res.get("refer")))
        for bar, p in zip(self.bars, c["probabilities"], strict=True):
            bar.setValue(round(100 * p))
        codes = res.get("warning_codes") or []
        texts = [self.t.get(f"w_{c}", w) for c, w in zip(codes, res["warnings"], strict=False)] or res["warnings"]
        self.warnings.setText("\n".join(f"⚠ {w}" for w in texts))
        maps = res.get("heatmaps", {})
        self.gs_panel.set_heatmap(maps.get("gs"))
        self.pd_panel.set_heatmap(maps.get("pd") if self.pd_panel.path else None)
        self.toggle_heatmap(self.heatmap_box.isChecked())
        self.gs_fix.setCurrentIndex(res["gs"]["grade"])
        self.pd_fix.setCurrentIndex(res["pd"]["grade"] if self.pd_panel.path else 0)
        ai = {"gs": res["gs"]["grade"], "pd": res["pd"]["grade"] if self.pd_panel.path else 0,
              "combined": c["grade"]}
        self.result["ai"] = ai
        self.case_index = self.cases.add(self.patient.text().strip(), self.joint.currentText(),
                                         self.side.currentText(), ai)
        self.submit_btn.setEnabled(True)
        self._show_cases()
        spoken = a11y.spoken_result(res, self.t)
        self.result_summary.setText(spoken)
        self.result_summary.setAccessibleDescription(spoken)
        self.result_summary.setFocus(Qt.FocusReason.OtherFocusReason)
        a11y.announce(self.result_summary, spoken)
        self._goal("analyzed")
        return res

    def export_pdf(self, path: str | None = None) -> Path | None:
        if not self.result:
            return None
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, self.t["export_pdf"], "", self.t["pdf_files"])
            if not path:
                return None
        fields = {self.t["patient"]: self.patient.text().strip()}
        html_text = report_html(self.result["res"], self.t, self.result["gs"], self.result["pd"], fields)
        try:
            out = write_pdf(html_text, path)
        except OSError as exc:
            self._error(exc)
            return None
        self.status.setText(self.t["pdf_saved"].format(path=out))
        self._goal("pdf_exported")
        return out

    def toggle_heatmap(self, on: bool) -> None:
        self.gs_panel.show_heatmap(on)
        self.pd_panel.show_heatmap(on)

    def submit(self) -> dict | None:
        """Send the selected grades for the loaded images (3 clicks: GS grade, PD grade, send)."""
        if not self.gs_panel.path:
            QMessageBox.information(self, self.t["title"], self.t["need_gs"])
            return None
        if self._blocked():
            return None
        same_case = bool(self.result) and (self.result["gs"], self.result["pd"]) == \
            (self.gs_panel.path, self.pd_panel.path) and self.case_index is not None
        patient = self.patient.text().strip()
        if not corr.PSEUDONYM.match(patient):
            self._tell(self.t["need_patient"])
            self.patient.setFocus()
            return None
        label = {"gs": self.gs_fix.currentIndex(), "pd": self.pd_fix.currentIndex()}
        try:
            res = self.client.correct(
                self.gs_panel.path, self.pd_panel.path, patient_id=patient,
                joint=self.joint.currentText(), side=self.side.currentText(),
                gs_grade=label["gs"], pd_grade=label["pd"],
            )
            previous = self.cases.cases[self.case_index]["image_id"] if same_case else None
            if previous:  # sent again after changing a grade: the new label replaces the old one
                self.client.undo_correction(previous)
        except (ApiError, OSError) as exc:
            self._error(exc)
            return None
        if not same_case:
            self.case_index = self.cases.add(patient, self.joint.currentText(), self.side.currentText(), None)
        self.cases.update(self.case_index, label=label, image_id=res["image_id"], patient=patient,
                          joint=self.joint.currentText(), side=self.side.currentText())
        self.last_correction = (res["image_id"], self.case_index)
        self.undo_btn.setEnabled(True)
        self.submit_btn.setEnabled(False)  # no duplicate: changing a grade enables it again
        ai = self.cases.cases[self.case_index]["ai"]
        confirm = ai is not None and label == {"gs": ai["gs"], "pd": ai["pd"]}
        self._tell(self.t["sent_confirm"] if confirm else self.t["sent_fix"].format(
            gs=a11y.grade_text(label["gs"], self.t), pd=a11y.grade_text(label["pd"], self.t)))
        self._show_cases()
        self._goal("correction_sent")
        return res

    def undo(self) -> dict | None:
        """Withdraw the last correction sent from this window (until a model is trained with it)."""
        if not self.last_correction:
            return None
        image_id, index = self.last_correction
        try:
            res = self.client.undo_correction(image_id)
        except ApiError as exc:
            self._error(exc)
            return None
        self.cases.update(index, label=None, image_id=None)
        self.last_correction = None
        self.undo_btn.setEnabled(False)
        self.submit_btn.setEnabled(True)
        self._tell(self.t["undone"])
        self._show_cases()
        self._goal("undone")
        return res

    def check_inputs(self) -> list[dict]:
        """Size and modality checks of the loaded images, shown under them and read aloud."""
        self.input_issues = corr.check_inputs(self.gs_panel.path, self.pd_panel.path, self.t)
        text = "\n".join(("✖ " if i["block"] else "⚠ ") + i["text"] for i in self.input_issues)
        self.input_note.setText(text)
        if text:
            a11y.announce(self.input_note, text)
        self.submit_btn.setEnabled(True)
        return self.input_issues

    def _blocked(self) -> bool:
        blocking = [i["text"] for i in self.input_issues if i["block"]]
        if blocking:
            self._tell(blocking[0])
        return bool(blocking)

    def _show_cases(self) -> None:
        self.today_list.clear()
        self.today_list.addItems(self.cases.lines(self.t))
        self.today_box.setTitle(self.t["today_title"].format(n=len(self.cases.cases), k=self.cases.corrected()))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


_SERVERS: dict[str, object] = {}


def start_local_server(models_dir: str = "models", data_dir: str | None = None,
                       workspace: str | None = None) -> str:
    """Start the API in a background thread on a free localhost port; return its URL."""
    import uvicorn

    from rausnet.serve.api import create_app, resolve_model_dir

    port = _free_port()
    app = create_app(resolve_model_dir(models_dir), data_dir, models_dir=models_dir, workspace=workspace)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    url = f"http://127.0.0.1:{port}"
    _SERVERS[url] = server
    client = ApiClient(url)
    for _ in range(100):
        try:
            client.health()
            return url
        except Exception:  # server still starting
            time.sleep(0.1)
    raise RuntimeError("local API did not start")


def stop_local_server(url: str) -> None:
    server = _SERVERS.pop(url, None)
    if server is not None:
        server.should_exit = True


def connect_workspace(workspace: str, previous_url: list[str]) -> ApiClient:
    """Start a local API on ``workspace`` and stop the previous one (``previous_url`` is updated)."""
    ws = Path(workspace)
    url = start_local_server(str(ws / "models"), str(ws / "data"), workspace=str(ws))
    if previous_url:
        stop_local_server(previous_url[0])
    previous_url[:] = [url]
    return ApiClient(url)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="RAUSNet desktop application.")
    parser.add_argument("--server", default=None, help="API URL; default: start a local one")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--data-dir", default=os.environ.get("RAUSNET_DATA_DIR"))
    parser.add_argument("--workspace", default=os.environ.get("RAUSNET_WORKSPACE"),
                        help="shared workspace (global model + corrections); overrides the two folders above")
    parser.add_argument("--lang", default="es", choices=sorted(STRINGS))
    parser.add_argument("--study", default=None, metavar="P01",
                        help="usability study: log actions (no file names or typed text) for this participant code")
    parser.add_argument("--study-dir", default=None, help="folder for study logs (default: local app data/study)")
    args = parser.parse_args(argv)
    app = QApplication(sys.argv[:1])
    app.setOrganizationName("RAUSNet")
    app.setApplicationName("desktop")
    study = None
    if args.study:
        folder = args.study_dir or str(Path(QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.AppLocalDataLocation)) / "study")
        study = StudyLog(folder, args.study)
        app.installEventFilter(study)
        print(f"study log: {study.path}", file=sys.stderr)
    if args.server:
        window = MainWindow(ApiClient(args.server), args.lang, study=study)
    else:
        workspace = args.workspace or settings().value("workspace") or None
        if workspace and not args.workspace and workspace_problem(workspace):
            print(f"remembered workspace {workspace} is not usable; using {args.models_dir}", file=sys.stderr)
            workspace = None
        current: list[str] = []
        if workspace:
            client = connect_workspace(workspace, current)
        else:
            current.append(start_local_server(args.models_dir, args.data_dir))
            client = ApiClient(current[0])
        window = MainWindow(client, args.lang, connect=lambda ws: connect_workspace(ws, current),
                            workspace=workspace, study=study)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
