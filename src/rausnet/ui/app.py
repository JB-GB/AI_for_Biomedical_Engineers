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
import os
import socket
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QByteArray, QSettings, Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from rausnet.ui.client import ApiClient, ApiError
from rausnet.ui.improve_view import STRINGS as IMPROVE_STRINGS
from rausnet.ui.improve_view import ImproveView
from rausnet.ui.report import report_html, write_pdf

STRINGS = {
    "es": {
        "title": "RAUSNet — Graduación de sinovitis por ultrasonido",
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
        "heatmap": "Mostrar mapa de calor (Grad-CAM): zonas que más influyeron en el grado",
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
        "heatmap": "Show heat map (Grad-CAM): regions that most influenced the grade",
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

PREVIEW = 260
POLL_MS = 120_000  # workspace status refresh (new model banner, images waiting)


def settings() -> QSettings:
    return QSettings("RAUSNet", "desktop")


def workspace_problem(path: str | Path) -> str | None:
    """None if ``path`` holds a servable global model, else the reason."""
    from rausnet.serve.api import resolve_model_dir

    try:
        resolve_model_dir(Path(path) / "models")
    except (FileNotFoundError, ValueError, OSError) as exc:
        return str(exc)
    return None


class ImagePanel(QGroupBox):
    def __init__(self, title: str, t: dict):
        super().__init__(title)
        self.t, self.path = t, None
        self.original: QPixmap | None = None
        self.heatmap: QPixmap | None = None
        self.view = QLabel(t["no_image"])
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setFixedSize(PREVIEW, PREVIEW)
        self.view.setStyleSheet("background:#111; color:#aaa; border-radius:4px;")
        self.button = QPushButton(t["load"])
        self.button.clicked.connect(self.choose)
        lay = QVBoxLayout(self)
        lay.addWidget(self.view)
        lay.addWidget(self.button)

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
        else:
            self.view.setText(self.t["no_image"])

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
                 connect: Callable[[str], ApiClient] | None = None, workspace: str | None = None):
        super().__init__()
        self.client, self.t, self.connect, self.workspace = client, STRINGS[lang], connect, workspace
        self.setWindowTitle(self.t["title"])
        info = client.model()

        self.banner = QLabel()
        validated = info.get("clinically_validated", False)
        self.banner.setText(self.t["banner_validated" if validated else "banner_unvalidated"])
        self.banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.banner.setStyleSheet(
            "font-weight:bold; padding:8px; border-radius:4px; color:#fff; background:"
            + ("#0b7a0b;" if validated else "#b3261e;")
        )
        self.model_label = QLabel()
        self.model_label.setStyleSheet("color:#555;")
        self._show_model(info)

        # new global model available in the workspace
        self.new_model_text = QLabel("")
        self.new_model_btn = QPushButton(self.t["use_new"])
        self.new_model_btn.setMinimumHeight(36)
        self.new_model_btn.setAccessibleName(self.t["use_new"])
        self.new_model_btn.clicked.connect(lambda: self.adopt_new_model())
        self.new_model_bar = QWidget()
        self.new_model_bar.setObjectName("newModelBar")
        self.new_model_bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.new_model_bar.setStyleSheet("#newModelBar {background:#e8f0fe; border:1px solid #1a56db;"
                                         " border-radius:4px;}")
        bar = QHBoxLayout(self.new_model_bar)
        bar.addWidget(self.new_model_text, 1)
        bar.addWidget(self.new_model_btn)
        self.new_model_bar.hide()

        self.gs_panel = ImagePanel(self.t["gs_image"], self.t)
        self.pd_panel = ImagePanel(self.t["pd_image"], self.t)
        self.analyze_btn = QPushButton(self.t["analyze"])
        self.analyze_btn.setMinimumHeight(40)
        self.analyze_btn.setStyleSheet("font-size:15px; font-weight:bold;")
        self.analyze_btn.clicked.connect(self.analyze)
        self.heatmap_box = QCheckBox(self.t["heatmap"])
        self.heatmap_box.toggled.connect(self.toggle_heatmap)
        self.result: dict | None = None
        self.export_btn = QPushButton(self.t["export_pdf"])
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(lambda: self.export_pdf())

        images = QHBoxLayout()
        images.addWidget(self.gs_panel)
        images.addWidget(self.pd_panel)

        # Result panel
        self.combined = QLabel("-")
        self.combined.setStyleSheet("font-size:40px; font-weight:bold;")
        self.confidence = QLabel("-")
        self.gs_grade = QLabel("-")
        self.pd_grade = QLabel("-")
        self.bars = []
        result = QGroupBox(self.t["result"])
        form = QFormLayout(result)
        form.addRow(self.t["combined"], self.combined)
        form.addRow(self.t["confidence"], self.confidence)
        form.addRow(self.t["gs_grade"], self.gs_grade)
        form.addRow(self.t["pd_grade"], self.pd_grade)
        grid = QGridLayout()
        for g in range(4):
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setFormat("%p%")
            self.bars.append(bar)
            grid.addWidget(QLabel(f"{g}"), g, 0)
            grid.addWidget(bar, g, 1)
        form.addRow(QLabel(self.t["probabilities"]))
        form.addRow(grid)
        self.warnings = QLabel("")
        self.warnings.setWordWrap(True)
        self.warnings.setStyleSheet("color:#8a5300;")
        form.addRow(self.warnings)

        # Correction panel
        corr = QGroupBox(self.t["correction"])
        cform = QFormLayout(corr)
        self.patient = QLineEdit()
        self.patient.setPlaceholderText("HOSP-0001")
        self.joint = QComboBox()
        self.joint.addItems(info.get("joints", ["MCP2", "MCP3", "WRIST"]))
        self.side = QComboBox()
        self.side.addItems(["L", "R"])
        self.gs_fix = QComboBox()
        self.pd_fix = QComboBox()
        for box in (self.gs_fix, self.pd_fix):
            box.addItems(["0", "1", "2", "3"])
        self.submit_btn = QPushButton(self.t["submit"])
        self.submit_btn.clicked.connect(self.submit)
        self.status = QLabel("")
        for label, widget in ((self.t["patient"], self.patient), (self.t["joint"], self.joint),
                              (self.t["side"], self.side), (self.t["gs_grade"], self.gs_fix),
                              (self.t["pd_grade"], self.pd_fix)):
            cform.addRow(label, widget)
        cform.addRow(self.submit_btn)
        cform.addRow(self.status)

        right = QVBoxLayout()
        right.addWidget(result)
        right.addWidget(corr)
        left = QVBoxLayout()
        left.addLayout(images)
        left.addWidget(self.analyze_btn)
        left.addWidget(self.heatmap_box)
        left.addWidget(self.export_btn)
        left.addStretch()
        body = QHBoxLayout()
        body.addLayout(left)
        body.addLayout(right)
        analyze_tab = QWidget()
        analyze_tab.setLayout(body)

        self.improve_view = ImproveView(client, self.t)
        self.improve_view.status_changed.connect(self._on_status)
        self.tabs = QTabWidget()
        self.tabs.addTab(analyze_tab, self.t["tab_analyze"])
        self.tabs.addTab(self.improve_view, self.t["tab_improve"])
        self.tabs.addTab(self._settings_tab(), self.t["tab_settings"])
        self.tabs.currentChanged.connect(lambda i: self.improve_view.refresh() if i == 1 else None)

        root = QVBoxLayout()
        root.addWidget(self.banner)
        root.addWidget(self.model_label)
        root.addWidget(self.new_model_bar)
        root.addWidget(self.tabs)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

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
        self.ws_btn.setMinimumHeight(40)
        self.ws_btn.setAccessibleName(self.t["choose_folder"])
        self.ws_btn.setAccessibleDescription(self.t["ws_folder"])
        self.ws_btn.clicked.connect(lambda: self.choose_workspace())
        self.ws_msg = QLabel("" if self.connect else self.t["ws_remote"])
        self.ws_msg.setWordWrap(True)
        self.ws_btn.setEnabled(self.connect is not None)
        form.addRow(self.t["ws_folder"], self.ws_label)
        form.addRow(self.ws_btn)
        form.addRow(self.ws_msg)
        return tab

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
        self._show_model(self.client.model())
        self.result = None  # a report must describe the model that produced it
        self.export_btn.setEnabled(False)
        self.status.setText(self.t["adopted"].format(version=res["version"]))
        self.improve_view.refresh()
        return res

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
        try:
            res = self.client.predict(self.gs_panel.path, self.pd_panel.path, explain=True)
        except (ApiError, OSError) as exc:
            self._error(exc)
            return None
        # the report must match the images it describes, so keep the paths with the result
        self.result = {"res": res, "gs": self.gs_panel.path, "pd": self.pd_panel.path}
        self.export_btn.setEnabled(True)
        c = res["combined"]
        self.combined.setText(str(c["grade"]))
        self.confidence.setText(f"{c['confidence']:.0%}")
        self.gs_grade.setText(str(res["gs"]["grade"]))
        self.pd_grade.setText(str(res["pd"]["grade"]))
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
        return out

    def toggle_heatmap(self, on: bool) -> None:
        self.gs_panel.show_heatmap(on)
        self.pd_panel.show_heatmap(on)

    def submit(self) -> dict | None:
        if not self.gs_panel.path:
            QMessageBox.information(self, self.t["title"], self.t["need_gs"])
            return None
        try:
            res = self.client.correct(
                self.gs_panel.path, self.pd_panel.path, patient_id=self.patient.text().strip(),
                joint=self.joint.currentText(), side=self.side.currentText(),
                gs_grade=int(self.gs_fix.currentText()), pd_grade=int(self.pd_fix.currentText()),
            )
        except (ApiError, OSError) as exc:
            self._error(exc)
            return None
        self.status.setText(self.t["sent"].format(image_id=res["image_id"]))
        return res


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
    args = parser.parse_args(argv)
    app = QApplication(sys.argv[:1])
    if args.server:
        window = MainWindow(ApiClient(args.server), args.lang)
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
                            workspace=workspace)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
