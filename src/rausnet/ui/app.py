"""Desktop application for non-technical users.

    rausnet-desktop                 # starts a private local API with the current registry model
    rausnet-desktop --server URL    # uses an existing API (e.g. a lab server)
    rausnet-desktop --lang en       # English interface (default: Spanish)

Workflow: load the grey-scale image (and the Power Doppler image) -> Analyze ->
read grades, probabilities and warnings -> optionally submit a corrected label,
which is stored for the next retraining round.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
from pathlib import Path

from PySide6.QtCore import QByteArray, Qt
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
    QVBoxLayout,
    QWidget,
)

from rausnet.ui.client import ApiClient, ApiError

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
    },
}

PREVIEW = 260


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
    def __init__(self, client: ApiClient, lang: str = "es"):
        super().__init__()
        self.client, self.t = client, STRINGS[lang]
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
        self.model_label = QLabel(self.t["model"].format(version=info["version"], stage=info["stage"]))
        self.model_label.setStyleSheet("color:#555;")

        self.gs_panel = ImagePanel(self.t["gs_image"], self.t)
        self.pd_panel = ImagePanel(self.t["pd_image"], self.t)
        self.analyze_btn = QPushButton(self.t["analyze"])
        self.analyze_btn.setMinimumHeight(40)
        self.analyze_btn.setStyleSheet("font-size:15px; font-weight:bold;")
        self.analyze_btn.clicked.connect(self.analyze)
        self.heatmap_box = QCheckBox(self.t["heatmap"])
        self.heatmap_box.toggled.connect(self.toggle_heatmap)

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
        left.addStretch()
        body = QHBoxLayout()
        body.addLayout(left)
        body.addLayout(right)
        root = QVBoxLayout()
        root.addWidget(self.banner)
        root.addWidget(self.model_label)
        root.addLayout(body)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

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


def start_local_server(models_dir: str = "models", data_dir: str | None = None) -> str:
    """Start the API in a background thread on a free localhost port; return its URL."""
    import uvicorn

    from rausnet.serve.api import create_app, resolve_model_dir

    port = _free_port()
    app = create_app(resolve_model_dir(models_dir), data_dir)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    url = f"http://127.0.0.1:{port}"
    client = ApiClient(url)
    for _ in range(100):
        try:
            client.health()
            return url
        except Exception:  # server still starting
            time.sleep(0.1)
    raise RuntimeError("local API did not start")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="RAUSNet desktop application.")
    parser.add_argument("--server", default=None, help="API URL; default: start a local one")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--data-dir", default=os.environ.get("RAUSNET_DATA_DIR"))
    parser.add_argument("--lang", default="es", choices=sorted(STRINGS))
    args = parser.parse_args(argv)
    url = args.server or start_local_server(args.models_dir, args.data_dir)
    app = QApplication(sys.argv[:1])
    window = MainWindow(ApiClient(url), args.lang)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
