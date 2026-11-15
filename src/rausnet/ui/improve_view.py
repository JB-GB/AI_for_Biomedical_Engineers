"""'Mejorar el modelo' tab: state of the shared workspace and one-click improvement on Colab.

Everything comes from the API (``GET /workspace/status``), so the panel works the same with the
private local API and with a lab server. Plain words, no ML jargon; every control has an
accessible name for screen readers.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from rausnet.improve import colab_url
from rausnet.ui.client import ApiClient, ApiError

STRINGS = {
    "es": {
        "tab_analyze": "Analizar",
        "tab_improve": "Mejorar el modelo",
        "tab_settings": "Ajustes",
        "global_model": "Modelo global",
        "version": "Versión",
        "quality": "Calidad",
        "q_very_high": "muy alta", "q_high": "alta", "q_moderate": "moderada", "q_low": "baja",
        "q_unknown": "sin medir",
        "quality_tip": "Concordancia con los expertos en el conjunto de prueba (QWK {qwk:.3f})",
        "progress": "{n} de {m} imágenes nuevas para la próxima mejora",
        "progress_ready": "{n} imágenes nuevas: lista para mejorar",
        "progress_first": "Aún no hay modelo global: la primera mejora lo crea",
        "incomplete": "{n} lote(s) aún sincronizándose: se usarán cuando estén completos",
        "running": "Mejora en curso en {where} desde {since}",
        "idle": "Ninguna mejora en curso",
        "colab": "Mejorar en Colab",
        "colab_tip": "Abre el cuaderno en Google Colab: pulsa «Ejecutar todo» y permite el acceso a Drive",
        "colab_early": ("Hay {n} de {m} imágenes nuevas. Con tan pocas, la mejora terminará sin cambios. "
                        "¿Abrir Colab de todos modos?"),
        "refresh": "Actualizar",
        "history": "Últimas mejoras",
        "no_history": "Todavía no se ha ejecutado ninguna mejora.",
        "h_nothing": "sin cambios ({reason})",
        "h_promoted": "nuevo modelo global {version}",
        "h_few": "{n} imágenes nuevas", "h_needed": ", se necesitan {m}",
        "h_kept": "se mantuvo el modelo anterior: el nuevo no fue mejor",
        "no_workspace": ("Sin carpeta compartida. Elige la carpeta del modelo global (por ejemplo, "
                         "RAUSNet en Google Drive) en «Ajustes»."),
        "new_model": "Hay un modelo global nuevo ({version}).",
        "use_new": "Usar el nuevo modelo",
        "adopted": "Ahora se usa el modelo {version}.",
        "where_colab": "Colab", "where_runner": "el servidor del laboratorio", "where_local": "un equipo local",
    },
    "en": {
        "tab_analyze": "Analyze",
        "tab_improve": "Improve the model",
        "tab_settings": "Settings",
        "global_model": "Global model",
        "version": "Version",
        "quality": "Quality",
        "q_very_high": "very high", "q_high": "high", "q_moderate": "moderate", "q_low": "low",
        "q_unknown": "not measured",
        "quality_tip": "Agreement with the experts on the test set (QWK {qwk:.3f})",
        "progress": "{n} of {m} new images for the next improvement",
        "progress_ready": "{n} new images: ready to improve",
        "progress_first": "No global model yet: the first improvement creates it",
        "incomplete": "{n} batch(es) still syncing: used once complete",
        "running": "Improvement running on {where} since {since}",
        "idle": "No improvement running",
        "colab": "Improve on Colab",
        "colab_tip": "Opens the notebook in Google Colab: press 'Run all' and allow Drive access",
        "colab_early": ("There are {n} of {m} new images. With so few the run will end without changes. "
                        "Open Colab anyway?"),
        "refresh": "Refresh",
        "history": "Recent improvements",
        "no_history": "No improvement has run yet.",
        "h_nothing": "no change ({reason})",
        "h_promoted": "new global model {version}",
        "h_few": "{n} new images", "h_needed": ", {m} needed",
        "h_kept": "previous model kept: the new one was not better",
        "no_workspace": ("No shared folder. Choose the global model folder (for example RAUSNet on Google "
                         "Drive) in 'Settings'."),
        "new_model": "A new global model is available ({version}).",
        "use_new": "Use the new model",
        "adopted": "Now using model {version}.",
        "where_colab": "Colab", "where_runner": "the lab server", "where_local": "a local computer",
    },
}


def quality_words(qwk: float | None, t: dict) -> str:
    if qwk is None:
        return t["q_unknown"]
    for limit, key in ((0.9, "q_very_high"), (0.8, "q_high"), (0.6, "q_moderate")):
        if qwk >= limit:
            return t[key]
    return t["q_low"]


def where_words(runtime: str | None, t: dict) -> str:
    if runtime == "colab":
        return t["where_colab"]
    if runtime == "github-runner":
        return t["where_runner"]
    return t["where_local"]


def history_line(entry: dict, t: dict) -> str:
    when = f"{entry.get('time', '?')} · {where_words(entry.get('runtime'), t)}"
    if entry.get("action") == "retrained":
        what = t["h_promoted"].format(version=entry.get("version")) if entry.get("promoted") else t["h_kept"]
    else:
        reason = entry.get("reason", "")
        if "new_images" in entry:  # localized; the logged reason is English
            reason = t["h_few"].format(n=entry["new_images"])
            reason += t["h_needed"].format(m=entry["min_new"]) if "min_new" in entry else ""
        what = t["h_nothing"].format(reason=reason)
    return f"{when}: {what}"


class ImproveView(QWidget):
    """Shows the workspace state; ``status_changed`` carries each status (or ``None`` without workspace)."""

    status_changed = Signal(object)

    def __init__(self, client: ApiClient, t: dict, branch: str = "PIA_devl",
                 open_url: Callable[[str], object] | None = None):
        super().__init__()
        self.client, self.t, self.branch = client, t, branch
        self.open_url = open_url or (lambda url: QDesktopServices.openUrl(QUrl(url)))
        self.last: dict | None = None

        self.notice = QLabel("")
        self.notice.setWordWrap(True)
        self.version = QLabel("-")
        self.quality = QLabel("-")
        card = QGroupBox(t["global_model"])
        form = QFormLayout(card)
        form.addRow(t["version"], self.version)
        form.addRow(t["quality"], self.quality)

        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setMinimumHeight(22)
        self.progress_text = QLabel("")
        self.incomplete = QLabel("")
        self.incomplete.setObjectName("incomplete")
        self.running = QLabel("")

        self.colab_btn = QPushButton(t["colab"])
        self.colab_btn.setMinimumHeight(48)
        self.colab_btn.setToolTip(t["colab_tip"])
        self.colab_btn.clicked.connect(lambda: self.open_colab())
        self.refresh_btn = QPushButton(t["refresh"])
        self.refresh_btn.setMinimumHeight(44)
        self.refresh_btn.clicked.connect(lambda: self.refresh())
        buttons = QHBoxLayout()
        buttons.addWidget(self.colab_btn, 2)
        buttons.addWidget(self.refresh_btn, 1)

        self.history = QLabel("")
        self.history.setWordWrap(True)
        self.scaled_fonts = [(self.colab_btn, 1.2, True), (self.progress_text, 1.2, False)]
        hist = QGroupBox(t["history"])
        QVBoxLayout(hist).addWidget(self.history)

        for widget, name, desc in (
            (self.progress, t["tab_improve"], ""),  # replaced by the progress sentence on refresh
            (self.colab_btn, t["colab"], t["colab_tip"]),
            (self.refresh_btn, t["refresh"], ""),
        ):
            widget.setAccessibleName(name)
            widget.setAccessibleDescription(desc)

        lay = QVBoxLayout(self)
        lay.addWidget(self.notice)
        lay.addWidget(card)
        lay.addWidget(self.progress_text)
        lay.addWidget(self.progress)
        lay.addWidget(self.incomplete)
        lay.addWidget(self.running)
        lay.addLayout(buttons)
        lay.addWidget(hist)
        lay.addStretch()

    def set_client(self, client: ApiClient) -> None:
        self.client = client
        self.refresh()

    def refresh(self) -> dict | None:
        t = self.t
        try:
            st = self.client.workspace_status()
        except ApiError as exc:
            self.last = None
            no_ws = str(exc).startswith("503")
            self.notice.setText(t["no_workspace"] if no_ws else f"{t.get('error', 'Error')}: {exc}")
            for w in (self.progress_text, self.incomplete, self.running, self.history):
                w.setText("")
            self.version.setText("-")
            self.quality.setText("-")
            self.progress.setValue(0)
            self.colab_btn.setEnabled(False)
            self.status_changed.emit(None)
            return None
        self.last = st
        self.notice.setText("")
        self.colab_btn.setEnabled(True)
        cur = st.get("current_model")
        n, m = st["pending_images"], max(1, st["min_new_images"])
        self.progress.setRange(0, m)
        self.progress.setValue(min(n, m))
        if cur is None:
            self.version.setText("-")
            self.quality.setText(t["q_unknown"])
            text = t["progress_first"]
        else:
            self.version.setText(f"{cur['version']} ({cur.get('created', '')[:10]})")
            qwk = (cur.get("test_metrics") or {}).get("qwk")
            self.quality.setText(quality_words(qwk, t))
            self.quality.setToolTip(t["quality_tip"].format(qwk=qwk) if qwk is not None else "")
            text = t["progress_ready" if st["ready"] else "progress"].format(n=n, m=m)
        self.progress_text.setText(text)
        self.progress.setAccessibleName(text)
        k = len(st.get("incomplete_batches") or [])
        self.incomplete.setText(t["incomplete"].format(n=k) if k else "")
        run = st.get("running")
        self.running.setText(t["running"].format(where=where_words(run.get("runtime"), t), since=run.get("started"))
                             if run else t["idle"])
        hist = st.get("history") or []
        self.history.setText("\n".join(history_line(e, t) for e in reversed(hist[-5:])) or t["no_history"])
        self.status_changed.emit(st)
        return st

    def open_colab(self, confirm: bool = True) -> str | None:
        st = self.last
        if confirm and st is not None and not st["ready"]:
            answer = QMessageBox.question(
                self, self.t["colab"], self.t["colab_early"].format(n=st["pending_images"], m=st["min_new_images"]))
            if answer != QMessageBox.StandardButton.Yes:
                return None
        url = colab_url(branch=self.branch)
        self.open_url(url)
        return url
