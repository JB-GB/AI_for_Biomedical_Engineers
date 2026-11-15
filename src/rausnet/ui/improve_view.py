"""'Mejorar el modelo' tab: shared workspace state, one-click improvement (Colab or this computer),
version history and roll back (lead option).

Everything comes from the API (``GET /workspace/status``), so the panel works the same with the
private local API and with a lab server. Plain words, no ML jargon; every control has an
accessible name for screen readers.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from rausnet.improve import colab_url
from rausnet.ui.accessibility import announce
from rausnet.ui.client import ApiClient, ApiError
from rausnet.ui.local_improve import LocalImprover, availability

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
        "local": "Mejorar en este equipo",
        "local_gpu": "Usará la tarjeta gráfica de este equipo ({name})",
        "local_cpu": "Sin tarjeta gráfica: será lento (modo de prueba)",
        "local_no_gpu": "Este equipo no tiene una tarjeta gráfica compatible: usa «Mejorar en Colab»",
        "local_no_torch": "Este equipo no tiene instalado el entrenamiento: usa «Mejorar en Colab»",
        "local_bundle": "Solo en la instalación completa: usa «Mejorar en Colab»",
        "local_no_config": "Falta la configuración de entrenamiento: usa «Mejorar en Colab»",
        "local_checking": "Comprobando la tarjeta gráfica…",
        "local_remote": "La carpeta compartida no está en este equipo: usa «Mejorar en Colab»",
        "local_early": ("Hay {n} de {m} imágenes nuevas. Con tan pocas, la mejora terminará sin cambios. "
                        "¿Empezar de todos modos?"),
        "local_busy": "Ya hay una mejora en curso en {where}. Espera a que termine.",
        "running_here": "Mejora en curso en este equipo",
        "cancel": "Cancelar mejora",
        "cancel_confirm": ("¿Cancelar la mejora? Se descarta lo entrenado hasta ahora; "
                           "el modelo en uso no cambia."),
        "stage_starting": "Preparando…", "stage_checking": "Buscando imágenes nuevas…",
        "stage_staging": "Copiando los datos a este equipo…", "stage_training": "Entrenando…",
        "stage_epoch": "Entrenando: ronda {epoch} de {epochs}",
        "stage_publishing": "Publicando el modelo nuevo…", "stage_done": "Terminado",
        "run_promoted": "Mejora terminada: nuevo modelo global {version}.",
        "run_kept": "Mejora terminada: se mantuvo el modelo anterior (el nuevo no fue mejor).",
        "run_nothing": "Mejora terminada sin cambios: {reason}.",
        "run_cancelled": "Mejora cancelada. El modelo en uso no cambió.",
        "run_failed": "La mejora falló y el modelo en uso no cambió. Detalle: {error}",
        "versions": "Versiones del modelo",
        "col_version": "Versión", "col_date": "Fecha", "col_status": "Estado", "col_quality": "Calidad",
        "s_current": "en uso", "s_archived": "anterior", "s_rejected": "no aprobada",
        "s_served": " (este equipo)",
        "rollback": "Volver a esta versión",
        "rollback_tip": "Solo responsables: vuelve a usar la versión elegida en todos los equipos",
        "rollback_pick": "Elige en la lista una versión anterior.",
        "rollback_ask": ("¿Volver a la versión {version} en todos los equipos?\n"
                         "Motivo (opcional, sin datos de pacientes):"),
        "rolled_back": "Ahora se usa la versión {version} (antes {previous}).",
        "h_rolled": "se volvió a la versión {version} (antes {previous})",
        "admin": "Opciones de responsable (volver a una versión anterior)",
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
        "local": "Improve on this computer",
        "local_gpu": "Uses this computer's graphics card ({name})",
        "local_cpu": "No graphics card: it will be slow (trial mode)",
        "local_no_gpu": "This computer has no supported graphics card: use 'Improve on Colab'",
        "local_no_torch": "Training is not installed on this computer: use 'Improve on Colab'",
        "local_bundle": "Only in the full installation: use 'Improve on Colab'",
        "local_no_config": "The training configuration is missing: use 'Improve on Colab'",
        "local_checking": "Checking the graphics card…",
        "local_remote": "The shared folder is not on this computer: use 'Improve on Colab'",
        "local_early": ("There are {n} of {m} new images. With so few the run will end without changes. "
                        "Start anyway?"),
        "local_busy": "An improvement is already running on {where}. Wait until it ends.",
        "running_here": "Improvement running on this computer",
        "cancel": "Cancel improvement",
        "cancel_confirm": "Cancel the improvement? Training so far is discarded; the model in use does not change.",
        "stage_starting": "Preparing…", "stage_checking": "Looking for new images…",
        "stage_staging": "Copying the data to this computer…", "stage_training": "Training…",
        "stage_epoch": "Training: round {epoch} of {epochs}",
        "stage_publishing": "Publishing the new model…", "stage_done": "Finished",
        "run_promoted": "Improvement finished: new global model {version}.",
        "run_kept": "Improvement finished: previous model kept (the new one was not better).",
        "run_nothing": "Improvement finished without changes: {reason}.",
        "run_cancelled": "Improvement cancelled. The model in use did not change.",
        "run_failed": "The improvement failed and the model in use did not change. Detail: {error}",
        "versions": "Model versions",
        "col_version": "Version", "col_date": "Date", "col_status": "Status", "col_quality": "Quality",
        "s_current": "in use", "s_archived": "previous", "s_rejected": "not approved",
        "s_served": " (this computer)",
        "rollback": "Go back to this version",
        "rollback_tip": "Leads only: serve the chosen version again on every computer",
        "rollback_pick": "Choose a previous version in the list.",
        "rollback_ask": "Go back to version {version} on every computer?\nReason (optional, no patient data):",
        "rolled_back": "Now using version {version} (before {previous}).",
        "h_rolled": "went back to version {version} (before {previous})",
        "admin": "Lead options (go back to a previous version)",
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


def run_message(summary: dict | None, error: str, t: dict) -> str:
    """Plain sentence for the end of a local improvement run."""
    if error == "cancelled":
        return t["run_cancelled"]
    if summary is None:
        last = error.strip().splitlines()[-1][:200] if error.strip() else "?"
        return t["run_failed"].format(error=last)
    if summary.get("action") == "retrained":
        return t["run_promoted"].format(version=summary.get("version")) if summary.get("promoted") else t["run_kept"]
    return t["run_nothing"].format(reason=nothing_reason(summary, t))


def nothing_reason(entry: dict, t: dict) -> str:
    reason = entry.get("reason", "")
    if "new_images" in entry:  # localized; the logged reason is English
        reason = t["h_few"].format(n=entry["new_images"])
        reason += t["h_needed"].format(m=entry["min_new"]) if "min_new" in entry else ""
    return reason


def stage_words(stage: str, info: dict, t: dict) -> str:
    if stage == "training" and info.get("epoch"):
        return t["stage_epoch"].format(epoch=info["epoch"], epochs=info.get("epochs", "?"))
    return t.get(f"stage_{stage}", t["stage_starting"])


def history_line(entry: dict, t: dict) -> str:
    when = f"{entry.get('time', '?')} · {where_words(entry.get('runtime'), t)}"
    if entry.get("action") == "rolled back":
        what = t["h_rolled"].format(version=entry.get("version"), previous=entry.get("previous") or "-")
    elif entry.get("action") == "retrained":
        what = t["h_promoted"].format(version=entry.get("version")) if entry.get("promoted") else t["h_kept"]
    else:
        what = t["h_nothing"].format(reason=nothing_reason(entry, t))
    return f"{when}: {what}"


class ImproveView(QWidget):
    """Shows the workspace state; ``status_changed`` carries each status (or ``None`` without workspace),
    ``model_switched`` the API's answer after a roll back (the served model changed)."""

    status_changed = Signal(object)
    model_switched = Signal(dict)

    def __init__(self, client: ApiClient, t: dict, branch: str = "PIA_devl",
                 open_url: Callable[[str], object] | None = None, runner: LocalImprover | None = None):
        super().__init__()
        self.client, self.t, self.branch = client, t, branch
        self.open_url = open_url or (lambda url: QDesktopServices.openUrl(QUrl(url)))
        self.last: dict | None = None
        self.admin = False
        self.goal: Callable[[str], object] = lambda _name: None  # usability study hook (see ui.study)
        self.runner = runner or LocalImprover(self)
        self.runner.progress.connect(self._on_progress)
        self.runner.finished.connect(self._on_finished)
        self.runner.gpu_ready.connect(lambda _gpu: self._update_local())

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
        self.local_btn = QPushButton(t["local"])
        self.local_btn.setMinimumHeight(48)
        self.local_btn.clicked.connect(lambda: self.start_local())
        self.refresh_btn = QPushButton(t["refresh"])
        self.refresh_btn.setMinimumHeight(44)
        self.refresh_btn.clicked.connect(lambda: self.refresh())
        buttons = QHBoxLayout()
        buttons.addWidget(self.colab_btn, 2)
        buttons.addWidget(self.local_btn, 2)
        buttons.addWidget(self.refresh_btn, 1)
        self.local_note = QLabel("")
        self.local_note.setWordWrap(True)

        # local run: stage sentence, bar and cancel, visible only while running or just after
        self.run_text = QLabel("")
        self.run_text.setWordWrap(True)
        self.run_bar = QProgressBar()
        self.run_bar.setRange(0, 100)
        self.run_bar.setMinimumHeight(22)
        self.cancel_btn = QPushButton(t["cancel"])
        self.cancel_btn.setMinimumHeight(44)
        self.cancel_btn.clicked.connect(lambda: self.cancel_local())
        run_row = QHBoxLayout()
        run_row.addWidget(self.run_bar, 3)
        run_row.addWidget(self.cancel_btn, 1)
        self.run_bar.hide()
        self.cancel_btn.hide()

        self.history = QLabel("")
        self.history.setWordWrap(True)
        hist = QGroupBox(t["history"])
        QVBoxLayout(hist).addWidget(self.history)

        self.versions = QTableWidget(0, 4)
        self.versions.setHorizontalHeaderLabels([t["col_version"], t["col_date"], t["col_status"], t["col_quality"]])
        self.versions.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.versions.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.versions.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.versions.verticalHeader().hide()
        self.versions.setCornerButtonEnabled(False)
        self.versions.horizontalHeader().setStretchLastSection(True)
        self.versions.setMinimumHeight(140)
        self.versions.itemSelectionChanged.connect(self._update_rollback)
        self.rollback_btn = QPushButton(t["rollback"])
        self.rollback_btn.setMinimumHeight(44)
        self.rollback_btn.setToolTip(t["rollback_tip"])
        self.rollback_btn.clicked.connect(lambda: self.rollback())
        self.rollback_btn.hide()
        self.version_rows: list[dict] = []
        vers = QGroupBox(t["versions"])
        vlay = QVBoxLayout(vers)
        vlay.addWidget(self.versions)
        vlay.addWidget(self.rollback_btn)

        self.scaled_fonts = [(self.colab_btn, 1.2, True), (self.local_btn, 1.2, True),
                             (self.progress_text, 1.2, False), (self.run_text, 1.1, False)]
        for widget, name, desc in (
            (self.progress, t["tab_improve"], ""),  # replaced by the progress sentence on refresh
            (self.colab_btn, t["colab"], t["colab_tip"]),
            (self.local_btn, t["local"], ""),
            (self.refresh_btn, t["refresh"], ""),
            (self.run_bar, t["stage_starting"], ""),
            (self.cancel_btn, t["cancel"], ""),
            (self.versions, t["versions"], ""),
            (self.rollback_btn, t["rollback"], t["rollback_tip"]),
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
        lay.addWidget(self.local_note)
        lay.addWidget(self.run_text)
        lay.addLayout(run_row)
        lay.addWidget(hist)
        lay.addWidget(vers, 1)

    def set_client(self, client: ApiClient) -> None:
        self.client = client
        self.refresh()

    def set_admin(self, on: bool) -> None:
        self.admin = on
        self.rollback_btn.setVisible(on)
        self._update_rollback()

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
            self._show_versions(None)
            self._update_local()
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
        if self.runner.running:  # our child may not hold the lock yet
            self.running.setText(t["running_here"])
        else:
            self.running.setText(t["running"].format(where=where_words(run.get("runtime"), t),
                                                     since=run.get("started")) if run else t["idle"])
        hist = st.get("history") or []
        self.history.setText("\n".join(history_line(e, t) for e in reversed(hist[-5:])) or t["no_history"])
        try:
            self._show_versions(self.client.model_versions())
        except ApiError:
            self._show_versions(None)
        self.runner.detect_gpu()
        self._update_local()
        self.status_changed.emit(st)
        return st

    def _ask_early(self, title: str, key: str) -> bool:
        """Below the threshold a run ends without changes: ask first. True = go ahead."""
        st = self.last
        if st is None or st["ready"]:
            return True
        answer = QMessageBox.question(self, title, self.t[key].format(n=st["pending_images"], m=st["min_new_images"]))
        return answer == QMessageBox.StandardButton.Yes

    def open_colab(self, confirm: bool = True) -> str | None:
        if confirm and not self._ask_early(self.t["colab"], "colab_early"):
            return None
        url = colab_url(branch=self.branch)
        self.open_url(url)
        self.goal("colab_opened")
        return url

    # ---- local improvement
    def local_state(self) -> tuple[bool, str]:
        """``(can_run, explanation)`` for the local button."""
        if self.last is None or not Path(self.last["workspace"]).is_dir():
            return False, self.t["local_remote"] if self.last else ""
        ok, key = availability(self.runner.gpu)
        name = (self.runner.gpu or {}).get("name") or ""
        return ok, self.t[key].format(name=name)

    def _update_local(self) -> None:
        ok, why = self.local_state()
        busy = self.runner.running
        self.local_btn.setEnabled(ok and not busy)
        self.local_btn.setToolTip(why)
        self.local_btn.setAccessibleDescription(why)
        self.local_note.setText("" if ok or busy else why)

    def start_local(self, confirm: bool = True) -> bool:
        ok, why = self.local_state()
        if not ok or self.runner.running:
            self.local_note.setText(why)
            return False
        st = self.last
        if st.get("running"):
            self.local_note.setText(self.t["local_busy"].format(where=where_words(st["running"].get("runtime"),
                                                                                   self.t)))
            return False
        if confirm and not self._ask_early(self.t["local"], "local_early"):
            return False
        self.run_bar.setValue(0)
        self.run_bar.show()
        self.cancel_btn.show()
        self.run_text.setText(self.t["stage_starting"])
        started = self.runner.start(st["workspace"])
        if started:
            self.running.setText(self.t["running_here"])
        self._update_local()
        if started:
            self.cancel_btn.setFocus()
            self.goal("local_started")
        return started

    def cancel_local(self, confirm: bool = True) -> None:
        if not self.runner.running:
            return
        if confirm:
            answer = QMessageBox.question(self, self.t["cancel"], self.t["cancel_confirm"])
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.runner.cancel()

    def _on_progress(self, percent: int, stage: str, info: dict) -> None:
        text = stage_words(stage, info, self.t)
        self.run_bar.setValue(percent)
        self.run_bar.setAccessibleName(text)
        if text != self.run_text.text():
            self.run_text.setText(text)
            if stage != "training" or not info.get("epoch"):  # one announcement per stage, not per epoch
                announce(self.run_text, text)

    def _on_finished(self, summary: dict | None, error: str) -> None:
        text = run_message(summary, error, self.t)
        self.run_text.setText(text)
        self.run_bar.hide()
        self.cancel_btn.hide()
        announce(self.run_text, text)
        self.refresh()  # shows the new-model banner when a version was published
        self.local_btn.setFocus()

    # ---- versions and roll back
    def _show_versions(self, data: dict | None) -> None:
        t = self.t
        rows = (data or {}).get("versions") or []
        served = (data or {}).get("served_version")
        self.version_rows = rows
        self.versions.setRowCount(len(rows))
        for i, v in enumerate(rows):
            status = t.get(f"s_{v['status']}", v["status"]) + (t["s_served"] if v["version"] == served else "")
            cells = (v["version"], (v.get("created") or "")[:10], status, quality_words(v.get("test_qwk"), t))
            for j, text in enumerate(cells):
                self.versions.setItem(i, j, QTableWidgetItem(text))
        self.versions.resizeColumnsToContents()
        self._update_rollback()

    def selected_version(self) -> dict | None:
        rows = self.versions.selectionModel().selectedRows() if self.versions.selectionModel() else []
        return self.version_rows[rows[0].row()] if rows else None

    def _update_rollback(self) -> None:
        v = self.selected_version()
        self.rollback_btn.setEnabled(self.admin and v is not None and v["status"] == "archived" and v["servable"])

    def select_version(self, version: str) -> bool:
        for i, v in enumerate(self.version_rows):
            if v["version"] == version:
                self.versions.selectRow(i)
                return True
        return False

    def rollback(self, version: str | None = None, reason: str | None = None) -> dict | None:
        """Lead only: serve an archived version again on every computer. Asks for confirmation and an
        optional reason unless ``reason`` is given (tests, scripts)."""
        if not self.admin:
            return None
        if version is None:
            v = self.selected_version()
            if v is None or v["status"] != "archived":
                self.run_text.setText(self.t["rollback_pick"])
                return None
            version = v["version"]
        if reason is None:
            reason, ok = QInputDialog.getText(self, self.t["rollback"], self.t["rollback_ask"].format(version=version))
            if not ok:
                return None
        try:
            res = self.client.rollback(version, reason)
        except ApiError as exc:
            self.run_text.setText(f"{self.t.get('error', 'Error')}: {exc}")
            announce(self.run_text, self.run_text.text())
            return None
        text = self.t["rolled_back"].format(version=res["version"], previous=res.get("previous") or "-")
        self.run_text.setText(text)
        announce(self.run_text, text)
        self.model_switched.emit(res)
        self.refresh()
        self.goal("rolled_back")
        return res
