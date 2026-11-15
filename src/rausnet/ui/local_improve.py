"""'Mejorar en este equipo': run ``rausnet.improve run`` on this computer from the desktop app.

The run is a child process (``QProcess``), not a thread: training can be cancelled at any moment by
killing the process, and its memory (GPU included) is freed when it ends. Progress comes from the
``RAUSNET_PROGRESS`` lines printed by ``python -m rausnet.improve run --progress`` and from the
per-epoch lines of the training loop. After a cancel or a crash the runner removes the workspace
lock if that process held it, and always deletes its staging folder. Publishing in
``rausnet.improve`` is ordered (registry last), so a killed run never leaves a half model in use.

``RAUSNET_GPU`` overrides detection: ``off`` hides local training, ``cpu`` allows it without a GPU
(slow; for small trials).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

from rausnet.improve import LOCK_FILE, PROGRESS_PREFIX

EPOCH_LINE = re.compile(r"^epoch=(\d+)\b")
# share of the bar for each stage; training fills the space between "training" and "publishing"
STAGE_PERCENT = {"checking": 2, "staging": 5, "training": 10, "publishing": 95, "done": 100}


def config_path() -> Path | None:
    """Retraining recipe of the installed source tree (absent in the inference-only bundle)."""
    cfg = Path(__file__).resolve().parents[3] / "configs" / "retrain" / "default.yaml"
    return cfg if cfg.exists() else None


def availability(gpu: dict | None) -> tuple[bool, str]:
    """``(can_run, reason_key)``; the key names a UI string explaining the state."""
    mode = os.environ.get("RAUSNET_GPU", "").lower()
    if mode == "off" or getattr(sys, "frozen", False):
        return False, "local_bundle"
    if config_path() is None:
        return False, "local_no_config"
    if gpu is None:
        return False, "local_checking"
    if not gpu.get("torch"):
        return False, "local_no_torch"
    if gpu.get("cuda") or mode == "cpu":
        return True, "local_gpu" if gpu.get("cuda") else "local_cpu"
    return False, "local_no_gpu"


def release_lock_of(workspace: str | Path, run_id: str) -> bool:
    """Remove the workspace lock if run ``run_id`` held it (it was killed before releasing it).

    The run id travels in ``RAUSNET_RUN_ID``: on Windows a venv ``python.exe`` is a launcher, so the
    process id seen here is not the one that wrote the lock."""
    lock = Path(workspace) / LOCK_FILE
    try:
        held = json.loads(lock.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    if not run_id or held.get("run_id") != run_id:
        return False
    lock.unlink(missing_ok=True)
    return True


class LocalImprover(QObject):
    """One improvement run at a time. ``progress(percent, stage, info)``; ``finished(summary, error)``
    with ``summary`` the logged run summary, or ``None`` with ``error`` "cancelled" or a message."""

    progress = Signal(int, str, dict)
    finished = Signal(object, str)
    gpu_ready = Signal(dict)

    def __init__(self, parent: QObject | None = None, command=None):
        super().__init__(parent)
        self.command = command or self.default_command
        self.proc: QProcess | None = None
        self.gpu: dict | None = None
        self._probe: QProcess | None = None
        self._buf, self._tail = "", []
        self._epochs, self._stage, self._summary = 0, "", None
        self._workspace = self._staging = None
        self._cancelled, self.run_id = False, ""

    # ---- capability check (in a child process: importing PyTorch here would freeze the window)
    def detect_gpu(self) -> None:
        mode = os.environ.get("RAUSNET_GPU", "").lower()
        if mode == "off" or getattr(sys, "frozen", False) or self._probe is not None or self.gpu is not None:
            return
        self._probe = QProcess(self)
        self._probe.finished.connect(self._probe_done)
        self._probe.start(sys.executable, ["-m", "rausnet.improve", "gpu"])

    def _probe_done(self, code: int, _status) -> None:
        out = bytes(self._probe.readAllStandardOutput()).decode("utf-8", "replace").strip().splitlines()
        try:
            self.gpu = json.loads(out[-1]) if code == 0 and out else {"torch": False, "cuda": False}
        except json.JSONDecodeError:
            self.gpu = {"torch": False, "cuda": False}
        self._probe = None
        self.gpu_ready.emit(self.gpu)

    # ---- run
    @staticmethod
    def default_command(workspace: str, staging: str) -> tuple[str, list[str]]:
        return sys.executable, ["-m", "rausnet.improve", "run", "--workspace", workspace, "--progress",
                                "--staging", staging, "--config", str(config_path())]

    @property
    def running(self) -> bool:
        return self.proc is not None

    def start(self, workspace: str | Path) -> bool:
        if self.running:
            return False
        self._workspace = str(workspace)
        self._staging = tempfile.mkdtemp(prefix="rausnet_local_")
        self._buf, self._tail, self._epochs, self._stage, self._summary = "", [], 0, "", None
        self._cancelled, self.run_id = False, uuid.uuid4().hex
        self.proc = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("RAUSNET_RUN_ID", self.run_id)
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        self.proc.setProcessEnvironment(env)
        self.proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(self._done)
        self.proc.errorOccurred.connect(self._failed_to_start)
        program, args = self.command(self._workspace, self._staging)
        self.proc.start(program, args)
        self.progress.emit(0, "starting", {})
        return True

    def cancel(self) -> None:
        if not self.running:
            return
        self._cancelled = True
        self.proc.kill()
        self.proc.waitForFinished(10_000)

    def _read(self) -> None:
        self._buf += bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        *lines, self._buf = self._buf.split("\n")
        for line in (x.strip() for x in lines):
            if line:
                self._line(line)

    def _line(self, line: str) -> None:
        self._tail = [*self._tail, line][-20:]
        if line.startswith(PROGRESS_PREFIX):
            try:
                info = json.loads(line[len(PROGRESS_PREFIX):])
            except json.JSONDecodeError:
                return
            self._stage = info.pop("stage", "")
            if self._stage == "training":
                self._epochs = int(info.get("epochs") or 0)
            if self._stage == "done":
                self._summary = info.get("summary")
            self.progress.emit(STAGE_PERCENT.get(self._stage, 0), self._stage, info)
            return
        m = EPOCH_LINE.match(line)
        if m and self._stage == "training" and self._epochs:
            epoch = min(int(m.group(1)), self._epochs)
            lo, hi = STAGE_PERCENT["training"], STAGE_PERCENT["publishing"]
            self.progress.emit(lo + (hi - lo) * epoch // self._epochs, "training",
                               {"epoch": epoch, "epochs": self._epochs})

    def _failed_to_start(self, error) -> None:
        if error == QProcess.ProcessError.FailedToStart and self.proc is not None:
            self._finish(None, f"could not start: {self.proc.errorString()}")

    def _done(self, code: int, _status) -> None:
        if self.proc is None:
            return
        self._read()
        if self._buf.strip():
            self._line(self._buf.strip())
        if self._cancelled:
            self._finish(None, "cancelled")
        elif code == 0 and self._summary is not None:
            self._finish(self._summary, "")
        else:
            self._finish(None, "\n".join(self._tail[-5:]) or f"exit code {code}")

    def _finish(self, summary: dict | None, error: str) -> None:
        if error and self._workspace:
            release_lock_of(self._workspace, self.run_id)
        if self._staging:
            shutil.rmtree(self._staging, ignore_errors=True)
        proc, self.proc, self._staging = self.proc, None, None
        if proc is not None:
            proc.deleteLater()
        self.finished.emit(summary, error)
