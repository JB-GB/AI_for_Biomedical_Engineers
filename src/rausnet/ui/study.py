"""Usability study support (interface phase F5): opt-in action log, task metrics and SUS scores.

During a session the facilitator starts the app with ``rausnet-desktop --study P01``. The app then
writes one JSON line per event to ``<folder>/P01_<date>_<time>.jsonl``:

- ``click`` (button, tab, list, menu), ``key`` (shortcuts and keyboard activation only: never the
  letters typed), ``drop`` (number of files only): the actions counted against the click budgets;
- ``goal``: a task finished (``analyzed``, ``pdf_exported``, ``correction_sent``, ``model_adopted``...);
- ``mark``: the facilitator pressed ``Ctrl+Shift+N`` when reading the next task (not counted).

Clicks inside list views count once per click (the view, not the item); a click on a combo box and
then on its item counts 2, like any other choice.

No file names, typed text or patient data are logged: only control names and times. The scoring
helpers below turn the logs and the SUS answers into the tables for the paper
(``scripts/usability_report.py``).
"""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QComboBox,
    QTabBar,
    QWidget,
)

# protocol tasks, in the order the facilitator reads them: (id, goal event, in-app action budget)
TASKS = (
    ("check_case", "analyzed", 2),        # drop both images -> Analizar
    ("export_pdf", "pdf_exported", 2),    # Exportar PDF -> save
    ("correct", "correction_sent", 3),    # GS grade -> PD grade -> Enviar
    ("undo", "undone", 1),                # Deshacer
    ("colab", "colab_opened", 1),         # Mejorar en Colab (Run all and Drive happen in the browser)
    ("adopt", "model_adopted", 1),        # Usar el nuevo modelo
)
ACTIONS = ("click", "key", "drop")
INTERACTIVE = (QAbstractButton, QComboBox, QTabBar, QAbstractItemView)
COUNTED_KEYS = {Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space, Qt.Key.Key_Tab, Qt.Key.Key_Backtab,
                Qt.Key.Key_F1, Qt.Key.Key_Escape}
SUS_PASS = 68.0
# two-sided 95 % Student t critical values, df 1..30 (no SciPy dependency)
_T95 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145,
        2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048,
        2.045, 2.042)


class StudyLog(QObject):
    """Application-wide event filter that writes the action log of one participant."""

    def __init__(self, folder: str | Path, participant: str, window: QWidget | None = None):
        super().__init__()
        if not participant.replace("-", "").replace("_", "").isalnum() or len(participant) > 20:
            raise ValueError("participant code: letters, digits, - or _ (e.g. P01), not a name")
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / f"{participant}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.participant, self.window = participant, window
        self.t0 = time.monotonic()
        self._seen: tuple = ()
        self._last_drop = 0.0
        self.write("start", participant=participant)

    def write(self, kind: str, **info) -> None:
        entry = {"t": round(time.monotonic() - self.t0, 2), "kind": kind, **info}
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def goal(self, name: str) -> None:
        self.write("goal", goal=name)

    def mark(self) -> None:
        self.write("mark")

    def shortcut(self, keys: str) -> None:
        """Called by the window when one of its shortcuts fires (shortcuts never reach widgets)."""
        self.write("key", key=keys)

    @staticmethod
    def _name(w: QWidget) -> str:
        text = w.text() if isinstance(w, QAbstractButton) else ""
        return (w.accessibleName() or text or type(w).__name__)[:60]

    def _first(self, kind: str, stamp: float) -> bool:
        """True once per physical event: Qt hands the same event to parent widgets (propagation),
        always with the same timestamp."""
        if self._seen == (kind, stamp):
            return False
        self._seen = (kind, stamp)
        return True

    def eventFilter(self, obj, event) -> bool:
        # Widget level (not the native window): QTest and accessibility tools deliver there directly.
        if not isinstance(obj, QWidget):
            return False
        et = event.type()
        if et == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
            target = obj
            while target is not None and not isinstance(target, INTERACTIVE):
                target = target.parentWidget()
            if target is not None and self._first("click", event.timestamp()):  # dialogs count too
                self.write("click", target=self._name(target))
        elif et == QEvent.Type.KeyPress and not event.isAutoRepeat():
            ctrl = event.modifiers() & Qt.KeyboardModifier.ControlModifier
            if event.key() in COUNTED_KEYS and not ctrl and self._first("key", event.timestamp()):
                self.write("key", key=_key_name(event.key()))  # letters typed are never logged
        elif et == QEvent.Type.Drop and event.mimeData().hasUrls():
            now = time.monotonic()  # drop events carry no timestamp: one drop per 0.3 s
            if now - self._last_drop > 0.3:
                self._last_drop = now
                self.write("drop", files=len(event.mimeData().urls()))
        return False


def _key_name(key: int) -> str:
    from PySide6.QtGui import QKeySequence

    return QKeySequence(key).toString() or str(key)


# ---- analysis
@dataclass
class TaskResult:
    participant: str
    task: str
    completed: bool
    actions: int
    seconds: float | None
    budget: int

    @property
    def within_budget(self) -> bool:
        return self.completed and self.actions <= self.budget


def read_log(path: str | Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def task_results(events: list[dict], participant: str, tasks=TASKS) -> list[TaskResult]:
    """Split one session into the protocol tasks.

    A task starts at its facilitator mark (or the end of the previous task without marks) and ends at
    the first matching goal event; the actions in between are counted (marks never count). A task
    without its goal before the next mark is not completed.
    """
    marks = [i for i, e in enumerate(events) if e["kind"] == "mark"]
    out, cursor = [], 0
    for k, (task, goal, budget) in enumerate(tasks):
        if marks:
            if k >= len(marks):
                out.append(TaskResult(participant, task, False, 0, None, budget))
                continue
            start, stop = marks[k], marks[k + 1] if k + 1 < len(marks) else len(events)
        else:
            start, stop = cursor, len(events)
        end = next((i for i in range(start, stop) if events[i]["kind"] == "goal" and events[i]["goal"] == goal), None)
        span = events[start:(end if end is not None else stop)]
        n = sum(e["kind"] in ACTIONS for e in span)
        secs = round(events[end]["t"] - events[start]["t"], 1) if end is not None else None
        out.append(TaskResult(participant, task, end is not None, n, secs, budget))
        if end is not None:
            cursor = end + 1
    return out


def sus_score(answers: list[int]) -> float:
    """System Usability Scale (Brooke 1996): 10 items answered 1-5, result 0-100."""
    if len(answers) != 10 or any(a not in (1, 2, 3, 4, 5) for a in answers):
        raise ValueError("SUS needs 10 answers from 1 to 5")
    odd = sum(a - 1 for a in answers[0::2])
    even = sum(5 - a for a in answers[1::2])
    return (odd + even) * 2.5


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    """Mean and two-sided 95 % t confidence interval (``nan`` bounds for a single value)."""
    m = statistics.fmean(values)
    if len(values) < 2:
        return m, math.nan, math.nan
    df = len(values) - 1
    t = _T95[df - 1] if df <= len(_T95) else 1.96
    half = t * statistics.stdev(values) / math.sqrt(len(values))
    return m, m - half, m + half


def summary_table(results: list[TaskResult]) -> list[dict]:
    """Per task: completion rate, median actions, budget met (median), median seconds."""
    rows = []
    for task, _goal, budget in TASKS:
        rs = [r for r in results if r.task == task]
        if not rs:
            continue
        done = [r for r in rs if r.completed]
        med_actions = statistics.median([r.actions for r in done]) if done else None
        rows.append({"task": task, "participants": len(rs), "completed": len(done),
                     "completion": len(done) / len(rs), "median_actions": med_actions, "budget": budget,
                     "budget_met": med_actions is not None and med_actions <= budget,
                     "median_seconds": statistics.median([r.seconds for r in done]) if done else None})
    return rows
