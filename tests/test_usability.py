"""Interface phase F5: usability-study log and SUS / task scoring."""

import importlib.util
import json
import os
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("fastapi")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from fastapi.testclient import TestClient
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from rausnet.data.manifest import read_manifest
from rausnet.serve.api import create_app
from rausnet.ui.app import MainWindow
from rausnet.ui.client import ApiClient
from rausnet.ui.study import TASKS, StudyLog, mean_ci, read_log, summary_table, sus_score, task_results

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _report_module():
    spec = importlib.util.spec_from_file_location("usability_report", ROOT / "scripts/usability_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- scoring
def test_sus_score_and_confidence_interval():
    assert sus_score([3] * 10) == 50.0
    assert sus_score([5, 1] * 5) == 100.0 and sus_score([1, 5] * 5) == 0.0
    with pytest.raises(ValueError):
        sus_score([3] * 9)
    with pytest.raises(ValueError):
        sus_score([0] + [3] * 9)
    m, lo, hi = mean_ci([70.0, 80.0, 90.0])
    half = 4.303 * 10 / 3 ** 0.5  # t(0.975, df 2) * sd / sqrt(n)
    assert m == 80.0 and lo == pytest.approx(80 - half) and hi == pytest.approx(80 + half)


def _events(*items):
    t, out = 0.0, []
    for kind, extra in items:
        t += 1.5
        out.append({"t": t, "kind": kind, **extra})
    return out


def test_task_results_split_by_facilitator_marks():
    click, mark = ("click", {"target": "x"}), ("mark", {})
    events = _events(("start", {}),
                     mark, ("drop", {"files": 2}), click, ("goal", {"goal": "analyzed"}),          # 2 actions
                     mark, click, ("key", {"key": "Return"}), click, ("goal", {"goal": "pdf_exported"}),  # 3
                     mark, click, click)                                                         # never sent
    res = task_results(events, "P01")
    assert [(r.task, r.completed, r.actions) for r in res[:3]] == [
        ("check_case", True, 2), ("export_pdf", True, 3), ("correct", False, 2)]
    assert res[0].within_budget and not res[1].within_budget
    assert res[0].seconds == 4.5
    assert all(not r.completed for r in res[3:]) and len(res) == len(TASKS)
    rows = {r["task"]: r for r in summary_table(res)}
    assert rows["check_case"]["budget_met"] and not rows["export_pdf"]["budget_met"]
    assert rows["correct"]["completion"] == 0 and rows["correct"]["median_actions"] is None


def test_report_applies_the_acceptance_rule(tmp_path):
    report = _report_module()
    good = tmp_path / "P01_20261114_100000.jsonl"
    lines = [{"t": 0, "kind": "start"}]
    for i, (_task, goal, budget) in enumerate(TASKS):
        lines += [{"t": 10 * i + 1, "kind": "mark"}] + [{"t": 10 * i + 2, "kind": "click", "target": "b"}] * budget
        lines.append({"t": 10 * i + 5, "kind": "goal", "goal": goal})
    good.write_text("\n".join(json.dumps(x) for x in lines))
    sus = [{"participant": f"P0{k}", **{f"q{i}": str(v) for i, v in enumerate([5, 2] * 5, 1)}} for k in (1, 2, 3)]
    text, ok = report.build(sus, [str(good)])
    assert ok and "mean 87.5" in text and "| check_case | 1/1 | 2 | 2 | yes |" in text
    low = [{**row, "q1": "1", "q3": "1", "q5": "1"} for row in sus]  # 57.5: below 68
    text, ok = report.build(low, [str(good)])
    assert not ok and "NOT met" in text


# ---- study log in the real window
def test_study_log_counts_actions_and_keeps_no_identifiers(qapp, onnx_model, synth_small, tmp_path):
    with pytest.raises(ValueError):
        StudyLog(tmp_path, "Juan Pérez")
    log = StudyLog(tmp_path / "study", "P07")
    w = MainWindow(ApiClient(http=TestClient(create_app(onnx_model[0], tmp_path / "data"))), "es", study=log)
    qapp.installEventFilter(log)
    try:
        w.show()
        assert QTest.qWaitForWindowExposed(w)
        assert "P07" in w.windowTitle()
        r = read_manifest(synth_small / "manifest.csv")[0]
        log.mark()
        w.load_files([str(synth_small / r.gs_path), str(synth_small / r.pd_path)])
        btn = w.analyze_btn
        QTest.mouseClick(btn, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                         QPoint(btn.width() // 2, btn.height() // 2))
        log.mark()
        w.patient.setFocus()
        QTest.keyClicks(w.patient, "HOSP-0042")  # typed text is never logged
        w._shortcut("Ctrl+Z", lambda: None)
        text = log.path.read_text(encoding="utf-8")
    finally:
        qapp.removeEventFilter(log)
        w.close()
    events = read_log(log.path)
    kinds = [e["kind"] for e in events]
    assert kinds[0] == "start" and kinds.count("mark") == 2
    assert {"kind": "goal", "goal": "analyzed"}.items() <= next(e for e in events if e["kind"] == "goal").items()
    clicks = [e for e in events if e["kind"] == "click"]
    assert len(clicks) == 1 and clicks[0]["target"].startswith("Analizar")
    assert {"kind": "key", "key": "Ctrl+Z"}.items() <= events[-1].items()
    assert r.gs_path.split("/")[-1] not in text and "HOSP" not in text and "0042" not in text
    assert task_results(events, "P07")[0].completed
