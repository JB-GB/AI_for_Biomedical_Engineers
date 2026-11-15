"""Automated accessibility audit of the desktop app (interface phase F5), see ``rausnet.ui.audit``.

Usage: python scripts/a11y_audit.py [--images GS PD] [--models-dir models] [--lang es]
       [--out docs/paper/a11y_audit.md]

Audits the window as it opens and, with ``--images``, again after analysing one case (result,
correction selectors and today's cases visible). Exit code 1 when any error is found.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images", nargs=2, metavar=("GS", "PD"))
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--lang", default="es")
    parser.add_argument("--out", default="docs/paper/a11y_audit.md")
    args = parser.parse_args(argv)

    tmp = tempfile.mkdtemp(prefix="a11y_")  # never touch the user's preferences or today's cases
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, tmp)
    import os

    os.environ["RAUSNET_CASES_FILE"] = str(Path(tmp) / "cases.json")
    os.environ.setdefault("RAUSNET_GPU", "off")

    from rausnet.ui.app import MainWindow, start_local_server
    from rausnet.ui.audit import audit_window, dedupe, report
    from rausnet.ui.client import ApiClient

    app = QApplication.instance() or QApplication([])
    window = MainWindow(ApiClient(start_local_server(args.models_dir)), args.lang)
    findings = audit_window(window)
    if args.images:
        window.load_files(args.images)
        window.patient.setText("HOSP-0001")
        window.analyze()
        after = audit_window(window)
        for f in after:
            f.where = "after analysis: " + f.where
        findings = dedupe(findings + after)
    text = report(findings)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    window.close()
    app.processEvents()
    return 1 if any(f.severity == "error" for f in findings) else 0


if __name__ == "__main__":
    sys.exit(main())
