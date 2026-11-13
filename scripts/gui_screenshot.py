"""Render the desktop app end to end (local API + current model) and save a screenshot.

Usage: python scripts/gui_screenshot.py GS_IMAGE [PD_IMAGE] [--out docs/paper/figures/gui_desktop.png] [--lang es]
       [--heatmap]
"""

from __future__ import annotations

import argparse

from PySide6.QtWidgets import QApplication

from rausnet.ui.app import MainWindow, start_local_server
from rausnet.ui.client import ApiClient


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("gs")
    parser.add_argument("pd", nargs="?")
    parser.add_argument("--out", default="docs/paper/figures/gui_desktop.png")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--lang", default="es")
    parser.add_argument("--heatmap", action="store_true", help="show the Grad-CAM overlay")
    args = parser.parse_args(argv)

    app = QApplication([])
    window = MainWindow(ApiClient(start_local_server(args.models_dir)), args.lang)
    window.gs_panel.set_path(args.gs)
    window.pd_panel.set_path(args.pd)
    result = window.analyze()
    window.patient.setText("HOSP-0001")
    window.heatmap_box.setChecked(args.heatmap)
    window.resize(1100, 720)
    app.processEvents()
    # grab() renders without showing the window; native platform keeps system fonts.
    window.grab().save(args.out)
    print(f"combined grade {result['combined']['grade']} -> {args.out}")


if __name__ == "__main__":
    main()
