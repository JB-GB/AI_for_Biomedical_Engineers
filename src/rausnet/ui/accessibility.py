"""Accessibility helpers for the desktop app (interface phase F2).

- ``pair_images``: decide which dropped file is grey-scale (GS) and which is Power Doppler (PD),
  from the file name and, failing that, from the colour content (Doppler images carry colour).
- Words for grades and confidence, and one spoken sentence per result for screen readers.
- Themes (standard / high contrast) as one style sheet keyed by object names, visible focus, and
  an in-app text size that scales every font, including the large result labels.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QToolTip, QWidget

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
TEXT_SCALES = (100, 125, 150, 175, 200)

STRINGS = {
    "es": {
        "g0": "normal", "g1": "leve", "g2": "moderada", "g3": "grave",
        "c_very_high": "muy alta", "c_high": "alta", "c_medium": "media", "c_low": "baja",
        "spoken": "Grado combinado {g}, {words}. Confianza {conf}, {pct}. {refer}",
        "refer_yes": "Derivar a un reumatólogo.",
        "refer_no": "No requiere derivación.",
        "no_result": "Sin resultado todavía: carga las imágenes y pulsa Analizar (Ctrl+Intro).",
        "result_summary": "Resumen del resultado",
        "drop_hint": "Sin imagen\n(arrastra aquí la imagen o ambas)",
        "paired": "Imágenes cargadas: GS {gs} · PD {pd}.",
        "paired_guess": "No se pudo distinguir GS de PD: comprueba que cada imagen esté en su lugar.",
        "not_image": "Solo se admiten imágenes PNG, JPG, BMP o TIFF.",
        "none": "ninguna",
        "preview": "Vista previa: {title}",
        "bar": "Probabilidad del grado combinado {g}",
        "help_title": "Ayuda y atajos de teclado",
        "help_text": ("1. Arrastra las imágenes GS y PD a la ventana (o Ctrl+O).\n"
                      "2. Pulsa Analizar (Ctrl+Intro). El resultado se lee en voz alta.\n"
                      "3. Exporta el informe PDF (Ctrl+P) o corrige el grado (Ctrl+K): un clic por grado "
                      "y Enviar. Ctrl+Z deshace el último envío.\n\n"
                      "Ctrl+O abrir imágenes · Ctrl+Intro analizar · Ctrl+P PDF · Ctrl+K corregir · "
                      "Ctrl+Z deshacer · "
                      "F1 esta ayuda · Tab / Mayús+Tab moverse entre controles."),
        "help_hint": "F1: ayuda y atajos de teclado",
        "open_images": "Abrir imágenes GS y PD",
        "text_size": "Tamaño del texto",
        "high_contrast": "Alto contraste (texto claro sobre fondo negro)",
        "palette": "Colores del mapa de calor",
        "pal_warm": "Cálidos: amarillo a rojo (predeterminado)",
        "pal_cividis": "Aptos para daltonismo: azul a amarillo",
        "sc_analyze": "Ctrl+Intro", "sc_open": "Ctrl+O", "sc_pdf": "Ctrl+P", "sc_correct": "Ctrl+K",
    },
    "en": {
        "g0": "normal", "g1": "mild", "g2": "moderate", "g3": "severe",
        "c_very_high": "very high", "c_high": "high", "c_medium": "medium", "c_low": "low",
        "spoken": "Combined grade {g}, {words}. Confidence {conf}, {pct}. {refer}",
        "refer_yes": "Refer to a rheumatologist.",
        "refer_no": "No referral needed.",
        "no_result": "No result yet: load the images and press Analyze (Ctrl+Enter).",
        "result_summary": "Result summary",
        "drop_hint": "No image\n(drop one or both images here)",
        "paired": "Images loaded: GS {gs} · PD {pd}.",
        "paired_guess": "Could not tell GS from PD: check that each image is in its place.",
        "not_image": "Only PNG, JPG, BMP or TIFF images are accepted.",
        "none": "none",
        "preview": "Preview: {title}",
        "bar": "Probability of combined grade {g}",
        "help_title": "Help and keyboard shortcuts",
        "help_text": ("1. Drop the GS and PD images on the window (or Ctrl+O).\n"
                      "2. Press Analyze (Ctrl+Enter). The result is read aloud.\n"
                      "3. Export the PDF report (Ctrl+P) or correct the grade (Ctrl+K): one click per "
                      "grade and Send. Ctrl+Z undoes the last one.\n\n"
                      "Ctrl+O open images · Ctrl+Enter analyze · Ctrl+P PDF · Ctrl+K correct · "
                      "Ctrl+Z undo · "
                      "F1 this help · Tab / Shift+Tab move between controls."),
        "help_hint": "F1: help and keyboard shortcuts",
        "open_images": "Open GS and PD images",
        "text_size": "Text size",
        "high_contrast": "High contrast (light text on black)",
        "palette": "Heat-map colours",
        "pal_warm": "Warm: yellow to red (default)",
        "pal_cividis": "Colour-blind safe: blue to yellow",
        "sc_analyze": "Ctrl+Enter", "sc_open": "Ctrl+O", "sc_pdf": "Ctrl+P", "sc_correct": "Ctrl+K",
    },
}

_PD = re.compile(r"(?<![a-z])(pd|doppler|power|colou?r|cdus|pdus)(?![a-z])")
_GS = re.compile(r"(?<![a-z])(gs|grey|gray|greyscale|grayscale|bmode|b-mode|b_mode)(?![a-z])")


def is_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_SUFFIXES


def name_kind(path: str | Path) -> str | None:
    """'gs' / 'pd' from the file name (or its folder), None if it does not say."""
    for part in (Path(path).stem.lower(), Path(path).parent.name.lower()):
        pd, gs = bool(_PD.search(part)), bool(_GS.search(part))
        if pd != gs:
            return "pd" if pd else "gs"
    return None


def has_colour(path: str | Path, tolerance: int = 24, min_pixels: int = 6) -> bool:
    """True if the image has coloured pixels (Doppler signal, colour box or scale bar).

    A PD image of grade 0 without colour box looks grey-scale: then the pairing is a guess."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as img:
        if img.mode in ("L", "LA", "I", "I;16", "F", "1"):
            return False
        img = img.convert("RGB")
        img.thumbnail((512, 512), Image.NEAREST)  # nearest: small colour spots are not averaged away
        rgb = np.asarray(img, np.int16)
    spread = rgb.max(axis=2) - rgb.min(axis=2)
    return int((spread > tolerance).sum()) >= min_pixels


def pair_images(paths: Iterable[str | Path]) -> dict:
    """Assign up to two image files to GS and PD: ``{"gs": path|None, "pd": path|None, "guessed": bool}``."""
    files = [str(p) for p in paths if is_image(p)][:2]
    if not files:
        return {"gs": None, "pd": None, "guessed": False}
    if len(files) == 1:
        f = files[0]
        kind = name_kind(f) or ("pd" if has_colour(f) else "gs")
        return {"gs": f if kind == "gs" else None, "pd": f if kind == "pd" else None, "guessed": False}
    a, b = files
    ka, kb = name_kind(a), name_kind(b)
    if ka == kb or (ka is None and kb is None):  # names do not decide: Doppler images are coloured
        ca, cb = has_colour(a), has_colour(b)
        if ca == cb:
            return {"gs": a, "pd": b, "guessed": True}
        return {"gs": b if ca else a, "pd": a if ca else b, "guessed": False}
    swap = ka == "pd" or kb == "gs"
    return {"gs": b if swap else a, "pd": a if swap else b, "guessed": False}


def grade_text(grade: int, t: dict) -> str:
    return f"{grade} · {t[f'g{grade}']}"


def confidence_words(c: float, t: dict) -> str:
    for limit, key in ((0.9, "c_very_high"), (0.75, "c_high"), (0.5, "c_medium")):
        if c >= limit:
            return t[key]
    return t["c_low"]


def percent(c: float) -> str:
    return f"{round(100 * c)} %"


def spoken_result(res: dict, t: dict) -> str:
    c = res["combined"]
    return t["spoken"].format(g=c["grade"], words=t[f"g{c['grade']}"], conf=confidence_words(c["confidence"], t),
                              pct=percent(c["confidence"]),
                              refer=t["refer_yes" if res.get("refer") else "refer_no"]).strip()


def announce(widget: QWidget, text: str) -> None:
    """Ask screen readers to read ``text`` now (Qt >= 6.8); silently skipped elsewhere."""
    try:
        from PySide6.QtGui import QAccessible, QAccessibleAnnouncementEvent

        QAccessible.updateAccessibility(QAccessibleAnnouncementEvent(widget, text))
    except (ImportError, RuntimeError, AttributeError):
        pass


_STANDARD = """
#banner { font-weight:bold; padding:8px; border-radius:4px; color:#ffffff; background:#b3261e; }
#banner[validated="true"] { background:#0b6b0b; }
#modelLabel { color:#4d4d4d; }
#newModelBar { background:#e8f0fe; border:1px solid #1a56db; border-radius:4px; }
#imageView { background:#111111; color:#d0d0d0; border-radius:4px; }
#warnings, #incomplete { color:#7a4a00; }
#referBanner { background:#fff4e5; color:#4a2900; border:2px solid #b45309; border-radius:4px; padding:8px;
               font-weight:bold; }
QPushButton { border:1px solid #767676; border-radius:4px; padding:6px 14px; background:#f5f5f5; color:#111111; }
QPushButton:hover { background:#e6e6e6; }
QPushButton:pressed { background:#d6d6d6; }
QPushButton:disabled { color:#6b6b6b; background:#f0f0f0; border-color:#b0b0b0; }
QPushButton:focus, QLineEdit:focus, #resultSummary:focus { border:3px solid #1a56db; }
QCheckBox:focus { border:2px solid #1a56db; }
#resultSummary { padding:4px; border:1px solid transparent; }
"""

_HIGH_CONTRAST = """
QWidget { background:#000000; color:#ffffff; }
QToolTip { background:#000000; color:#ffffff; border:1px solid #ffffff; }
#banner { font-weight:bold; padding:8px; border-radius:4px; color:#000000; background:#ff8a80; }
#banner[validated="true"] { background:#69f0ae; }
#modelLabel { color:#ffffff; }
#newModelBar { background:#000000; border:2px solid #ffffff; border-radius:4px; }
#imageView { background:#000000; color:#ffffff; border:1px solid #ffffff; }
#warnings, #incomplete { color:#ffd60a; }
#referBanner { background:#000000; color:#ffd60a; border:3px solid #ffd60a; padding:8px; font-weight:bold; }
QPushButton { background:#000000; color:#ffffff; border:2px solid #ffffff; border-radius:4px; padding:6px 14px; }
QPushButton:hover { background:#1f1f1f; }
QPushButton:disabled { color:#a6a6a6; border-color:#a6a6a6; }
QLineEdit, QComboBox { background:#000000; color:#ffffff; border:2px solid #ffffff; padding:2px 6px; }
QComboBox QAbstractItemView { background:#000000; color:#ffffff; selection-background-color:#ffd60a;
                              selection-color:#000000; }
QPushButton:focus, QLineEdit:focus, QComboBox:focus, #resultSummary:focus { border:3px solid #ffd60a; }
QCheckBox:focus { border:2px solid #ffd60a; }
QCheckBox::indicator { border:2px solid #ffffff; width:14px; height:14px; }
QCheckBox::indicator:checked { background:#ffd60a; }
QProgressBar { border:2px solid #ffffff; color:#ffffff; text-align:center; }
QProgressBar::chunk { background:#ffd60a; }
QGroupBox { border:1px solid #ffffff; margin-top:14px; padding-top:6px; }
QGroupBox::title { subcontrol-origin:margin; left:8px; padding:0 4px; }
QTabWidget::pane { border:1px solid #ffffff; }
QTabBar::tab { background:#000000; color:#ffffff; border:1px solid #ffffff; padding:6px 14px; }
QTabBar::tab:selected { background:#ffffff; color:#000000; }
QTabBar::tab:focus { border:3px solid #ffd60a; }
#resultSummary { padding:4px; border:1px solid transparent; }
"""


def stylesheet(high_contrast: bool = False) -> str:
    return _HIGH_CONTRAST if high_contrast else _STANDARD


_BASE_POINT_SIZE: list[float] = []


def apply_text_scale(percent_: int, scaled: Iterable[tuple[QWidget, float, bool]] = (),
                     root: QWidget | None = None) -> float:
    """Scale text to ``percent_`` of the system size and return the new base point size.

    ``scaled`` lists widgets with their own relative size (factor, bold). With ``root`` every widget
    under it gets the font explicitly: under a style sheet Qt does not pass font changes from the
    application or a parent down to child widgets, so changing only the application font left most
    of the window at 100 % (found with the F5 screenshot at 200 %).
    """
    app = QApplication.instance()
    if not _BASE_POINT_SIZE:
        _BASE_POINT_SIZE.append(app.font().pointSizeF() if app.font().pointSizeF() > 0 else 9.0)
    base = _BASE_POINT_SIZE[0] * percent_ / 100
    font = QFont(app.font())
    font.setPointSizeF(base)
    app.setFont(font)
    QToolTip.setFont(font)
    special = {id(w): (factor, bold) for w, factor, bold in scaled}
    widgets = ([root, *root.findChildren(QWidget)] if root is not None else []) + [w for w, _f, _b in scaled]
    for widget in widgets:
        factor, bold = special.get(id(widget), (1.0, None))
        f = QFont(widget.font())
        f.setPointSizeF(base * factor)
        if bold is not None:
            f.setBold(bold)
        widget.setFont(f)
    return base
