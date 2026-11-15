"""Correction workflow (interface phase F3): one-click grade selectors, today's cases, input checks.

Nothing identifying is kept: today's cases hold time, pseudonym, joint, side and grades only (no
file names or paths, which may contain names) and are discarded on a new day.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from PySide6.QtCore import QStandardPaths, Signal
from PySide6.QtWidgets import QButtonGroup, QGridLayout, QRadioButton, QWidget

from rausnet.ui.accessibility import grade_text, has_colour

MIN_SIDE = 64
PSEUDONYM = re.compile(r"^[A-Za-z0-9_-]{1,40}$")  # same rule as the API
MAX_BYTES = 20 * 1024 * 1024  # same limit as the API
SIZE_SEP = " x "

STRINGS = {
    "es": {
        "undo": "Deshacer",
        "need_patient": ("Escribe el seudónimo del paciente (letras, números, «-» o «_», sin nombres) "
                         "para guardar la corrección."),
        "sc_undo": "Ctrl+Z",
        "undone": "Corrección deshecha: no se usará para mejorar el modelo.",
        "sent_confirm": "Confirmación guardada (los grados de la IA eran correctos).",
        "sent_fix": "Corrección guardada: GS {gs} · PD {pd}.",
        "today": "Casos de hoy",
        "today_title": "Casos de hoy ({n}, {k} corregidos)",
        "today_line": "{time} · {patient} · {joint} {side} · IA {ai}{label}",
        "today_fixed": " → corregido {label}",
        "today_ok": " · confirmado",
        "no_patient": "sin seudónimo",
        "i_gs_colour": "La imagen GS tiene color: ¿es en realidad la imagen Power Doppler?",
        "i_pd_grey": ("La imagen PD no tiene color. Si no hay señal Doppler es normal; si no, comprueba que "
                      "sea la imagen Power Doppler."),
        "i_size": "GS y PD tienen tamaños distintos ({gs} y {pd}): deben ser del mismo examen.",
        "i_small": "La imagen {which} es demasiado pequeña ({size}); se necesitan al menos {min} píxeles por lado.",
        "i_large": "La imagen {which} pesa más de 20 MB; expórtala en PNG o JPG.",
        "i_unreadable": "No se pudo abrir la imagen {which}.",
    },
    "en": {
        "undo": "Undo",
        "need_patient": "Type the patient pseudonym (letters, digits, '-' or '_', no names) to save the correction.",
        "sc_undo": "Ctrl+Z",
        "undone": "Correction undone: it will not be used to improve the model.",
        "sent_confirm": "Confirmation saved (the AI grades were right).",
        "sent_fix": "Correction saved: GS {gs} · PD {pd}.",
        "today": "Today's cases",
        "today_title": "Today's cases ({n}, {k} corrected)",
        "today_line": "{time} · {patient} · {joint} {side} · AI {ai}{label}",
        "today_fixed": " → corrected {label}",
        "today_ok": " · confirmed",
        "no_patient": "no pseudonym",
        "i_gs_colour": "The GS image has colour: is it actually the Power Doppler image?",
        "i_pd_grey": ("The PD image has no colour. Without Doppler signal this is normal; otherwise check "
                      "that it is the Power Doppler image."),
        "i_size": "GS and PD differ in size ({gs} and {pd}): they must come from the same scan.",
        "i_small": "The {which} image is too small ({size}); at least {min} pixels per side are needed.",
        "i_large": "The {which} image is larger than 20 MB; export it as PNG or JPG.",
        "i_unreadable": "The {which} image could not be opened.",
    },
}


class GradeSelector(QWidget):
    """Four radio buttons (0 to 3 with words): one click per grade, arrow keys inside the group."""

    changed = Signal(int)

    def __init__(self, title: str, t: dict):
        super().__init__()
        self.group = QButtonGroup(self)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        for g in range(4):
            b = QRadioButton(grade_text(g, t))
            b.setMinimumHeight(44)
            b.setAccessibleName(f"{title}: {grade_text(g, t)}")
            self.group.addButton(b, g)
            grid.addWidget(b, g // 2, g % 2)
        self.group.button(0).setChecked(True)
        self.group.idClicked.connect(self.changed.emit)

    def buttons(self) -> list[QRadioButton]:
        return [self.group.button(g) for g in range(4)]

    def currentIndex(self) -> int:  # same interface as QComboBox
        return self.group.checkedId()

    def setCurrentIndex(self, grade: int) -> None:
        self.group.button(grade).setChecked(True)

    def focus(self) -> None:
        self.group.checkedButton().setFocus()


def cases_file() -> Path:
    """Where today's cases are kept on this computer (``RAUSNET_CASES_FILE`` overrides)."""
    if os.environ.get("RAUSNET_CASES_FILE"):
        return Path(os.environ["RAUSNET_CASES_FILE"])
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation)
    return Path(base or ".") / "cases_today.json"


class TodayCases:
    """Cases analysed today on this computer; a new day starts empty."""

    FIELDS = ("time", "patient", "joint", "side", "ai", "label", "image_id")

    def __init__(self, path: str | Path | None = None, today: str | None = None):
        self.path = Path(path) if path else cases_file()
        self.today = today or time.strftime("%Y-%m-%d")
        self.cases: list[dict] = []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("date") == self.today:
                self.cases = [{k: c.get(k) for k in self.FIELDS} for c in data.get("cases", [])]
        except (FileNotFoundError, json.JSONDecodeError, AttributeError):
            pass

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"date": self.today, "cases": self.cases}, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    def add(self, patient: str, joint: str, side: str, ai: dict | None) -> int:
        self.cases.append({"time": time.strftime("%H:%M"), "patient": patient, "joint": joint, "side": side,
                           "ai": ai, "label": None, "image_id": None})
        self.save()
        return len(self.cases) - 1

    def update(self, i: int, **fields) -> None:
        self.cases[i].update({k: v for k, v in fields.items() if k in self.FIELDS})
        self.save()

    def corrected(self) -> int:
        return sum(bool(c["label"]) and c["label"] != _grades(c["ai"]) for c in self.cases)

    def lines(self, t: dict) -> list[str]:
        out = []
        for c in reversed(self.cases):
            ai = c["ai"] or {}
            label = ""
            if c["label"]:
                same = c["label"] == _grades(ai)
                label = t["today_ok"] if same else t["today_fixed"].format(
                    label=f"GS {c['label']['gs']} · PD {c['label']['pd']}")
            out.append(t["today_line"].format(time=c["time"], patient=c["patient"] or t["no_patient"],
                                              joint=c["joint"], side=c["side"],
                                              ai=f"GS {ai.get('gs', '-')} · PD {ai.get('pd', '-')}", label=label))
        return out


def _grades(ai: dict | None) -> dict | None:
    return {"gs": ai["gs"], "pd": ai["pd"]} if ai else None


def check_inputs(gs: str | None, pd: str | None, t: dict) -> list[dict]:
    """Problems with the loaded images: ``[{"code", "block", "text"}]``; blocking ones stop the analysis."""
    from PIL import Image, UnidentifiedImageError

    out, sizes = [], {}
    for which, path in (("GS", gs), ("PD", pd)):
        if not path:
            continue
        if Path(path).stat().st_size > MAX_BYTES:
            out.append({"code": "large", "block": True, "text": t["i_large"].format(which=which)})
            continue
        try:
            with Image.open(path) as img:
                sizes[which] = img.size
        except (UnidentifiedImageError, OSError):
            out.append({"code": "unreadable", "block": True, "text": t["i_unreadable"].format(which=which)})
            continue
        if min(sizes[which]) < MIN_SIDE:
            out.append({"code": "small", "block": True, "text": t["i_small"].format(
                which=which, size=SIZE_SEP.join(map(str, sizes[which])), min=MIN_SIDE)})
    if len(sizes) == 2 and sizes["GS"] != sizes["PD"]:
        out.append({"code": "size", "block": True, "text": t["i_size"].format(
            gs=SIZE_SEP.join(map(str, sizes["GS"])), pd=SIZE_SEP.join(map(str, sizes["PD"])))})
    if "GS" in sizes and has_colour(gs):
        out.append({"code": "gs_colour", "block": False, "text": t["i_gs_colour"]})
    if "PD" in sizes and not has_colour(pd):
        out.append({"code": "pd_grey", "block": False, "text": t["i_pd_grey"]})
    return out
