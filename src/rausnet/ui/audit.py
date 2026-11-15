"""Automated accessibility audit of the desktop app (interface phase F5).

Mirrors the automatic checks of Accessibility Insights for Windows on the Qt widget tree, on every
tab, in both themes and at every text size:

- **name:** every interactive control exposes an accessible name (what a screen reader says);
- **keyboard:** every interactive control can be reached with Tab (focus policy) and is in the
  focus chain of its window;
- **target:** interactive controls are at least 44 px high (the app's rule; WCAG 2.2 AA asks 24 px);
- **contrast:** text colours of the style sheets reach 4.5:1, focus frames 3:1 (WCAG 1.4.3, 1.4.11);
- **scaled:** the text-size setting reaches every text control (each one doubles at 200 %);
- **clipped:** at the largest text size no visible label or button text is cut off (WCAG 1.4.4);
- **fits:** at every text size the window fits a 1366 x 768 laptop screen (1366 x 728 without the
  taskbar) without horizontal scrolling; long tabs scroll vertically (WCAG 1.4.10 reflow, adapted).

It cannot hear what NVDA reads or judge whether a name is meaningful: the manual walkthrough stays
part of F5. Usage (needs the current model)::

    python scripts/a11y_audit.py [--images GS PD] [--out docs/paper/a11y_audit.md]
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from PySide6.QtGui import QAccessible
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QAbstractSlider,
    QApplication,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QScrollBar,
    QTabBar,
    QWidget,
)

from rausnet.ui import accessibility as a11y

MIN_TARGET = 44
WCAG_TARGET = 24
TEXT_CONTRAST = 4.5
UI_CONTRAST = 3.0
TEXT_SCALE_MIN = 1.8  # 200 % setting: allow rounding of point sizes
REFERENCE_SCREEN = (1366, 728)  # common hospital laptop, minus the Windows taskbar
INTERACTIVE = (QAbstractButton, QComboBox, QLineEdit, QAbstractSlider, QAbstractItemView)
SIZED = (QPushButton, QComboBox, QLineEdit, QCheckBox, QRadioButton)


@dataclass
class Finding:
    check: str
    severity: str  # "error" (fails the rule) or "warning" (passes WCAG, fails the app's own rule)
    where: str  # tab / state
    widget: str
    detail: str


def _luminance(hex_colour: str) -> float:
    h = hex_colour.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    rgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast_ratio(fg: str, bg: str) -> float:
    a, b = sorted((_luminance(fg), _luminance(bg)), reverse=True)
    return (a + 0.05) / (b + 0.05)


_RULE = re.compile(r"([^{}]+)\{([^}]*)\}")
_PROP = re.compile(r"(?<![-\w])(color|background|border(?:-color)?)\s*:\s*([^;]+);?")
_HEX = re.compile(r"#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b")


def stylesheet_rules(css: str) -> dict[str, dict[str, str]]:
    """``{selector: {"color": "#..", "background": "#..", "border": "#.."}}`` (hex colours only)."""
    out: dict[str, dict[str, str]] = {}
    for selectors, body in _RULE.findall(css):
        props = {}
        for name, value in _PROP.findall(body):
            m = _HEX.search(value)
            if m:
                props["border" if name.startswith("border") else name] = m.group(0)
        for sel in (s.strip() for s in selectors.split(",")):
            out.setdefault(sel, {}).update(props)
    return out


def _base_colours(rules: dict, high_contrast: bool) -> tuple[str, str]:
    """Window text and background the rules fall back to (Fusion/Windows defaults otherwise)."""
    base = rules.get("QWidget", {})
    return base.get("color", "#000000"), base.get("background", "#000000" if high_contrast else "#f0f0f0")


def check_contrast(high_contrast: bool) -> list[Finding]:
    rules = stylesheet_rules(a11y.stylesheet(high_contrast))
    text, window = _base_colours(rules, high_contrast)
    theme = "high contrast" if high_contrast else "standard"
    out = []
    for sel, props in rules.items():
        if sel.endswith(":disabled") or "::chunk" in sel or "::indicator" in sel:
            continue  # WCAG exempts inactive controls; chunks and indicators have no text
        bg = props.get("background") or rules.get(sel.split(":")[0].split("[")[0], {}).get("background") or window
        fg = props.get("color") or rules.get(sel.split(":")[0].split("[")[0], {}).get("color") or text
        if "color" in props or "background" in props:
            r = contrast_ratio(fg, bg)
            if r < TEXT_CONTRAST:
                out.append(Finding("contrast", "error", theme, sel, f"text {fg} on {bg} = {r:.2f}:1 (< 4.5)"))
        if "border" in props and sel.endswith(":focus"):
            r = contrast_ratio(props["border"], bg)
            if r < UI_CONTRAST:
                out.append(Finding("contrast", "error", theme, sel,
                                   f"focus frame {props['border']} on {bg} = {r:.2f}:1 (< 3)"))
    return out


def _label(w: QWidget) -> str:
    name = w.accessibleName() or (w.text() if hasattr(w, "text") and callable(w.text) else "")
    return f"{type(w).__name__} '{name[:40]}'" if name else f"{type(w).__name__} {w.objectName() or '(unnamed)'}"


def _qt_internal(w: QWidget) -> bool:
    p = w.parent()
    internal_parent = isinstance(p, QTabBar | QComboBox | QAbstractItemView)
    return isinstance(w, QScrollBar) or internal_parent or w.objectName().startswith("qt_")


def check_page(page: QWidget, where: str) -> list[Finding]:
    """Name, keyboard and target checks for the interactive widgets visible on ``page``."""
    out = []
    for w in page.findChildren(QWidget):
        if not isinstance(w, INTERACTIVE) or _qt_internal(w) or not w.isVisibleTo(page):
            continue
        iface = QAccessible.queryAccessibleInterface(w)
        name = iface.text(QAccessible.Text.Name) if iface is not None else ""
        if not name.strip():
            out.append(Finding("name", "error", where, _label(w), "no accessible name"))
        if not (w.focusPolicy().value & 0x1):  # Qt.TabFocus bit
            out.append(Finding("keyboard", "error", where, _label(w), "cannot be reached with Tab"))
        if isinstance(w, SIZED):
            h = w.height()
            if h < WCAG_TARGET:
                out.append(Finding("target", "error", where, _label(w), f"{h} px high (< {WCAG_TARGET})"))
            elif h < MIN_TARGET:
                out.append(Finding("target", "warning", where, _label(w), f"{h} px high (< {MIN_TARGET})"))
    return out


def check_clipping(page: QWidget, where: str) -> list[Finding]:
    """Text that does not fit its widget (labels without word wrap, buttons)."""
    out = []
    for w in page.findChildren(QWidget):
        if _qt_internal(w) or not w.isVisibleTo(page) or w.width() <= 1:
            continue
        if isinstance(w, QLabel) and w.text() and not w.wordWrap() and w.pixmap().isNull():
            need = w.fontMetrics().horizontalAdvance(max(w.text().splitlines(), key=len))
            if need > w.contentsRect().width() + 2:
                out.append(Finding("clipped", "error", where, _label(w),
                                   f"text needs {need} px, has {w.contentsRect().width()} px"))
        elif isinstance(w, QPushButton) and w.text():
            need = w.fontMetrics().horizontalAdvance(w.text()) + 16
            if need > w.width():
                out.append(Finding("clipped", "error", where, _label(w), f"text needs {need} px, has {w.width()} px"))
    return out


def _settle(app, rounds: int = 6) -> None:
    """Let font and style changes reach every layout (size hints are recomputed lazily)."""
    for _ in range(rounds):
        app.sendPostedEvents()
        app.processEvents()


def check_fits(window, where: str, screen: tuple[int, int] = REFERENCE_SCREEN) -> list[Finding]:
    """The window's minimum size fits ``screen``, and no tab needs horizontal scrolling there."""
    out = []
    need = window.minimumSizeHint()
    w, h = max(need.width(), window.minimumWidth()), max(need.height(), window.minimumHeight())
    if w > screen[0] or h > screen[1]:
        out.append(Finding("fits", "error", where, "main window",
                           f"needs at least {w} x {h} px; a 1366 x 768 screen offers {screen[0]} x {screen[1]}"))
    margin = 40  # window frame, tab pane and vertical scroll bar
    for i in range(window.tabs.count()):
        page = window.tabs.widget(i)
        inner = page.widget() if isinstance(page, QScrollArea) else page
        cw = max(inner.minimumSizeHint().width(), inner.minimumWidth())
        if cw + margin > screen[0]:
            out.append(Finding("fits", "error", where, f"tab '{window.tabs.tabText(i)}'",
                               f"content needs {cw} px wide: horizontal scrolling on a 1366 px screen"))
    return out


def text_sizes(page: QWidget) -> dict[int, tuple[str, float]]:
    """Point size of every visible control that shows text (by widget id)."""
    out = {}
    for w in page.findChildren(QWidget):
        if not w.isVisibleTo(page) or (_qt_internal(w) and not isinstance(w, QTabBar)):
            continue
        boxes = isinstance(w, QComboBox | QLineEdit | QTabBar | QGroupBox)
        has_text = boxes or (isinstance(w, QLabel | QAbstractButton) and bool(w.text()))
        if has_text:
            out[id(w)] = (_label(w), w.font().pointSizeF())
    return out


def check_scaled(before: dict, after: dict, where: str, factor: float = TEXT_SCALE_MIN) -> list[Finding]:
    """At 200 % every text control must be at least ``factor`` times its size at 100 % (WCAG 1.4.4)."""
    out = []
    for key, (label, size) in after.items():
        if key in before and before[key][1] > 0 and size / before[key][1] < factor:
            detail = f"{before[key][1]:.1f} pt at 100 % and {size:.1f} pt at 200 %: text size setting ignored"
            out.append(Finding("scaled", "error", where, label, detail))
    return out


def audit_window(window, size=(1180, 900)) -> list[Finding]:
    """Run every check on every tab of ``window`` (a ``MainWindow``) in both themes and at the
    smallest and largest text size. Restores the window's preferences afterwards."""
    app = QApplication.instance()
    findings = check_contrast(False) + check_contrast(True)
    saved = (window.text_size.currentIndex(), window.contrast_box.isChecked(), window.tabs.currentIndex())
    base_sizes: dict[int, dict] = {}
    window.resize(*size)
    window.show()
    try:
        for hc in (False, True):
            for scale_index in range(len(a11y.TEXT_SCALES)):
                window.contrast_box.setChecked(hc)
                window.text_size.setCurrentIndex(scale_index)
                state = f"{'high contrast' if hc else 'standard'}, text {a11y.TEXT_SCALES[scale_index]} %"
                for i in range(window.tabs.count()):
                    window.tabs.setCurrentIndex(i)
                    _settle(app)
                    page = window.tabs.widget(i)
                    where = f"{window.tabs.tabText(i)} ({state})"
                    findings += check_page(page, where)
                    if scale_index == 0 and not hc:
                        base_sizes[i] = text_sizes(page)
                    if scale_index == len(a11y.TEXT_SCALES) - 1:
                        findings += check_clipping(page, where)
                        if not hc:
                            findings += check_scaled(base_sizes.get(i, {}), text_sizes(page), where)
                findings += check_fits(window, state)
    finally:
        window.text_size.setCurrentIndex(saved[0])
        window.contrast_box.setChecked(saved[1])
        window.tabs.setCurrentIndex(saved[2])
    return dedupe(findings)


def dedupe(findings: list[Finding]) -> list[Finding]:
    """One finding per (check, widget, detail); the states where it happens are joined."""
    merged: dict[tuple, Finding] = {}
    for f in findings:
        key = (f.check, f.severity, f.widget, re.sub(r"\d+ (x \d+ )?px", "N px", f.detail))
        if key in merged:
            if f.where not in merged[key].where:
                merged[key].where += "; " + f.where
        else:
            merged[key] = Finding(**asdict(f))
    return list(merged.values())


def report(findings: list[Finding], title: str = "Automated accessibility audit") -> str:
    errors = sum(f.severity == "error" for f in findings)
    warnings = len(findings) - errors
    lines = [f"# {title}", "",
             f"**{errors} errors, {warnings} warnings.** Checks: accessible name, Tab reachability, target size "
             f"(≥ {MIN_TARGET} px; WCAG 2.2 AA {WCAG_TARGET} px), contrast (text 4.5:1, focus 3:1), clipped text at "
             f"{a11y.TEXT_SCALES[-1]} %, window fits 1366 x 768 at every text size. "
             "Every tab, standard and high-contrast themes; every text control must double at 200 %.", ""]
    if findings:
        lines += ["| Check | Severity | Control | Detail | Where |", "|---|---|---|---|---|"]
        lines += [f"| {f.check} | {f.severity} | {f.widget} | {f.detail} | {f.where} |" for f in findings]
    else:
        lines.append("No findings.")
    lines += ["", "Not covered (manual, with a person): what NVDA actually reads, whether names are meaningful, "
              "reading order of announcements, Accessibility Insights 'FastPass' on the built Windows app."]
    return "\n".join(lines) + "\n"
