"""PDF report of one analysis, built with Qt only (QTextDocument HTML -> QPdfWriter).

``report_html`` is pure: it turns the ``/predict`` result dict the desktop app receives
into HTML, with input images and Grad-CAM overlays embedded as data URLs.
``write_pdf`` renders that HTML to a PDF file.
"""

from __future__ import annotations

import base64
import html
from datetime import datetime
from pathlib import Path

THUMB = 170
MIME = {".jpg": "jpeg", ".jpeg": "jpeg", ".bmp": "bmp", ".tif": "tiff", ".tiff": "tiff"}


def _file_data_url(path: str | Path) -> str:
    p = Path(path)
    mime = MIME.get(p.suffix.lower(), "png")
    return f"data:image/{mime};base64," + base64.b64encode(p.read_bytes()).decode("ascii")


def _img(src: str, caption: str) -> str:
    return (f'<td align="center"><img src="{html.escape(src, quote=True)}" width="{THUMB}"><br>'
            f"<small>{html.escape(caption)}</small></td>")


def report_html(result: dict, t: dict, gs_path: str | Path | None = None, pd_path: str | Path | None = None,
                fields: dict[str, str] | None = None, when: datetime | None = None) -> str:
    """Build the report HTML.

    ``t`` is the app's string table for one language (``STRINGS[lang]``); ``fields`` maps
    already-translated labels to user-entered values (empty values are skipped)."""
    e = html.escape
    when = when or datetime.now()
    c = result["combined"]
    rows = [(t["r_date"], when.strftime("%Y-%m-%d %H:%M")),
            (t["r_model"], f"{result.get('model_version', '?')} ({result.get('stage', '?')})")]
    rows += [(k, v) for k, v in (fields or {}).items() if v]
    meta = "".join(f"<tr><td><b>{e(k)}</b></td><td>{e(str(v))}</td></tr>" for k, v in rows)

    def conf(part: dict) -> str:
        return f"{max(part['probabilities']):.0%}"

    grades = (
        f"<tr><th align='left'>{e(t['r_item'])}</th><th>{e(t['r_grade'])}</th><th>{e(t['confidence'])}</th></tr>"
        f"<tr><td><b>{e(t['combined'])}</b></td><td align='center'><b>{c['grade']}</b></td>"
        f"<td align='center'>{c['confidence']:.0%}</td></tr>"
        f"<tr><td>{e(t['gs_grade'])}</td><td align='center'>{result['gs']['grade']}</td>"
        f"<td align='center'>{conf(result['gs'])}</td></tr>"
        f"<tr><td>{e(t['pd_grade'])}</td><td align='center'>{result['pd']['grade']}</td>"
        f"<td align='center'>{conf(result['pd'])}</td></tr>"
    )
    probs = " · ".join(f"{g}: {p:.0%}" for g, p in enumerate(c["probabilities"]))

    refer = ""
    if result.get("refer"):
        thr = result.get("referral_threshold")
        extra = f" ({t['confidence']} {c['confidence']:.0%} &lt; {thr:.0%})" if thr is not None else ""
        refer = (f"<p style='color:#b3261e; font-size:14pt;'><b>⚠ {e(t['r_refer'])}</b>{extra}</p>")
    codes = result.get("warning_codes") or []
    warnings = result.get("warnings", [])
    # the referral is shown prominently above, so it is not repeated in the list
    texts = [t.get(f"w_{k}", w) for k, w in zip(codes, warnings, strict=False)
             if not (k == "refer_specialist" and refer)] if codes else warnings
    warns = "".join(f"<li>{e(w)}</li>" for w in texts)

    rows_img = []  # one row per modality: input image, then its Grad-CAM overlay
    maps = result.get("heatmaps") or {}
    for key, path, label in (("gs", gs_path, t["gs_image"]), ("pd", pd_path, t["pd_image"])):
        if path:
            cells = _img(_file_data_url(path), label)
            if maps.get(key):
                cells += _img(maps[key], f"{label} — Grad-CAM")
            rows_img.append(f"<tr>{cells}</tr>")
    images = (f"<h3>{e(t['r_images'])}</h3><table cellpadding='4'>{''.join(rows_img)}</table>"
              if rows_img else "")

    return (
        "<html><body style='font-family:sans-serif; font-size:10pt;'>"
        f"<h2>{e(t['r_title'])}</h2><p style='color:#b3261e;'>{e(t['r_disclaimer'])}</p>"
        f"<table cellpadding='3'>{meta}</table>"
        f"<h3>{e(t['result'])}</h3>"
        f"<table border='1' cellspacing='0' cellpadding='5'>{grades}</table>"
        f"<p>{e(t['probabilities'])}: {probs}</p>"
        f"{refer}"
        + (f"<ul style='color:#8a5300;'>{warns}</ul>" if warns else "")
        + images
        + f"<hr><p style='color:#b3261e;'><b>{e(t['r_disclaimer'])}</b></p>"
        "</body></html>"
    )


def write_pdf(html_text: str, path: str | Path) -> Path:
    """Render ``html_text`` to an A4 PDF at ``path`` (needs a QGuiApplication)."""
    from PySide6.QtCore import QMarginsF
    from PySide6.QtGui import QPageLayout, QPageSize, QPdfWriter, QTextDocument

    path = Path(path)
    writer = QPdfWriter(str(path))
    writer.setPageLayout(QPageLayout(QPageSize(QPageSize.PageSizeId.A4), QPageLayout.Orientation.Portrait,
                                     QMarginsF(15, 15, 15, 15), QPageLayout.Unit.Millimeter))
    writer.setResolution(150)
    doc = QTextDocument()
    doc.setHtml(html_text)
    doc.print_(writer)
    del writer  # flush and close the file
    if not path.is_file() or path.stat().st_size == 0:
        raise OSError(f"could not write PDF: {path}")
    return path
