"""Dataset manifest: one CSV row per joint image pair (GS + optional PD).

The manifest is the single entry point for any data source (synthetic, public,
hospital). New hospital data only needs a manifest in this format to be used
for retraining. Paths are relative to the data root and never point inside the repo.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from rausnet.grading import combined_score, validate_grade

JOINTS = ("MCP2", "MCP3", "WRIST")
SIDES = ("L", "R")
SOURCES = ("synthetic", "public", "hospital")


@dataclass(frozen=True)
class Record:
    image_id: str
    patient_id: str
    joint: str
    side: str
    gs_path: str
    pd_path: str
    gs_grade: int
    pd_grade: int
    source: str
    device: str = ""

    def __post_init__(self) -> None:
        if not self.image_id or not self.patient_id:
            raise ValueError("image_id and patient_id are required")
        if self.joint not in JOINTS:
            raise ValueError(f"joint must be one of {JOINTS}, got {self.joint!r}")
        if self.side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {self.side!r}")
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got {self.source!r}")
        if not self.gs_path:
            raise ValueError("gs_path is required")
        for path in (self.gs_path, self.pd_path):
            if path and Path(path).is_absolute():
                raise ValueError(f"paths must be relative to the data root: {path!r}")
        validate_grade(self.gs_grade, "gs_grade")
        validate_grade(self.pd_grade, "pd_grade")
        combined_score(self.gs_grade, self.pd_grade)

    @property
    def combined_grade(self) -> int:
        return combined_score(self.gs_grade, self.pd_grade)


FIELDNAMES = [f.name for f in fields(Record)]
_INT_FIELDS = {"gs_grade", "pd_grade"}


def read_manifest(path: str | Path) -> list[Record]:
    """Read and validate a manifest CSV. Raises ValueError with the offending line."""
    records: list[Record] = []
    seen: set[str] = set()
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = set(FIELDNAMES) - set(reader.fieldnames or []) - {"device"}
        if missing:
            raise ValueError(f"manifest is missing columns: {sorted(missing)}")
        for line, row in enumerate(reader, start=2):
            try:
                values = {k: (row.get(k) or "").strip() for k in FIELDNAMES}
                for key in _INT_FIELDS:
                    values[key] = int(values[key])
                record = Record(**values)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line}: {exc}") from exc
            if record.image_id in seen:
                raise ValueError(f"{path}:{line}: duplicate image_id {record.image_id!r}")
            seen.add(record.image_id)
            records.append(record)
    return records


def write_manifest(records: list[Record], path: str | Path) -> None:
    """Write records to a manifest CSV."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
