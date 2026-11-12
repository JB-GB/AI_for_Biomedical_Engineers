"""Stable patient-level train/val/test split.

Each patient is assigned by hashing ``salt + patient_id``. The assignment never
depends on the rest of the dataset, so when new data arrives existing patients
keep their split and the test set stays frozen across retraining rounds.
All images of one patient always land in the same split (no leakage).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from rausnet.data.manifest import Record

SPLITS = ("train", "val", "test")
DEFAULT_SALT = "rausnet-split-v1"


def assign_split(
    patient_id: str,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    salt: str = DEFAULT_SALT,
) -> str:
    """Return 'train', 'val' or 'test' for a patient, deterministically."""
    if len(ratios) != 3 or any(r < 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must be three non-negative values summing to 1, got {ratios}")
    digest = hashlib.sha256(f"{salt}:{patient_id}".encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64
    if u < ratios[0]:
        return "train"
    if u < ratios[0] + ratios[1]:
        return "val"
    return "test"


def split_records(
    records: Iterable[Record],
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    salt: str = DEFAULT_SALT,
) -> dict[str, list[Record]]:
    """Group records by split using their patient id."""
    out: dict[str, list[Record]] = {name: [] for name in SPLITS}
    for record in records:
        out[assign_split(record.patient_id, ratios, salt)].append(record)
    return out
