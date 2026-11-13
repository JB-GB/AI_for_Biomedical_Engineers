"""Merge manifests from several sources into one dataset version.

Each source manifest stores paths relative to its own folder. Merged manifests
store paths relative to the common data root, so one dataset version can mix
synthetic, public and hospital images.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path, PurePosixPath

from rausnet.data.manifest import Record, read_manifest, write_manifest


def load_source(data_root: str | Path, manifest: str | Path) -> list[Record]:
    """Read a source manifest and rebase its paths onto ``data_root``."""
    root = Path(data_root).resolve()
    path = Path(manifest)
    path = (path if path.is_absolute() else root / path).resolve()
    try:
        rel = PurePosixPath(path.parent.relative_to(root).as_posix())
    except ValueError as exc:
        raise ValueError(f"manifest {path} is outside the data root {root}") from exc
    out = []
    for r in read_manifest(path):
        gs = (rel / r.gs_path).as_posix()
        pd = (rel / r.pd_path).as_posix() if r.pd_path else ""
        for p in (gs, pd):
            if p and not (root / p).is_file():
                raise FileNotFoundError(f"{path}: image not found: {p}")
        out.append(replace(r, gs_path=gs, pd_path=pd))
    return out


def merge_records(*groups: list[Record]) -> list[Record]:
    """Concatenate record groups; identical duplicates are dropped, conflicting ones rejected."""
    merged: dict[str, Record] = {}
    for group in groups:
        for r in group:
            prev = merged.get(r.image_id)
            if prev is None:
                merged[r.image_id] = r
            elif prev != r:
                raise ValueError(f"image_id {r.image_id!r} appears with different content")
    return list(merged.values())


def write_dataset(data_root: str | Path, name: str, records: list[Record]) -> str:
    """Write ``datasets/<name>/manifest.csv`` under the data root; return its relative path."""
    rel = f"datasets/{name}/manifest.csv"
    path = Path(data_root) / rel
    if path.exists():
        raise FileExistsError(f"dataset version already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=False)
    write_manifest(records, path)
    return rel
