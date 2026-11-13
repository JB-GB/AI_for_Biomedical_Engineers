"""Model registry: versioned models with provenance and promotion status.

``<models_dir>/registry.json`` lists every trained version. Exactly one version
is ``current`` (the one served); replaced versions become ``archived`` and
candidates that failed the promotion rule stay as ``rejected`` for audit.

Usage: python -m rausnet.registry list [--models-dir models]
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

STATUSES = ("current", "archived", "rejected")


@dataclass
class ModelEntry:
    version: str
    created: str
    model_path: str  # relative to models_dir
    dataset_manifest: str  # relative to the data root
    dataset_sha256: str
    stage: str
    parent: str | None
    status: str = "rejected"
    test_metrics: dict = field(default_factory=dict)
    decision: str = ""


class Registry:
    def __init__(self, models_dir: str | Path):
        self.dir = Path(models_dir)
        self.path = self.dir / "registry.json"
        self.entries: list[ModelEntry] = []
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.entries = [ModelEntry(**e) for e in data["models"]]
        if sum(e.status == "current" for e in self.entries) > 1:
            raise ValueError(f"{self.path}: more than one current model")

    def current(self) -> ModelEntry | None:
        return next((e for e in self.entries if e.status == "current"), None)

    def get(self, version: str) -> ModelEntry:
        for e in self.entries:
            if e.version == version:
                return e
        raise KeyError(version)

    def next_version(self) -> str:
        return f"v{len(self.entries) + 1:04d}"

    def add(self, entry: ModelEntry, promote: bool) -> None:
        if any(e.version == entry.version for e in self.entries):
            raise ValueError(f"version {entry.version} already registered")
        if promote:
            for e in self.entries:
                if e.status == "current":
                    e.status = "archived"
        entry.status = "current" if promote else "rejected"
        self.entries.append(entry)
        self.save()

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"models": [asdict(e) for e in self.entries]}, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Inspect the model registry.")
    parser.add_argument("command", choices=["list"])
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args(argv)
    reg = Registry(args.models_dir)
    if not reg.entries:
        print(f"no models registered in {reg.path}")
        return
    for e in reg.entries:
        qwk = e.test_metrics.get("qwk")
        qwk_txt = f"{qwk:.3f}" if isinstance(qwk, float) else "-"
        print(f"{e.version}  {e.status:8s}  stage={e.stage}  test_qwk={qwk_txt}  parent={e.parent}  {e.created}")


if __name__ == "__main__":
    main()
