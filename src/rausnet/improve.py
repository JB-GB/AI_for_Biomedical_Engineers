"""Continuous improvement of the global model from a shared workspace (Colab or any machine).

    python -m rausnet.improve run    --workspace /content/drive/MyDrive/RAUSNet
    python -m rausnet.improve status --workspace <path>          # JSON, for people and the GUI
    python -m rausnet.improve init   --workspace <path> [--models models --data-root J:/rausnet_data]

The workspace is one folder shared by every site, e.g. a Google Drive folder synced on the
hospital PCs with Google Drive for desktop::

    <workspace>/
      models/            global registry: the desktop apps serve its current model
      data/              data root: incoming/<batch>/manifest.csv, incoming/corrections/<date>/ ...
      data/datasets/     immutable dataset manifests, one per model version
      workspace.json     synthetic seed spec (regenerated, never uploaded)
      improve_log.jsonl  one line per run (no patient data)
      .improve.lock      held while a run is in progress

``run`` finds incoming batches with images the current model has not seen (clinician corrections
from the desktop app land in ``data/incoming/corrections``), retrains (fine-tune + promotion on
the frozen test split + calibration + ONNX export, see ``rausnet.retrain``) in a local staging copy
(fast disk, nothing half-written in the workspace) and publishes the result: dataset manifest,
then the version folder, then ``registry.json`` last, so readers always see a complete model.
An empty workspace bootstraps itself: synthetic seed data (stage S2) and a first model.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import tempfile
import time
from pathlib import Path

DEFAULT_CONFIG = "configs/retrain/default.yaml"
WORKSPACE_FILE = "workspace.json"
LOG_FILE = "improve_log.jsonl"
LOCK_FILE = ".improve.lock"
DEFAULT_SPEC = {"synthetic": {"dir": "synth", "patients": 200, "seed": 0}}


class WorkspaceLocked(RuntimeError):
    pass


class Workspace:
    def __init__(self, path: str | Path):
        self.root = Path(path)
        self.models = self.root / "models"
        self.data = self.root / "data"
        self.spec_path = self.root / WORKSPACE_FILE
        self.log_path = self.root / LOG_FILE
        self.lock_path = self.root / LOCK_FILE

    def ensure(self) -> Workspace:
        for d in (self.models, self.data / "incoming" / "corrections"):
            d.mkdir(parents=True, exist_ok=True)
        if not self.spec_path.exists():
            self.spec_path.write_text(json.dumps(DEFAULT_SPEC, indent=2), encoding="utf-8")
        return self

    @property
    def spec(self) -> dict:
        return json.loads(self.spec_path.read_text(encoding="utf-8")) if self.spec_path.exists() else DEFAULT_SPEC

    def log(self, entry: dict) -> None:
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    def history(self, n: int = 10) -> list[dict]:
        if not self.log_path.exists():
            return []
        lines = self.log_path.read_text(encoding="utf-8").splitlines()
        return [json.loads(x) for x in lines[-n:] if x.strip()]


class Lock:
    """Exclusive run lock in the workspace; a lock older than ``stale_hours`` is taken over
    (a Colab session that disconnected never releases it)."""

    def __init__(self, ws: Workspace, stale_hours: float = 6.0):
        self.path, self.stale = ws.lock_path, stale_hours * 3600

    def info(self) -> dict | None:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def __enter__(self):
        held = self.info()
        if held and time.time() - held.get("time", 0) < self.stale:
            raise WorkspaceLocked(f"another improvement run is in progress since {held.get('started')} "
                                  f"on {held.get('runtime')}; try again later")
        if held:
            self.path.unlink(missing_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise WorkspaceLocked("another improvement run just started") from exc
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"time": time.time(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "runtime": runtime_name()}, fh)
        return self

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)
        return False


def runtime_name() -> str:
    """Where the run happens, without personal data (no user or host names)."""
    if os.environ.get("COLAB_RELEASE_TAG") or importlib.util.find_spec("google.colab"):
        return "colab"
    if os.environ.get("GITHUB_ACTIONS"):
        return "github-runner"
    return f"local-{platform.system().lower()}"


def _sync(src: Path, dst: Path) -> int:
    """Copy files missing in ``dst`` or with a different size; return how many were copied."""
    n = 0
    for f in src.rglob("*"):
        if not f.is_file():
            continue
        t = dst / f.relative_to(src)
        if not t.exists() or t.stat().st_size != f.stat().st_size:
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, t)
            n += 1
    return n


def _publish_dir(src: Path, dst: Path) -> None:
    """Copy a folder next to its destination, then rename: readers never see it half-written."""
    tmp = dst.with_name(dst.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(src, tmp)
    if dst.exists():
        shutil.rmtree(dst)
    os.replace(tmp, dst)


def _current(models_dir: Path):
    from rausnet.registry import Registry

    return Registry(models_dir).current()


def pending_batches(data_root: Path, current_manifest: str | None) -> dict:
    """Incoming manifests holding images the current dataset does not contain.

    Returns ``{"ready": [(manifest, n_new)], "incomplete": [(manifest, reason)]}``. A batch whose
    images are not all present yet (a sync still in progress) is reported, not used.
    """
    from rausnet.data.manifest import read_manifest
    from rausnet.data.merge import load_source

    seen = {r.image_id for r in read_manifest(data_root / current_manifest)} if current_manifest else set()
    out = {"ready": [], "incomplete": []}
    incoming = data_root / "incoming"
    for m in sorted(incoming.rglob("manifest.csv")) if incoming.is_dir() else []:
        rel = m.relative_to(data_root).as_posix()
        try:
            n_new = sum(r.image_id not in seen for r in load_source(data_root, rel))
        except (FileNotFoundError, ValueError) as exc:
            out["incomplete"].append((rel, f"{type(exc).__name__}: {exc}"))
            continue
        if n_new:
            out["ready"].append((rel, n_new))
    return out


def status(workspace: str | Path) -> dict:
    ws = Workspace(workspace)
    cur = _current(ws.models) if (ws.models / "registry.json").exists() else None
    pend = pending_batches(ws.data, cur.dataset_manifest if cur else None) if ws.data.is_dir() else \
        {"ready": [], "incomplete": []}
    return {
        "workspace": str(ws.root),
        "current_model": None if cur is None else {"version": cur.version, "created": cur.created,
                                                   "test_metrics": cur.test_metrics, "decision": cur.decision},
        "pending_images": sum(n for _, n in pend["ready"]),
        "pending_batches": pend["ready"],
        "incomplete_batches": pend["incomplete"],
        "running": Lock(ws).info(),
        "history": ws.history(),
    }


def _ensure_synthetic(spec: dict, data_root: Path) -> str | None:
    """Regenerate the deterministic synthetic seed set in ``data_root`` if missing."""
    s = spec.get("synthetic")
    if not s:
        return None
    out = data_root / s["dir"]
    if not (out / "manifest.csv").exists():
        from rausnet.data.synthetic import main as synth

        synth(["--out", str(out), "--patients", str(s["patients"]), "--seed", str(s["seed"]),
               "--size", str(s.get("size", 256))])
    return f"{s['dir']}/manifest.csv"


def improve(workspace: str | Path, cfg: dict, min_new_images: int | None = None,
            staging: str | Path | None = None) -> dict:
    """One improvement cycle; returns (and logs) a summary. Never leaves a partial model behind."""
    ws = Workspace(workspace).ensure()
    opts = cfg.get("improve") or {}
    min_new = opts.get("min_new_images", 20) if min_new_images is None else min_new_images
    started = time.time()
    summary = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "runtime": runtime_name()}
    with Lock(ws, opts.get("lock_stale_hours", 6.0)):
        stage = Path(staging) if staging else Path(tempfile.mkdtemp(prefix="rausnet_stage_"))
        try:
            _cycle(ws, cfg, stage, min_new, summary)
        finally:
            if not staging:
                shutil.rmtree(stage, ignore_errors=True)
    summary["minutes"] = round((time.time() - started) / 60, 1)
    ws.log(summary)
    return summary


def _cycle(ws: Workspace, cfg: dict, stage: Path, min_new: int, summary: dict) -> None:
    from rausnet.retrain import retrain

    s_models, s_data = stage / "models", stage / "data"
    cur = _current(ws.models) if (ws.models / "registry.json").exists() else None
    summary["previous"] = cur.version if cur else None
    if cur is not None:  # cheap check on the workspace first: most runs have nothing to do
        n_ws = sum(n for _, n in pending_batches(ws.data, cur.dataset_manifest)["ready"])
        if n_ws < min_new:
            summary.update(new_images=n_ws, batches=[], action="nothing to do",
                           reason=f"{n_ws} new images (< {min_new})")
            return

    # stage: registry + current version, all workspace data, regenerated synthetic seed
    s_models.mkdir(parents=True, exist_ok=True)
    if cur:
        shutil.copy2(ws.models / "registry.json", s_models / "registry.json")
        _sync(ws.models / Path(cur.model_path).parent, s_models / Path(cur.model_path).parent)
    _sync(ws.data, s_data)
    seed = _ensure_synthetic(ws.spec, s_data)

    pend = pending_batches(s_data, cur.dataset_manifest if cur else None)
    add = [m for m, _ in pend["ready"]]
    n_new = sum(n for _, n in pend["ready"])
    if cur is None and seed and seed not in add:
        add.insert(0, seed)  # bootstrap: first model on the synthetic seed (+ any incoming data)
    summary.update(new_images=n_new, batches=add, incomplete=[m for m, _ in pend["incomplete"]])
    if cur is not None and n_new < min_new:
        summary.update(action="nothing to do", reason=f"{n_new} new images (< {min_new})")
    elif not add:
        summary.update(action="nothing to do", reason="no data")
    else:
        run_cfg = json.loads(json.dumps(cfg))
        run_cfg["models_dir"], run_cfg["output_dir"] = str(s_models), str(stage / "outputs")
        run_cfg["data"]["root"] = str(s_data)
        entry = retrain(run_cfg, add)
        # publish: dataset manifest -> version folder -> registry.json (readers switch last)
        new_ds = Path(entry.dataset_manifest).parent
        _publish_dir(s_data / new_ds, ws.data / new_ds)
        _publish_dir(s_models / entry.version, ws.models / entry.version)
        tmp = ws.models / "registry.json.partial"
        shutil.copy2(s_models / "registry.json", tmp)
        os.replace(tmp, ws.models / "registry.json")
        report = json.loads((s_models / entry.version / "retrain_report.json").read_text(encoding="utf-8"))
        pp = report.get("postprocess")
        if isinstance(pp, dict):  # staging paths vanish: keep file names relative to the version folder
            pp = {k: Path(v).name if k in ("calibration", "onnx", "sidecar") else v for k, v in pp.items()}
        summary.update(action="retrained", version=entry.version, promoted=entry.status == "current",
                       reason=report.get("reason"), test_qwk=entry.test_metrics.get("qwk"), postprocess=pp)


def init(workspace: str | Path, models: str | Path | None = None, data_root: str | Path | None = None) -> dict:
    """Create the workspace; optionally import a local registry (current version) and its data.

    Synthetic images are not copied (regenerated from ``workspace.json``); incoming batches and
    the current dataset manifest are.
    """
    ws = Workspace(workspace).ensure()
    copied = {"models": 0, "data": 0}
    if models:
        cur = _current(Path(models))
        if cur is None:
            raise ValueError(f"no current model in {models}")
        if (ws.models / "registry.json").exists():
            raise FileExistsError(f"{ws.models} already has a registry; not overwriting the global model")
        if not data_root:
            raise ValueError("--data-root is needed to import the dataset of the current model")
        root = Path(data_root)
        copied["data"] += _sync(root / Path(cur.dataset_manifest).parent, ws.data / Path(cur.dataset_manifest).parent)
        if (root / "incoming").is_dir():
            copied["data"] += _sync(root / "incoming", ws.data / "incoming")
        copied["models"] = _sync(Path(models) / cur.version, ws.models / cur.version)
        shutil.copy2(Path(models) / "registry.json", ws.models / "registry.json")
    return {"workspace": str(ws.root), "copied_files": copied}


def main(argv=None) -> None:
    from rausnet.train import load_config

    parser = argparse.ArgumentParser(description="Continuous improvement of the global model.")
    parser.add_argument("command", choices=["run", "status", "init"])
    parser.add_argument("--workspace", default=os.environ.get("RAUSNET_WORKSPACE"),
                        help="shared workspace folder (default: $RAUSNET_WORKSPACE)")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--min-new", type=int, default=None, help="minimum new images to retrain")
    parser.add_argument("--staging", default=None, help="local working folder (default: temporary)")
    parser.add_argument("--models", default=None, help="init: local registry to import")
    parser.add_argument("--data-root", default=None, help="init: local data root to import from")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    if not args.workspace:
        parser.error("pass --workspace or set RAUSNET_WORKSPACE")
    if args.command == "status":
        print(json.dumps(status(args.workspace), indent=2))
    elif args.command == "init":
        print(json.dumps(init(args.workspace, args.models, args.data_root), indent=2))
    else:
        os.environ.setdefault("RAUSNET_DATA_DIR", str(Path(args.workspace) / "data"))
        cfg = load_config(args.config, overrides=args.set)
        print(json.dumps(improve(args.workspace, cfg, args.min_new, args.staging), indent=2))


if __name__ == "__main__":
    main()
