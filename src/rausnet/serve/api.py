"""HTTP inference API (FastAPI + ONNX Runtime, no PyTorch).

    rausnet-api [--model-dir models/v0003] [--host 127.0.0.1] [--port 8000]

Without ``--model-dir`` the current model of the registry in ``models/`` is served
(it must have been exported with ``python -m rausnet.export``).

Endpoints:
    GET  /health       liveness
    GET  /model        model sidecar (version, stage, validation status, metrics)
    POST /predict      multipart: gs (required), pd (optional) -> grades and probabilities;
                       ?explain=true adds Grad-CAM overlays as base64 PNG data URLs
                       (&palette=cividis for a colour-blind-safe ramp; default warm)
    POST /corrections  multipart: gs, pd, patient_id, joint, side, gs_grade, pd_grade
                       -> stored in <data_dir>/incoming/corrections/<date>/ for the next retrain
    DELETE /corrections/{image_id}  undo a correction (files + manifest row) while no dataset uses it
    GET  /workspace/status  shared workspace: global model, images waiting, run in progress, history
                       (``--workspace`` only) plus the version this API serves
    POST /model/reload switch to the registry's current model if it changed (between requests)
    GET  /model/versions   registered versions of the global model, newest first (``--workspace``)
    POST /model/rollback   JSON {version, reason}: serve an archived version again, logged (``--workspace``)
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import os
import re
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from rausnet import improve
from rausnet.data.manifest import FIELDNAMES, Record
from rausnet.registry import Registry
from rausnet.serve.inference import Predictor

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
PSEUDONYM = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
CORRECTION_ID = re.compile(r"^corr-[0-9a-f]{12}$")


class RollbackRequest(BaseModel):
    version: str = Field(pattern=r"^v[0-9]{4}$")
    reason: str = Field("", max_length=200)


def _ids_in(manifest: Path) -> set[str]:
    with open(manifest, newline="", encoding="utf-8") as fh:
        return {row.get("image_id", "") for row in csv.DictReader(fh)}


def resolve_model_dir(models_dir: str | Path = "models") -> Path:
    """Folder of the registry's current model; it must contain model.onnx and model.json."""
    reg = Registry(models_dir)
    cur = reg.current()
    if cur is None:
        raise FileNotFoundError(f"no current model in {reg.path}")
    folder = (reg.dir / cur.model_path).parent
    if not (folder / "model.onnx").exists():
        raise FileNotFoundError(f"{folder} has no model.onnx; run: python -m rausnet.export")
    return folder


async def _read_image(upload: UploadFile, field: str) -> Image.Image:
    data = await upload.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"{field}: file larger than {MAX_UPLOAD_BYTES // 2**20} MB")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(422, f"{field}: not a readable image") from exc
    return img


def _png_data_url(rgb) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def create_app(model_dir: str | Path, data_dir: str | Path | None = None, models_dir: str | Path | None = None,
               workspace: str | Path | None = None, min_new_images: int = improve.DEFAULT_MIN_NEW) -> FastAPI:
    """``models_dir`` (registry) enables ``/model/reload``; ``workspace`` enables ``/workspace/status``."""
    state = {"predictor": Predictor(model_dir), "dir": Path(model_dir).resolve()}
    lock, swap = threading.Lock(), threading.Lock()
    app = FastAPI(title="RAUSNet", version=state["predictor"].info.get("rausnet_version", "0"),
                  description="EULAR-OMERACT synovitis grading. Research use only, not for diagnosis.")

    @app.get("/health")
    def health():
        return {"status": "ok", "model_version": state["predictor"].info["version"]}

    @app.get("/model")
    def model_info():
        return state["predictor"].info

    def _swap_to_current() -> bool:
        """Load the registry's current model and switch to it; the caller holds ``swap``."""
        try:
            folder = resolve_model_dir(models_dir).resolve()
        except FileNotFoundError as exc:
            raise HTTPException(409, str(exc)) from exc
        if folder == state["dir"]:
            return False
        try:
            predictor = Predictor(folder)  # load fully before switching: never a half model
        except Exception as exc:  # keep serving the previous model
            raise HTTPException(409, f"new model could not be loaded: {exc}") from exc
        state.update(predictor=predictor, dir=folder)
        return True

    @app.post("/model/reload")
    def reload_model():
        if models_dir is None:
            raise HTTPException(503, "reload is disabled: the API serves a fixed --model-dir")
        with swap:
            changed = _swap_to_current()
        return {"changed": changed, "version": state["predictor"].info["version"]}

    @app.get("/model/versions")
    def model_versions():
        if workspace is None:
            raise HTTPException(503, "no shared workspace: start the API with --workspace")
        return {"served_version": state["predictor"].info["version"], "versions": improve.versions(workspace)}

    @app.post("/model/rollback")
    def model_rollback(req: RollbackRequest):
        if workspace is None or models_dir is None:
            raise HTTPException(503, "no shared workspace: start the API with --workspace")
        with swap:
            reg = Registry(models_dir)
            try:
                folder = (reg.dir / reg.get(req.version).model_path).parent.resolve()
            except KeyError as exc:
                raise HTTPException(404, f"unknown version {req.version}") from exc
            try:  # load before the registry changes: a broken old version changes nothing
                predictor = Predictor(folder)
            except Exception as exc:
                raise HTTPException(409, f"{req.version} could not be loaded: {exc}") from exc
            try:
                entry = improve.rollback(workspace, req.version, req.reason)
            except (ValueError, improve.WorkspaceLocked) as exc:
                raise HTTPException(409, str(exc)) from exc
            state.update(predictor=predictor, dir=folder)
        return {**entry, "served_version": state["predictor"].info["version"]}

    @app.get("/workspace/status")
    def workspace_status():
        if workspace is None:
            raise HTTPException(503, "no shared workspace: start the API with --workspace")
        out = improve.status(workspace, min_new_images)
        out["served_version"] = state["predictor"].info["version"]
        cur = out["current_model"]
        out["new_model"] = bool(cur and cur["version"] != out["served_version"])
        return out

    @app.post("/predict")
    async def predict(gs: UploadFile = File(...), pd: UploadFile | None = File(None), explain: bool = False,
                      palette: str = "warm"):
        gs_img = await _read_image(gs, "gs")
        pd_img = await _read_image(pd, "pd") if pd is not None else None
        try:
            result = state["predictor"].predict(gs_img, pd_img, explain=explain, palette=palette)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        if "heatmaps" in result:
            result["heatmaps"] = {k: _png_data_url(v) for k, v in result["heatmaps"].items()}
        return result

    @app.post("/corrections")
    async def corrections(
        gs: UploadFile = File(...), pd: UploadFile | None = File(None),
        patient_id: str = Form(...), joint: str = Form(...), side: str = Form(...),
        gs_grade: int = Form(...), pd_grade: int = Form(...), device: str = Form(""),
    ):
        if data_dir is None:
            raise HTTPException(503, "corrections are disabled: start the API with --data-dir")
        if not PSEUDONYM.match(patient_id):
            raise HTTPException(422, "patient_id must be a pseudonym (letters, digits, '-', '_'; no names)")
        if pd is None and pd_grade != 0:
            raise HTTPException(422, "pd_grade requires a PD image")
        gs_img = await _read_image(gs, "gs")
        pd_img = await _read_image(pd, "pd") if pd is not None else None
        image_id = f"corr-{uuid.uuid4().hex[:12]}"
        try:
            record = Record(image_id, patient_id, joint, side, f"gs/{image_id}.png",
                            f"pd/{image_id}.png" if pd_img else "", gs_grade, pd_grade, "hospital", device)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        folder = Path(data_dir) / "incoming" / "corrections" / time.strftime("%Y%m%d")
        with lock:
            (folder / "gs").mkdir(parents=True, exist_ok=True)
            # Re-encoding as PNG drops embedded metadata (EXIF/text chunks).
            gs_img.convert("L").save(folder / record.gs_path)
            if pd_img is not None:
                (folder / "pd").mkdir(exist_ok=True)
                pd_img.convert("RGB").save(folder / record.pd_path)
            manifest = folder / "manifest.csv"
            new = not manifest.exists()
            with open(manifest, "a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
                if new:
                    writer.writeheader()
                writer.writerow(asdict(record))
        return {"image_id": image_id, "manifest": manifest.relative_to(data_dir).as_posix()}

    @app.delete("/corrections/{image_id}")
    def undo_correction(image_id: str):
        if data_dir is None:
            raise HTTPException(503, "corrections are disabled: start the API with --data-dir")
        if not CORRECTION_ID.match(image_id):
            raise HTTPException(422, "not a correction id")
        root = Path(data_dir)
        with lock:
            # a correction already used to train a model is part of that dataset: never remove it
            for ds in sorted((root / "datasets").glob("*/manifest.csv")):
                if image_id in _ids_in(ds):
                    raise HTTPException(409, f"{image_id} is already part of {ds.parent.name}; it cannot be undone")
            for manifest in sorted((root / "incoming" / "corrections").glob("*/manifest.csv")):
                with open(manifest, newline="", encoding="utf-8") as fh:
                    reader = csv.DictReader(fh)
                    fields, rows = reader.fieldnames or FIELDNAMES, list(reader)
                keep = [r for r in rows if r.get("image_id") != image_id]
                if len(keep) == len(rows):
                    continue
                gone = next(r for r in rows if r.get("image_id") == image_id)
                tmp = manifest.with_suffix(".tmp")
                with open(tmp, "w", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(keep)
                os.replace(tmp, manifest)  # manifest first: a sync never sees a row without its image
                for rel in (gone.get("gs_path"), gone.get("pd_path")):
                    if rel:
                        (manifest.parent / rel).unlink(missing_ok=True)
                if not keep:
                    manifest.unlink()
                return {"removed": image_id}
        raise HTTPException(404, f"{image_id} not found")

    return app


def main(argv=None) -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Serve the RAUSNet inference API.")
    parser.add_argument("--model-dir", default=None, help="folder with model.onnx + model.json")
    parser.add_argument("--models-dir", default="models", help="registry folder (used if --model-dir is absent)")
    parser.add_argument("--data-dir", default=os.environ.get("RAUSNET_DATA_DIR"),
                        help="data root for clinician corrections (default: $RAUSNET_DATA_DIR)")
    parser.add_argument("--workspace", default=os.environ.get("RAUSNET_WORKSPACE"),
                        help="shared workspace (global model + corrections); sets both folders below")
    parser.add_argument("--min-new", type=int, default=improve.DEFAULT_MIN_NEW,
                        help="images needed before an improvement run (shown in /workspace/status)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    if args.workspace and not args.model_dir:
        args.models_dir, args.data_dir = str(Path(args.workspace) / "models"), str(Path(args.workspace) / "data")
    model_dir = Path(args.model_dir) if args.model_dir else resolve_model_dir(args.models_dir)
    app = create_app(model_dir, args.data_dir, models_dir=None if args.model_dir else args.models_dir,
                     workspace=None if args.model_dir else args.workspace, min_new_images=args.min_new)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
