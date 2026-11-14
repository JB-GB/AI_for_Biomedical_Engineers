"""HTTP inference API (FastAPI + ONNX Runtime, no PyTorch).

    rausnet-api [--model-dir models/v0003] [--host 127.0.0.1] [--port 8000]

Without ``--model-dir`` the current model of the registry in ``models/`` is served
(it must have been exported with ``python -m rausnet.export``).

Endpoints:
    GET  /health       liveness
    GET  /model        model sidecar (version, stage, validation status, metrics)
    POST /predict      multipart: gs (required), pd (optional) -> grades and probabilities;
                       ?explain=true adds Grad-CAM overlays as base64 PNG data URLs
    POST /corrections  multipart: gs, pd, patient_id, joint, side, gs_grade, pd_grade
                       -> stored in <data_dir>/incoming/corrections/<date>/ for the next retrain
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

from rausnet.data.manifest import FIELDNAMES, Record
from rausnet.registry import Registry
from rausnet.serve.inference import Predictor

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
PSEUDONYM = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


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


def create_app(model_dir: str | Path, data_dir: str | Path | None = None) -> FastAPI:
    predictor = Predictor(model_dir)
    lock = threading.Lock()
    app = FastAPI(title="RAUSNet", version=predictor.info.get("rausnet_version", "0"),
                  description="EULAR-OMERACT synovitis grading. Research use only, not for diagnosis.")

    @app.get("/health")
    def health():
        return {"status": "ok", "model_version": predictor.info["version"]}

    @app.get("/model")
    def model_info():
        return predictor.info

    @app.post("/predict")
    async def predict(gs: UploadFile = File(...), pd: UploadFile | None = File(None), explain: bool = False):
        gs_img = await _read_image(gs, "gs")
        pd_img = await _read_image(pd, "pd") if pd is not None else None
        try:
            result = predictor.predict(gs_img, pd_img, explain=explain)
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
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    if args.workspace and not args.model_dir:
        args.models_dir, args.data_dir = str(Path(args.workspace) / "models"), str(Path(args.workspace) / "data")
    model_dir = Path(args.model_dir) if args.model_dir else resolve_model_dir(args.models_dir)
    uvicorn.run(create_app(model_dir, args.data_dir), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
