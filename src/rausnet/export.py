"""Export a registry model to ONNX plus a JSON sidecar (the training/serving contract).

    python -m rausnet.export [--models-dir models] [--version v0003]

Writes ``<models_dir>/<version>/model.onnx`` and ``model.json``. The graph outputs
logits and exact Grad-CAM maps for both heads (see ``rausnet.explain``). If
``calibration.json`` exists (``python -m rausnet.calibrate``) its temperatures are
copied into the sidecar. The export is checked against PyTorch on random inputs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import rausnet
from rausnet.data.dataset import MEAN, STD
from rausnet.evaluate import load_model
from rausnet.explain import CamNet
from rausnet.registry import Registry

SIDECAR_FORMAT = 1
OPSET = 17


def export_onnx(checkpoint: str | Path, out_path: str | Path, image_size: int) -> float:
    """Export ``checkpoint`` (logits + CAMs) to ONNX with a dynamic batch axis.

    Returns the max |torch - onnx| difference over all outputs.
    """
    import onnxruntime as ort

    net, _ = load_model(checkpoint, torch.device("cpu"))
    model = CamNet(net).eval()
    dummy = torch.randn(2, 4, image_size, image_size)
    names = ["gs_logits", "pd_logits", "gs_cam", "pd_cam"]
    torch.onnx.export(
        model, (dummy,), str(out_path), input_names=["input"], output_names=names,
        dynamic_axes={"input": {0: "batch"}, **{n: {0: "batch"} for n in names}},
        opset_version=OPSET, dynamo=False,
    )
    check = torch.randn(3, 4, image_size, image_size, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref = [t.numpy() for t in model(check)]
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"input": check.numpy()})
    return float(max(np.abs(r - g).max() for r, g in zip(ref, got, strict=True)))


def export_version(models_dir: str | Path = "models", version: str | None = None) -> Path:
    reg = Registry(models_dir)
    entry = reg.get(version) if version else reg.current()
    if entry is None:
        raise ValueError(f"no current model in {reg.path}")
    ckpt_path = reg.dir / entry.model_path
    cfg = torch.load(ckpt_path, map_location="cpu", weights_only=False)["config"]
    size = cfg["data"]["image_size"]
    out = ckpt_path.parent / "model.onnx"
    diff = export_onnx(ckpt_path, out, size)
    if diff > 1e-3:
        raise RuntimeError(f"ONNX output differs from PyTorch by {diff:.2e}")
    calib_path = out.parent / "calibration.json"
    calib = json.loads(calib_path.read_text()) if calib_path.exists() else None
    sidecar = {
        "format_version": SIDECAR_FORMAT,
        "rausnet_version": rausnet.__version__,
        "version": entry.version,
        "status_at_export": entry.status,
        "exported": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stage": entry.stage,
        "clinically_validated": False,
        "intended_use": "Research use only. Not a medical device. Not for diagnosis.",
        "backbone": cfg["model"]["backbone"],
        "input": {
            "name": "input", "channels": ["gs", "pd_r", "pd_g", "pd_b"], "size": size,
            "mean": MEAN, "std": STD, "resize": "bilinear, align_corners=false, no antialias",
            "missing_pd": "replicate GS into PD channels",
        },
        "outputs": {"gs_logits": "N x 4 (EULAR-OMERACT GS 0-3)", "pd_logits": "N x 4 (PD 0-3)",
                    "gs_cam": "N x 4 x h x w", "pd_cam": "N x 4 x h x w"},
        "cam": {"method": "exact Grad-CAM (global average pooling + linear head)", "relu": True},
        "temperature": calib["temperature"] if calib else {"gs": 1.0, "pd": 1.0},
        "calibration": None if calib is None else {
            "method": calib["method"],
            "test_combined_ece": [calib["test"][k]["combined"]["ece"] for k in ("before", "after")],
        },
        "combined_rule": "0 if GS == 0 else max(GS, PD)",
        "joints": ["MCP2", "MCP3", "WRIST"],
        "dataset_sha256": entry.dataset_sha256,
        "test_metrics": entry.test_metrics,
        "onnx_opset": OPSET,
        "parity_max_abs_diff": diff,
    }
    (out.parent / "model.json").write_text(json.dumps(sidecar, indent=2))
    print(f"exported {entry.version} -> {out} (max |diff| {diff:.2e}, calibrated: {calib is not None})")
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Export a registry model to ONNX.")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--version", default=None, help="default: current model")
    args = parser.parse_args(argv)
    export_version(args.models_dir, args.version)


if __name__ == "__main__":
    main()
