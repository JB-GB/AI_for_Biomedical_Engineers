"""Export a registry model to ONNX plus a JSON sidecar (the training/serving contract).

    python -m rausnet.export [--models-dir models] [--version v0003]

Writes ``<models_dir>/<version>/model.onnx`` and ``model.json``. The export is
checked against PyTorch on random inputs; the max absolute logit difference is
stored in the sidecar.
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
from rausnet.registry import Registry

SIDECAR_FORMAT = 1
OPSET = 17


def export_onnx(checkpoint: str | Path, out_path: str | Path, image_size: int) -> float:
    """Export ``checkpoint`` to ONNX with a dynamic batch axis; return max |torch - onnx| logit diff."""
    import onnxruntime as ort

    model, _ = load_model(checkpoint, torch.device("cpu"))
    model.eval()
    dummy = torch.randn(2, 4, image_size, image_size)
    torch.onnx.export(
        model, (dummy,), str(out_path), input_names=["input"], output_names=["gs_logits", "pd_logits"],
        dynamic_axes={"input": {0: "batch"}, "gs_logits": {0: "batch"}, "pd_logits": {0: "batch"}},
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
        "outputs": {"gs_logits": "N x 4 (EULAR-OMERACT GS 0-3)", "pd_logits": "N x 4 (PD 0-3)"},
        "combined_rule": "0 if GS == 0 else max(GS, PD)",
        "joints": ["MCP2", "MCP3", "WRIST"],
        "dataset_sha256": entry.dataset_sha256,
        "test_metrics": entry.test_metrics,
        "onnx_opset": OPSET,
        "parity_max_abs_diff": diff,
    }
    (out.parent / "model.json").write_text(json.dumps(sidecar, indent=2))
    print(f"exported {entry.version} -> {out} (max |diff| {diff:.2e})")
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Export a registry model to ONNX.")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--version", default=None, help="default: current model")
    args = parser.parse_args(argv)
    export_version(args.models_dir, args.version)


if __name__ == "__main__":
    main()
