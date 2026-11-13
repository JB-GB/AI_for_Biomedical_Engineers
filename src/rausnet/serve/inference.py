"""ONNX Runtime inference without PyTorch.

Preprocessing reproduces ``rausnet.data.dataset.load_input`` exactly (verified by
tests): GS as luminance, PD as RGB, bilinear resize with align_corners=False and
no antialiasing, then (x - mean) / std from the model sidecar.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from rausnet.metrics import combined_probabilities

ImageInput = str | Path | bytes | Image.Image | np.ndarray

WARNINGS = {
    "not_validated": "Model not clinically validated: research use only, not for diagnosis.",
    "no_pd": "No Power Doppler image: PD and combined grades are unreliable.",
}


def _resize_axis(x: np.ndarray, out: int, axis: int) -> np.ndarray:
    n = x.shape[axis]
    # float32 arithmetic, as in torch, so results match to ~1e-7
    scale = np.float32(n / out)
    src = np.maximum((np.arange(out, dtype=np.float32) + np.float32(0.5)) * scale - np.float32(0.5), np.float32(0))
    i0 = np.minimum(np.floor(src).astype(int), n - 1)
    i1 = np.minimum(i0 + 1, n - 1)
    w = (src - i0).astype(np.float32)
    shape = [1] * x.ndim
    shape[axis] = out
    w = w.reshape(shape)
    return np.take(x, i0, axis=axis) * (1 - w) + np.take(x, i1, axis=axis) * w


def resize_bilinear(x: np.ndarray, size: int) -> np.ndarray:
    """Resize (C, H, W) to (C, size, size) like torch bilinear, align_corners=False, no antialias."""
    return _resize_axis(_resize_axis(x, size, 1), size, 2).astype(np.float32)


def _open(img: ImageInput) -> Image.Image:
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, np.ndarray):
        return Image.fromarray(img)
    if isinstance(img, bytes):
        return Image.open(io.BytesIO(img))
    return Image.open(img)


def preprocess(gs: ImageInput, pd: ImageInput | None, size: int, mean: float, std: float) -> np.ndarray:
    """Return a (1, 4, size, size) float32 tensor."""
    g = np.asarray(_open(gs).convert("L"), np.float32) / 255.0
    if pd is not None:
        p = np.asarray(_open(pd).convert("RGB"), np.float32) / 255.0
        if p.shape[:2] != g.shape:
            raise ValueError(f"GS and PD images must have the same size: {g.shape} vs {p.shape[:2]}")
    else:
        p = np.repeat(g[:, :, None], 3, axis=2)
    x = np.concatenate([g[None], p.transpose(2, 0, 1)], axis=0)
    return ((resize_bilinear(x, size) - mean) / std)[None].astype(np.float32)


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _round(p: np.ndarray) -> list[float]:
    return [round(float(v), 4) for v in p]


class Predictor:
    """Loads ``model.onnx`` + ``model.json`` from a model folder and grades images."""

    def __init__(self, model_dir: str | Path):
        import onnxruntime as ort

        self.dir = Path(model_dir)
        self.info = json.loads((self.dir / "model.json").read_text())
        wanted = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        providers = [p for p in wanted if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(str(self.dir / "model.onnx"), providers=providers)

    def predict(self, gs: ImageInput, pd: ImageInput | None = None) -> dict:
        spec = self.info["input"]
        x = preprocess(gs, pd, spec["size"], spec["mean"], spec["std"])
        lg, lp = self.session.run(None, {spec["name"]: x})
        p_gs, p_pd = _softmax(lg), _softmax(lp)
        p_comb = combined_probabilities(p_gs, p_pd)
        codes = []
        if not self.info.get("clinically_validated", False):
            codes.append("not_validated")
        if pd is None:
            codes.append("no_pd")
        return {
            "model_version": self.info["version"],
            "stage": self.info["stage"],
            "gs": {"grade": int(p_gs[0].argmax()), "probabilities": _round(p_gs[0])},
            "pd": {"grade": int(p_pd[0].argmax()), "probabilities": _round(p_pd[0])},
            "combined": {"grade": int(p_comb[0].argmax()), "probabilities": _round(p_comb[0]),
                         "confidence": round(float(p_comb[0].max()), 4)},
            "warnings": [WARNINGS[c] for c in codes],
            "warning_codes": codes,
        }
