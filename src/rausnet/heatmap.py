"""Heat-map rendering for class activation maps (numpy + PIL only, used by the API)."""

from __future__ import annotations

import numpy as np


def normalize_cam(cam: np.ndarray) -> np.ndarray:
    """ReLU and scale a (h, w) map to [0, 1]; all-zero maps stay zero."""
    cam = np.maximum(cam, 0)
    peak = cam.max()
    return cam / peak if peak > 0 else cam


# Sequential warm ramp (transparent -> amber -> red): one hue family, no rainbow.
_RAMP = np.array([[255, 214, 102], [245, 140, 40], [214, 39, 40]], dtype=np.float32)


def overlay(gray_or_rgb: np.ndarray, cam01: np.ndarray, alpha: float = 0.55, floor: float = 0.2) -> np.ndarray:
    """Blend a [0,1] CAM (any size) onto an image (H,W) or (H,W,3); returns uint8 RGB.

    Values below ``floor`` are left transparent so the anatomy stays readable.
    """
    from PIL import Image

    base = gray_or_rgb if gray_or_rgb.ndim == 3 else np.repeat(gray_or_rgb[:, :, None], 3, axis=2)
    h, w = base.shape[:2]
    cam = np.asarray(Image.fromarray((cam01 * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR),
                     np.float32) / 255.0
    pos = np.clip(cam, 0, 1) * (len(_RAMP) - 1)
    lo = np.floor(pos).astype(int).clip(0, len(_RAMP) - 2)
    t = (pos - lo)[..., None]
    colour = _RAMP[lo] * (1 - t) + _RAMP[lo + 1] * t
    a = (alpha * np.clip((cam - floor) / (1 - floor), 0, 1))[..., None]
    return (base.astype(np.float32) * (1 - a) + colour * a).clip(0, 255).astype(np.uint8)
