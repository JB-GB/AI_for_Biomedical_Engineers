"""Heat-map rendering for class activation maps (numpy + PIL only, used by the API)."""

from __future__ import annotations

import numpy as np


def normalize_cam(cam: np.ndarray) -> np.ndarray:
    """ReLU and scale a (h, w) map to [0, 1]; all-zero maps stay zero."""
    cam = np.maximum(cam, 0)
    peak = cam.max()
    return cam / peak if peak > 0 else cam


# Sequential ramps, no rainbow. "warm": transparent -> amber -> red (one hue family).
# "cividis": dark blue -> grey -> yellow, readable with colour-vision deficiency and distinct from
# the red/yellow Power Doppler signal.
PALETTES = {
    "warm": np.array([[255, 214, 102], [245, 140, 40], [214, 39, 40]], dtype=np.float32),
    "cividis": np.array([[0, 34, 78], [65, 77, 108], [124, 123, 120], [188, 175, 111], [255, 234, 70]],
                        dtype=np.float32),
}


def overlay(gray_or_rgb: np.ndarray, cam01: np.ndarray, alpha: float = 0.55, floor: float = 0.2,
            palette: str = "warm") -> np.ndarray:
    """Blend a [0,1] CAM (any size) onto an image (H,W) or (H,W,3); returns uint8 RGB.

    Values below ``floor`` are left transparent so the anatomy stays readable.
    """
    from PIL import Image

    ramp = PALETTES[palette]
    base = gray_or_rgb if gray_or_rgb.ndim == 3 else np.repeat(gray_or_rgb[:, :, None], 3, axis=2)
    h, w = base.shape[:2]
    cam = np.asarray(Image.fromarray((cam01 * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR),
                     np.float32) / 255.0
    pos = np.clip(cam, 0, 1) * (len(ramp) - 1)
    lo = np.floor(pos).astype(int).clip(0, len(ramp) - 2)
    t = (pos - lo)[..., None]
    colour = ramp[lo] * (1 - t) + ramp[lo + 1] * t
    a = (alpha * np.clip((cam - floor) / (1 - floor), 0, 1))[..., None]
    return (base.astype(np.float32) * (1 - a) + colour * a).clip(0, 255).astype(np.uint8)
