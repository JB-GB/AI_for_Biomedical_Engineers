"""Synthetic dorsal longitudinal ultrasound of MCP 2-3 and wrist joints (stage S2).

Each image has a known EULAR-OMERACT grey-scale (GS) and Power Doppler (PD)
grade. It is used to train the grading head and to verify the full pipeline
before clinical data exists. Synthetic results are NOT clinical evidence.

Grade definitions implemented (simplified from EULAR-OMERACT):
- GS 0: no synovial hypertrophy.
- GS 1: small hypoechoic area, thin and narrow.
- GS 2: moderate, flat upper surface.
- GS 3: large, convex (bulging) upper surface and wider extent.
- PD 0: no signal; PD 1: <= 3 isolated spots;
  PD 2: < 50 % of the synovium; PD 3: > 50 % of the synovium.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rausnet.data.manifest import JOINTS
from rausnet.grading import validate_grade

GENERATOR_VERSION = "1"
PD_THRESHOLD = 0.3


@dataclass(frozen=True)
class SyntheticImage:
    gs: np.ndarray  # (H, W) uint8 grey-scale image
    pd: np.ndarray  # (H, W, 3) uint8 grey-scale with Power Doppler colour overlay
    mask: np.ndarray  # (H, W) bool synovial region


def _gaussian_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur with reflect padding (numpy only)."""
    if sigma <= 0:
        return a
    r = max(1, int(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    out = a.astype(np.float32)
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r, r)
        p = np.pad(out, pad, mode="reflect")
        n = out.shape[axis]
        acc = np.zeros_like(out)
        for i, weight in enumerate(k):
            acc += weight * (p[i : i + n] if axis == 0 else p[:, i : i + n])
        out = acc
    return out


def _bone_layout(joint: str, rng: np.random.Generator):
    """Return bone segments (x0, x1, base_y, height, t0, t1) and synovial window."""
    yb = rng.uniform(0.52, 0.62)
    if joint in ("MCP2", "MCP3"):
        xj = rng.uniform(0.50, 0.62)
        g = rng.uniform(0.015, 0.03)
        segments = [
            (0.0, xj - g, yb, rng.uniform(0.08, 0.13), 0.35, 1.0),  # metacarpal head
            (xj + g, 1.0, yb + rng.uniform(0.0, 0.03), rng.uniform(0.02, 0.05), 0.0, 0.6),  # phalanx
        ]
        window = (xj - g - rng.uniform(0.12, 0.20), xj + g + rng.uniform(0.04, 0.08))
    else:
        g = rng.uniform(0.012, 0.025)
        a = rng.uniform(0.30, 0.38)
        b = a + rng.uniform(0.20, 0.26)
        segments = [
            (0.0, a - g, yb, rng.uniform(0.01, 0.03), 0.4, 0.9),  # radius
            (a + g, b - g, yb - 0.01, rng.uniform(0.05, 0.08), 0.0, 1.0),  # lunate
            (b + g, 1.0, yb, rng.uniform(0.04, 0.07), 0.0, 0.7),  # capitate
        ]
        window = (a - g - rng.uniform(0.05, 0.10), b + g + rng.uniform(0.03, 0.06))
    return segments, window


def _add_blob(pd_map, cy, cx, radius, amp, yy, xx):
    pd_map += amp * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * radius**2)))


def _render_doppler(mask, pd_grade, rng, shape, distractor_band):
    """Return a 0-1 Power Doppler intensity map satisfying the PD grade definition."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    area = int(mask.sum())
    for _ in range(30):
        pd_map = np.zeros(shape, np.float32)
        if pd_grade > 0 and area > 0:
            ys, xs = np.nonzero(mask)
            if pd_grade == 1:
                for _ in range(int(rng.integers(1, 4))):
                    i = rng.integers(len(ys))
                    _add_blob(pd_map, ys[i], xs[i], rng.uniform(1.2, 2.2), rng.uniform(0.7, 1.0), yy, xx)
            else:
                target = rng.uniform(0.15, 0.40) if pd_grade == 2 else rng.uniform(0.60, 0.90)
                r = float(np.clip(np.sqrt(area) * 0.12, 1.5, 7.0))
                for _ in range(400):
                    if ((pd_map > PD_THRESHOLD) & mask).sum() >= target * area:
                        break
                    i = rng.integers(len(ys))
                    _add_blob(pd_map, ys[i], xs[i], r * rng.uniform(0.6, 1.2), rng.uniform(0.6, 1.0), yy, xx)
        pd_map = np.where(mask, pd_map, 0.0)
        if distractor_band is not None and rng.random() < 0.15:
            y0, y1, x0, x1 = distractor_band
            cy = rng.uniform(y0, y1) * h
            cx = rng.uniform(x0, x1) * w
            for dx in np.linspace(-6, 6, 9):
                _add_blob(pd_map, cy, cx + dx, 1.3, 0.9, yy, xx)
        coverage = ((pd_map > PD_THRESHOLD) & mask).sum() / max(area, 1)
        ok = {
            0: coverage == 0,
            1: 0 < coverage < 0.5,
            2: 0 < coverage < 0.5,
            3: coverage > 0.5,
        }[pd_grade]
        if ok:
            return np.clip(pd_map, 0.0, 1.0)
    raise RuntimeError("could not satisfy PD grade definition")


def render_joint(
    joint: str, gs_grade: int, pd_grade: int, rng: np.random.Generator, size: int = 256
) -> SyntheticImage:
    """Render one synthetic joint with the requested GS and PD grades."""
    if joint not in JOINTS:
        raise ValueError(f"joint must be one of {JOINTS}, got {joint!r}")
    validate_grade(gs_grade, "gs_grade")
    validate_grade(pd_grade, "pd_grade")
    if gs_grade == 0 and pd_grade > 0:
        raise ValueError("PD > 0 requires GS >= 1")

    h = w = size
    x = (np.arange(w) + 0.5) / w
    yy = ((np.arange(h) + 0.5) / h)[:, None]

    skin = rng.uniform(0.02, 0.035)
    t0 = rng.uniform(0.09, 0.14)
    t1 = t0 + rng.uniform(0.04, 0.07)
    cortex = rng.uniform(0.006, 0.010)
    cart = rng.uniform(0.008, 0.015)

    segments, (wl, wr) = _bone_layout(joint, rng)
    ybone = np.full(w, np.nan)
    for x0, x1, base, height, ta, tb in segments:
        sel = (x >= x0) & (x < x1)
        t = ta + (tb - ta) * (x[sel] - x0) / (x1 - x0)
        ybone[sel] = base - height * np.sin(np.pi * t) ** 0.7
    has_bone = ~np.isnan(ybone)
    floor = np.interp(x, x[has_bone], ybone[has_bone]) - cart

    # Synovial region: thickness profile depends on GS grade.
    mask = np.zeros((h, w), bool)
    if gs_grade > 0:
        scale = {1: 0.6, 2: 1.0, 3: 1.25}[gs_grade]
        centre, half = (wl + wr) / 2, (wr - wl) / 2 * scale
        thick = {1: (0.02, 0.035), 2: (0.05, 0.07), 3: (0.09, 0.13)}[gs_grade]
        power = {1: 1.0, 2: 0.35, 3: 0.8}[gs_grade]
        t = np.clip((x - (centre - half)) / (2 * half), 0.0, 1.0)
        win = np.sin(np.pi * t)
        top = np.maximum(floor - rng.uniform(*thick) * win**power, t1 + 0.01)
        mask = (yy > top[None, :]) & (yy < floor[None, :]) & (win > 0.02)[None, :]

    yb = ybone[None, :]
    low1 = _gaussian_blur(rng.standard_normal((h, w)), size * 0.05)
    low1 /= np.abs(low1).max() + 1e-6
    tissue = 0.32 + 0.08 * low1
    tissue = np.where(yy < skin, 0.05, tissue)
    tissue = np.where((yy >= skin) & (yy < skin + 0.012), 0.9, tissue)
    stripes = 0.5 + 0.5 * np.sin(2 * np.pi * (yy * rng.uniform(60, 90) + 0.3 * low1))
    tissue = np.where((yy > t0) & (yy < t1), 0.55 + 0.25 * stripes, tissue)
    gap = ~has_bone[None, :] & (yy > floor[None, :] + cart)
    tissue = np.where(gap, np.where(yy < floor[None, :] + 0.08, 0.12, 0.03), tissue)
    tissue = np.where((yy >= yb - cart) & (yy < yb - cortex), 0.04, tissue)
    tissue = np.where(np.abs(yy - yb) <= cortex, 1.0, tissue)
    tissue = np.where(yy > yb + cortex, 0.025, tissue)
    tissue = np.where(mask, rng.uniform(0.06, 0.14), tissue)

    speckle = rng.rayleigh(1.0, (h, w)) / 1.2533
    img = _gaussian_blur(tissue * speckle, 0.7)
    img *= np.exp(-rng.uniform(0.6, 1.4) * yy) * rng.uniform(0.8, 1.2)
    img = np.log1p(20 * np.clip(img, 0, None)) / np.log1p(20)
    gs = (np.clip(img, 0, 1) * 255).astype(np.uint8)

    band = (skin + 0.015, t0 - 0.01, wl, wr) if t0 - skin > 0.04 else None
    pd_map = _render_doppler(mask, pd_grade, rng, (h, w), band)
    rgb = np.repeat(gs[:, :, None], 3, axis=2).astype(np.float32)
    colour = pd_map > PD_THRESHOLD
    v = pd_map[colour]
    rgb[colour] = np.stack([np.full_like(v, 255), 80 + 175 * v, 30 * (1 - v)], axis=1)
    bx0, bx1 = int(max(wl - 0.05, 0) * w), int(min(wr + 0.05, 1) * w) - 1
    by0, by1 = int(skin * h), int(min(np.nanmax(ybone) + 0.15, 1) * h) - 1
    rgb[[by0, by1], bx0 : bx1 + 1] = 160
    rgb[by0 : by1 + 1, [bx0, bx1]] = 160
    pd = rgb.astype(np.uint8)

    if rng.random() < 0.5:
        gs, pd, mask = gs[:, ::-1], pd[:, ::-1], mask[:, ::-1]
    return SyntheticImage(np.ascontiguousarray(gs), np.ascontiguousarray(pd), np.ascontiguousarray(mask))

