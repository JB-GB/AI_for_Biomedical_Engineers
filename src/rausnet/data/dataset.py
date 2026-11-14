"""PyTorch dataset over a manifest: 4-channel input (GS + PD RGB) and two labels."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from rausnet.data.manifest import Record

IGNORE_INDEX = -1
MEAN, STD = 0.5, 0.25


def load_input(gs_path: str | Path, pd_path: str | Path | None, size: int) -> torch.Tensor:
    """Load a GS image and optional PD image as a normalised (4, size, size) tensor.

    Without a PD image the GS image is replicated into the PD channels.
    Shared by training and inference so preprocessing never diverges.
    """
    gs = np.asarray(Image.open(gs_path).convert("L"), np.float32) / 255.0
    if pd_path:
        pd = np.asarray(Image.open(pd_path).convert("RGB"), np.float32) / 255.0
    else:
        pd = np.repeat(gs[:, :, None], 3, axis=2)
    if pd.shape[:2] != gs.shape:
        raise ValueError(f"GS and PD sizes differ: {gs.shape} vs {pd.shape[:2]}")
    x = torch.from_numpy(np.concatenate([gs[None], pd.transpose(2, 0, 1)], axis=0))
    x = F.interpolate(x[None], size=(size, size), mode="bilinear", align_corners=False)[0]
    return (x - MEAN) / STD


def _modality_noise(sigma: float, hw: tuple[int, int]) -> torch.Tensor:
    """Independent noise for the GS channel and the PD image (shared by its RGB channels).

    GS and PD are separate acquisitions, so their backgrounds never match pixel for pixel.
    Synthetic PD images are an exact copy of GS plus colour; keeping that identity lets the
    network learn a shortcut that fails on unaugmented images (validation collapsed with a
    shared field). One field per modality breaks the identity without inventing colour.
    """
    gs, pd = sigma * torch.randn((1, *hw)), sigma * torch.randn((1, *hw))
    return torch.cat([gs, pd.expand(3, -1, -1)], 0)


def _augment(x: torch.Tensor) -> torch.Tensor:
    """Ultrasound-plausible augmentations applied identically to all channels.

    Uses the global torch RNG; DataLoader seeds each worker from it, so runs are
    reproducible after ``torch.manual_seed``.
    """
    size = x.shape[-1]
    if torch.rand(1) < 0.5:
        x = x.flip(-1)
    scale = 0.85 + 0.15 * torch.rand(1).item()
    crop = max(8, int(size * scale))
    top = int(torch.randint(0, size - crop + 1, (1,)))
    left = int(torch.randint(0, size - crop + 1, (1,)))
    x = x[:, top : top + crop, left : left + crop]
    x = F.interpolate(x[None], size=(size, size), mode="bilinear", align_corners=False)[0]
    gain = 0.8 + 0.4 * torch.rand(1)
    x = (x * STD + MEAN) * gain
    # random noise level including ~0: a fixed level made the network depend on the noise itself
    # (clean validation images collapsed to chance; logbook M4-T4.2)
    x = x + _modality_noise(0.03 * torch.rand(1).item(), x.shape[-2:])
    return (x.clamp(0, 1) - MEAN) / STD


def _uniform(lo: float, hi: float) -> float:
    return lo + (hi - lo) * torch.rand(1).item()


def _augment_us(x: torch.Tensor) -> torch.Tensor:
    """Ultrasound acquisition variability, on top of flip/crop (T4.2).

    Every transform is grade-preserving: geometry and intensity change, the size of the
    synovial region and of the Doppler signal relative to it do not.
    - probe tilt: rotation up to +/-10 degrees, zoom 0.75-1 (depth setting);
    - time-gain compensation: gain ramp along depth (rows), +/-25 % top to bottom;
    - dynamic range / contrast: gamma 0.7-1.4;
    - multiplicative speckle: gamma-distributed (shape 8-30), one field per modality (see
      ``_modality_noise``), shared by the PD RGB channels so no colour is invented;
    - Doppler gain: colour excess over the grey background scaled 0.75-1.25.
    """
    size = x.shape[-1]
    img = x * STD + MEAN                                   # back to [0, 1]
    if torch.rand(1) < 0.5:
        img = img.flip(-1)
    angle = torch.deg2rad(torch.tensor(_uniform(-10, 10)))
    zoom = _uniform(0.75, 1.0)
    shift = (torch.rand(2) * 2 - 1) * (1 - zoom)
    cos, sin = torch.cos(angle) * zoom, torch.sin(angle) * zoom
    theta = torch.tensor([[cos, -sin, shift[0]], [sin, cos, shift[1]]], dtype=img.dtype)[None]
    grid = F.affine_grid(theta, (1, *img.shape), align_corners=False)
    img = F.grid_sample(img[None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0]

    gs, pd = img[:1], img[1:]
    colour = pd - gs                                        # Doppler signal over the B-mode background
    gs = gs * (0.8 + 0.4 * torch.rand(1))                   # overall gain
    tgc = 1 + _uniform(-0.25, 0.25) * torch.linspace(-1, 1, size, dtype=img.dtype)[:, None]
    gs = (gs * tgc).clamp(0, 1) ** _uniform(0.7, 1.4)
    k = _uniform(8, 30)
    speckle = torch.distributions.Gamma(k, k).sample((2, size, size)).to(img.dtype)  # one field per modality
    img = torch.cat([gs * speckle[:1], gs * speckle[1:] + colour * _uniform(0.75, 1.25)], 0)
    return (img.clamp(0, 1) - MEAN) / STD


AUGMENTATIONS = {"basic": _augment, "us": _augment_us}


class JointDataset(Dataset):
    def __init__(self, records: Sequence[Record], root: str | Path, size: int = 224,
                 train: bool = False, augment: str = "basic"):
        if augment not in AUGMENTATIONS:
            raise ValueError(f"augment must be one of {sorted(AUGMENTATIONS)}, got {augment!r}")
        self.records = list(records)
        self.root = Path(root)
        self.size = size
        self.train = train
        self.augment = AUGMENTATIONS[augment]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        r = self.records[i]
        x = load_input(self.root / r.gs_path, self.root / r.pd_path if r.pd_path else None, self.size)
        if self.train:
            x = self.augment(x)
        pd_label = r.pd_grade if r.pd_path else IGNORE_INDEX
        return x, r.gs_grade, pd_label
