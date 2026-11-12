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
    x = x + 0.02 * torch.randn(x.shape)
    return (x.clamp(0, 1) - MEAN) / STD


class JointDataset(Dataset):
    def __init__(self, records: Sequence[Record], root: str | Path, size: int = 224,
                 train: bool = False):
        self.records = list(records)
        self.root = Path(root)
        self.size = size
        self.train = train

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        r = self.records[i]
        x = load_input(self.root / r.gs_path, self.root / r.pd_path if r.pd_path else None, self.size)
        if self.train:
            x = _augment(x)
        pd_label = r.pd_grade if r.pd_path else IGNORE_INDEX
        return x, r.gs_grade, pd_label
