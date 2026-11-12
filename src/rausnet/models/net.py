"""Two-head network: shared backbone, separate GS and PD 4-class heads.

Predicting GS and PD separately mirrors the EULAR-OMERACT procedure; the
combined grade is derived with the scoring rule instead of being learned.
"""

from __future__ import annotations

import timm
import torch
from torch import nn

N_CLASSES = 4


class GradingNet(nn.Module):
    def __init__(self, backbone: str = "efficientnet_b0", pretrained: bool = True,
                 in_chans: int = 4, dropout: float = 0.2):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, in_chans=in_chans,
                                          num_classes=0)
        features = self.backbone.num_features
        self.gs_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(features, N_CLASSES))
        self.pd_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(features, N_CLASSES))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.backbone(x)
        return self.gs_head(f), self.pd_head(f)
