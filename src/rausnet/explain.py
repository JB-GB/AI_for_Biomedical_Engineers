"""Class activation maps for the two-head grading network.

For a backbone that ends in global *average* pooling followed by a linear head,
Grad-CAM is exactly the class activation map: the gradient of logit c with
respect to feature map A_k is W[c, k] / (h * w) at every position, so

    GradCAM_c = ReLU( sum_k W[c, k] * A_k )   (up to a positive constant).

This lets the maps be computed in the forward pass (and exported to ONNX), with no
gradients at inference time. ``gradcam_autograd`` is the reference used in tests.
"""

from __future__ import annotations

import torch
from torch import nn

from rausnet.models import GradingNet


def _linear(head: nn.Sequential) -> nn.Linear:
    return next(m for m in head if isinstance(m, nn.Linear))


class CamNet(nn.Module):
    """Wraps a GradingNet: returns logits and per-class CAMs for both heads.

    Outputs: gs_logits (N,4), pd_logits (N,4), gs_cam (N,4,h,w), pd_cam (N,4,h,w).
    CAMs are raw (signed, unnormalised); the spatial mean of a CAM plus the head
    bias equals the logit.
    """

    def __init__(self, net: GradingNet):
        super().__init__()
        pool = getattr(net.backbone, "global_pool", None)
        if pool is None or getattr(pool, "pool_type", "avg") != "avg":
            raise ValueError("exact CAM requires a global average pooling backbone")
        self.net = net

    def forward(self, x: torch.Tensor):
        feats = self.net.backbone.forward_features(x)
        pooled = self.net.backbone.forward_head(feats, pre_logits=True)
        out = []
        for head in (self.net.gs_head, self.net.pd_head):
            lin = _linear(head)
            out.append(head(pooled))
            out.append(torch.einsum("kc,nchw->nkhw", lin.weight, feats))
        gs_logits, gs_cam, pd_logits, pd_cam = out
        return gs_logits, pd_logits, gs_cam, pd_cam


def gradcam_autograd(net: GradingNet, x: torch.Tensor, head: str, cls: int) -> torch.Tensor:
    """Reference Grad-CAM via autograd on the last feature map. Returns (N, h, w) before ReLU."""
    net.eval()
    feats = net.backbone.forward_features(x)
    feats.retain_grad()
    pooled = net.backbone.forward_head(feats, pre_logits=True)
    logits = (net.gs_head if head == "gs" else net.pd_head)(pooled)
    logits[:, cls].sum().backward()
    weights = feats.grad.mean(dim=(2, 3), keepdim=True)
    return (weights * feats).sum(dim=1).detach()
