import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("timm")

import torch

from rausnet.explain import CamNet, gradcam_autograd
from rausnet.heatmap import normalize_cam, overlay
from rausnet.models import GradingNet


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return GradingNet("resnet18", pretrained=False).eval()


def test_cam_mean_plus_bias_equals_logit(net):
    x = torch.randn(2, 4, 64, 64)
    with torch.no_grad():
        gs_logits, pd_logits, _, pd_cam = CamNet(net)(x)
        ref_gs, ref_pd = net(x)
    assert torch.allclose(gs_logits, ref_gs, atol=1e-5) and torch.allclose(pd_logits, ref_pd, atol=1e-5)
    bias = net.pd_head[1].bias
    assert torch.allclose(pd_cam.mean(dim=(2, 3)) + bias, pd_logits, atol=1e-4)


@pytest.mark.parametrize(("head", "cls"), [("gs", 0), ("gs", 3), ("pd", 2)])
def test_cam_equals_autograd_gradcam(net, head, cls):
    x = torch.randn(1, 4, 64, 64)
    with torch.no_grad():
        outs = CamNet(net)(x)
    cam = (outs[2] if head == "gs" else outs[3])[0, cls]
    ref = gradcam_autograd(net, x, head, cls)[0]
    h, w = cam.shape
    # Grad-CAM weights are W/(h*w): identical maps up to that positive constant.
    assert torch.allclose(cam / (h * w), ref, atol=1e-5)


def test_normalize_and_overlay():
    cam = np.array([[-1.0, 0.0], [0.5, 2.0]], np.float32)
    n = normalize_cam(cam)
    assert n.min() == 0 and n.max() == 1 and n[1, 0] == 0.25
    assert normalize_cam(np.zeros((2, 2))).max() == 0
    img = np.full((20, 30), 100, np.uint8)
    out = overlay(img, n)
    assert out.shape == (20, 30, 3) and out.dtype == np.uint8
    assert (out[0, 0] == 100).all()          # zero activation stays transparent
    assert out[-1, -1, 0] > out[-1, -1, 2]    # hot region tinted warm
