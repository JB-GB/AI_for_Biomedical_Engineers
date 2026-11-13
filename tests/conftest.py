import json

import pytest


@pytest.fixture(scope="session")
def synth_small(tmp_path_factory):
    """Tiny synthetic dataset (2 patients, 96 px)."""
    from rausnet.data.synthetic import generate_dataset

    root = tmp_path_factory.mktemp("synth_small")
    generate_dataset(root, n_patients=2, seed=4, size=96)
    return root


@pytest.fixture(scope="session")
def onnx_model(tmp_path_factory):
    """Untrained ResNet18 grading net exported to ONNX with a sidecar. Returns (dir, torch model)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    pytest.importorskip("onnxruntime")
    from rausnet.data.dataset import MEAN, STD
    from rausnet.export import export_onnx
    from rausnet.models import GradingNet

    d = tmp_path_factory.mktemp("onnx_model")
    torch.manual_seed(0)
    net = GradingNet("resnet18", pretrained=False)
    cfg = {"model": {"backbone": "resnet18", "pretrained": False, "dropout": 0.2}}
    torch.save({"state_dict": net.state_dict(), "config": cfg}, d / "model.pt")
    assert export_onnx(d / "model.pt", d / "model.onnx", 64) < 1e-4
    sidecar = {"version": "vtest", "stage": "S2", "clinically_validated": False,
               "input": {"name": "input", "size": 64, "mean": MEAN, "std": STD}}
    (d / "model.json").write_text(json.dumps(sidecar))
    return d, net
