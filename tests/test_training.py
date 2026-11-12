import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from rausnet.data.dataset import IGNORE_INDEX, JointDataset, load_input  # noqa: E402
from rausnet.data.manifest import read_manifest  # noqa: E402
from rausnet.data.synthetic import generate_dataset  # noqa: E402
from rausnet.models import GradingNet  # noqa: E402
from rausnet.train import load_config, train  # noqa: E402


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    generate_dataset(root, n_patients=20, seed=5, size=64)
    return root


def test_dataset_item(synth):
    records = read_manifest(synth / "manifest.csv")
    ds = JointDataset(records, synth, size=48, train=True)
    x, gs, pd = ds[0]
    assert x.shape == (4, 48, 48) and x.dtype == torch.float32
    assert gs == records[0].gs_grade and pd == records[0].pd_grade


def test_missing_pd_uses_gs_and_ignore_label(synth):
    from dataclasses import replace

    r = replace(read_manifest(synth / "manifest.csv")[0], pd_path="", pd_grade=0)
    x, _, pd = JointDataset([r], synth, size=32)[0]
    assert pd == IGNORE_INDEX
    assert torch.allclose(x[1], x[0]) and torch.allclose(x[3], x[0])


def test_load_input_shape(synth):
    r = read_manifest(synth / "manifest.csv")[0]
    assert load_input(synth / r.gs_path, synth / r.pd_path, 40).shape == (4, 40, 40)


def test_model_outputs():
    model = GradingNet("resnet18", pretrained=False)
    lg, lp = model(torch.randn(2, 4, 64, 64))
    assert lg.shape == (2, 4) and lp.shape == (2, 4)


def test_train_smoke(synth, tmp_path):
    cfg = load_config("configs/train/baseline.yaml", overrides=[
        f"data.root={synth.as_posix()}", "data.image_size=64", "model.backbone=resnet18",
        "model.pretrained=false", "train.epochs=1", "train.batch_size=16", "train.num_workers=0",
        "train.amp=false", f"output_dir={tmp_path.as_posix()}",
    ])
    run = train(cfg)
    assert (run / "model.pt").exists() and (run / "history.csv").exists()
    card = json.loads((run / "model_card.json").read_text())
    assert card["stage"] == "S2" and card["clinically_validated"] is False
    assert sum(card["split_sizes"].values()) == 120


def test_unresolved_env_var_rejected(monkeypatch):
    monkeypatch.delenv("RAUSNET_DATA_DIR", raising=False)
    with pytest.raises(ValueError, match="environment variable"):
        load_config("configs/train/baseline.yaml")
