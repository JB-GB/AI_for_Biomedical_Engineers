import json
import math

import pytest

pytest.importorskip("torch")
pytest.importorskip("timm")

from rausnet.data.synthetic import generate_dataset
from rausnet.registry import Registry
from rausnet.retrain import bootstrap_decision, promotion_decision, retrain
from rausnet.train import load_config


def rep(qwk):
    return {"combined": {"qwk": qwk}}


def test_promotion_rule():
    assert promotion_decision(rep(0.5), None) == (True, "first model")
    assert promotion_decision(rep(0.80), rep(0.80))[0]
    assert promotion_decision(rep(0.79), rep(0.80), tolerance=0.02)[0]
    assert not promotion_decision(rep(0.75), rep(0.80), tolerance=0.02)[0]
    assert not promotion_decision(rep(math.nan), rep(0.80))[0]


def test_bootstrap_decision():
    assert bootstrap_decision(None) == (True, "first model")
    noise = {"a": 0.90, "b": 0.89, "diff": -0.01, "ci_low": -0.05, "ci_high": 0.03}
    worse = {"a": 0.90, "b": 0.70, "diff": -0.20, "ci_low": -0.30, "ci_high": -0.10}
    assert bootstrap_decision(noise)[0]
    assert not bootstrap_decision(worse)[0]
    assert not bootstrap_decision({**noise, "b": math.nan})[0]


def test_retrain_lifecycle(tmp_path):
    root = tmp_path / "data"
    for name, seed, n in (("synthA", 1, 20), ("synthB", 2, 8), ("synthC", 3, 8)):
        generate_dataset(root / name, n_patients=n, seed=seed, size=64)

    def cfg(tolerance):
        return load_config("configs/retrain/default.yaml", overrides=[
            f"data.root={root.as_posix()}", "data.image_size=64", "model.backbone=resnet18",
            "model.pretrained=false", "train.epochs=1", "train.finetune_epochs=1", "train.batch_size=16",
            "train.amp=false", f"output_dir={(tmp_path / 'out').as_posix()}",
            f"models_dir={(tmp_path / 'models').as_posix()}", "promotion.rule=tolerance",
            f"promotion.tolerance={tolerance}", "promotion.n_boot=50",
        ])

    first = retrain(cfg(0.0), ["synthA/manifest.csv"])
    assert first.status == "current" and first.parent is None and first.stage == "S2"

    second = retrain(cfg(10.0), ["synthB/manifest.csv"])  # huge tolerance -> always promoted
    assert second.status == "current" and second.parent == "v0001"

    third = retrain(cfg(-10.0), ["synthC/manifest.csv"])  # impossible bar -> rejected
    assert third.status == "rejected"

    reg = Registry(tmp_path / "models")
    assert [e.status for e in reg.entries] == ["archived", "current", "rejected"]
    assert (tmp_path / "models/v0002/model.pt").exists()
    report = json.loads((tmp_path / "models/v0003/retrain_report.json").read_text())
    assert report["parent"] == "v0002" and report["new_images"] == 48 and report["current"] is not None
    assert report["paired_bootstrap"]["n_groups"] == report["candidate"]["n_patients"]
    assert len(report["candidate"]["combined"]["qwk_ci95"]) == 2

    with pytest.raises(ValueError, match="no new images"):
        retrain(cfg(0.0), ["synthA/manifest.csv"])

    fourth = retrain(cfg(10.0), [], from_scratch=True)  # new recipe, same data
    report = json.loads((tmp_path / "models/v0004/retrain_report.json").read_text())
    assert fourth.status == "current" and report["mode"] == "from_scratch" and report["new_images"] == 0
    assert fourth.dataset_sha256 == Registry(tmp_path / "models").get("v0002").dataset_sha256
