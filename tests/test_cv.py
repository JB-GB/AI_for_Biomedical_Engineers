import json

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("timm")

from rausnet.cv import assign_fold, compare, fold_splits, make_folds, run_experiment
from rausnet.data.manifest import read_manifest
from rausnet.data.synthetic import generate_dataset
from rausnet.train import load_config


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("cv_synth")
    generate_dataset(root, n_patients=24, seed=5, size=64)
    return root


def test_folds_are_patient_level_and_disjoint(synth):
    records = read_manifest(synth / "manifest.csv")
    folds = make_folds(records, 4)
    assert sum(len(f) for f in folds) == len(records)
    owners = [{r.patient_id for r in f} for f in folds]
    assert all(not (a & b) for i, a in enumerate(owners) for b in owners[i + 1:])
    assert assign_fold("P1", 4) == assign_fold("P1", 4)
    s = fold_splits(folds, 3)
    assert s["val"] is folds[0] and s["predict"] is folds[3]
    assert len(s["train"]) == len(folds[1]) + len(folds[2])


def test_k_too_small_rejected(synth):
    with pytest.raises(ValueError, match="k must be"):
        make_folds(read_manifest(synth / "manifest.csv"), 2)


def test_run_experiment_covers_pool_once(synth, tmp_path):
    cfg = load_config("configs/train/baseline.yaml", overrides=[
        f"data.root={synth.as_posix()}", "data.image_size=48", "model.backbone=resnet18",
        "model.pretrained=false", "train.epochs=1", "train.batch_size=16", "train.amp=false",
    ])
    report = run_experiment(cfg, "tiny", 3, "rausnet-cv-v1", tmp_path)
    oof = np.load(tmp_path / "tiny" / "oof.npz")
    assert len(set(oof["image_id"])) == len(oof["image_id"]) == report["n_images"]
    assert sorted(set(oof["fold"].tolist())) == [0, 1, 2]
    assert json.loads((tmp_path / "tiny" / "cv_report.json").read_text())["k"] == 3
    assert not list((tmp_path / "tiny").glob("fold*/model.pt"))  # fold weights deleted
    same = compare(tmp_path, ["tiny"], "tiny", n_boot=50)["experiments"]["tiny"]
    assert same["diff"] == 0
