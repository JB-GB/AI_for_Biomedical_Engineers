import json
import time

import pytest

from rausnet.data.manifest import read_manifest, write_manifest
from rausnet.data.synthetic import generate_dataset
from rausnet.improve import Lock, Workspace, WorkspaceLocked, init, pending_batches, status
from rausnet.registry import ModelEntry, Registry


def test_pending_batches_counts_unseen_and_flags_incomplete(tmp_path):
    root = tmp_path / "data"
    generate_dataset(root / "incoming/b1", n_patients=2, seed=1, size=64)
    recs = read_manifest(root / "incoming/b1/manifest.csv")
    # current dataset already holds the first 5 images of b1 (paths rebased onto the data root)
    seen = [r.__class__(**{**r.__dict__, "gs_path": f"incoming/b1/{r.gs_path}",
                           "pd_path": f"incoming/b1/{r.pd_path}" if r.pd_path else ""}) for r in recs[:5]]
    (root / "datasets/v0001").mkdir(parents=True)
    write_manifest(seen, root / "datasets/v0001/manifest.csv")
    # a batch still syncing: manifest present, image missing
    generate_dataset(root / "incoming/b2", n_patients=1, seed=2, size=64)
    next((root / "incoming/b2/gs").iterdir()).unlink()

    pend = pending_batches(root, "datasets/v0001/manifest.csv")
    assert pend["ready"] == [("incoming/b1/manifest.csv", len(recs) - 5)]
    assert [m for m, _ in pend["incomplete"]] == ["incoming/b2/manifest.csv"]
    assert pending_batches(root, None)["ready"][0][1] == len(recs)


def test_lock_is_exclusive_and_stale_lock_is_taken_over(tmp_path):
    ws = Workspace(tmp_path).ensure()
    with Lock(ws):
        with pytest.raises(WorkspaceLocked), Lock(ws):
            pass
    assert not ws.lock_path.exists()
    ws.lock_path.write_text(json.dumps({"time": time.time() - 7 * 3600, "started": "old", "runtime": "colab"}))
    with Lock(ws, stale_hours=6):
        assert Lock(ws).info()["runtime"] != "colab" or Lock(ws).info()["started"] != "old"
    assert not ws.lock_path.exists()


def test_init_imports_current_model_without_synthetic_images(tmp_path):
    models, data = tmp_path / "local_models", tmp_path / "local_data"
    generate_dataset(data / "synth", n_patients=1, seed=0, size=64)
    generate_dataset(data / "incoming/b1", n_patients=1, seed=1, size=64)
    (data / "datasets/v0001").mkdir(parents=True)
    write_manifest(read_manifest(data / "synth/manifest.csv"), data / "datasets/v0001/manifest.csv")
    (models / "v0001").mkdir(parents=True)
    (models / "v0001/model.onnx").write_bytes(b"onnx")
    Registry(models).add(ModelEntry("v0001", "now", "v0001/model.pt", "datasets/v0001/manifest.csv", "x",
                                    "S2", None), promote=True)
    ws = tmp_path / "ws"
    init(ws, models, data)
    assert (ws / "models/v0001/model.onnx").exists() and (ws / "models/registry.json").exists()
    assert (ws / "data/datasets/v0001/manifest.csv").exists() and (ws / "data/incoming/b1/manifest.csv").exists()
    assert not (ws / "data/synth").exists()  # regenerated on demand, never uploaded
    with pytest.raises(FileExistsError):
        init(ws, models, data)  # never overwrite the global registry
    assert status(ws)["current_model"]["version"] == "v0001"


def test_improve_bootstraps_publishes_and_skips_until_enough_new_data(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    from rausnet.improve import improve
    from rausnet.train import load_config

    ws = Workspace(tmp_path / "ws").ensure()
    ws.spec_path.write_text(json.dumps({"synthetic": {"dir": "synth", "patients": 12, "seed": 0, "size": 64}}))
    cfg = load_config("configs/retrain/default.yaml", overrides=[
        f"data.root={(tmp_path / 'unused').as_posix()}", "data.image_size=64", "model.backbone=resnet18",
        "model.pretrained=false", "train.epochs=1", "train.finetune_epochs=1", "train.batch_size=16",
        "train.amp=false", "promotion.rule=tolerance", "promotion.tolerance=10", "promotion.n_boot=50",
        "postprocess.export=false", "improve.min_new_images=20"])

    first = improve(ws.root, cfg)
    assert first["action"] == "retrained" and first["version"] == "v0001" and first["promoted"]
    assert Registry(ws.models).current().version == "v0001"
    assert (ws.models / "v0001/model.pt").exists() and (ws.models / "v0001/calibration.json").exists()
    assert (ws.data / "datasets/v0001/manifest.csv").exists() and not (ws.data / "synth").exists()
    assert not ws.lock_path.exists() and not list(ws.root.rglob("*.partial"))

    assert improve(ws.root, cfg)["action"] == "nothing to do"
    generate_dataset(ws.data / "incoming/corrections/20261114", n_patients=4, seed=7, size=64)  # 24 images
    assert status(ws.root)["pending_images"] == 24
    second = improve(ws.root, cfg)
    assert second["action"] == "retrained" and second["version"] == "v0002" and second["new_images"] == 24
    assert Registry(ws.models).current().version == "v0002" and status(ws.root)["pending_images"] == 0
    assert [h["action"] for h in ws.history()] == ["retrained", "nothing to do", "retrained"]
