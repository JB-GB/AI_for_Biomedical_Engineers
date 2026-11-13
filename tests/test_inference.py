import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("timm")
pytest.importorskip("onnxruntime")

import torch

from rausnet.data.dataset import MEAN, STD, load_input
from rausnet.data.manifest import read_manifest
from rausnet.serve.inference import Predictor, preprocess, resize_bilinear


@pytest.mark.parametrize("size", [64, 224, 300])
def test_resize_matches_torch(size):
    x = np.random.default_rng(0).random((4, 97, 131), dtype=np.float32)
    ref = torch.nn.functional.interpolate(torch.from_numpy(x)[None], size=(size, size), mode="bilinear",
                                          align_corners=False)[0].numpy()
    assert np.allclose(resize_bilinear(x, size), ref, atol=1e-5)


def test_preprocess_matches_training(synth_small):
    synth = synth_small
    r = read_manifest(synth / "manifest.csv")[0]
    for pd in (synth / r.pd_path, None):
        ref = load_input(synth / r.gs_path, pd, 64).numpy()
        got = preprocess(synth / r.gs_path, pd, 64, MEAN, STD)[0]
        assert np.allclose(got, ref, atol=1e-5)


def test_predictor_matches_torch(synth_small, onnx_model):
    synth = synth_small
    d, net = onnx_model
    r = read_manifest(synth / "manifest.csv")[0]
    out = Predictor(d).predict(synth / r.gs_path, (synth / r.pd_path).read_bytes())
    net.eval()
    with torch.no_grad():
        lg, _ = net(load_input(synth / r.gs_path, synth / r.pd_path, 64)[None])
    assert np.allclose(out["gs"]["probabilities"], lg.softmax(1)[0].numpy(), atol=1e-3)
    assert sum(out["combined"]["probabilities"]) == pytest.approx(1, abs=1e-3)
    assert sum(out["pd"]["probabilities"]) == pytest.approx(1, abs=1e-3)
    assert out["pd"]["probabilities"][0] >= out["gs"]["probabilities"][0] - 1e-3  # GS0 mass goes to PD0
    assert any("not clinically validated" in w for w in out["warnings"])


def test_size_mismatch_rejected(synth_small):
    synth = synth_small
    r = read_manifest(synth / "manifest.csv")[0]
    small = np.zeros((10, 10, 3), np.uint8)
    with pytest.raises(ValueError, match="same size"):
        preprocess(synth / r.gs_path, small, 64, MEAN, STD)


def test_predictor_heatmaps_and_temperature(synth_small, onnx_model, tmp_path):
    import json
    import shutil

    d, _ = onnx_model
    r = read_manifest(synth_small / "manifest.csv")[0]
    out = Predictor(d).predict(synth_small / r.gs_path, synth_small / r.pd_path, explain=True)
    assert out["heatmaps"]["gs"].shape == (96, 96, 3) and out["heatmaps"]["pd"].dtype == np.uint8

    hot = tmp_path / "hot"
    shutil.copytree(d, hot)
    info = json.loads((hot / "model.json").read_text())
    info["temperature"] = {"gs": 100.0, "pd": 100.0}  # extreme T -> near-uniform probabilities
    (hot / "model.json").write_text(json.dumps(info))
    flat = Predictor(hot).predict(synth_small / r.gs_path, synth_small / r.pd_path)
    assert max(flat["gs"]["probabilities"]) < 0.3 and flat["calibrated"]
    assert flat["gs"]["grade"] == Predictor(d).predict(synth_small / r.gs_path, synth_small / r.pd_path)["gs"]["grade"]
