import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from rausnet.data.manifest import read_manifest
from rausnet.serve.api import create_app


@pytest.fixture()
def client(onnx_model, tmp_path):
    return TestClient(create_app(onnx_model[0], data_dir=tmp_path)), tmp_path


def files(synth, with_pd=True):
    r = read_manifest(synth / "manifest.csv")[0]
    out = {"gs": ("gs.png", (synth / r.gs_path).read_bytes(), "image/png")}
    if with_pd:
        out["pd"] = ("pd.png", (synth / r.pd_path).read_bytes(), "image/png")
    return out


def test_health_and_model(client):
    c, _ = client
    assert c.get("/health").json()["model_version"] == "vtest"
    assert c.get("/model").json()["clinically_validated"] is False


def test_predict(client, synth_small):
    c, _ = client
    res = c.post("/predict", files=files(synth_small))
    assert res.status_code == 200
    body = res.json()
    assert body["combined"]["grade"] in range(4) and len(body["pd"]["probabilities"]) == 4
    gs_only = c.post("/predict", files=files(synth_small, with_pd=False)).json()
    assert any("No Power Doppler" in w for w in gs_only["warnings"])


def test_predict_rejects_non_image(client):
    c, _ = client
    res = c.post("/predict", files={"gs": ("x.png", b"not an image", "image/png")})
    assert res.status_code == 422


def test_correction_stored_and_valid(client, synth_small):
    c, data = client
    form = {"patient_id": "HOSP-0001", "joint": "MCP2", "side": "L", "gs_grade": "2", "pd_grade": "1"}
    res = c.post("/corrections", files=files(synth_small), data=form)
    assert res.status_code == 200, res.text
    manifest = data / res.json()["manifest"]
    recs = read_manifest(manifest)
    assert recs[0].source == "hospital" and recs[0].gs_grade == 2
    assert (manifest.parent / recs[0].gs_path).exists() and (manifest.parent / recs[0].pd_path).exists()
    c.post("/corrections", files=files(synth_small), data=form)
    assert len(read_manifest(manifest)) == 2


@pytest.mark.parametrize("bad", [{"patient_id": "Juan Perez"}, {"joint": "KNEE"}, {"gs_grade": "0", "pd_grade": "2"}])
def test_correction_validation(client, synth_small, bad):
    c, _ = client
    form = {"patient_id": "HOSP-0001", "joint": "MCP2", "side": "L", "gs_grade": "2", "pd_grade": "1", **bad}
    assert c.post("/corrections", files=files(synth_small), data=form).status_code == 422


def test_corrections_disabled_without_data_dir(onnx_model, synth_small):
    c = TestClient(create_app(onnx_model[0]))
    form = {"patient_id": "H1", "joint": "MCP2", "side": "L", "gs_grade": "1", "pd_grade": "0"}
    assert c.post("/corrections", files=files(synth_small), data=form).status_code == 503


def test_predict_with_heatmaps(client, synth_small):
    import base64
    import io

    from PIL import Image

    c, _ = client
    body = c.post("/predict", params={"explain": "true"}, files=files(synth_small)).json()
    for key in ("gs", "pd"):
        url = body["heatmaps"][key]
        assert url.startswith("data:image/png;base64,")
        img = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
        assert img.size == (96, 96) and img.mode == "RGB"
    assert "heatmaps" not in c.post("/predict", files=files(synth_small)).json()
