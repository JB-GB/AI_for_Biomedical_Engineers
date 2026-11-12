import numpy as np
import pytest

from rausnet.data.synthetic import render_joint


def colour_inside(img):
    rgb = img.pd.astype(int)
    coloured = (rgb[..., 0] - rgb[..., 2]) > 100
    return (coloured & img.mask).sum() / max(img.mask.sum(), 1)


def render(joint, gs, pd, seed=0, size=128):
    return render_joint(joint, gs, pd, np.random.default_rng(seed), size=size)


def test_shapes_and_types():
    img = render("MCP2", 2, 1)
    assert img.gs.shape == (128, 128) and img.gs.dtype == np.uint8
    assert img.pd.shape == (128, 128, 3) and img.pd.dtype == np.uint8
    assert img.mask.dtype == bool


def test_deterministic():
    a, b = render("WRIST", 3, 2, seed=7), render("WRIST", 3, 2, seed=7)
    assert np.array_equal(a.gs, b.gs) and np.array_equal(a.pd, b.pd)


@pytest.mark.parametrize("joint", ["MCP2", "MCP3", "WRIST"])
def test_synovium_grows_with_gs(joint):
    areas = [np.mean([render(joint, g, 0, s).mask.sum() for s in range(8)]) for g in range(4)]
    assert areas[0] == 0
    assert areas[1] < areas[2] < areas[3]


@pytest.mark.parametrize("joint", ["MCP3", "WRIST"])
@pytest.mark.parametrize("gs", [1, 2, 3])
def test_pd_coverage_matches_grade(joint, gs):
    for seed in range(4):
        assert colour_inside(render(joint, gs, 0, seed)) == 0
        assert 0 < colour_inside(render(joint, gs, 1, seed)) < 0.5
        assert 0 < colour_inside(render(joint, gs, 2, seed)) < 0.5
        assert colour_inside(render(joint, gs, 3, seed)) > 0.5


def test_invalid_inputs():
    with pytest.raises(ValueError):
        render("KNEE", 1, 0)
    with pytest.raises(ValueError):
        render("MCP2", 0, 1)


def test_generate_dataset_round_trip(tmp_path):
    import json

    from rausnet.data.manifest import read_manifest
    from rausnet.data.synthetic import generate_dataset

    records = generate_dataset(tmp_path, n_patients=3, seed=1, size=64)
    assert len(records) == 18
    assert read_manifest(tmp_path / "manifest.csv") == records
    assert all((tmp_path / r.gs_path).exists() and (tmp_path / r.pd_path).exists() for r in records)
    info = json.loads((tmp_path / "dataset_info.json").read_text())
    assert info["images"] == 18 and sum(info["combined_grade_counts"].values()) == 18


def test_generate_dataset_reproducible(tmp_path):
    from rausnet.data.synthetic import generate_dataset

    generate_dataset(tmp_path / "a", 2, seed=3, size=64)
    generate_dataset(tmp_path / "b", 2, seed=3, size=64)
    assert (tmp_path / "a/manifest.csv").read_bytes() == (tmp_path / "b/manifest.csv").read_bytes()
    assert (tmp_path / "a/gs").iterdir() and all(
        (tmp_path / "a/gs" / f.name).read_bytes() == f.read_bytes() for f in (tmp_path / "b/gs").iterdir()
    )
