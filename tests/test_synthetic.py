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
