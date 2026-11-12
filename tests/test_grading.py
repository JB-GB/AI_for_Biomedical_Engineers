import pytest

from rausnet.grading import combined_score, validate_grade


@pytest.mark.parametrize(
    ("gs", "pd", "expected"),
    [(0, 0, 0), (1, 0, 1), (1, 1, 1), (1, 2, 2), (2, 1, 2), (1, 3, 3), (2, 3, 3), (3, 0, 3)],
)
def test_combined_score(gs, pd, expected):
    assert combined_score(gs, pd) == expected


def test_pd_without_gs_is_rejected():
    with pytest.raises(ValueError):
        combined_score(0, 2)


@pytest.mark.parametrize("bad", [-1, 4, 1.0, "1", True, None])
def test_invalid_grade(bad):
    with pytest.raises(ValueError):
        validate_grade(bad)
