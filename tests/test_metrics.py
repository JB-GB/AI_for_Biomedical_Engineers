import numpy as np
import pytest

from rausnet.metrics import (
    balanced_accuracy,
    classification_report,
    combined_probabilities,
    confusion_matrix,
    quadratic_weighted_kappa,
    roc_auc,
)


def test_confusion_matrix():
    cm = confusion_matrix([0, 1, 2, 3, 3], [0, 1, 1, 3, 2])
    assert cm.sum() == 5 and cm[2, 1] == 1 and cm[3, 2] == 1


def test_qwk_perfect_and_ordinal():
    y = [0, 1, 2, 3, 0, 1, 2, 3]
    assert quadratic_weighted_kappa(y, y) == pytest.approx(1.0)
    near = quadratic_weighted_kappa(y, [0, 1, 2, 2, 0, 1, 2, 2])
    far = quadratic_weighted_kappa(y, [0, 1, 2, 0, 0, 1, 2, 0])
    assert near > far


def test_qwk_known_value():
    # Hand-computed: observed weighted disagreement 1/9, expected 8/9 -> 1 - 1/8.
    assert quadratic_weighted_kappa([0, 1, 2, 3], [0, 1, 2, 2]) == pytest.approx(0.875)


def test_balanced_accuracy_ignores_missing_classes():
    assert balanced_accuracy([0, 0, 1, 1], [0, 0, 1, 0]) == pytest.approx(0.75)


def test_auc():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert np.isnan(roc_auc([1, 1], [0.1, 0.2]))


def test_combined_probabilities_rule():
    p_gs = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0.5, 0.5, 0]], float)
    p_pd = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0]], float)
    out = combined_probabilities(p_gs, p_pd)
    assert np.allclose(out.sum(axis=1), 1)
    assert np.allclose(out[0], [0, 0, 0, 1])  # GS1 + PD3 -> 3
    assert np.allclose(out[1], [1, 0, 0, 0])  # GS0 -> 0 regardless of PD
    assert np.allclose(out[2], [0, 0.5, 0.5, 0])


def test_report_keys():
    probs = np.eye(4)[[0, 1, 2, 3]]
    rep = classification_report([0, 1, 2, 3], probs)
    assert rep["accuracy"] == 1.0 and rep["qwk"] == pytest.approx(1.0) and rep["auc_ge2"] == 1.0
