import numpy as np
import pytest

from rausnet.metrics import (
    balanced_accuracy,
    classification_report,
    combined_probabilities,
    confusion_matrix,
    consistent_pd_probabilities,
    coverage_curve,
    quadratic_weighted_kappa,
    referral_outcome,
    referral_threshold,
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



def test_consistent_pd_probabilities():
    p_gs = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0.4, 0.6, 0, 0]], float)
    p_pd = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    out = consistent_pd_probabilities(p_gs, p_pd)
    assert np.allclose(out.sum(axis=1), 1)
    assert np.allclose(out[0], [1, 0, 0, 0])  # certain GS0 -> PD0 whatever the PD head says
    assert np.allclose(out[1], [0, 0, 1, 0])  # synovium present -> PD head unchanged
    assert np.allclose(out[2], [0.4, 0, 0, 0.6])
    assert np.allclose(p_pd[0], [0, 0, 0, 1])  # input not modified


def test_report_keys():
    probs = np.eye(4)[[0, 1, 2, 3]]
    rep = classification_report([0, 1, 2, 3], probs)
    assert rep["accuracy"] == 1.0 and rep["qwk"] == pytest.approx(1.0) and rep["auc_ge2"] == 1.0


def _toy(seed=0, n_patients=40, per=6):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 4, n_patients * per)
    groups = np.repeat(np.arange(n_patients), per)
    return y, groups, rng


def test_bootstrap_ci_contains_point_estimate():
    from rausnet.metrics import bootstrap_ci

    y, groups, rng = _toy()
    pred = np.where(rng.random(len(y)) < 0.8, y, rng.integers(0, 4, len(y)))
    lo, hi = bootstrap_ci(y, pred, groups, n_boot=300)
    assert lo <= quadratic_weighted_kappa(y, pred) <= hi < 1


def test_paired_bootstrap_identical_models():
    from rausnet.metrics import paired_bootstrap

    y, groups, _ = _toy()
    res = paired_bootstrap(y, y, y, groups, n_boot=200)
    assert res["diff"] == 0 and res["ci_low"] == 0 and res["ci_high"] == 0 and res["n_groups"] == 40


def test_paired_bootstrap_detects_worse_model():
    from rausnet.metrics import paired_bootstrap

    y, groups, rng = _toy(1)
    good = np.where(rng.random(len(y)) < 0.95, y, rng.integers(0, 4, len(y)))
    bad = np.where(rng.random(len(y)) < 0.4, y, rng.integers(0, 4, len(y)))
    res = paired_bootstrap(y, good, bad, groups, n_boot=300)
    assert res["diff"] < 0 and res["ci_high"] < 0


def test_paired_bootstrap_is_deterministic():
    from rausnet.metrics import paired_bootstrap

    y, groups, rng = _toy(2)
    p = np.where(rng.random(len(y)) < 0.7, y, 0)
    assert paired_bootstrap(y, y, p, groups, n_boot=100) == paired_bootstrap(y, y, p, groups, n_boot=100)


def test_calibration_metrics():
    from rausnet.metrics import expected_calibration_error, negative_log_likelihood, reliability_bins

    y = np.array([0, 1, 2, 3])
    perfect = np.eye(4)[y]
    assert expected_calibration_error(y, perfect) == 0
    assert negative_log_likelihood(y, perfect) == pytest.approx(0, abs=1e-9)
    # Always 90 % confident but only 50 % right -> ECE 0.4
    probs = np.full((4, 4), 0.1 / 3)
    probs[np.arange(4), [0, 1, 0, 0]] = 0.9
    assert expected_calibration_error(y, probs) == pytest.approx(0.4)
    assert sum(b["n"] for b in reliability_bins(y, probs)) == 4


def test_referral_threshold_picks_max_coverage_meeting_target():
    # confidences 0.9 (right), 0.8 (right), 0.7 (wrong), 0.6 (right)
    probs = np.array([[0.9, 0.1, 0, 0], [0.8, 0.2, 0, 0], [0.7, 0.3, 0, 0], [0.6, 0.4, 0, 0]])
    y = np.array([0, 0, 1, 0])
    r = referral_threshold(y, probs, target_accuracy=0.95)
    assert r["threshold"] == 0.8 and r["coverage"] == 0.5 and r["accuracy_accepted"] == 1.0
    r = referral_threshold(y, probs, target_accuracy=0.75)
    assert r["threshold"] == 0.6 and r["coverage"] == 1.0
    out = referral_outcome(y, probs, 0.8)
    assert out["n_referred"] == 2 and out["accuracy_referred"] == 0.5 and out["accuracy_accepted"] == 1.0


def test_referral_threshold_never_splits_ties_and_can_refer_all():
    probs = np.array([[0.7, 0.3, 0, 0], [0.7, 0.3, 0, 0]])
    r = referral_threshold(np.array([0, 1]), probs, target_accuracy=0.9)
    assert r["coverage"] == 0.0 and r["threshold"] > 1.0


def test_coverage_curve_accuracy_rises_as_coverage_falls():
    probs = np.array([[0.9, 0.1, 0, 0], [0.8, 0.2, 0, 0], [0.7, 0.3, 0, 0], [0.6, 0.4, 0, 0]])
    curve = coverage_curve(np.array([0, 0, 1, 1]), probs, [0.5, 1.0])
    assert curve[0] == {"coverage": 0.5, "accuracy": 1.0, "min_confidence": 0.8}
    assert curve[1]["accuracy"] == 0.5
