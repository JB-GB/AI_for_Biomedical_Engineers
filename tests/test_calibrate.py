import numpy as np
import pytest

from rausnet.calibrate import apply_temperature, cross_fit_gain, fit_temperature
from rausnet.metrics import negative_log_likelihood


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def test_apply_temperature_keeps_argmax_and_matches_logit_scaling():
    z = np.random.default_rng(0).normal(size=(50, 4)) * 3
    p = softmax(z)
    for t in (0.5, 1.0, 3.0):
        q = apply_temperature(p, t)
        assert np.allclose(q, softmax(z / t), atol=1e-9)
        assert (q.argmax(1) == p.argmax(1)).all()


def test_fit_temperature_recovers_overconfidence():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(4000, 4)) * 2
    y = np.array([rng.choice(4, p=row) for row in softmax(z)])  # labels drawn from calibrated probs
    overconfident = softmax(z * 3)  # same model, logits x3 -> optimal T ~ 3
    t = fit_temperature(overconfident, y)
    assert t == pytest.approx(3, rel=0.1)
    assert negative_log_likelihood(y, apply_temperature(overconfident, t)) < negative_log_likelihood(y, overconfident)


def test_fit_temperature_ignores_missing_labels():
    p = softmax(np.random.default_rng(2).normal(size=(10, 4)))
    y = np.array([0, 1, 2, 3, -1, -1, 0, 1, 2, 3])
    assert 0.05 <= fit_temperature(p, y) <= 20


def _sharp_probs(n, conf, acc, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 4, n)
    pred = np.where(rng.random(n) < acc, y, (y + 1) % 4)
    probs = np.full((n, 4), (1 - conf) / 3)
    probs[np.arange(n), pred] = conf
    return probs, y


def test_cross_fit_gain_detects_overconfidence_and_calibrated_models():
    groups = [f"P{i // 2}" for i in range(2000)]
    probs, y = _sharp_probs(2000, conf=0.99, acc=0.7, seed=1)  # very overconfident -> scaling helps
    assert cross_fit_gain(probs, y, groups)["helps"]
    probs, y = _sharp_probs(2000, conf=0.7, acc=0.7, seed=2)   # already calibrated -> no gain
    assert not cross_fit_gain(probs, y, groups)["helps"]
