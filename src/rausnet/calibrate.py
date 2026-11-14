"""Temperature scaling of the GS and PD heads (confidence calibration).

    python -m rausnet.calibrate [--models-dir models] [--version v0003] [--data-root DIR]

Fits one temperature per head on the validation split of the model's dataset by
minimising negative log-likelihood, then reports ECE/NLL before and after on
validation and test. Writes ``<models_dir>/<version>/calibration.json``; the ONNX
export copies the temperatures into the sidecar so the API serves calibrated
probabilities. Temperature scaling never changes the predicted grade.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from rausnet.metrics import (
    combined_probabilities,
    consistent_pd_probabilities,
    expected_calibration_error,
    negative_log_likelihood,
    reliability_bins,
)


def apply_temperature(probs: np.ndarray, t: float) -> np.ndarray:
    """Rescale probabilities as softmax(log p / T) (equivalent to dividing logits by T)."""
    z = np.log(np.clip(probs, 1e-12, 1)) / t
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(probs: np.ndarray, y: np.ndarray, lo: float = 0.05, hi: float = 20.0) -> float:
    """Temperature minimising NLL (golden-section search on log T)."""
    mask = y >= 0
    probs, y = probs[mask], y[mask]
    f = lambda log_t: negative_log_likelihood(y, apply_temperature(probs, float(np.exp(log_t))))  # noqa: E731
    a, b = np.log(lo), np.log(hi)
    g = (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    for _ in range(80):
        if f(c) < f(d):
            b, d = d, c
            c = b - g * (b - a)
        else:
            a, c = c, d
            d = a + g * (b - a)
    return float(np.exp((a + b) / 2))


def cross_fit_gain(probs: np.ndarray, y: np.ndarray, groups) -> dict:
    """Does temperature scaling help on data it was not fitted on?

    Patients are split into two halves by hash; T is fitted on one half and scored (NLL) on
    the other, both ways. ``helps`` is True when the cross-fitted NLL beats T = 1.
    """
    import hashlib

    groups = np.asarray(groups)
    half = np.array([hashlib.sha256(f"calib:{g}".encode()).digest()[0] % 2 for g in groups])
    labelled = y >= 0
    nll_t, nll_1 = 0.0, 0.0
    for h in (0, 1):
        fit, score = (half == h) & labelled, (half != h) & labelled
        if not fit.any() or not score.any():
            return {"helps": False, "reason": "too few patients for cross-fitting"}
        t = fit_temperature(probs[fit], y[fit])
        w = score.sum()
        nll_t += w * negative_log_likelihood(y[score], apply_temperature(probs[score], t))
        nll_1 += w * negative_log_likelihood(y[score], probs[score])
    n = labelled.sum()
    return {"helps": bool(nll_t < nll_1), "cross_fit_nll": float(nll_t / n), "nll_t1": float(nll_1 / n)}


def _summary(p_gs, p_pd, y_gs, y_pd) -> dict:
    from rausnet.evaluate import combined_labels

    has_pd = y_pd >= 0
    p_comb = combined_probabilities(p_gs, p_pd)
    p_pd_rule = consistent_pd_probabilities(p_gs, p_pd)
    y_comb = combined_labels(y_gs, y_pd)
    return {
        "gs": {"ece": expected_calibration_error(y_gs, p_gs), "nll": negative_log_likelihood(y_gs, p_gs)},
        "pd": {"ece": expected_calibration_error(y_pd[has_pd], p_pd_rule[has_pd]),
               "nll": negative_log_likelihood(y_pd[has_pd], p_pd_rule[has_pd])},
        "combined": {"ece": expected_calibration_error(y_comb, p_comb), "nll": negative_log_likelihood(y_comb, p_comb),
                     "bins": reliability_bins(y_comb, p_comb)},
    }


def calibrate(models_dir: str | Path = "models", version: str | None = None, data_root: str | None = None,
              target_accuracy: float = 0.95, gate: bool = True) -> dict:
    """Fit per-head temperatures (kept only where cross-fitting shows they help, if ``gate``) and
    the "refer to specialist" confidence threshold on the validation split; report both on test."""
    import torch

    from rausnet.evaluate import predict_checkpoint
    from rausnet.registry import Registry
    from rausnet.train import load_config

    reg = Registry(models_dir)
    entry = reg.get(version) if version else reg.current()
    if entry is None:
        raise ValueError(f"no current model in {reg.path}")
    ckpt = reg.dir / entry.model_path
    cfg = torch.load(ckpt, map_location="cpu", weights_only=False)["config"]
    cfg["data"]["manifest"] = entry.dataset_manifest
    if data_root:
        cfg["data"]["root"] = data_root
    cfg = load_config(None, cfg)

    val_recs, (vg, vp, vyg, vyp) = predict_checkpoint(ckpt, cfg, "val")
    _, (tg, tp, tyg, typ) = predict_checkpoint(ckpt, cfg, "test")
    # a PD head trained with pd_mask_gs0 never saw GS 0 joints: fit its temperature where PD is defined
    vyp_fit = np.where(vyg == 0, -1, vyp) if cfg["train"].get("pd_mask_gs0") else vyp
    t_gs, t_pd = fit_temperature(vg, vyg), fit_temperature(vp, vyp_fit)
    patients = [r.patient_id for r in val_recs]
    gating = {"gs": cross_fit_gain(vg, vyg, patients), "pd": cross_fit_gain(vp, vyp_fit, patients)}
    fitted = {"gs": t_gs, "pd": t_pd}
    if gate:
        t_gs = t_gs if gating["gs"]["helps"] else 1.0
        t_pd = t_pd if gating["pd"]["helps"] else 1.0

    def scaled(g, p):
        return apply_temperature(g, t_gs), apply_temperature(p, t_pd)

    from rausnet.evaluate import combined_labels
    from rausnet.metrics import referral_outcome, referral_threshold

    v_comb = combined_probabilities(*scaled(vg, vp))
    t_comb = combined_probabilities(*scaled(tg, tp))
    ref = referral_threshold(combined_labels(vyg, vyp), v_comb, target_accuracy)
    result = {
        "version": entry.version,
        "method": "temperature scaling per head, NLL on validation split"
                  + (", kept only if 2-fold patient cross-fitting improves NLL" if gate else ""),
        "temperature": {"gs": t_gs, "pd": t_pd},
        "fitted_temperature": fitted,
        "gating": gating,
        "referral": {**ref, "val_outcome": referral_outcome(combined_labels(vyg, vyp), v_comb, ref["threshold"]),
                     "test_outcome": referral_outcome(combined_labels(tyg, typ), t_comb, ref["threshold"])},
        "val": {"before": _summary(vg, vp, vyg, vyp), "after": _summary(*scaled(vg, vp), vyg, vyp)},
        "test": {"before": _summary(tg, tp, tyg, typ), "after": _summary(*scaled(tg, tp), tyg, typ)},
    }
    out = ckpt.parent / "calibration.json"
    out.write_text(json.dumps(result, indent=2))
    b, a = result["test"]["before"]["combined"], result["test"]["after"]["combined"]
    print(f"{entry.version}: T_gs={t_gs:.3f} T_pd={t_pd:.3f} | test combined ECE {b['ece']:.4f} -> {a['ece']:.4f}, "
          f"NLL {b['nll']:.4f} -> {a['nll']:.4f} -> {out}")
    to = result["referral"]["test_outcome"]
    print(f"refer below confidence {ref['threshold']:.3f} (target accuracy {target_accuracy:.0%} on val, coverage "
          f"{ref['coverage']:.0%}) | test: coverage {to['coverage']:.0%}, accuracy accepted {to['accuracy_accepted']}, "
          f"referred {to['n_referred']} with accuracy {to['accuracy_referred']}")
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Fit temperature scaling for a registry model.")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--version", default=None)
    parser.add_argument("--data-root", default=os.environ.get("RAUSNET_DATA_DIR"))
    parser.add_argument("--target-accuracy", type=float, default=0.95,
                        help="accuracy required on automatically graded cases (val split)")
    parser.add_argument("--no-gate", action="store_true", help="always apply the fitted temperatures")
    args = parser.parse_args(argv)
    calibrate(args.models_dir, args.version, args.data_root, args.target_accuracy, not args.no_gate)


if __name__ == "__main__":
    main()
