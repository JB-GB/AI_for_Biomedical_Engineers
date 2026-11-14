"""Patient-level k-fold cross-validation for model selection (T4.3 / T4.6).

    python -m rausnet.cv --config configs/cv/compare.yaml [--only b0_us ...]

Only the development pool (train + val of the frozen hash split) is used: the test split is
never touched, so choosing a backbone or augmentation does not leak into the final estimate.
Patients are assigned to ``k`` folds by hash (``fold_salt``). For fold ``i`` the model trains
on ``k - 2`` folds, early-stops on fold ``i + 1`` and predicts fold ``i``; the k
out-of-fold (OOF) predictions cover every development image exactly once and are never used
for early stopping or calibration.

Each experiment writes ``<output_dir>/<experiment>/`` with ``oof.npz`` and ``cv_report.json``
(OOF metrics, per-fold combined QWK, patient-level bootstrap CI). ``comparison.json`` adds a
paired bootstrap of every experiment against the ``reference`` experiment on the same images.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from rausnet.data.manifest import Record, read_manifest
from rausnet.data.split import split_records


def assign_fold(patient_id: str, k: int, salt: str = "rausnet-cv-v1") -> int:
    """Deterministic fold of a patient (independent of the train/val/test salt)."""
    digest = hashlib.sha256(f"{salt}:{patient_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % k


def make_folds(records: list[Record], k: int, salt: str = "rausnet-cv-v1") -> list[list[Record]]:
    if k < 3:
        raise ValueError("k must be >= 3 (one fold to predict, one to early-stop, the rest to train)")
    folds: list[list[Record]] = [[] for _ in range(k)]
    for r in records:
        folds[assign_fold(r.patient_id, k, salt)].append(r)
    if any(not f for f in folds):
        raise ValueError(f"empty fold with k={k}: too few patients")
    return folds


def fold_splits(folds: list[list[Record]], i: int) -> dict[str, list[Record]]:
    """Train on all folds except ``i`` and ``i + 1``; early-stop on ``i + 1``; predict ``i``."""
    k = len(folds)
    stop = (i + 1) % k
    train = [r for j, f in enumerate(folds) if j not in (i, stop) for r in f]
    return {"train": train, "val": folds[stop], "test": [], "predict": folds[i]}


def development_pool(cfg: dict) -> list[Record]:
    d = cfg["data"]
    records = read_manifest(Path(d["root"]) / d["manifest"])
    splits = split_records(records, tuple(d["split_ratios"]), d["split_salt"])
    return splits["train"] + splits["val"]


def oof_report(oof: dict, n_boot: int = 2000) -> dict:
    from rausnet.evaluate import combined_labels, evaluate_predictions
    from rausnet.metrics import bootstrap_ci, combined_probabilities, quadratic_weighted_kappa

    p_gs, p_pd, y_gs, y_pd = oof["p_gs"], oof["p_pd"], oof["y_gs"], oof["y_pd"]
    report = evaluate_predictions(p_gs, p_pd, y_gs, y_pd)
    y_comb = combined_labels(y_gs, y_pd)
    y_hat = combined_probabilities(p_gs, p_pd).argmax(1)
    report["combined"]["qwk_ci95"] = bootstrap_ci(y_comb, y_hat, list(oof["patient_id"]), n_boot=n_boot)
    per_fold = [quadratic_weighted_kappa(y_comb[oof["fold"] == f], y_hat[oof["fold"] == f])
                for f in np.unique(oof["fold"])]
    report["per_fold_combined_qwk"] = per_fold
    report["per_fold_mean_sd"] = [float(np.mean(per_fold)), float(np.std(per_fold, ddof=1))]
    report["n_images"], report["n_patients"] = len(y_gs), len(set(oof["patient_id"]))
    return report


def run_experiment(cfg: dict, name: str, k: int, salt: str, out_dir: Path) -> dict:
    import torch
    from torch.utils.data import DataLoader

    from rausnet.data.dataset import JointDataset
    from rausnet.evaluate import load_model, predict
    from rausnet.train import pick_device, train

    exp_dir = out_dir / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    folds = make_folds(development_pool(cfg), k, salt)
    device = pick_device()
    parts = []
    for i in range(k):
        splits = fold_splits(folds, i)
        run_cfg = copy.deepcopy(cfg)
        run_cfg["run_name"], run_cfg["output_dir"] = f"fold{i}", str(exp_dir)
        target = splits.pop("predict")
        print(f"[{name}] fold {i + 1}/{k}: train {len(splits['train'])}, early-stop {len(splits['val'])}, "
              f"predict {len(target)} images", flush=True)
        run = train(run_cfg, splits=splits)
        model, _ = load_model(run / "model.pt", device)
        loader = DataLoader(JointDataset(target, cfg["data"]["root"], cfg["data"]["image_size"]),
                            batch_size=cfg["train"]["batch_size"])
        p_gs, p_pd, y_gs, y_pd = predict(model, loader, device, cfg["train"].get("amp", False))
        parts.append({"p_gs": p_gs, "p_pd": p_pd, "y_gs": y_gs, "y_pd": y_pd, "fold": np.full(len(target), i),
                      "image_id": np.array([r.image_id for r in target]),
                      "patient_id": np.array([r.patient_id for r in target])})
        (run / "model.pt").unlink()  # fold models are not reused; keep disk usage small
        del model
        torch.cuda.empty_cache()
    oof = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    order = np.argsort(oof["image_id"])
    oof = {key: v[order] for key, v in oof.items()}
    np.savez(exp_dir / "oof.npz", **oof)
    report = {"experiment": name, "k": k, "fold_salt": salt, **oof_report(oof)}
    (exp_dir / "cv_report.json").write_text(json.dumps(report, indent=2))
    c = report["combined"]
    print(f"[{name}] OOF combined QWK {c['qwk']:.4f} CI {c['qwk_ci95']}", flush=True)
    return report


def compare(out_dir: Path, names: list[str], reference: str, n_boot: int = 2000) -> dict:
    """Paired patient-level bootstrap of each experiment's OOF combined QWK minus the reference."""
    from rausnet.evaluate import combined_labels
    from rausnet.metrics import combined_probabilities, paired_bootstrap

    def load(n):
        return dict(np.load(out_dir / n / "oof.npz"))

    ref = load(reference)
    y = combined_labels(ref["y_gs"], ref["y_pd"])
    ref_hat = combined_probabilities(ref["p_gs"], ref["p_pd"]).argmax(1)
    result = {"reference": reference, "experiments": {}}
    for n in names:
        o = load(n)
        if not np.array_equal(o["image_id"], ref["image_id"]):
            raise ValueError(f"{n}: OOF images differ from the reference")
        hat = combined_probabilities(o["p_gs"], o["p_pd"]).argmax(1)
        result["experiments"][n] = paired_bootstrap(y, ref_hat, hat, list(ref["patient_id"]), n_boot=n_boot)
    (out_dir / "comparison.json").write_text(json.dumps(result, indent=2))
    return result


def main(argv=None) -> None:
    from rausnet.train import load_config

    parser = argparse.ArgumentParser(description="Patient-level k-fold cross-validation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--only", nargs="*", default=None, help="run only these experiments")
    parser.add_argument("--compare-only", action="store_true", help="skip training, recompute comparison")
    args = parser.parse_args(argv)
    spec = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(spec["output_dir"])
    names = list(spec["experiments"])
    if not args.compare_only:
        for name in args.only or names:
            base = yaml.safe_load(Path(spec["base"]).read_text())
            overrides = [*spec.get("overrides", []), *spec["experiments"][name]]
            cfg = load_config(None, base, overrides)
            run_experiment(cfg, name, spec["folds"], spec.get("fold_salt", "rausnet-cv-v1"), out_dir)
    done = [n for n in names if (out_dir / n / "oof.npz").exists()]
    if spec["reference"] in done:
        result = compare(out_dir, done, spec["reference"])
        for n, b in result["experiments"].items():
            print(f"{n:>18}: OOF QWK {b['b']:.4f}  diff vs {spec['reference']} {b['diff']:+.4f} "
                  f"[{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]")


if __name__ == "__main__":
    main()
