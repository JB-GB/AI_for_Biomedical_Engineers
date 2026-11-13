"""Retraining loop with a promotion rule.

    python -m rausnet.retrain --config configs/retrain/default.yaml --add hospital/batch1/manifest.csv

Steps:
1. Merge the current model's dataset with the new source manifests into a new
   dataset version (``<data_root>/datasets/<version>/manifest.csv``).
2. Fine-tune from the current model (or train from scratch if none exists).
3. Evaluate candidate and current model on the SAME frozen test split
   (hash-based patient split; existing test patients never move).
4. Promotion rule (``promotion.rule``):
   - ``bootstrap`` (default): paired patient-level bootstrap of candidate - current
     combined QWK; reject only if the whole 95 % CI is below 0 (significantly worse).
   - ``tolerance``: promote if candidate >= current - tolerance.
5. Register the version (current / rejected) with a retrain report.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

from rausnet.data.manifest import read_manifest
from rausnet.data.merge import load_source, merge_records, write_dataset
from rausnet.evaluate import combined_labels, evaluate_predictions, predict_checkpoint
from rausnet.metrics import bootstrap_ci, combined_probabilities, paired_bootstrap
from rausnet.registry import ModelEntry, Registry
from rausnet.train import load_config, train

STAGE_BY_SOURCE = {"synthetic": "S2", "public": "S1", "hospital": "S3"}
STAGE_ORDER = ["S0", "S1", "S2", "S3"]


def promotion_decision(candidate: dict, current: dict | None, metric: str = "qwk",
                       tolerance: float = 0.0) -> tuple[bool, str]:
    """Return (promote, reason) comparing combined-grade metrics."""
    cand = candidate["combined"][metric]
    if cand is None or math.isnan(cand):
        return False, f"candidate {metric} is undefined"
    if current is None:
        return True, "first model"
    cur = current["combined"][metric]
    if cur is None or math.isnan(cur):
        return True, f"current {metric} undefined on this split"
    if cand >= cur - tolerance:
        return True, f"candidate {metric} {cand:.4f} >= current {cur:.4f} - {tolerance}"
    return False, f"candidate {metric} {cand:.4f} < current {cur:.4f} - {tolerance}"


def bootstrap_decision(boot: dict | None) -> tuple[bool, str]:
    """Promote unless the candidate is significantly worse (CI of the difference entirely < 0)."""
    if boot is None:
        return True, "first model"
    if math.isnan(boot["b"]):
        return False, "candidate qwk is undefined"
    ci = f"diff {boot['diff']:+.4f}, 95% CI [{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}]"
    if boot["ci_high"] < 0:
        return False, f"candidate significantly worse ({ci})"
    return True, f"candidate not significantly worse ({ci})"


def _report(records, preds, n_boot: int, seed: int) -> dict:
    report = evaluate_predictions(*preds)
    y = combined_labels(preds[2], preds[3])
    y_hat = combined_probabilities(preds[0], preds[1]).argmax(axis=1)
    report["combined"]["qwk_ci95"] = bootstrap_ci(y, y_hat, [r.patient_id for r in records],
                                                  n_boot=n_boot, seed=seed)
    report["n_patients"] = len({r.patient_id for r in records})
    return report


def dataset_stage(records) -> str:
    """Highest data stage present: hospital data makes a model S3 (still not clinically validated)."""
    stages = {STAGE_BY_SOURCE[r.source] for r in records}
    return max(stages, key=STAGE_ORDER.index)


def _summary(report: dict | None) -> dict:
    if report is None:
        return {}
    c = report["combined"]
    return {k: c[k] for k in ("qwk", "accuracy", "balanced_accuracy", "auc_ge2", "n")}


def retrain(cfg: dict, add: list[str]) -> ModelEntry:
    data_root = Path(cfg["data"]["root"])
    reg = Registry(cfg["models_dir"])
    cur = reg.current()

    previous = read_manifest(data_root / cur.dataset_manifest) if cur else []
    new = merge_records(*(load_source(data_root, m) for m in add))
    combined = merge_records(previous, new)
    if cur and len(combined) == len(previous):
        raise ValueError("no new images: every --add record is already in the current dataset")

    version = reg.next_version()
    manifest_rel = write_dataset(data_root, version, combined)

    run_cfg = copy.deepcopy(cfg)
    run_cfg["run_name"] = f"retrain_{version}"
    run_cfg["stage"] = dataset_stage(combined)
    run_cfg["data"]["manifest"] = manifest_rel
    t = run_cfg["train"]
    cur_ckpt = None
    if cur:
        import torch

        cur_ckpt = reg.dir / cur.model_path
        run_cfg["model"] = torch.load(cur_ckpt, map_location="cpu", weights_only=False)["config"]["model"]
        t["init_checkpoint"] = str(cur_ckpt)
        t["epochs"] = t.get("finetune_epochs", t["epochs"])
        t["lr"] = t.get("finetune_lr", t["lr"])
    else:
        t["init_checkpoint"] = None

    run_dir = train(run_cfg)

    promo = cfg.get("promotion", {})
    sources = promo.get("sources")
    n_boot, seed = promo.get("n_boot", 2000), promo.get("seed", 0)
    recs, cand_preds = predict_checkpoint(run_dir / "model.pt", run_cfg, "test", sources)
    cand_report = _report(recs, cand_preds, n_boot, seed)
    cur_report = boot = None
    if cur:
        _, cur_preds = predict_checkpoint(cur_ckpt, run_cfg, "test", sources)
        cur_report = _report(recs, cur_preds, n_boot, seed)
        y = combined_labels(cand_preds[2], cand_preds[3])
        boot = paired_bootstrap(
            y, combined_probabilities(*cur_preds[:2]).argmax(1), combined_probabilities(*cand_preds[:2]).argmax(1),
            [r.patient_id for r in recs], n_boot=n_boot, seed=seed,
        )
    rule = promo.get("rule", "bootstrap")
    if rule == "bootstrap":
        promote, reason = bootstrap_decision(boot)
    elif rule == "tolerance":
        promote, reason = promotion_decision(cand_report, cur_report, promo.get("metric", "qwk"),
                                             promo.get("tolerance", 0.0))
    else:
        raise ValueError(f"unknown promotion rule {rule!r}")

    vdir = reg.dir / version
    vdir.mkdir(parents=True, exist_ok=False)
    for name in ("model.pt", "model_card.json", "config.yaml", "history.csv"):
        if (run_dir / name).exists():
            shutil.copy2(run_dir / name, vdir / name)
    report = {
        "version": version, "parent": cur.version if cur else None, "run_dir": str(run_dir),
        "dataset_manifest": manifest_rel, "images": len(combined), "new_images": len(combined) - len(previous),
        "promotion_rule": promo, "promoted": promote, "reason": reason,
        "paired_bootstrap": boot, "candidate": cand_report, "current": cur_report,
    }
    (vdir / "retrain_report.json").write_text(json.dumps(report, indent=2))

    entry = ModelEntry(
        version=version,
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
        model_path=f"{version}/model.pt",
        dataset_manifest=manifest_rel,
        dataset_sha256=hashlib.sha256((data_root / manifest_rel).read_bytes()).hexdigest(),
        stage=run_cfg["stage"],
        parent=cur.version if cur else None,
        test_metrics=_summary(cand_report),
        decision=reason,
    )
    reg.add(entry, promote)
    print(f"{version}: {'PROMOTED' if promote else 'REJECTED'} ({reason})")
    return entry


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Retrain with new data and apply the promotion rule.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--add", action="append", required=True, metavar="MANIFEST",
                        help="source manifest inside the data root (repeatable)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    retrain(load_config(args.config, overrides=args.set), args.add)


if __name__ == "__main__":
    main()
