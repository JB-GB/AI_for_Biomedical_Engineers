"""Compare two training runs, e.g. the same config on local GPU, Colab and the self-hosted runner (T7.4).

Usage: python scripts/compare_runs.py <run_a> <run_b> [--bootstrap] [--data-root DIR] [--out report.json]

Checks that both runs used the same data (manifest hash, split sizes) and prints their test
metrics side by side. With ``--bootstrap`` both checkpoints are re-run on run A's test split and
the combined-QWK difference gets a paired patient-level 95 % CI ("equivalent" when the CI
contains 0). ``--data-root`` remaps data.root when runs come from another machine.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

KEYS = ("qwk", "accuracy", "balanced_accuracy", "auc_ge2")


def load(run: Path) -> dict:
    card = json.loads((run / "model_card.json").read_text())
    test = json.loads((run / "test_metrics.json").read_text())
    return {"card": card, "test": test}


def compare(run_a: Path, run_b: Path, bootstrap: bool = False, data_root: str | None = None) -> dict:
    a, b = load(run_a), load(run_b)
    same_data = (a["card"]["manifest_sha256"] == b["card"]["manifest_sha256"]
                 and a["card"]["split_sizes"] == b["card"]["split_sizes"])
    report = {
        "runs": [str(run_a), str(run_b)],
        "same_data": same_data,
        "environment": [{k: r["card"].get(k) for k in ("device", "torch", "git_commit", "best_epoch")} for r in (a, b)],
        "test_combined": {k: [a["test"]["combined"][k], b["test"]["combined"][k]] for k in KEYS},
    }
    if bootstrap:
        import yaml

        from rausnet.evaluate import combined_labels, predict_checkpoint
        from rausnet.metrics import combined_probabilities, paired_bootstrap
        from rausnet.train import load_config

        cfg = yaml.safe_load((run_a / "config.yaml").read_text())
        if data_root:
            cfg["data"]["root"] = data_root
        cfg = load_config(None, cfg)
        preds = []
        for run in (run_a, run_b):
            recs, (pg, pp, yg, yp) = predict_checkpoint(run / "model.pt", cfg, "test")
            preds.append(combined_probabilities(pg, pp).argmax(1))
        y = combined_labels(yg, yp)
        boot = paired_bootstrap(y, preds[0], preds[1], [r.patient_id for r in recs])
        boot["equivalent"] = bool(boot["ci_low"] <= 0 <= boot["ci_high"])
        report["paired_bootstrap"] = boot
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    report = compare(args.run_a, args.run_b, args.bootstrap, args.data_root)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text)
    if not report["same_data"]:
        raise SystemExit("runs used different data: metrics are not comparable")


if __name__ == "__main__":
    main()
