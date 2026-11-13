"""Prediction and evaluation of a trained run on a patient-level split.

Usage: python -m rausnet.evaluate --run outputs/<run_dir> [--split test]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from rausnet.metrics import classification_report, combined_probabilities


@torch.no_grad()
def predict(model, loader: DataLoader, device: torch.device, amp: bool = False):
    """Return (p_gs, p_pd, y_gs, y_pd) as numpy arrays."""
    model.eval()
    p_gs, p_pd, y_gs, y_pd = [], [], [], []
    for x, gs, pd in loader:
        with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
            lg, lp = model(x.to(device, non_blocking=True))
        p_gs.append(lg.float().softmax(1).cpu())
        p_pd.append(lp.float().softmax(1).cpu())
        y_gs.append(gs)
        y_pd.append(pd)
    cat = lambda t: torch.cat(t).numpy()  # noqa: E731
    return cat(p_gs), cat(p_pd), cat(y_gs), cat(y_pd)


def evaluate_predictions(p_gs, p_pd, y_gs, y_pd) -> dict:
    """Reports for GS head, PD head (labelled rows only) and the combined grade."""
    has_pd = y_pd >= 0
    p_comb = combined_probabilities(p_gs, p_pd)
    # Combined ground truth: GS 0 -> 0, else max(GS, PD); PD missing -> GS.
    y_pd_filled = np.where(has_pd, y_pd, 0)
    y_comb = np.where(y_gs == 0, 0, np.maximum(y_gs, y_pd_filled))
    report = {
        "gs": classification_report(y_gs, p_gs),
        "combined": classification_report(y_comb, p_comb),
    }
    if has_pd.any():
        report["pd"] = classification_report(y_pd[has_pd], p_pd[has_pd])
    return report


def load_model(checkpoint: str | Path, device: torch.device):
    """Load a checkpoint written by ``rausnet.train``; returns (model, checkpoint config)."""
    from rausnet.train import build_model

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_model({"model": {**cfg["model"], "pretrained": False}}).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model, cfg


def evaluate_checkpoint(checkpoint: str | Path, cfg: dict, split: str = "test",
                        sources: list[str] | None = None) -> dict:
    """Evaluate ``checkpoint`` on a split of the dataset described by ``cfg['data']``.

    The split is computed from ``cfg`` (not from the checkpoint), so two models can
    be compared on exactly the same images. ``sources`` restricts evaluation to
    records from those sources (e.g. ``["hospital"]``). Adds ``by_source`` reports.
    """
    from rausnet.data.dataset import JointDataset
    from rausnet.data.manifest import read_manifest
    from rausnet.data.split import split_records
    from rausnet.train import pick_device

    d = cfg["data"]
    root = Path(d["root"])
    records = split_records(read_manifest(root / d["manifest"]), tuple(d["split_ratios"]),
                            d["split_salt"])[split]
    if sources:
        records = [r for r in records if r.source in sources]
    if not records:
        raise ValueError(f"no records in split '{split}' for sources {sources}")
    device = pick_device()
    model, _ = load_model(checkpoint, device)
    loader = DataLoader(JointDataset(records, root, d["image_size"]), batch_size=cfg["train"]["batch_size"])
    preds = predict(model, loader, device, cfg["train"].get("amp", False))
    report = evaluate_predictions(*preds)
    report["n_patients"] = len({r.patient_id for r in records})
    src = np.array([r.source for r in records])
    if len(set(src)) > 1:
        report["by_source"] = {s: evaluate_predictions(*(p[src == s] for p in preds)) for s in sorted(set(src))}
    return report


def main(argv=None) -> None:
    from rausnet.train import load_config

    parser = argparse.ArgumentParser(description="Evaluate a trained run.")
    parser.add_argument("--run", required=True, help="run directory containing model.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args(argv)

    run = Path(args.run)
    cfg = torch.load(run / "model.pt", map_location="cpu", weights_only=False)["config"]
    report = evaluate_checkpoint(run / "model.pt", load_config(None, cfg), args.split)
    out = run / f"{args.split}_metrics.json"
    out.write_text(json.dumps(report, indent=2))
    c = report["combined"]
    print(f"{args.split}: combined QWK={c['qwk']:.3f} acc={c['accuracy']:.3f} "
          f"bal_acc={c['balanced_accuracy']:.3f} AUC(>=2)={c['auc_ge2']:.3f} -> {out}")


if __name__ == "__main__":
    main()
