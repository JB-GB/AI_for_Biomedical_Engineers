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


def main(argv=None) -> None:
    from rausnet.train import build_loaders, build_model, load_config, pick_device

    parser = argparse.ArgumentParser(description="Evaluate a trained run.")
    parser.add_argument("--run", required=True, help="run directory containing model.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args(argv)

    run = Path(args.run)
    ckpt = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    cfg["model"]["pretrained"] = False
    device = pick_device()
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    loaders, _ = build_loaders(load_config(None, cfg))
    report = evaluate_predictions(*predict(model, loaders[args.split], device, cfg["train"]["amp"]))
    out = run / f"{args.split}_metrics.json"
    out.write_text(json.dumps(report, indent=2))
    c = report["combined"]
    print(f"{args.split}: combined QWK={c['qwk']:.3f} acc={c['accuracy']:.3f} "
          f"bal_acc={c['balanced_accuracy']:.3f} AUC(>=2)={c['auc_ge2']:.3f} -> {out}")


if __name__ == "__main__":
    main()
