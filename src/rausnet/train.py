"""Config-driven training of the two-head grading network.

Usage:
    python -m rausnet.train --config configs/train/baseline.yaml [--set train.epochs=5 ...]
    python -m rausnet.train --resume outputs/<run_dir>      # continue an interrupted run (e.g. Colab)

Outputs (in ``output_dir/<run_name>_<timestamp>/``):
    model.pt          best checkpoint (by validation combined-grade QWK) + config
    model_card.json   provenance: stage, data hash, split sizes, git commit, metrics
    history.csv       per-epoch losses and validation metrics
    test_metrics.json final evaluation of the best checkpoint on the frozen test split
    last.pt           model + optimiser state after the latest epoch, for ``--resume``
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

import rausnet
from rausnet.data.dataset import IGNORE_INDEX, JointDataset
from rausnet.data.manifest import read_manifest
from rausnet.data.split import split_records
from rausnet.evaluate import evaluate_predictions, predict
from rausnet.models import GradingNet


def _set_dotted(cfg: dict, key: str, value) -> None:
    node = cfg
    *parents, leaf = key.split(".")
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = value


def load_config(path: str | Path | None, base: dict | None = None,
                overrides: list[str] | None = None) -> dict:
    """Load YAML (or copy ``base``), apply ``key=value`` overrides, expand env vars in data.root."""
    cfg = copy.deepcopy(base) if base is not None else yaml.safe_load(Path(path).read_text())
    for item in overrides or []:
        key, _, raw = item.partition("=")
        _set_dotted(cfg, key.strip(), yaml.safe_load(raw))
    root = os.path.expandvars(str(cfg["data"]["root"]))
    if "$" in root:
        raise ValueError(f"unresolved environment variable in data.root: {root!r}")
    cfg["data"]["root"] = root
    return cfg


def pick_device() -> torch.device:
    """CUDA when available; ``RAUSNET_DEVICE=cpu`` (or ``cuda``) forces a device, e.g. for
    cross-runtime equivalence checks (an empty CUDA_VISIBLE_DEVICES is ignored on Windows)."""
    forced = os.environ.get("RAUSNET_DEVICE", "").strip().lower()
    if forced:
        if forced == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("RAUSNET_DEVICE=cuda but CUDA is not available")
        return torch.device(forced)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(cfg: dict) -> GradingNet:
    m = cfg["model"]
    return GradingNet(m["backbone"], m.get("pretrained", True), dropout=m.get("dropout", 0.2))


def build_loaders(cfg: dict):
    d, t = cfg["data"], cfg["train"]
    root = Path(d["root"])
    records = read_manifest(root / d["manifest"])
    splits = split_records(records, tuple(d["split_ratios"]), d["split_salt"])
    for name in ("train", "val"):
        if not splits[name]:
            raise ValueError(f"split '{name}' is empty; add more patients")
    workers = t.get("num_workers", 0)
    if workers > 0:
        # Spawned workers (Windows) re-import numpy; one BLAS thread each avoids
        # "OpenBLAS error: Memory allocation still failed".
        for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(var, "1")
    loaders = {}
    for name, recs in splits.items():
        ds = JointDataset(recs, root, d["image_size"], train=name == "train")
        loaders[name] = DataLoader(
            ds, batch_size=t["batch_size"], shuffle=name == "train", num_workers=workers,
            pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0 and bool(recs),
        )
    return loaders, splits


def pd_target(gs: int, pd: int, mask_gs0: bool) -> int:
    """PD training label; ignored on GS 0 joints when ``mask_gs0`` (no synovium, PD undefined)."""
    return IGNORE_INDEX if mask_gs0 and gs == 0 else pd


def mask_pd_targets(gs: torch.Tensor, pd: torch.Tensor) -> torch.Tensor:
    """Batch version of ``pd_target`` with masking on."""
    return torch.where(gs == 0, torch.full_like(pd, IGNORE_INDEX), pd)


def class_weights(labels: list[int], enabled: bool) -> torch.Tensor:
    counts = np.bincount([y for y in labels if y >= 0], minlength=4).astype(float)
    if not enabled:
        return torch.ones(4)
    w = np.where(counts > 0, counts.sum() / (4 * np.maximum(counts, 1)), 0.0)
    return torch.tensor(w, dtype=torch.float32)


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _save_atomic(obj, path: Path) -> None:
    """torch.save through a temporary file, so a disconnect never leaves a truncated checkpoint."""
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train(cfg: dict, resume: str | Path | None = None) -> Path:
    """Train from ``cfg``; with ``resume`` (a run directory) continue that run from ``last.pt``
    using its saved config. Resumed runs are not bit-identical (data order RNG is not restored)."""
    if resume is not None:
        cfg = yaml.safe_load((Path(resume) / "config.yaml").read_text())
    t = cfg["train"]
    seed_everything(t["seed"])
    device = pick_device()
    loaders, splits = build_loaders(cfg)

    model = build_model(cfg).to(device)
    if t.get("init_checkpoint") and resume is None:
        state = torch.load(t["init_checkpoint"], map_location="cpu", weights_only=False)
        model.load_state_dict(state["state_dict"])

    train_recs = splits["train"]
    w_gs = class_weights([r.gs_grade for r in train_recs], t["class_weighting"]).to(device)
    mask_gs0 = t.get("pd_mask_gs0", False)
    w_pd = class_weights([pd_target(r.gs_grade, r.pd_grade if r.pd_path else IGNORE_INDEX, mask_gs0)
                          for r in train_recs], t["class_weighting"]).to(device)
    loss_gs = nn.CrossEntropyLoss(weight=w_gs)
    loss_pd = nn.CrossEntropyLoss(weight=w_pd, ignore_index=IGNORE_INDEX)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["epochs"])
    amp = t["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best, best_epoch, history, start, stopped = -float("inf"), -1, [], 1, False
    if resume is not None:
        run_dir = Path(resume)
        last = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(last["state_dict"])
        opt.load_state_dict(last["optimizer"])
        sched.load_state_dict(last["scheduler"])
        scaler.load_state_dict(last["scaler"])
        best, best_epoch, history, stopped = last["best"], last["best_epoch"], last["history"], last["stopped"]
        start = last["epoch"] + 1
        print(f"resuming {run_dir} at epoch {start}", flush=True)
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = Path(cfg["output_dir"]) / f"{cfg['run_name']}_{stamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    for epoch in range(start, t["epochs"] + 1):
        if stopped:
            break
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, gs, pd in loaders["train"]:
            x, gs, pd = x.to(device, non_blocking=True), gs.to(device), pd.to(device)
            if mask_gs0:
                pd = mask_pd_targets(gs, pd)
            with torch.autocast(device.type, enabled=amp):
                lg, lp = model(x)
                loss = loss_gs(lg.float(), gs)
                if (pd != IGNORE_INDEX).any():
                    loss = loss + loss_pd(lp.float(), pd)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(x)
            n += len(x)
        sched.step()

        val = evaluate_predictions(*predict(model, loaders["val"], device, amp))
        row = {
            "epoch": epoch, "train_loss": total / n, "seconds": round(time.time() - t0, 1),
            "val_qwk": val["combined"]["qwk"], "val_acc": val["combined"]["accuracy"],
            "val_bal_acc": val["combined"]["balanced_accuracy"],
            "val_gs_qwk": val["gs"]["qwk"], "val_pd_qwk": val.get("pd", {}).get("qwk"),
        }
        history.append(row)
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
              flush=True)
        score = row["val_qwk"] if not np.isnan(row["val_qwk"]) else -row["train_loss"]
        if score > best:
            best, best_epoch = score, epoch
            _save_atomic({"state_dict": model.state_dict(), "config": cfg, "epoch": epoch}, run_dir / "model.pt")
        elif epoch - best_epoch >= t["patience"]:
            print(f"early stopping at epoch {epoch} (best {best_epoch})")
            stopped = True
        _save_atomic({"state_dict": model.state_dict(), "optimizer": opt.state_dict(),
                      "scheduler": sched.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
                      "best": best, "best_epoch": best_epoch, "history": history, "stopped": stopped},
                     run_dir / "last.pt")

    with open(run_dir / "history.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    ckpt = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test = None
    if splits["test"]:
        test = evaluate_predictions(*predict(model, loaders["test"], device, amp))
        (run_dir / "test_metrics.json").write_text(json.dumps(test, indent=2))

    manifest = Path(cfg["data"]["root"]) / cfg["data"]["manifest"]
    card = {
        "rausnet_version": rausnet.__version__,
        "stage": cfg.get("stage"),
        "clinically_validated": False,
        "git_commit": _git_commit(),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "split_patients": {k: len({r.patient_id for r in v}) for k, v in splits.items()},
        "best_epoch": best_epoch,
        "val_combined_qwk": best,
        "test_combined": None if test is None else {k: test["combined"][k] for k in
                                                    ("qwk", "accuracy", "balanced_accuracy", "auc_ge2")},
        "init_checkpoint": t.get("init_checkpoint"),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
        "torch": torch.__version__,
    }
    (run_dir / "model_card.json").write_text(json.dumps(card, indent=2))
    print(f"run saved to {run_dir}")
    return run_dir


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train the grading network.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config")
    group.add_argument("--resume", metavar="RUN_DIR", help="continue an interrupted run from its last.pt")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override config values, e.g. train.epochs=5")
    args = parser.parse_args(argv)
    if args.resume:
        train({}, resume=args.resume)
    else:
        train(load_config(args.config, overrides=args.set))


if __name__ == "__main__":
    main()
