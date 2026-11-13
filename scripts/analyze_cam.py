"""Grad-CAM localisation analysis on the synthetic test split (masks are known there).

Usage: python scripts/analyze_cam.py [--models-dir models] [--version v0003] [--data-root DIR]
       [--figure docs/paper/figures/gradcam_examples.png]

For each test image with synovitis (GS >= 1) it measures, for the predicted grade's CAM:
- energy ratio: share of positive CAM inside the synovium vs the synovium's share of the image
  (the value a uniform map would get);
- pointing game: whether the CAM peak falls inside the synovium dilated by one CAM cell.
It also inspects PD false positives (predicted PD > true PD): where the PD CAM points and
whether the image contains the out-of-synovium distractor vessel.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

from rausnet.data.dataset import load_input
from rausnet.data.manifest import read_manifest
from rausnet.data.split import split_records
from rausnet.evaluate import load_model
from rausnet.explain import CamNet
from rausnet.heatmap import normalize_cam, overlay
from rausnet.registry import Registry


def upsample(cam: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    img = Image.fromarray(normalize_cam(cam).astype(np.float32), mode="F")
    return np.asarray(img.resize(size, Image.BILINEAR), np.float32).clip(0, 1)


def coloured_outside(pd_rgb: np.ndarray, mask: np.ndarray) -> bool:
    rgb = pd_rgb.astype(int)
    return bool((((rgb[..., 0] - rgb[..., 2]) > 100) & ~mask).any())


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--version", default=None)
    parser.add_argument("--data-root", default=os.environ.get("RAUSNET_DATA_DIR"))
    parser.add_argument("--figure", default="docs/paper/figures/gradcam_examples.png")
    args = parser.parse_args(argv)

    reg = Registry(args.models_dir)
    entry = reg.get(args.version) if args.version else reg.current()
    ckpt = reg.dir / entry.model_path
    root = Path(args.data_root)
    net, cfg = load_model(ckpt, torch.device("cpu"))
    cam_net = CamNet(net).eval()
    d = cfg["data"]
    records = read_manifest(root / entry.dataset_manifest)
    recs = split_records(records, tuple(d["split_ratios"]), d["split_salt"])["test"]

    rows, examples = [], {"correct": [], "pd_false_positive": []}
    for r in recs:
        mask_path = root / r.gs_path.replace("/gs/", "/masks/")
        if r.source != "synthetic" or not mask_path.exists():
            continue
        x = load_input(root / r.gs_path, root / r.pd_path, d["image_size"])[None]
        with torch.no_grad():
            lg, lp, cg, cp = (t.numpy()[0] for t in cam_net(x))
        g_hat, p_hat = int(lg.argmax()), int(lp.argmax())
        mask = np.asarray(Image.open(mask_path)) > 0
        h, w = mask.shape
        pd_rgb = np.asarray(Image.open(root / r.pd_path).convert("RGB"))
        row = {"image_id": r.image_id, "gs": r.gs_grade, "pd": r.pd_grade, "gs_hat": g_hat, "pd_hat": p_hat,
               "distractor": coloured_outside(pd_rgb, mask)}
        if mask.any():
            cell = 2 * round(max(h, w) / cg.shape[-1] / 2) + 1
            dilated = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(cell))) > 0
            for head, cam in (("gs", cg[g_hat]), ("pd", cp[p_hat])):
                m = upsample(cam, (w, h))
                total = m.sum()
                row[f"{head}_energy_in"] = float((m * mask).sum() / total) if total > 0 else None
                peak = np.unravel_index(m.argmax(), m.shape)
                row[f"{head}_pointing_hit"] = bool(dilated[peak]) if total > 0 else None
            row["mask_fraction"] = float(mask.mean())
            row["dilated_fraction"] = float(dilated.mean())  # chance level of the pointing game
        if p_hat > r.pd_grade:
            m = upsample(cp[p_hat], (w, h))
            row["pd_cam_energy_outside"] = float((m * ~mask).sum() / max(m.sum(), 1e-9))
            examples["pd_false_positive"].append((r, cg[g_hat], cp[p_hat]))
        elif g_hat == r.gs_grade and p_hat == r.pd_grade and r.gs_grade >= 1 and r.pd_grade >= 1:
            examples["correct"].append((r, cg[g_hat], cp[p_hat]))
        rows.append(row)

    syn = [x for x in rows if "gs_energy_in" in x]

    def mean(key, items):
        vals = [x[key] for x in items if x.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    fp = [x for x in rows if x["pd_hat"] > x["pd"]]
    summary = {
        "version": entry.version,
        "images_with_synovitis": len(syn),
        "mask_fraction_mean": mean("mask_fraction", syn),
        "gs_energy_in_mean": mean("gs_energy_in", syn),
        "pd_energy_in_mean_pd_ge1": mean("pd_energy_in", [x for x in syn if x["pd"] >= 1]),
        "pointing_chance_mean": mean("dilated_fraction", syn),
        "gs_pointing_hit_rate": mean("gs_pointing_hit", syn),
        "pd_pointing_hit_rate_pd_ge1": mean("pd_pointing_hit", [x for x in syn if x["pd"] >= 1]),
        "pd_false_positives": len(fp),
        "pd_false_positives_with_distractor": sum(x["distractor"] for x in fp),
        "distractor_rate_all_test": float(np.mean([x["distractor"] for x in rows])),
        "pd_fp_cam_energy_outside_mean": mean("pd_cam_energy_outside", fp),
        "pd_false_positive_cases": fp,
    }
    out = ckpt.parent / "cam_analysis.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "pd_false_positive_cases"}, indent=2))

    picks = examples["correct"][:2] + examples["pd_false_positive"][:2]
    if picks:
        tiles = []
        for r, cg_, cp_ in picks:
            gs = np.asarray(Image.open(root / r.gs_path).convert("L"))
            pd = np.asarray(Image.open(root / r.pd_path).convert("RGB"))
            tiles.append([np.repeat(gs[..., None], 3, 2), overlay(gs, normalize_cam(cg_)), pd,
                          overlay(pd, normalize_cam(cp_))])
        h, w = tiles[0][0].shape[:2]
        pad = 6
        canvas = np.full((len(tiles) * (h + pad), 4 * (w + pad), 3), 255, np.uint8)
        for i, row_tiles in enumerate(tiles):
            for j, t in enumerate(row_tiles):
                canvas[i * (h + pad):i * (h + pad) + h, j * (w + pad):j * (w + pad) + w] = t
        Image.fromarray(canvas).save(args.figure)
        labels = [f"{r.image_id}: GS{r.gs_grade}/PD{r.pd_grade}" for r, _, _ in picks]
        print(f"figure rows (cols: GS, GS-CAM, PD, PD-CAM): {labels} -> {args.figure}")


if __name__ == "__main__":
    main()
