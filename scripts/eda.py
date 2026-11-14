"""Exploratory data analysis of a manifest: class balance, patients, image stats, PD colour.

Usage: python scripts/eda.py --manifest DATA_ROOT/datasets/vXXXX/manifest.csv --root DATA_ROOT --out DIR
       [--split-salt rausnet-split-v1] [--limit N]
Writes ``eda_summary.json`` and PNG figures to DIR (read-only on the data).
Requires the ``paper`` extra (matplotlib).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from matplotlib.ticker import PercentFormatter
from PIL import Image

from rausnet.data.manifest import read_manifest
from rausnet.data.split import DEFAULT_SALT, SPLITS, assign_split

GRADES = range(4)
COLOUR_CHROMA = 40  # max(RGB) - min(RGB) above this is a Doppler colour pixel


def _style():
    """Reuse the paper figure style (rcParams and colours) from make_figures.py."""
    spec = importlib.util.spec_from_file_location("make_figures", Path(__file__).with_name("make_figures.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _counts(values) -> dict[str, int]:
    c = Counter(values)
    return {str(g): c.get(g, 0) for g in GRADES}


def _balance(records) -> dict:
    return {"n": len(records), "gs": _counts(r.gs_grade for r in records),
            "pd": _counts(r.pd_grade for r in records if r.pd_path),
            "combined": _counts(r.combined_grade for r in records)}


def _describe(xs) -> dict:
    if not xs:
        return {"n": 0}
    a = np.asarray(xs, np.float64)
    return {"n": len(xs), "mean": round(float(a.mean()), 4), "std": round(float(a.std()), 4),
            "min": round(float(a.min()), 4), "p25": round(float(np.percentile(a, 25)), 4),
            "median": round(float(np.median(a)), 4), "p75": round(float(np.percentile(a, 75)), 4),
            "max": round(float(a.max()), 4)}


def colour_fraction(rgb: np.ndarray) -> float:
    """Fraction of pixels carrying Doppler colour (chroma above ``COLOUR_CHROMA``)."""
    a = rgb.astype(np.int16)
    return float(((a.max(axis=2) - a.min(axis=2)) > COLOUR_CHROMA).mean())


def analyse(manifest: str | Path, root: str | Path, split_salt: str = DEFAULT_SALT, limit: int = 0) -> dict:
    records = read_manifest(manifest)
    root = Path(root)
    splits = {r.image_id: assign_split(r.patient_id, salt=split_salt) for r in records}
    groups = {"split": defaultdict(list), "source": defaultdict(list), "joint": defaultdict(list)}
    for r in records:
        groups["split"][splits[r.image_id]].append(r)
        groups["source"][r.source].append(r)
        groups["joint"][r.joint].append(r)
    per_patient = Counter(r.patient_id for r in records)
    patients_per_split = {s: len({r.patient_id for r in groups["split"][s]}) for s in SPLITS}

    sizes, missing = Counter(), 0
    gs_mean, gs_std, gs_mean_by_grade = [], [], defaultdict(list)
    pd_frac_by_grade = defaultdict(list)
    for r in records[:limit] if limit else records:
        gs_file = root / r.gs_path
        if not gs_file.is_file():
            missing += 1
            continue
        with Image.open(gs_file) as im:
            gs = np.asarray(im.convert("L"), np.float32) / 255.0
            sizes[f"{im.width}x{im.height}"] += 1
        gs_mean.append(float(gs.mean()))
        gs_std.append(float(gs.std()))
        gs_mean_by_grade[r.gs_grade].append(float(gs.mean()))
        if r.pd_path:
            pd_file = root / r.pd_path
            if not pd_file.is_file():
                missing += 1
                continue
            with Image.open(pd_file) as im:
                pd_frac_by_grade[r.pd_grade].append(colour_fraction(np.asarray(im.convert("RGB"))))

    return {
        "manifest": Path(manifest).as_posix(),
        "split_salt": split_salt,
        "n_records": len(records),
        "n_patients": len(per_patient),
        "patients_per_split": patients_per_split,
        "class_balance": {
            "all": _balance(records),
            **{key: {name: _balance(rs) for name, rs in sorted(g.items())} for key, g in groups.items()},
        },
        "images_per_patient": {**_describe(list(per_patient.values())),
                               "histogram": dict(sorted(Counter(per_patient.values()).items()))},
        "side": dict(Counter(r.side for r in records)),
        "images": {
            "analysed": len(gs_mean), "missing_files": missing, "sizes": dict(sizes.most_common()),
            "gs_intensity_mean": _describe(gs_mean), "gs_intensity_std": _describe(gs_std),
            "gs_intensity_mean_by_gs_grade": {str(g): _describe(gs_mean_by_grade[g]) for g in GRADES},
        },
        "pd_colour_fraction_by_pd_grade": {str(g): _describe(pd_frac_by_grade[g]) for g in GRADES},
        "_raw": {"pd_frac": {g: pd_frac_by_grade[g] for g in GRADES}, "per_patient": list(per_patient.values()),
                 "gs_by_grade": {g: gs_mean_by_grade[g] for g in GRADES}},
    }


def balance_figure(summary: dict, path: Path, mf) -> None:
    """Grade distribution per split for GS, PD and combined (share of images)."""
    bal = summary["class_balance"]["split"]
    fig, axes = mf.plt.subplots(1, 3, figsize=(9.6, 3.0))
    width = 0.8 / len(SPLITS)
    for ax, (key, title) in zip(axes, (("gs", "GS grade"), ("pd", "PD grade"), ("combined", "Combined grade")),
                                strict=True):
        for i, (split, color) in enumerate(zip(SPLITS, mf.SERIES, strict=True)):
            counts = bal.get(split, {}).get(key, {})
            total = sum(counts.values()) or 1
            ax.bar([g + (i - 1) * width for g in GRADES], [counts.get(str(g), 0) / total for g in GRADES],
                   width=width * 0.95, color=color, label=split)
        ax.set_xticks(list(GRADES))
        ax.set_title(title, loc="left")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.grid(axis="y", color=mf.GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Share of images")
    axes[-1].legend(frameon=False, fontsize=8)
    _save(fig, path, mf)


def pd_colour_figure(summary: dict, path: Path, mf) -> None:
    data = [summary["_raw"]["pd_frac"][g] for g in GRADES]
    fig, ax = mf.plt.subplots(figsize=(4.4, 3.2))
    keep = [g for g in GRADES if data[g]]
    if keep:
        ax.boxplot([data[g] for g in keep], positions=keep, widths=0.5, showfliers=False,
                   medianprops={"color": mf.SERIES[1], "lw": 2}, boxprops={"color": mf.TEXT_2},
                   whiskerprops={"color": mf.TEXT_2}, capprops={"color": mf.TEXT_2})
    ax.set_xticks(list(GRADES))
    ax.set_xlabel("PD grade")
    ax.set_ylabel("Colour pixel fraction")
    ax.set_title("Doppler colour per PD grade", loc="left")
    ax.grid(axis="y", color=mf.GRID, lw=0.8)
    ax.set_axisbelow(True)
    _save(fig, path, mf)


def patients_figure(summary: dict, path: Path, mf) -> None:
    hist = summary["images_per_patient"]["histogram"]
    fig, ax = mf.plt.subplots(figsize=(4.4, 3.0))
    ax.bar(list(hist), list(hist.values()), color=mf.SERIES[0], width=0.8)
    ax.set_xlabel("Images per patient")
    ax.set_ylabel("Patients")
    ax.set_title("Images per patient", loc="left")
    ax.grid(axis="y", color=mf.GRID, lw=0.8)
    ax.set_axisbelow(True)
    _save(fig, path, mf)


def _save(fig, path: Path, mf) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    mf.plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", required=True, help="data root the manifest paths are relative to")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split-salt", default=DEFAULT_SALT)
    ap.add_argument("--limit", type=int, default=0, help="analyse pixels of the first N records only (0 = all)")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = analyse(args.manifest, args.root, args.split_salt, args.limit)
    if not args.no_figures:
        mf = _style()
        balance_figure(summary, out / "eda_class_balance.png", mf)
        pd_colour_figure(summary, out / "eda_pd_colour.png", mf)
        patients_figure(summary, out / "eda_images_per_patient.png", mf)
    summary.pop("_raw")
    (out / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    ipp = summary["images_per_patient"]
    print(f"{summary['n_records']} images, {summary['n_patients']} patients "
          f"(median {ipp['median']:g} images/patient), wrote {out}")


if __name__ == "__main__":
    main()
