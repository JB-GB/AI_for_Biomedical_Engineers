"""Paper figures for a training run: combined-grade confusion matrix and training curves.

Usage: python scripts/make_figures.py outputs/<run_dir> [--out docs/paper/figures] [--prefix baseline_s2]
Requires the ``paper`` extra (matplotlib).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
TEXT, TEXT_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3, fixed order
BLUES = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f3f8fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "text.color": TEXT, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})


def confusion_figure(metrics: dict, path: Path, title: str) -> None:
    cm = metrics["confusion_matrix"]
    n = len(cm)
    rows = [sum(r) or 1 for r in cm]
    norm = [[v / rows[i] for v in r] for i, r in enumerate(cm)]
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(norm, cmap=BLUES, vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            dark = norm[i][j] > 0.55
            ax.text(j, i, f"{cm[i][j]}\n{norm[i][j]:.0%}", ha="center", va="center", fontsize=9,
                    color="#ffffff" if dark else TEXT)
    ax.set_xticks(range(n), [f"{g}" for g in range(n)])
    ax.set_yticks(range(n), [f"{g}" for g in range(n)])
    ax.set_xlabel("Predicted grade")
    ax.set_ylabel("Reference grade")
    ax.set_title(title, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def curves_figure(history: list[dict], best_epoch: int, path: Path) -> None:
    epochs = [int(r["epoch"]) for r in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.2))
    ax1.plot(epochs, [float(r["train_loss"]) for r in history], color=SERIES[0], lw=2)
    ax1.set_title("Training loss (GS + PD)", loc="left")
    ax1.set_xlabel("Epoch")
    series = [("val_qwk", "Combined"), ("val_gs_qwk", "GS"), ("val_pd_qwk", "PD")]
    for (key, label), color in zip(series, SERIES, strict=True):
        ys = [float(r[key]) for r in history]
        ax2.plot(epochs, ys, color=color, lw=2, label=label)
        ax2.annotate(label, (epochs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points",
                     va="center", fontsize=9, color=TEXT_2)
    ax2.axvline(best_epoch, color=TEXT_2, lw=1, ls=":")
    ax2.annotate(f"best epoch {best_epoch}", (best_epoch, 0.02), xytext=(4, 0), textcoords="offset points",
                 fontsize=8, color=TEXT_2, xycoords=("data", "axes fraction"))
    ax2.set_ylim(0, 1)
    ax2.set_title("Validation quadratic weighted kappa", loc="left")
    ax2.set_xlabel("Epoch")
    ax2.legend(frameon=False, loc="lower right", fontsize=9)
    for ax in (ax1, ax2):
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run")
    parser.add_argument("--out", default="docs/paper/figures")
    parser.add_argument("--prefix", default=None)
    args = parser.parse_args(argv)
    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or run.name.rsplit("_", 1)[0]
    test = json.loads((run / "test_metrics.json").read_text())
    card = json.loads((run / "model_card.json").read_text())
    with open(run / "history.csv", newline="") as fh:
        history = list(csv.DictReader(fh))
    confusion_figure(test["combined"], out / f"{prefix}_confusion.png",
                     f"Combined grade, test set (n={test['combined']['n']})")
    curves_figure(history, card["best_epoch"], out / f"{prefix}_curves.png")
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
