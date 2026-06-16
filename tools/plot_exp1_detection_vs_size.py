"""Plot Exp1 — detection Recall vs object size, per class, on SAT-MTB.

Reads ``docs/figures/exp1_detection_recall_by_size.csv`` (from
``compute_exp1_detection_recall.py``) and draws a 2x2 panel, one per class.
Each panel shows the detectors that cover that class:

    car            -> HiEUM (specialist, solid)        + FairMOT/TGraM (JDT, dashed)
    airplane/ship/train -> Faster R-CNN (specialist)   + FairMOT/TGraM

So "which detector covers which size" is read off directly: HiEUM lives only in
the car panel, Faster R-CNN only in the others — no fictional pixel boundary,
since detector identity tracks class (and classes overlap in size).

Bins with fewer than MIN_GT GT boxes are dropped (SAT-MTB has a few cross-class
size outliers, e.g. <5 px "train" / >=40 px "car", that would be noise).

Run:
    python tools/plot_exp1_detection_vs_size.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import apply_neurips_style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

CSV = Path("/home/ziwen/code/esa_dlstem/docs/figures/exp1_detection_recall_by_size.csv")
OUT_DIR = Path("/home/ziwen/code/esa_dlstem/docs/figures")
MIN_GT = 50  # drop bins with too few GT boxes (size outliers / noise)

BIN_LABELS = ["<5", "5-8", "8-12", "12-20", "20-40", ">=40"]
CLASSES = ["car", "airplane", "ship", "train"]

# detector -> style. Specialist detectors solid; JDT dashed (matches Exp2 fig).
STYLE = {
    "HiEUM":      dict(color="#1f77b4", ls="-",  marker="o", label="HiEUM (car det.)"),
    "FasterRCNN": dict(color="#1f77b4", ls="-",  marker="o", label="Faster R-CNN"),
    "FairMOT":    dict(color="#d62728", ls="--", marker="v", label="FairMOT"),
    "TGraM":      dict(color="#ff7f0e", ls="--", marker="P", label="TGraM"),
}
PANEL_SPECIALIST = {"car": "HiEUM", "airplane": "FasterRCNN",
                    "ship": "FasterRCNN", "train": "FasterRCNN"}


def main():
    df = pd.read_csv(CSV)
    df = df[df.n_gt >= MIN_GT]
    apply_neurips_style(base_size=9.0)

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), sharey=True)
    for ax, cls in zip(axes.ravel(), CLASSES):
        sub = df[df["class"] == cls]
        dets = [PANEL_SPECIALIST[cls], "FairMOT", "TGraM"]
        for det in dets:
            s = sub[sub.detector == det].sort_values("bin_idx")
            if not len(s):
                continue
            ax.plot(s.bin_idx, s.recall, markersize=4.5, linewidth=1.4,
                    markeredgewidth=0.0, **STYLE[det])
        ax.set_title(cls, fontweight="bold")
        ax.set_xticks(range(len(BIN_LABELS)))
        ax.set_xticklabels(BIN_LABELS)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, axis="y")
        ax.legend(loc="upper left", frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel("object size  (px, $\\sqrt{wh}$)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Recall $\\uparrow$  (IoU $\\geq$ 0.5)")

    fig.suptitle("Exp1 — detection recall vs object size, per class (SAT-MTB)",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = OUT_DIR / f"exp1_detection_recall_by_size.{ext}"
        fig.savefig(p)
        print(f"wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
