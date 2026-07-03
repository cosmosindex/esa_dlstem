#!/usr/bin/env python
"""GT-oracle BIRDSAI tracker comparison figure: pure association vs object size.

Reads docs/use_case_results/birdsai_gt_oracle_assa_vs_size.csv (real TrackEval
HOTA/AssA per size bin, produced by _birdsai_gt_oracle_hota_by_size.py) and plots,
per tracker, AssA / IDF1 / IDsw vs object size. With GT boxes fed to every tracker
detection is oracle (DetA~1), so any size trend is the TRACKER's own association.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tools.plot_style import apply_neurips_style

CSV_IN = Path("docs/use_case_results/birdsai_gt_oracle_assa_vs_size.csv")
OUT_PNG = Path("docs/use_case_results/figures/birdsai_gt_oracle_size.png")
OUT_PDF = Path("docs/use_case_results/figures/birdsai_gt_oracle_size.pdf")

BINS = ["<14", "14-20", "20-28", "28-38", "38-50", ">=50"]
BIN_CENTERS = [10, 17, 24, 33, 44, 60]
TRK_LABEL = {"sort": "SORT", "ocsort": "OC-SORT", "bytetrack": "ByteTrack",
             "botsort": "BoT-SORT", "botsort_reid": "BoT-SORT+ReID", "tracktrack": "TrackTrack"}
ORDER = ["sort", "ocsort", "bytetrack", "botsort", "botsort_reid", "tracktrack"]
COLORS = {"sort": "#1f77b4", "ocsort": "#d62728", "bytetrack": "#2ca02c",
          "botsort": "#9467bd", "botsort_reid": "#8c564b", "tracktrack": "#ff7f0e"}


def main():
    # data[method][bin] = {AssA, IDF1, IDsw}
    data = defaultdict(dict)
    for r in csv.DictReader(open(CSV_IN)):
        if r["size_bin"] == "all":
            continue
        data[r["method"]][r["size_bin"]] = r

    def series(method, key, scale=1.0):
        return [float(data[method].get(b, {}).get(key, "nan")) * scale for b in BINS]

    apply_neurips_style(base_size=10)
    x = np.array(BIN_CENTERS, float)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    panels = [("AssA", "AssA (%)", 100.0, (0, 80)),
              ("IDF1", "IDF1 (%)", 100.0, (0, 85)),
              ("IDsw", "IDsw (count)", 1.0, None)]
    for ax, (key, title, scale, ylim) in zip(axes, panels):
        for m in ORDER:
            if m not in data:
                continue
            ax.plot(x, series(m, key, scale), "-o", ms=3, lw=1.4,
                    color=COLORS[m], label=TRK_LABEL[m])
        ax.set_title(title)
        ax.set_xlabel(r"object size $\sqrt{\mathrm{area}}$ (px)")
        ax.set_xticks(x); ax.set_xticklabels(BINS, fontsize=7, rotation=30)
        if ylim:
            ax.set_ylim(*ylim)
        ax.axvspan(x[0] - 4, 14, color="0.9", zorder=0)
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, loc="upper left", ncol=2)
    fig.suptitle("BIRDSAI GT-oracle: pure association vs object size "
                 "(GT boxes fed to every tracker; detection is oracle)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200); fig.savefig(OUT_PDF)
    print(f"wrote {OUT_PNG}\nwrote {OUT_PDF}")


if __name__ == "__main__":
    main()
