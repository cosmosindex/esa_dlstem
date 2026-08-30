"""
Car-part tracking versus object scale, with and without a detector.

The companion to \\cref{tab:mot_car_oracle}. Both panels bin ground-truth boxes
by $s=\\sqrt{\\mathrm{area}}$ and restrict HOTA's own per-detection quantities to
each bin, exactly as tools/compute_mot_size_curve.py does for the non-car part.
The x-range covers the whole car part: its largest test object is 24 px.

The point of the pair is that the two panels have the same shape and different
heights. Association does not degrade with scale anywhere above ~3 px in either
condition; what changes between them is only the level, and the level is set by
the detector. Plotting them on a shared y-axis is deliberate -- the gap is the
result, and a per-panel axis would hide it.

Usage:
    python tools/plot_wacv_car_oracle_size.py \\
        --detector docs/space_tracker/mot_size_curve_car.csv \\
        --oracle   docs/space_tracker/mot_size_curve_car_oracle.csv \\
        --out      wacv-2027-author-kit-template/plots/car_oracle_vs_size
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from plot_style import apply_neurips_style  # noqa: E402

HIST_COLOR = "#7A7A7A"
METHODS = [
    ("sort",         "SORT",          "-",  "o", "#1f77b4"),
    ("bytetrack",    "ByteTrack",     "-",  "s", "#2ca02c"),
    ("ocsort",       "OC-SORT",       "-",  "^", "#17becf"),
    ("botsort",      "BoT-SORT",      "-",  "D", "#9467bd"),
    ("botsort_reid", "BoT-SORT-ReID", "--", "X", "#8c564b"),
    ("tracktrack",   "TrackTrack",    "--", "*", "#e377c2"),
    ("masa",         "MASA",          "--", "p", "#bcbd22"),
]


def draw(ax, df, metric, x_max, title, show_ylabel):
    df = df[df.bin_hi <= x_max + 1e-9]
    counts = (df.groupby("bin_lo").agg(n=("n_gt", "first"), hi=("bin_hi", "first"))
                .sort_index())
    centers = (counts.index.values + counts.hi.values) / 2
    width = float(counts.hi.values[0] - counts.index.values[0])
    frac = counts.n.values / max(counts.n.values.sum(), 1)
    hb = ax.twinx()
    hb.bar(centers, frac, width=width * 0.95, color=HIST_COLOR, alpha=0.18,
           linewidth=0, zorder=1)
    hb.set_ylim(0, max(frac.max() * 1.15, 1e-3))
    hb.set_yticks([])
    hb.set_zorder(ax.get_zorder() - 1)
    ax.patch.set_visible(False)

    drawn = []
    for key, _, ls, mk, c in METHODS:
        sub = df[df.tracker.isin([key, key + "_oracle"])].sort_values("bin_lo")
        if sub.empty:
            continue
        drawn.append(key)
        ax.plot((sub.bin_lo.values + sub.bin_hi.values) / 2, sub[metric].values,
                ls=ls, marker=mk, color=c, markersize=3.0, linewidth=1.2,
                markeredgewidth=0.0, zorder=3)
    ax.set_xlim(0, x_max)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(r"object scale  $s=\sqrt{\mathrm{area}}$  [px]")
    if show_ylabel:
        ax.set_ylabel(rf"{metric} $\uparrow$")
    return drawn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", required=True, type=Path)
    ap.add_argument("--oracle", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--metric", default="AssA")
    ap.add_argument("--x-max", type=float, default=24.0)
    ap.add_argument("--figsize", default="3.35,3.9",
                    help="inches, W,H -- default fits one WACV column")
    args = ap.parse_args()

    apply_neurips_style(base_size=9.0)
    w, h = (float(v) for v in args.figsize.split(","))
    # Stacked, not side by side: this is a single-column figure, and putting the
    # two conditions one above the other on a shared x-axis is what makes
    # "same shape, different height" readable at this width.
    fig, axes = plt.subplots(2, 1, figsize=(w, h), sharex=True, sharey=True)
    drawn = draw(axes[0], pd.read_csv(args.detector), args.metric, args.x_max,
                 "(a) HiEUM detections", True)
    drawn += draw(axes[1], pd.read_csv(args.oracle), args.metric, args.x_max,
                  "(b) ground-truth boxes", True)
    axes[0].set_xlabel("")

    order = [m for m in METHODS if m[0] in set(drawn)]
    handles = [Line2D([0], [0], color=c, ls=ls, marker=mk, markersize=3.4,
                      linewidth=1.2, markeredgewidth=0.0)
               for _, _, ls, mk, c in order]
    labels = [n for _, n, _, _, _ in order]
    handles.append(Patch(facecolor=HIST_COLOR, alpha=0.30))
    labels.append("GT-box density")
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=3, frameon=False, fontsize=7.0, handlelength=1.8,
               handletextpad=0.5, columnspacing=1.2, labelspacing=0.35,
               borderaxespad=0.0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {args.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
