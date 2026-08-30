"""
MOT detection and association versus object scale (WACV figure).

The MOT counterpart of ``tools/plot_wacv_size_curve.py``. That figure asks
one question of the SOT half -- at what object scale does tracking stop working
-- and this asks it of the MOT half, for every benchmarked method rather than
for their mean, because on MOT the methods do not agree.

Two things force a two-column layout rather than one shared axis:

* The two halves live on disjoint scale ranges. **Every car in the test split
  is at most 24 px**, and 80% of them are below 8 px; airplane and ship run out
  to 164 px with a median of 31 px. Drawn on one axis, the entire car benchmark
  would collapse into the first two bins of the non-car one.
* They are scored against different detectors, so a shared curve would mix a
  detector comparison into a scale comparison.

Rows are the two HOTA axes, taken straight from
``tools/compute_mot_size_curve.py``: DetRe(s) is the fraction of ground-truth
boxes at scale s that any tracker output matched, AssA(s) is the mean HOTA
association score of those matches. Precision is not drawn, for the reason
given in that script -- a false positive has no ground-truth scale.

Display ranges are cut where the sample stops supporting a curve: the car panel
at 16 px (the 0.6% above it is 60 tracks) and the non-car panel at 48 px (the
12.6% above it is 28 tracks in 8 sequences, so a single object entering or
leaving view moves a whole bin). The underlying CSVs keep the full range.

Usage:
    python tools/plot_wacv_mot_size_curve.py \
        --car   docs/space_tracker/mot_size_curve_car.csv \
        --nocar docs/space_tracker/mot_size_curve_nocar.csv \
        --out   wacv-2027-author-kit-template/plots/mot_deta_assa_vs_size
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

# (csv key, display name, paradigm). Order controls the legend.
# Paradigm is encoded in the line style so the families separate in greyscale:
# TBD solid, learned association dash-dot, JDT dashed, query-based dotted.
METHODS = [
    ("sort",         "SORT",          "-",   "o", "#1f77b4"),
    ("bytetrack",    "ByteTrack",     "-",   "s", "#2ca02c"),
    ("ocsort",       "OC-SORT",       "-",   "^", "#17becf"),
    ("botsort",      "BoT-SORT",      "-",   "D", "#9467bd"),
    ("botsort_reid", "BoT-SORT-ReID", "-",   "X", "#8c564b"),
    ("tracktrack",   "TrackTrack",    "-.",  "*", "#e377c2"),
    ("masa",         "MASA",          "-.",  "p", "#bcbd22"),
    ("fairmot_all",  "FairMOT",       "--",  "v", "#d62728"),
    ("tgram_all",    "TGraM",         "--",  "P", "#ff7f0e"),
    ("motrv2",       "MOTRv2",        ":",   "h", "#7f7f7f"),
    ("motip",        "MOTIP",         ":",   "<", "#111111"),
]

PANELS = [
    ("car",   "(a) car",              16.0),
    ("nocar", "(b) airplane + ship",  48.0),
]
METRICS = [("DetRe", r"DetRe $\uparrow$"), ("AssA", r"AssA $\uparrow$")]


def draw_column(axes, df: pd.DataFrame, x_max: float, title: str,
                show_ylabel: bool) -> list[str]:
    """One benchmark half: DetRe on axes[0], AssA on axes[1]. Returns methods drawn."""
    df = df[df.bin_hi <= x_max + 1e-9]
    centers_all = None
    drawn = []

    # Box-count density, identical in both rows, drawn behind the curves. It is
    # what stops the eye from reading a two-track bin as a trend.
    counts = (df.groupby("bin_lo")
                .agg(n=("n_gt", "first"), hi=("bin_hi", "first"))
                .sort_index())
    centers_all = (counts.index.values + counts.hi.values) / 2
    width = float(counts.hi.values[0] - counts.index.values[0])
    frac = counts.n.values / max(counts.n.values.sum(), 1)

    for ax in axes:
        hb = ax.twinx()
        hb.bar(centers_all, frac, width=width * 0.95, color=HIST_COLOR,
               alpha=0.18, linewidth=0, zorder=1)
        hb.set_ylim(0, max(frac.max() * 1.15, 1e-3))
        hb.set_yticks([])
        hb.set_zorder(ax.get_zorder() - 1)
        ax.patch.set_visible(False)

    for key, name, ls, marker, color in METHODS:
        sub = df[df.tracker == key].sort_values("bin_lo")
        if sub.empty:
            continue
        drawn.append(key)
        centers = (sub.bin_lo.values + sub.bin_hi.values) / 2
        for ax, (metric, _) in zip(axes, METRICS):
            ax.plot(centers, sub[metric].values, ls=ls, marker=marker, color=color,
                    markersize=3.0, linewidth=1.2, markeredgewidth=0.0, zorder=3)

    for ax, (_, ylab) in zip(axes, METRICS):
        ax.set_xlim(0, x_max)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)
        if show_ylabel:
            ax.set_ylabel(ylab)
    axes[0].set_title(title, fontweight="bold")
    axes[1].set_xlabel(r"object scale  $s=\sqrt{\mathrm{area}}$  [px]")
    return drawn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--car", required=True, type=Path)
    ap.add_argument("--nocar", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--figsize", default="7.0,4.5")
    args = ap.parse_args()

    data = {"car": pd.read_csv(args.car), "nocar": pd.read_csv(args.nocar)}
    apply_neurips_style(base_size=9.0)

    w, h = (float(v) for v in args.figsize.split(","))
    fig, axes = plt.subplots(2, 2, figsize=(w, h), sharey="row")
    drawn: list[str] = []
    for j, (half, title, x_max) in enumerate(PANELS):
        drawn += draw_column(axes[:, j], data[half], x_max, title, show_ylabel=(j == 0))

    # s = 8 px marks the same "tiny" regime as the SOT figure, so the two can be
    # read against each other; s = 32 px is the small/large threshold used
    # elsewhere in the paper and falls inside the non-car range only -- the car
    # half has no large half at all.
    for ax in axes.ravel():
        ax.axvline(8, color="#444444", linestyle=(0, (4, 2)), linewidth=0.9, zorder=2)
    for ax in axes[:, 1]:
        ax.axvline(32, color="#444444", linestyle=(0, (1, 2)), linewidth=0.9, zorder=2)
    axes[0, 0].text(7.7, 0.97, r"tiny: $s<8$ px", fontsize=7.5, va="top",
                    ha="right", color="#444444", zorder=4)
    axes[0, 1].text(32.8, 0.10, r"$s=32$ px", fontsize=7.5, color="#444444", zorder=4)
    axes[0, 0].text(15.6, 0.78, "no car exceeds 24 px", ha="right", va="top",
                    fontsize=7.5, color="#444444", zorder=4)

    order = [m for m in METHODS if m[0] in set(drawn)]
    handles = [Line2D([0], [0], color=c, ls=ls, marker=mk, markersize=3.4,
                      linewidth=1.2, markeredgewidth=0.0) for _, _, ls, mk, c in order]
    labels = [n for _, n, _, _, _ in order]
    handles.append(Patch(facecolor=HIST_COLOR, alpha=0.30))
    labels.append("GT-box density")
    fig.tight_layout(rect=(0, 0.105, 1, 1))
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=6, frameon=False, fontsize=7.5, handlelength=2.0,
               handletextpad=0.5, columnspacing=1.2, labelspacing=0.35,
               borderaxespad=0.0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {args.out.with_suffix('.pdf')}")
    print(f"[save] {args.out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
