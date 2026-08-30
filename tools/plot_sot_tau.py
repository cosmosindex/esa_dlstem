"""
Ablation figure: what the confidence threshold tau actually does to each SOT tracker.

The reviewer asked whether tau=0.5 is a defensible operating point. Answering it
in a table means asking the reader to compare nine score distributions against a
scalar, which is exactly the comparison that is hard to do in numbers. The three
panels instead show:

(a) **Where each tracker's scores live.** A single global tau is only meaningful
    if the trackers are calibrated alike, and they are not. Three of them emit a
    constant 1.0 -- for those, tau is not a weak filter, it is *no filter at all*,
    at any value. Others put their 5th percentile above 0.9, so tau=0.5 is far
    outside their operating range. Only LoRAT has appreciable mass below 0.5.

(b) **What tau costs in SR.** Six curves are flat to three decimals up to 0.6;
    LoRAT falls off a cliff past 0.6. tau=0.5 sits in the flat region for every
    tracker, which is the argument that the main table is not threshold-tuned.

(c) **Why "no valid box" needed redefining.** The shipped numbers conflated the
    tracker declaring it lost the target (`declared`) with our own threshold
    removing a box the tracker did emit (`filtered`). SAM 2 is 29.0% declared /
    0.0% filtered; LoRAT is 0.8% declared / 16.6% filtered. Those are opposite
    failure modes that the old single "abstention" column reported identically.

Input CSVs come from tools/analyse_sot_tau.py.

Usage:
    python tools/plot_sot_tau.py \
        --csv-dir docs/space_tracker --prefix sot_tau \
        --out wacv-2027-author-kit-template/figures/sot_tau_ablation.pdf
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import apply_neurips_style

TAU_PAPER = 0.5

# key -> (display name, colour). Colour encodes the architecture family, because
# the panels are read family-wise: the constant-score group is not a coincidence,
# it is what Siamese correlation / one-stream ViT heads emit when the head has no
# calibrated confidence branch.
TRACKERS = [
    ("siamfc",     "SiamFC",      "#9ecae1"),
    ("siamrpn",    "SiamRPN++",   "#4292c6"),
    ("smalltrack", "SmallTrack",  "#08519c"),
    ("ostrack",    "OSTrack-384", "#fdae6b"),
    ("odtrack",    "ODTrack",     "#e6550d"),
    ("lorat",      "LoRAT-g378",  "#a63603"),
    ("sam2",       "SAM 2",       "#a1d99b"),
    ("samurai",    "SAMURAI",     "#41ab5d"),
    ("sam3",       "SAM 3",       "#006d2c"),
]
NAME = {k: n for k, n, _ in TRACKERS}
COLOR = {k: c for k, _, c in TRACKERS}


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def panel_scores(ax, dist: dict[str, dict], order: list[str]):
    """Horizontal min / central-90% / median summary of the score distribution.

    Deliberately NOT a Tukey box plot. These scores are so heavily massed at 1.0
    that the interquartile range degenerates: it is exactly 0.000 for SiamFC,
    OSTrack-384 and ODTrack, and 0.004-0.006 for SiamRPN++, SmallTrack and
    SAM 3 -- six of nine boxes would be narrower than the line used to draw
    them. The central 90% (p05-p95) is the narrowest conventional interval that
    stays visible for every tracker, so it is what we show; see
    docs/space_tracker/sot_tau_quantiles.csv for the full ladder.
    """
    for y, key in enumerate(order):
        d = dist[key]
        c = COLOR[key]
        if d["constant_score"] == "True":
            # A range plot would draw a zero-length bar and read as missing data.
            ax.plot([1.0], [y], marker="D", ms=5, color=c, zorder=3,
                    markeredgecolor="white", markeredgewidth=0.5)
            ax.annotate("constant", xy=(1.0, y), xytext=(-4, 0),
                        textcoords="offset points", ha="right", va="center",
                        fontsize=6.0, color=c, style="italic")
            continue
        lo, p05 = float(d["score_min"]), float(d["score_p05"])
        p50, p95 = float(d["score_p50"]), float(d["score_p95"])
        hi = float(d["score_max"])
        ax.plot([lo, hi], [y, y], lw=0.8, color=c, solid_capstyle="butt", zorder=2)
        ax.plot([p05, p95], [y, y], lw=4.0, color=c, alpha=0.55,
                solid_capstyle="butt", zorder=2)
        ax.plot([p50], [y], marker="o", ms=4, color=c, zorder=3,
                markeredgecolor="white", markeredgewidth=0.5)

    ax.axvspan(0.0, TAU_PAPER, color="0.5", alpha=0.10, lw=0, zorder=0)
    ax.axvline(TAU_PAPER, color="0.25", ls="--", lw=0.8, zorder=1)
    ax.annotate(rf"$\tau={TAU_PAPER}$", xy=(TAU_PAPER, -0.72), xytext=(3, 0),
                textcoords="offset points", ha="left", va="center",
                fontsize=7.0, color="0.25")
    ax.annotate("dropped", xy=(TAU_PAPER, -0.72), xytext=(-3, 0),
                textcoords="offset points", ha="right", va="center",
                fontsize=6.0, color="0.45", style="italic")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([NAME[k] for k in order])
    ax.set_xlim(-0.02, 1.06)
    ax.set_ylim(-1.1, len(order) + 2.9)    # spare rows for the stacked key below
    ax.invert_yaxis()
    ax.set_xlabel("predicted confidence")
    ax.set_title("(a) score distribution per tracker", loc="left")
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax.legend(handles=[
        Line2D([], [], color="0.35", lw=0.8, label="min-max"),
        Line2D([], [], color="0.35", lw=4.0, alpha=0.55, label="central 90%"),
        Line2D([], [], color="0.35", lw=0, marker="o", ms=4, label="median"),
    ], loc="lower right", bbox_to_anchor=(1.03, 0.02), frameon=False,
        handlelength=1.3, borderpad=0.0, labelspacing=0.35, ncols=1,
        handletextpad=0.4, fontsize=6.6)


def panel_sweep(ax, sweep: dict[str, list], order: list[str]):
    for key in order:
        rows = sweep[key]
        xs = [float(r["tau"]) for r in rows]
        ys = [float(r["SR"]) for r in rows]
        # Constant-score trackers are exactly horizontal; dashing them stops the
        # nine lines from looking like nine independently flat measurements.
        const = all(abs(v - ys[0]) < 1e-9 for v in ys)
        ax.plot(xs, ys, color=COLOR[key], lw=1.3,
                ls=(0, (4, 1.5)) if const else "-",
                marker="o" if not const else None, ms=2.4, zorder=3)
        if key == "lorat":
            # The one curve whose shape is the finding; labelling all nine here
            # only reproduces panel (a)'s legend on top of the data.
            ax.annotate(NAME[key], xy=(0.86, 0.095), xytext=(-1, 7),
                        textcoords="offset points", ha="right", va="bottom",
                        fontsize=6.5, color=COLOR[key])

    ax.axvline(TAU_PAPER, color="0.25", ls="--", lw=0.8, zorder=1)
    ax.annotate(rf"$\tau={TAU_PAPER}$", xy=(TAU_PAPER, 0.445), xytext=(-3, 0),
                textcoords="offset points", ha="right", va="top",
                fontsize=7.0, color="0.25")
    ax.set_xlim(-0.025, 0.925)
    ax.set_ylim(0, 0.45)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.annotate("dashed = constant score", xy=(0.025, 0.025),
                xycoords="axes fraction", fontsize=6.0, color="0.42",
                va="bottom")
    ax.set_xlabel(r"confidence threshold $\tau$")
    ax.set_ylabel("Success Rate")
    ax.set_title(r"(b) SR as a function of $\tau$", loc="left")
    ax.grid(axis="y", lw=0.4, alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def panel_split(ax, split: dict[str, list], order: list[str]):
    ys = list(range(len(order)))
    dec, fil = [], []
    for key in order:
        row = next(r for r in split[key] if abs(float(r["tau"]) - TAU_PAPER) < 1e-9)
        dec.append(float(row["declared_pct"]))
        fil.append(float(row["filtered_pct"]))

    for y, key, d, f in zip(ys, order, dec, fil):
        c = COLOR[key]
        ax.barh(y, d, color=c, height=0.62, zorder=3)
        ax.barh(y, f, left=d, color=c, height=0.62, alpha=0.35, zorder=3,
                hatch="///", edgecolor=c, lw=0.0)
        total = d + f
        if total > 0.4:
            ax.annotate(f"{total:.1f}", xy=(total, y), xytext=(3, 0),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=6.0, color="0.2")

    ax.set_yticks(ys)
    ax.set_yticklabels([NAME[k] for k in order])
    ax.set_ylim(-0.7, len(order) + 1.85)   # two spare rows: the key is two lines tall
    ax.invert_yaxis()
    ax.set_xlim(0, 38)
    ax.set_xlabel("frames with no scored box (%)")
    ax.set_title(rf"(c) abstention at $\tau={TAU_PAPER}$, split", loc="left")
    ax.grid(axis="x", lw=0.4, alpha=0.25)
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax.legend(handles=[
        Patch(facecolor="0.35", label="declared lost by tracker"),
        Patch(facecolor="0.35", alpha=0.35, hatch="///", edgecolor="0.35",
              label=r"removed by $\tau$"),
    ], loc="lower right", bbox_to_anchor=(1.01, 0.015), frameon=False,
        borderpad=0.0, labelspacing=0.2, handlelength=1.3, handletextpad=0.4)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-dir", default="docs/space_tracker")
    ap.add_argument("--prefix", default="sot_tau")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = Path(args.csv_dir)
    dist = {r["tracker"]: r for r in read_csv(d / f"{args.prefix}_score_dist.csv")}
    sweep, split = defaultdict(list), defaultdict(list)
    for r in read_csv(d / f"{args.prefix}_sweep.csv"):
        sweep[r["tracker"]].append(r)
    for r in read_csv(d / f"{args.prefix}_abstention_split.csv"):
        split[r["tracker"]].append(r)

    missing = [k for k, _, _ in TRACKERS if k not in dist]
    if missing:
        raise SystemExit(f"missing trackers in the CSVs: {missing}")

    # One order for all three panels -- by the SR the main table reports, so the
    # reader can carry a row from panel to panel without re-finding the name.
    def sr_at_paper_tau(key):
        return float(next(r for r in sweep[key]
                          if abs(float(r["tau"]) - TAU_PAPER) < 1e-9)["SR"])
    order = sorted((k for k, _, _ in TRACKERS), key=sr_at_paper_tau, reverse=True)

    apply_neurips_style(8.5)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.72),
                             gridspec_kw={"width_ratios": [1.02, 1.16, 0.95],
                                          "wspace": 0.44})
    panel_scores(axes[0], dist, order)
    panel_sweep(axes[1], sweep, order)
    panel_split(axes[2], split, order)

    # All three panels share one colour per tracker, so the key belongs to the
    # figure rather than to any panel. The handle's line style doubles as (b)'s
    # solid/dashed distinction, so one legend carries both encodings. Keep the
    # declaration order (grouped by architecture family), not the SR order used
    # inside the panels -- the colours were chosen family-wise and the key is
    # where that becomes visible.
    handles = [
        Line2D([], [], color=c, lw=1.6,
               ls=(0, (3.2, 1.4)) if dist[k]["constant_score"] == "True" else "-",
               label=nm)
        for k, nm, c in TRACKERS
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.055),
               ncols=9, frameon=False, handlelength=1.5, handletextpad=0.4,
               columnspacing=1.05, borderpad=0.0, fontsize=6.8)
    fig.subplots_adjust(bottom=0.195)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=220)
    print(f"wrote {out} and {out.with_suffix('.png')}")

    # Numbers the caption quotes, printed so the text can never drift from the figure.
    print("\ncaption facts:")
    const = [NAME[k] for k in order if dist[k]["constant_score"] == "True"]
    print(f"  constant-score trackers ({len(const)}): {', '.join(const)}")
    for k in order:
        s0 = sr_at_paper_tau(k)
        sz = float(next(r for r in sweep[k] if float(r["tau"]) == 0.0)["SR"])
        print(f"  {NAME[k]:12s} SR(0)={sz:.4f}  SR(0.5)={s0:.4f}  "
              f"delta={s0 - sz:+.4f}  p05={dist[k]['score_p05']}")
if __name__ == "__main__":
    main()
