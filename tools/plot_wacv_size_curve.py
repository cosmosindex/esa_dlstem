"""
Tracking accuracy versus ground-truth object scale, for both halves of the
benchmark (WACV figure).

Differences from the NeurIPS version (tools/plot_sot_sr_npr_unified.py):

1. **One dataset, not three.** The NeurIPS figure averaged the three source
   benchmarks with equal weight, which keeps the source boundaries alive inside
   the curve. Space-Tracker-SOT is a single dataset here: every released
   sequence contributes its frames directly, so SV248S naturally dominates the
   frame count. That is a property of the benchmark, not something to average
   away.
2. **Nine trackers, not seven.** SiamFC and SmallTrack were added to the
   evaluation after the NeurIPS submission.
3. **OOTB is scored axis-aligned**, for every tracker, matching the protocol of
   the headline table. Object scale on OOTB is therefore also measured on the
   enclosing axis-aligned box.

Per bin, each tracker's SR and NPR are computed over the frames it answered on,
and the curve is the mean across trackers. The grey histogram is the pooled
frame-count density, counted once per frame rather than once per tracker.

With ``--mot-nocar`` the MOT half is overlaid on the same axes, averaged over
its methods the same way, so that the scale at which tracking stops working can
be read once for the whole benchmark instead of twice. Only the airplane+ship
half of MOT is drawn: every car in the MOT test split is at most 24 px and 80%
of them are under 8 px, so the car half occupies the left quarter of this axis
and would say nothing the SOT curves do not already say. MOT contributes DetRe
and AssA -- the two axes of HOTA that are functions of ground-truth scale. HOTA's
third ingredient, precision, is not: a false positive has no ground-truth object
and so no ground-truth scale, and DetA cannot be formed per bin without
inventing one.

Usage:
    python tools/plot_wacv_size_curve.py \
        --runs    /path/to/SOT_whole_dataset_<date> \
        --release /path/to/release/space_tracker \
        --out     wacv-2027-author-kit-template/plots/tracking_accuracy_vs_size
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from tools.aggregate_sot_release import _aabb_rescore
from tools.make_wacv_sot_table import DATASETS, TRACKERS, find_run, released_sequences
from tools.plot_style import apply_neurips_style

# Thresholds -- kept in sync with lightning_modules/sot_metrics.py
SUCCESS_THRESHOLDS = np.linspace(0, 1, 21)
NORM_PRECISION_THRESHOLDS = np.linspace(0, 0.5, 21)

# The figure is 2x2 -- two tasks, two metrics each -- so the palette encodes
# that rather than giving four unrelated series four unrelated hues. Each task
# owns one hue family, including its density mark; within a family the two
# metrics differ by lightness. Task is thus coded twice, by hue family and by
# line style, and the four curves stay separable in greyscale because the two
# members of each family differ in luminance by roughly a factor of two.
SOT_DEEP, SOT_MID = "#14507D", "#6DA9D2"   # blues
MOT_DEEP, MOT_MID = "#A85A1E", "#E4A24A"   # ambers
METRIC_COLORS = {"SR": SOT_DEEP, "NPR": SOT_MID}
MOT_COLORS = {"DetRe": MOT_DEEP, "AssA": MOT_MID}
HIST_COLOR = SOT_DEEP
MOT_HIST_COLOR = MOT_DEEP

#: MOT methods averaged into the overlay, in the order they appear in the
#: headline table. Kept explicit rather than "everything in the CSV" so that a
#: method added to the CSV cannot silently change a published curve.
MOT_METHODS = ["sort", "bytetrack", "ocsort", "botsort", "botsort_reid",
               "tracktrack", "masa", "fairmot_all", "tgram_all", "motrv2",
               "motip"]


def mot_curves(csv_path: Path, x_max: float):
    """Mean DetRe and AssA over the MOT methods, plus the GT-box density.

    Returns bin centres of the CSV's own binning, which is coarser than the SOT
    binning and is left that way: the underlying HOTA decomposition is expensive
    to recompute and 4 px bins are what the released CSV holds.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    missing = set(MOT_METHODS) - set(df.tracker.unique())
    if missing:
        raise SystemExit(f"{csv_path}: missing methods {sorted(missing)}")
    df = df[df.tracker.isin(MOT_METHODS) & (df.bin_hi <= x_max + 1e-9)]

    g = df.groupby(["bin_lo", "bin_hi"])
    stat = g[["DetRe", "AssA"]].mean()
    n_gt = g["n_gt"].first()
    lo = np.array([i[0] for i in stat.index], dtype=float)
    hi = np.array([i[1] for i in stat.index], dtype=float)
    frac = n_gt.values / max(n_gt.values.sum(), 1)
    return ((lo + hi) / 2, stat["DetRe"].values, stat["AssA"].values,
            lo, hi, frac, len(MOT_METHODS))


def _size_px(rec: dict) -> float:
    """GT scale s = sqrt(area) from the stored box, which is already the AABB."""
    box = rec.get("gt_box")
    if box is None or len(box) != 4:
        return float("nan")
    w = float(box[2]) - float(box[0])
    h = float(box[3]) - float(box[1])
    return float(np.sqrt(w * h)) if w > 0 and h > 0 else float("nan")


def collect_bins(runs_root: Path, released: dict, bin_w: float, n_bins: int):
    """(tracker, bin) -> [(iou, ncle)], plus unique frame counts per bin."""
    buckets: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    frame_counts = np.zeros(n_bins, dtype=np.int64)
    seen: set[tuple[str, str, int]] = set()

    for key, name, *_ in TRACKERS:
        for ds in DATASETS:
            pim = find_run(runs_root, key, ds)
            if pim is None:
                raise FileNotFoundError(f"no run found for {key}/{ds}")
            keep = set(released[ds])
            rescore = ds == "ootb"
            for frame in json.loads(pim.read_text()):
                vid = frame["video_id"]
                if vid not in keep:
                    continue
                for rec in frame.get("sot_records", []):
                    s = _size_px(rec)
                    if not np.isfinite(s):
                        continue
                    b = int(s // bin_w)
                    if b >= n_bins:
                        continue
                    if rescore:
                        iou, _, ncle = _aabb_rescore(rec)
                    else:
                        iou = float(rec.get("best_iou", 0.0))
                        ncle = float(rec.get("norm_center_dist", 0.0))
                    if not np.isfinite(ncle):
                        ncle = 1e9
                    buckets[(key, b)].append((iou, ncle))
                    fk = (ds, vid, int(frame["frame_id"]))
                    if fk not in seen:
                        seen.add(fk)
                        frame_counts[b] += 1
        print(f"  {name:<12} done")
    return buckets, frame_counts


def curves(buckets, n_bins: int):
    keys = [t[0] for t in TRACKERS]
    sr = np.full((len(keys), n_bins), np.nan)
    npr = np.full((len(keys), n_bins), np.nan)
    for i, k in enumerate(keys):
        for b in range(n_bins):
            recs = buckets.get((k, b))
            if not recs:
                continue
            arr = np.asarray(recs, dtype=float)
            ious, ncles = arr[:, 0], arr[:, 1]
            sr[i, b] = np.mean([np.mean(ious >= t) for t in SUCCESS_THRESHOLDS])
            npr[i, b] = np.mean([np.mean(ncles <= t) for t in NORM_PRECISION_THRESHOLDS])
    with np.errstate(invalid="ignore"):
        return np.nanmean(sr, axis=0), np.nanmean(npr, axis=0)


def plot(sr, npr, frame_counts, out_stem: Path, x_max: float, bin_w: float,
         n_models: int, figsize: tuple[float, float], mot=None) -> None:
    apply_neurips_style(base_size=9.0)
    edges = np.arange(0.0, x_max + 1e-6, bin_w)
    centers = (edges[:-1] + edges[1:]) / 2
    frac = frame_counts / frame_counts.sum() if frame_counts.sum() else frame_counts

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax_hist = ax.twinx()
    ax_hist.bar(centers, frac, width=bin_w * 0.95, color=HIST_COLOR,
                alpha=0.14, linewidth=0, zorder=1)
    hist_top = max(frac.max() * 1.15, 1e-3)

    if mot is not None:
        m_c, m_det, m_ass, m_lo, m_hi, m_frac, n_mot = mot
        # The MOT density is drawn as an outline rather than a second set of
        # bars: two filled histograms on one axis read as one histogram. Both
        # are normalised within themselves, so the two outlines say "where does
        # each half put its data", never "which half has more".
        step_x = np.repeat(np.append(m_lo, m_hi[-1]), 2)[1:-1]
        step_y = np.repeat(m_frac * (hist_top / max(m_frac.max(), 1e-9)) * 0.85, 2)
        ax_hist.plot(step_x, step_y, color=MOT_HIST_COLOR, linewidth=0.9,
                     alpha=0.45, linestyle=(0, (2.5, 1.5)), zorder=1)

    ax_hist.set_ylim(0, hist_top)
    ax_hist.set_yticks([])
    ax_hist.set_zorder(ax.get_zorder() - 1)
    ax.patch.set_visible(False)

    ax.plot(centers, sr, color=METRIC_COLORS["SR"], marker="o", markersize=3.2,
            linewidth=1.5, label="SOT: SR", zorder=3)
    ax.plot(centers, npr, color=METRIC_COLORS["NPR"], marker="s", markersize=3.2,
            linewidth=1.5, label="SOT: NPR", zorder=3)
    if mot is not None:
        ax.plot(m_c, m_det, color=MOT_COLORS["DetRe"], marker="^", markersize=3.8,
                markerfacecolor="white", markeredgewidth=1.0, linewidth=1.4,
                linestyle=(0, (4, 1.6)), label="MOT: DetRe", zorder=3)
        ax.plot(m_c, m_ass, color=MOT_COLORS["AssA"], marker="D", markersize=3.2,
                markerfacecolor="white", markeredgewidth=1.0, linewidth=1.4,
                linestyle=(0, (4, 1.6)), label="MOT: AssA", zorder=3)

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(np.arange(0, x_max + 1, 8))
    ax.set_xlabel(r"object scale  $s=\sqrt{\mathrm{area}}$  [px]")
    ax.set_ylabel("mean over methods")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    ax.axvline(8, color="#444444", linestyle="--", linewidth=1.0, zorder=2)
    ax.text(7.7, 0.97, r"tiny: $s<8$ px", ha="right", va="top", fontsize=8,
            color="#444444", zorder=4)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor=HIST_COLOR, alpha=0.22))
    labels.append("SOT frame density")
    if mot is not None:
        handles.append(Line2D([0], [0], color=MOT_HIST_COLOR, linewidth=1.0,
                              alpha=0.55, linestyle=(0, (2.5, 1.5))))
        labels.append("MOT box density")
    # Outside the axes: with four curves there is no longer an empty corner.
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.30),
              ncol=3, frameon=False, fontsize=7.0, handlelength=1.8,
              handletextpad=0.5, columnspacing=1.0, labelspacing=0.35)
    fig.tight_layout()

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_stem.with_suffix('.pdf')}")
    print(f"[save] {out_stem.with_suffix('.png')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--release", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--x-max", type=float, default=32.0)
    ap.add_argument("--bin-w", type=float, default=2.0)
    ap.add_argument("--figsize", type=str, default="3.35,2.5",
                    help="inches, W,H -- default fits one WACV column")
    ap.add_argument("--mot-nocar", type=Path,
                    help="mot_size_curve_nocar.csv; overlays the MOT half")
    ap.add_argument("--cache", type=Path,
                    help="npz holding the SOT pass, which is the slow part")
    args = ap.parse_args()

    n_bins = int(round(args.x_max / args.bin_w))
    released = released_sequences(args.release)
    n_seq = sum(len(v) for v in released.values())
    print(f"Released sequences: {n_seq}")

    if args.cache and args.cache.exists():
        z = np.load(args.cache)
        sr, npr, frame_counts = z["sr"], z["npr"], z["frame_counts"]
        print(f"SOT pass loaded from {args.cache}")
    else:
        buckets, frame_counts = collect_bins(args.runs, released, args.bin_w, n_bins)
        sr, npr = curves(buckets, n_bins)
        if args.cache:
            args.cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.cache, sr=sr, npr=npr, frame_counts=frame_counts)
    print(f"frames in [0,{args.x_max:g}) px: {frame_counts.sum()}")
    for b in range(n_bins):
        lo = b * args.bin_w
        print(f"  s in [{lo:4.0f},{lo + args.bin_w:4.0f})  "
              f"SR={sr[b]:.3f}  NPR={npr[b]:.3f}  frames={frame_counts[b]}")
    mot = mot_curves(args.mot_nocar, args.x_max) if args.mot_nocar else None
    if mot is not None:
        print(f"MOT overlay: mean over {mot[6]} methods")
        for c, d, a in zip(mot[0], mot[1], mot[2]):
            print(f"  s ~ {c:5.1f} px  DetRe={d:.3f}  AssA={a:.3f}")

    w, h = (float(v) for v in args.figsize.split(","))
    plot(sr, npr, frame_counts, args.out, args.x_max, args.bin_w, len(TRACKERS),
         (w, h), mot=mot)


if __name__ == "__main__":
    main()
