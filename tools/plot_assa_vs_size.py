"""Plot association ability vs. object size for Exp2 (GT-box oracle).

Reads ``assa_vs_size_le32.csv`` (produced by ``compute_hota_by_size.py`` with
``--bins 0,5,8,12,20,32``), pools each metric across all 5 MOT datasets per size
bin, and draws AssA / IDF1 / IDsw vs. object size — the same three panels as the
BIRDSAI GT-oracle figure (``tools/plot_birdsai_gt_oracle.py``). Detection is
oracle (the SAME GT boxes are fed to every method), so any size trend is the
method's own **association** ability.

All nine benchmarked association methods are drawn, by paradigm:
  * TBD (solid)  — SORT / ByteTrack / OC-SORT / BoT-SORT / BoT-SORT+ReID / TrackTrack
  * JDT (dashed) — FairMOT / TGraM (learned stride-4 centre ReID)
  * query (dotted) — MOTRv2 (track queries)

The size axis stops at 32 px: above that every method saturates and the bins hold
only a handful of tracks, so the interesting range is the small-object end.

Pooling: AssA / IDF1 are ``n_gt_tracks``-weighted means over datasets (empty bins
dropped); IDsw is a raw count, so it is SUMMED over datasets and drawn on a log
axis (JDT switch counts are ~2 orders of magnitude above TBD).

Run:
    python tools/plot_assa_vs_size.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import apply_neurips_style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
ROOT = Path("/data/ESA_DLSTEM_2025/experiments/MOT/exp2_oracle_20260608")
CSV = ROOT / "assa_vs_size_le32.csv"
OUT_DIR = REPO / "docs" / "figures"

# bin_idx 0..4 -> label (compute_hota_by_size.py --bins 0,5,8,12,20,32)
BINS = [(0, "<5"), (1, "5–8"), (2, "8–12"), (3, "12–20"), (4, "20–32")]

# method -> (display name, paradigm). Order controls legend/z-order.
METHODS = [
    ("sort",         "SORT",          "TBD"),
    ("bytetrack",    "ByteTrack",     "TBD"),
    ("ocsort",       "OC-SORT",       "TBD"),
    ("botsort",      "BoT-SORT",      "TBD"),
    ("botsort_reid", "BoT-SORT+ReID", "TBD"),
    ("tracktrack",   "TrackTrack",    "TBD"),
    ("fairmot",      "FairMOT",       "JDT"),
    ("tgram",        "TGraM",         "JDT"),
    ("motrv2",       "MOTRv2",        "query"),
]
# TBD solid / JDT dashed / query-based dotted, so the paradigms read apart even
# in greyscale.
#
# Two pairs land on top of each other in every panel — BoT-SORT vs BoT-SORT+ReID
# (AssA differs by <0.05 pt: on 5-40 px crops the MOT17 ReID embedding adds
# nothing) and FairMOT vs TGraM (same JDT recipe, same stride-4 ReID floor). The
# lower member of each pair is drawn as a wide translucent halo so the pair reads
# as "identical" instead of "one line is missing".
STYLE = {
    "sort":         dict(color="#1f77b4", ls="-",  marker="o"),
    "bytetrack":    dict(color="#2ca02c", ls="-",  marker="s"),
    "ocsort":       dict(color="#17becf", ls="-",  marker="^"),
    "botsort":      dict(color="#9467bd", ls="-",  marker="D"),
    "botsort_reid": dict(color="#8c564b", ls="-",  marker="X"),
    "tracktrack":   dict(color="#e377c2", ls="-",  marker="*"),
    "fairmot":      dict(color="#d62728", ls="--", marker="v"),
    "tgram":        dict(color="#ff7f0e", ls="--", marker="P"),
    "motrv2":       dict(color="#7f7f7f", ls=":",  marker="h"),
}
HALO = {"botsort": dict(linewidth=3.2, alpha=0.45, markersize=6.5),
        "fairmot": dict(linewidth=3.2, alpha=0.45, markersize=6.5)}


def pooled(df: pd.DataFrame, metric: str, how: str) -> dict[str, np.ndarray]:
    """method -> array over size bins; ``how`` is "weighted" (rate) or "sum" (count)."""
    out = {}
    for key, _, _ in METHODS:
        vals = []
        for idx, _ in BINS:
            sub = df[(df.method == key) & (df.bin_idx == idx) & (df.n_gt_tracks > 0)]
            w = sub.n_gt_tracks.to_numpy(dtype=float)
            v = sub[metric].to_numpy(dtype=float)
            ok = w > 0
            if not ok.any():
                vals.append(np.nan)
            elif how == "sum":
                vals.append(float(v[ok].sum()))
            else:
                vals.append(float(np.average(v[ok], weights=w[ok])))
        out[key] = np.asarray(vals, dtype=float)
    return out


def main():
    df = pd.read_csv(CSV)
    apply_neurips_style(base_size=9.0)

    x = np.arange(len(BINS))
    labels = [lab for _, lab in BINS]
    panels = [
        ("AssA", "AssA (%) $\\uparrow$  (association accuracy)", "weighted", 100.0, False),
        ("IDF1", "IDF1 (%) $\\uparrow$", "weighted", 100.0, False),
        ("IDsw", "IDsw (count) $\\downarrow$", "sum", 1.0, True),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 2.9), sharex=True)
    for ax, (metric, ylabel, how, scale, logy) in zip(axes, panels):
        series = pooled(df, metric, how)
        for key, name, _ in METHODS:
            y = series[key] * scale
            if logy:                       # 0 switches is fine but unplottable in log
                y = np.where(y <= 0, np.nan, y)
            kw = dict(markersize=4.0, linewidth=1.3, markeredgewidth=0.0)
            kw.update(HALO.get(key, {}))
            ax.plot(x, y, label=name, **kw, **STYLE[key])
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("object size  (px, $\\sqrt{wh}$)")
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, 102)
        ax.grid(True, axis="y")
        ax.margins(x=0.03)

    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=5,
               frameon=False, bbox_to_anchor=(0.5, 0.0),
               columnspacing=1.2, handlelength=2.0)
    fig.subplots_adjust(bottom=0.32, wspace=0.30, left=0.07, right=0.99, top=0.97)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = OUT_DIR / f"exp2_assa_vs_size.{ext}"
        fig.savefig(p)
        print(f"wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
