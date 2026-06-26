"""Plot association ability vs. object size for Exp2 (GT-box oracle).

Reads ``assa_vs_size.csv`` (produced by ``compute_hota_by_size.py``), pools each
metric across all 5 MOT datasets per size bin (weighted by ``n_gt_tracks``,
empty bins dropped), and draws AssA & IDF1 vs. object-size line plots. Detection
is oracle (DetA approx 1.0) so AssA/IDF1 isolate pure **association** ability.

TBD motion trackers (SORT/ByteTrack/OC-SORT/BoT-SORT) are drawn solid; JDT
learned-ReID trackers (FairMOT/TGraM) dashed, to contrast the two paradigms.

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

ROOT = Path("/data/ESA_DLSTEM_2025/experiments/MOT/exp2_oracle_20260608")
CSV = ROOT / "assa_vs_size.csv"
OUT_DIR = Path("/home/anon/code/esa_dlstem/docs/figures")

# bin_idx 0..5 -> label (matches compute_hota_by_size.py)
BINS = [(0, "<5"), (1, "5–8"), (2, "8–12"),
        (3, "12–20"), (4, "20–40"), (5, "≥40")]

# method -> (display name, paradigm). Order controls legend/z-order.
METHODS = [
    ("sort",      "SORT",      "TBD"),
    ("bytetrack", "ByteTrack", "TBD"),
    ("ocsort",    "OC-SORT",   "TBD"),
    ("botsort",   "BoT-SORT",  "TBD"),
    ("fairmot",   "FairMOT",   "JDT"),
    ("tgram",     "TGraM",     "JDT"),
]
# Colorblind-friendly: TBD in blues/greens, JDT in warm reds/oranges.
STYLE = {
    "sort":      dict(color="#1f77b4", ls="-",  marker="o"),
    "bytetrack": dict(color="#2ca02c", ls="-",  marker="s"),
    "ocsort":    dict(color="#17becf", ls="-",  marker="^"),
    "botsort":   dict(color="#9467bd", ls="-",  marker="D"),
    "fairmot":   dict(color="#d62728", ls="--", marker="v"),
    "tgram":     dict(color="#ff7f0e", ls="--", marker="P"),
}


def pooled(df: pd.DataFrame, metric: str) -> dict[str, np.ndarray]:
    """method -> array over size bins of n_gt_tracks-weighted mean metric."""
    out = {}
    for key, _, _ in METHODS:
        vals = []
        for idx, _ in BINS:
            sub = df[(df.method == key) & (df.bin_idx == idx) & (df.n_gt_tracks > 0)]
            w = sub.n_gt_tracks.to_numpy(dtype=float)
            v = sub[metric].to_numpy(dtype=float)
            ok = w > 0
            vals.append(np.average(v[ok], weights=w[ok]) if ok.any() else np.nan)
        out[key] = np.asarray(vals, dtype=float)
    return out


def main():
    df = pd.read_csv(CSV)
    apply_neurips_style(base_size=9.0)

    x = np.arange(len(BINS))
    labels = [lab for _, lab in BINS]
    # both metrics are higher-is-better -> up-arrow next to the axis label
    panels = [("AssA", "AssA $\\uparrow$  (association accuracy)"),
              ("IDF1", "IDF1 $\\uparrow$")]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharex=True)
    for ax, (metric, ylabel) in zip(axes, panels):
        series = pooled(df, metric)
        for key, name, _ in METHODS:
            y = series[key]
            ax.plot(x, y, label=name, markersize=4.0, linewidth=1.3,
                    markeredgewidth=0.0, **STYLE[key])
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("object size  (px, $\\sqrt{wh}$)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.5, 1.02)
        ax.grid(True, axis="y")
        ax.margins(x=0.03)

    # single shared legend below the panels, grouped TBD then JDT
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=6,
               frameon=False, bbox_to_anchor=(0.5, -0.02),
               columnspacing=1.2, handlelength=2.0)
    fig.subplots_adjust(bottom=0.30, wspace=0.28, left=0.08, right=0.98, top=0.93)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = OUT_DIR / f"exp2_assa_vs_size.{ext}"
        fig.savefig(p)
        print(f"wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
