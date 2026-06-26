"""
Single-panel SR/NPR vs GT bbox size for the Space-Tracker-SOT benchmark.

Each metric (SR, NPR) is the equal-weight mean across OOTB / SatSOT / SV248S
of the per-bin per-model mean. A single faint histogram at the back shows the
combined frame-count density across the three datasets.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Match NeurIPS body text: Times-compatible serif at 10 pt, math in STIX.
mpl.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["STIXGeneral", "Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset":  "stix",
    "font.size":         10,        # body text in NeurIPS = 10 pt
    "axes.labelsize":    10,
    "axes.titlesize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,         # NeurIPS caption / footnote = 9 pt
    "figure.titlesize":  10,
    "pdf.fonttype":      42,        # embed TrueType so the PDF is searchable / vector
    "ps.fonttype":       42,
})

DEFAULT_SUMMARY = Path(
    "/work/anon/experiments/NeurIPS/SOT_whole_dataset_04_22/analysis/perf_vs_bbox_size_bin2/summary.json"
)
DEFAULT_OUT_DIR = Path(
    "/home/anon/code/esa_dlstem/docs/NeurIPS_results/SOT/attributes"
)
PAPER_PLOTS_DIR = Path(
    "/home/anon/code/esa_dlstem/Formatting Instructions For NeurIPS 2026/plots"
)

DATASET_ORDER = ("OOTB", "SatSOT", "SV248S")

METRIC_COLORS = {
    "SR":  "#2E5C8A",   # deep slate blue
    "NPR": "#C44E2D",   # warm terracotta
}
HIST_COLOR = "#7A7A7A"  # neutral grey for the density background


def _trim_to_grid(edges: np.ndarray, arr: np.ndarray, bin_w: float, n_bins: int) -> np.ndarray:
    """Take the first n_bins values of arr aligned to a uniform [0, n_bins*bin_w] grid."""
    edges = np.asarray(edges, dtype=float)
    arr = np.asarray(arr, dtype=float)
    if abs((edges[1] - edges[0]) - bin_w) > 1e-6:
        raise ValueError(f"bin width mismatch: got {edges[1] - edges[0]}, expected {bin_w}")
    out = np.full(n_bins, np.nan, dtype=float)
    take = min(n_bins, arr.shape[0])
    out[:take] = arr[:take]
    return out


def plot_unified(data: dict, out_path: Path, x_max: float = 32.0, bin_w: float = 2.0) -> None:
    n_bins = int(round(x_max / bin_w))
    grid_edges = np.arange(0.0, x_max + 1e-6, bin_w)
    centers = (grid_edges[:-1] + grid_edges[1:]) / 2

    # Stack per-dataset arrays onto the common grid.
    sr_stack, npr_stack, count_stack = [], [], []
    for ds in DATASET_ORDER:
        if ds not in data:
            continue
        edges = data[ds]["bin_edges_px"]
        sr_stack.append(_trim_to_grid(edges, data[ds]["SR"], bin_w, n_bins))
        npr_stack.append(_trim_to_grid(edges, data[ds]["NPR"], bin_w, n_bins))
        count_stack.append(_trim_to_grid(edges, data[ds]["count"], bin_w, n_bins))

    sr_mean = np.nanmean(np.vstack(sr_stack), axis=0)
    npr_mean = np.nanmean(np.vstack(npr_stack), axis=0)
    counts_total = np.nansum(np.vstack(count_stack), axis=0)
    counts_frac = counts_total / counts_total.sum() if counts_total.sum() > 0 else counts_total

    fig, ax = plt.subplots(1, 1, figsize=(8.0, 3.0))
    ax_hist = ax.twinx()

    # Histogram (combined frame-count density) sits behind the lines.
    ax_hist.bar(
        centers, counts_frac, width=bin_w * 0.95,
        color=HIST_COLOR, alpha=0.18, linewidth=0, zorder=1,
    )
    ax_hist.set_ylim(0, max(counts_frac.max() * 1.15, 1e-3))
    ax_hist.set_yticks([])
    ax_hist.set_zorder(ax.get_zorder() - 1)
    ax.patch.set_visible(False)

    # Two metric lines on top.
    ax.plot(
        centers, sr_mean,
        color=METRIC_COLORS["SR"], marker="o", markersize=4.5,
        linewidth=2.0, label="SR (Success AUC)", zorder=3,
    )
    ax.plot(
        centers, npr_mean,
        color=METRIC_COLORS["NPR"], marker="s", markersize=4.5,
        linewidth=2.0, label="NPR (Norm. Precision AUC)", zorder=3,
    )

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(np.arange(0, x_max + 1, 4))
    ax.set_xlabel(r"GT bbox size  $\sqrt{\mathrm{area}}$  [px]")
    ax.set_ylabel("Metric (mean over 7 models, 3 datasets)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    # Mark the tiny-object boundary at sqrt(area) = 8 px.
    ax.axvline(8, color="#444444", linestyle="--", linewidth=1.0, zorder=2)
    ax.text(
        7.7, 0.97, "tiny object",
        ha="right", va="top", fontsize=8, color="#444444",
        rotation=0, zorder=4,
    )

    # Legend below the figure (outside the plotting area).
    handles, labels = ax.get_legend_handles_labels()
    from matplotlib.patches import Patch
    handles.append(Patch(facecolor=HIST_COLOR, alpha=0.30,
                         label="frame-count density (3-dataset combined)"))
    labels.append("frame-count density (3-dataset combined)")
    fig.legend(
        handles, labels,
        loc="lower center", ncol=len(handles),
        framealpha=0.92,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout(rect=(0, 0.08, 1, 1.0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path.with_suffix('.png')}")
    print(f"[save] {out_path.with_suffix('.pdf')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--x-max", type=float, default=32.0)
    ap.add_argument("--bin-w", type=float, default=2.0)
    args = ap.parse_args()

    data = json.loads(args.summary.read_text())
    out_stem = args.out_dir / "sr_npr_vs_bbox_size_unified"
    plot_unified(data, out_stem, x_max=args.x_max, bin_w=args.bin_w)

    # Also drop a copy into the paper's plots directory so the .tex just works.
    PAPER_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        src = out_stem.with_suffix(ext)
        if src.exists():
            dst = PAPER_PLOTS_DIR / f"sr_npr_vs_bbox_size_unified{ext}"
            shutil.copy2(src, dst)
            print(f"[copy] {dst}")


if __name__ == "__main__":
    main()
