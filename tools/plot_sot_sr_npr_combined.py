"""
Combined SR / NPR vs GT bbox size across OOTB, SatSOT, SV248S.

One figure, two panels (SR | NPR). Each dataset is a coloured line. Frame
counts per dataset are shown as faint step-fills on a secondary y-axis so the
reader can tell where metric estimates are well- vs poorly-supported without
getting three separate gray backgrounds stacked on top of each other.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_SUMMARY = Path(
    "/work/ziwen/experiments/NeurIPS/SOT_whole_dataset_04_22/analysis/perf_vs_bbox_size_bin2/summary.json"
)
DEFAULT_OUT_DIR = Path(
    "/home/ziwen/code/esa_dlstem/docs/NeurIPS_results/SOT/attributes"
)

DATASET_ORDER = ("OOTB", "SatSOT", "SV248S")
DATASET_COLORS = {
    "OOTB":   "#1f77b4",
    "SatSOT": "#d62728",
    "SV248S": "#2ca02c",
}


def _trim(edges: np.ndarray, arr: np.ndarray, x_max: float) -> tuple[np.ndarray, np.ndarray]:
    keep = edges[1:] <= x_max + 1e-6
    last = int(np.argmax(keep[::-1]))
    sl = slice(0, len(keep) - last)
    centers = (edges[:-1] + edges[1:]) / 2
    return centers[sl], arr[sl]


def plot_combined(data: dict, out_path: Path, x_max: float = 32.0) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    metric_titles = {"SR": "SR (Success AUC)",
                     "NPR": "NPR (Norm. Precision AUC)"}

    # Shared right-side axis scale for per-dataset frame counts: use fraction
    # of that dataset's total inside [0, x_max] so scales line up.
    frac_axes = [ax.twinx() for ax in axes]

    for ax_idx, metric in enumerate(("SR", "NPR")):
        ax = axes[ax_idx]
        ax_frac = frac_axes[ax_idx]

        # Faint per-dataset frame-count fraction as background.
        for ds in DATASET_ORDER:
            if ds not in data:
                continue
            edges = np.asarray(data[ds]["bin_edges_px"], dtype=float)
            counts = np.asarray(data[ds]["count"], dtype=float)
            centers, counts_s = _trim(edges, counts, x_max)
            total = counts_s.sum()
            if total <= 0:
                continue
            frac = counts_s / total
            ax_frac.fill_between(
                centers, frac, step="mid",
                alpha=0.08, color=DATASET_COLORS[ds], linewidth=0,
            )

        ax_frac.set_ylim(0, None)
        ax_frac.set_yticks([])
        ax_frac.set_zorder(ax.get_zorder() - 1)
        ax.patch.set_visible(False)

        # Metric lines on top.
        for ds in DATASET_ORDER:
            if ds not in data:
                continue
            edges = np.asarray(data[ds]["bin_edges_px"], dtype=float)
            vals = np.asarray(data[ds][metric], dtype=float)
            centers, vals_s = _trim(edges, vals, x_max)
            ax.plot(
                centers, vals_s,
                color=DATASET_COLORS[ds], marker="o", markersize=4,
                linewidth=1.8, label=ds, zorder=3,
            )

        ax.set_xlim(0, x_max)
        ax.set_ylim(0, 1.0)
        ax.set_xticks(np.arange(0, x_max + 1, 4))
        ax.set_xlabel(r"GT bbox size  $\sqrt{\mathrm{area}}$  [px]")
        ax.set_title(metric_titles[metric], fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Metric (mean over 7 models)")

    # Single shared legend below the figure.
    handles, labels = axes[0].get_legend_handles_labels()
    # Add a proxy for the faded frame-count background.
    from matplotlib.patches import Patch
    handles.append(Patch(facecolor="#888888", alpha=0.18,
                         label="frame-count density"))
    labels.append("frame-count density (per-dataset, normalised)")
    fig.legend(
        handles, labels,
        loc="lower center", ncol=len(handles),
        fontsize=7, framealpha=0.92,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        "Mean SOT performance vs GT bbox size (bin = 2 px, √area ≤ 32 px)",
        fontsize=11, y=0.99,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(out_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path.with_suffix('.png')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--x-max", type=float, default=32.0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.summary.read_text())
    plot_combined(
        data, args.out_dir / "sr_npr_vs_bbox_size_combined",
        x_max=args.x_max,
    )


if __name__ == "__main__":
    main()
