"""
Replot SOT SR/NPR vs GT bbox size from a precomputed `summary.json`.

Reads the aggregated bins produced by `plot_sot_perf_vs_bbox_size.py` and
draws, per dataset, a bar chart of mean SR and NPR over 0–32 px (bin = 2 px),
with frame counts rendered as a gray background histogram.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_DIR = Path(
    "/work/ziwen/experiments/NeurIPS/SOT_whole_dataset_04_22/analysis/perf_vs_bbox_size_bin2"
)

METRIC_COLORS = {"SR": "#1f77b4", "NPR": "#2ca02c"}
METRIC_LABELS = {
    "SR":  "SR (Success AUC)",
    "NPR": "NPR (Norm. Precision AUC)",
}


def plot_dataset(
    ds: str,
    agg: dict,
    out_path: Path,
    x_max: float = 32.0,
    metrics: tuple[str, ...] = ("SR", "NPR"),
) -> None:
    edges = np.asarray(agg["bin_edges_px"], dtype=float)
    counts = np.asarray(agg["count"], dtype=np.int64)
    centers = (edges[:-1] + edges[1:]) / 2
    bin_px = float(edges[1] - edges[0])

    keep = edges[1:] <= x_max + 1e-6
    if not keep.any():
        print(f"[warn] {ds}: no bins ≤ {x_max} px")
        return
    last = int(np.argmax(keep[::-1]))
    sl = slice(0, len(keep) - last)

    centers_s = centers[sl]
    counts_s = counts[sl]
    edges_s = edges[: sl.stop + 1]

    fig, ax = plt.subplots(figsize=(9.5, 4.4))

    ax_bg = ax.twinx()
    ax_bg.fill_between(
        centers_s, counts_s, step="mid",
        alpha=0.20, color="#888888", linewidth=0, label="# frames",
    )
    ax_bg.set_ylabel("# frames per bin", color="#555555")
    ax_bg.tick_params(axis="y", colors="#555555")
    ax_bg.set_ylim(0, counts_s.max() * 1.25 if counts_s.max() > 0 else 1)
    ax_bg.set_zorder(ax.get_zorder() - 1)
    ax.patch.set_visible(False)

    n_metrics = len(metrics)
    width = bin_px / (n_metrics + 1.0)
    offsets = (np.linspace(-width / 2, width / 2, n_metrics)
               if n_metrics > 1 else np.array([0.0]))
    for off, key in zip(offsets, metrics):
        vals = np.asarray(agg[key], dtype=float)[sl]
        ax.bar(
            centers_s + off, np.nan_to_num(vals, nan=0.0),
            width=width * 0.95, color=METRIC_COLORS[key],
            edgecolor="white", linewidth=0.4,
            label=METRIC_LABELS[key], zorder=3,
        )

    ax.set_xlim(edges_s[0], edges_s[-1])
    ax.set_ylim(0, 1.0)
    ax.set_xlabel(r"GT bbox size  $\sqrt{\mathrm{area}}$  [px]")
    ax.set_ylabel("Metric (mean over models)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    ax.set_xticks(edges_s)
    ax.set_xticklabels([f"{int(e)}" for e in edges_s], fontsize=9)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_bg.get_legend_handles_labels()
    fig.legend(
        h1 + h2, l1 + l2,
        loc="lower center", ncol=len(h1) + len(h2),
        fontsize=7, framealpha=0.92,
        bbox_to_anchor=(0.5, -0.01),
    )

    n_models = len(agg.get("models", []))
    total_shown = int(counts_s.sum())
    ax.set_title(
        f"{ds} — mean SOT performance vs GT bbox size "
        f"(bin = {int(bin_px)} px, {n_models} models, "
        f"{total_shown:,} frames ≤ {int(x_max)} px)",
        fontsize=11,
    )

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path.with_suffix('.png')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", type=Path,
                    default=DEFAULT_DIR / "summary.json")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--x-max", type=float, default=32.0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.summary.read_text())

    for ds in ("OOTB", "SatSOT", "SV248S"):
        if ds not in data:
            print(f"[warn] {ds} missing from {args.summary}")
            continue
        out_path = args.out_dir / f"{ds.lower()}_sr_npr_vs_bbox_size"
        plot_dataset(ds, data[ds], out_path, x_max=args.x_max)


if __name__ == "__main__":
    main()
