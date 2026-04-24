"""
Plot average SOT performance (SR / PR / NPR) vs GT bbox size (√area, px)
for OOTB, SatSOT, SV248S. One figure per dataset, 3 grouped bars per bin.

Data sources:
  * Per-frame records:  <exp_root>/<model>/<run_dir>/per_image_metrics.json
  * GT bboxes (pixels): /data/ESA_DLSTEM_2025/data/trafic/{OOTB,SatSOT,SV248S}

Metric definitions (match lightning_modules/sot_metrics.py):
  * SR  = success_auc       : mean over t∈[0,1] step 0.05 of P(IoU ≥ t)
  * PR  = precision_auc     : mean over t∈{0..30} px       of P(CLE ≤ t)
  * NPR = norm_precision_auc: mean over t∈[0,0.5] step .025 of P(nCLE ≤ t)

For each dataset and 5-px size bin we compute each model's {SR, PR, NPR}
on the frames falling in that bin, then average across models.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Thresholds — kept in sync with lightning_modules/sot_metrics.py
# ---------------------------------------------------------------------------
SUCCESS_THRESHOLDS = np.linspace(0, 1, 21)           # SR  curve x-axis
PRECISION_THRESHOLDS = np.arange(0, 31, dtype=float)  # PR  curve x-axis (OOTB)
NORM_PRECISION_THRESHOLDS = np.linspace(0, 0.5, 21)   # NPR curve x-axis

EXP_ROOT = Path("/work/ziwen/experiments/NeurIPS/SOT_whole_dataset_04_22")
DATA_ROOTS = {
    "OOTB":   Path("/data/ESA_DLSTEM_2025/data/trafic/OOTB"),
    "SatSOT": Path("/data/ESA_DLSTEM_2025/data/trafic/SatSOT"),
    "SV248S": Path("/data/ESA_DLSTEM_2025/data/trafic/SV248S"),
}
# Folder → dataset name mapping used to bucket model runs.
# Keys are lowercase tokens that appear in run-dir names.
DS_TOKENS = {"ootb": "OOTB", "satsot": "SatSOT", "sv248s": "SV248S"}

MODEL_ORDER = ["siamrpn", "ostrack", "odtrack", "lorat", "samurai", "sam2", "sam3"]


# ---------------------------------------------------------------------------
# GT loaders — return {(video_id, frame_id): (w, h)} in pixels
# ---------------------------------------------------------------------------

def _load_gt_ootb(root: Path) -> dict[tuple[str, int], tuple[float, float]]:
    gt: dict[tuple[str, int], tuple[float, float]] = {}
    for seq in sorted(root.iterdir()):
        if not seq.is_dir() or seq.name == "anno":
            continue
        gt_file = seq / "groundtruth.txt"
        if not gt_file.exists():
            continue
        for fid, line in enumerate(gt_file.read_text().splitlines()):
            line = line.strip()
            if not line:
                continue
            vals = [float(v) for v in re.split(r"[,\t ]+", line)[:8]]
            if len(vals) < 8:
                continue
            xs, ys = vals[0::2], vals[1::2]
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)
            gt[(seq.name, fid)] = (w, h)
    return gt


def _load_gt_xywh(root: Path) -> dict[tuple[str, int], tuple[float, float]]:
    """SatSOT-style: one groundtruth.txt per sequence, rows `x,y,w,h`."""
    gt: dict[tuple[str, int], tuple[float, float]] = {}
    for seq in sorted(root.iterdir()):
        if not seq.is_dir():
            continue
        gt_file = seq / "groundtruth.txt"
        if not gt_file.exists():
            continue
        for fid, line in enumerate(gt_file.read_text().splitlines()):
            line = line.strip()
            if not line:
                continue
            tokens = re.split(r"[,\t ]+", line)[:4]
            try:
                vals = [float(v) for v in tokens]
            except ValueError:
                # SatSOT uses "none" on frames where the target is absent.
                continue
            if len(vals) < 4:
                continue
            gt[(seq.name, fid)] = (vals[2], vals[3])
    return gt


def _load_gt_sv248s(root: Path) -> dict[tuple[str, int], tuple[float, float]]:
    gt: dict[tuple[str, int], tuple[float, float]] = {}
    for video_dir in sorted(root.iterdir()):
        if not video_dir.is_dir():
            continue
        ann_dir = video_dir / "annotations"
        if not ann_dir.exists():
            continue
        for rect_file in sorted(ann_dir.glob("*.rect")):
            seq_id = rect_file.stem
            vid = f"{video_dir.name}/{seq_id}"
            for fid, line in enumerate(rect_file.read_text().splitlines()):
                line = line.strip()
                if not line:
                    continue
                vals = [float(v) for v in line.split(",")[:4]]
                if len(vals) < 4:
                    continue
                gt[(vid, fid)] = (vals[2], vals[3])
    return gt


GT_LOADERS = {
    "OOTB":   _load_gt_ootb,
    "SatSOT": _load_gt_xywh,
    "SV248S": _load_gt_sv248s,
}


# ---------------------------------------------------------------------------
# Class loaders — return {video_id: class_name} per dataset
# ---------------------------------------------------------------------------

def _cls_from_seqname(root: Path) -> dict[str, str]:
    """OOTB / SatSOT: class = sequence-name prefix (`car_1` → `car`)."""
    out: dict[str, str] = {}
    for seq in sorted(root.iterdir()):
        if not seq.is_dir() or seq.name == "anno":
            continue
        if not (seq / "groundtruth.txt").exists():
            continue
        out[seq.name] = re.sub(r"_\d+$", "", seq.name)
    return out


def _cls_sv248s(root: Path) -> dict[str, str]:
    """SV248S: class from `annotations/<seq>.abs` JSON (e.g. `car-large`)."""
    out: dict[str, str] = {}
    for video_dir in sorted(root.iterdir()):
        if not video_dir.is_dir():
            continue
        ann_dir = video_dir / "annotations"
        if not ann_dir.exists():
            continue
        for abs_path in sorted(ann_dir.glob("*.abs")):
            try:
                meta = json.loads(abs_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            cls = meta.get("details", {}).get("class_name", "unknown")
            out[f"{video_dir.name}/{abs_path.stem}"] = cls
    return out


CLS_LOADERS = {
    "OOTB":   _cls_from_seqname,
    "SatSOT": _cls_from_seqname,
    "SV248S": _cls_sv248s,
}

# Palette for class stacks (colourblind-friendly, stable ordering).
CLASS_PALETTE = [
    "#4C72B0", "#DD8452", "#55A467", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _detect_dataset(run_dir_name: str) -> str | None:
    low = run_dir_name.lower()
    for tok, name in DS_TOKENS.items():
        if f"_{tok}_" in low or low.endswith(f"_{tok}") or f"_{tok}20" in low:
            return name
    # Fallback: substring match after the last "first_frame".
    for tok, name in DS_TOKENS.items():
        if tok in low:
            return name
    return None


def _bin_edges(max_size: float, bin_px: int) -> np.ndarray:
    upper = int(np.ceil(max_size / bin_px) * bin_px)
    return np.arange(0, upper + bin_px, bin_px, dtype=float)


def _metrics_from(records: np.ndarray) -> dict[str, float]:
    """records: structured array with columns iou, cle, ncle."""
    ious, cles, ncles = records["iou"], records["cle"], records["ncle"]
    sr = float(np.mean([np.mean(ious >= t) for t in SUCCESS_THRESHOLDS]))
    pr = float(np.mean([np.mean(cles <= t) for t in PRECISION_THRESHOLDS]))
    npr = float(np.mean([np.mean(ncles <= t) for t in NORM_PRECISION_THRESHOLDS]))
    return {"SR": sr, "PR": pr, "NPR": npr}


def aggregate(
    exp_root: Path,
    gt_by_ds: dict[str, dict[tuple[str, int], tuple[float, float]]],
    bin_px: int,
    cls_by_ds: dict[str, dict[str, str]] | None = None,
) -> dict:
    """Returns nested dict: dataset → {metric → 1-D array over bins, 'count' → array,
    'edges' → bin edges, 'n_models' → int, 'class_count' → {cls: array}}.

    class_count is populated from the first model's run per dataset (same frames
    across models), so it reflects unique (video, frame) counts per class/bin.
    """
    # (dataset, model, bin_idx) → list of (iou, cle, ncle)
    buckets: dict[tuple[str, str, int], list[tuple[float, float, float]]] = defaultdict(list)
    # (dataset, cls, bin_idx) → unique-frame count (counted once, not per model)
    class_counts_raw: dict[tuple[str, str, int], int] = defaultdict(int)
    seen_frame: set[tuple[str, str, int]] = set()   # (ds, vid, fid)
    models_per_ds: dict[str, set[str]] = defaultdict(set)
    max_size_per_ds: dict[str, float] = defaultdict(float)
    cls_by_ds = cls_by_ds or {}

    for model_dir in sorted(exp_root.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        for run_dir in sorted(model_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            pim = run_dir / "per_image_metrics.json"
            if not pim.exists():
                continue
            ds = _detect_dataset(run_dir.name)
            if ds is None:
                print(f"[skip] cannot detect dataset for {run_dir}")
                continue
            gt = gt_by_ds[ds]
            data = json.loads(pim.read_text())
            kept = 0
            for entry in data:
                vid = entry["video_id"]
                fid = int(entry["frame_id"])
                wh = gt.get((vid, fid))
                if wh is None:
                    continue
                w, h = wh
                area = w * h
                if area <= 0 or not np.isfinite(area):
                    continue
                size = float(np.sqrt(area))
                b = int(size // bin_px)
                for rec in entry["sot_records"]:
                    iou = float(rec["best_iou"])
                    cle = float(rec["center_dist"])
                    ncle = float(rec["norm_center_dist"])
                    # Guard inf (no same-class prediction) → still counts as a
                    # failure. PR/NPR thresholds will exclude these naturally.
                    if not np.isfinite(cle):
                        cle = 1e9
                    if not np.isfinite(ncle):
                        ncle = 1e9
                    buckets[(ds, model, b)].append((iou, cle, ncle))
                    max_size_per_ds[ds] = max(max_size_per_ds[ds], size)
                    kept += 1
                # Count each unique (ds, vid, fid) once per class/bin, no
                # matter how many models processed it.
                key_frame = (ds, vid, fid)
                if key_frame not in seen_frame:
                    seen_frame.add(key_frame)
                    cls = cls_by_ds.get(ds, {}).get(vid, "unknown")
                    class_counts_raw[(ds, cls, b)] += 1
            models_per_ds[ds].add(model)
            print(f"  {ds:<6} {model:<10} {run_dir.name}: {kept} frames")

    out: dict[str, dict] = {}
    for ds in sorted(models_per_ds):
        edges = _bin_edges(max_size_per_ds[ds], bin_px)
        n_bins = len(edges) - 1
        models = sorted(models_per_ds[ds])

        # per-model per-bin metric arrays (NaN where model has no frames in bin)
        per_model = {m: {"SR": np.full(n_bins, np.nan),
                         "PR": np.full(n_bins, np.nan),
                         "NPR": np.full(n_bins, np.nan)} for m in models}
        counts = np.zeros(n_bins, dtype=np.int64)

        for b in range(n_bins):
            for m in models:
                recs = buckets.get((ds, m, b), [])
                if not recs:
                    continue
                arr = np.array(recs, dtype=[("iou", "f8"), ("cle", "f8"), ("ncle", "f8")])
                metrics = _metrics_from(arr)
                for k, v in metrics.items():
                    per_model[m][k][b] = v
                counts[b] += len(recs)

        # Per-class frame counts for this dataset
        classes_here = sorted({c for (d, c, _b) in class_counts_raw if d == ds})
        class_count = {c: np.zeros(n_bins, dtype=np.int64) for c in classes_here}
        for (d, c, b), n in class_counts_raw.items():
            if d == ds and b < n_bins:
                class_count[c][b] = n

        agg = {"edges": edges, "count": counts, "models": models,
               "n_models": len(models), "class_count": class_count}
        for k in ("SR", "PR", "NPR"):
            stacked = np.stack([per_model[m][k] for m in models], axis=0)  # (M, B)
            with np.errstate(invalid="ignore"):
                agg[k] = np.nanmean(stacked, axis=0)
                agg[f"{k}_std"] = np.nanstd(stacked, axis=0)
        out[ds] = agg
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

METRIC_COLORS = {
    "SR":  "#1f77b4",  # blue
    "PR":  "#d62728",  # red
    "NPR": "#2ca02c",  # green
}
METRIC_LABELS = {
    "SR":  "SR (Success AUC, IoU∈[0,1])",
    "PR":  "PR (Precision AUC, CLE≤30 px)",
    "NPR": "NPR (Norm. Precision AUC, nCLE≤0.5)",
}


def plot_dataset(
    ds: str, agg: dict, bin_px: int, out_path: Path,
    min_count: int = 30, x_max: float | None = None,
    coverage: float = 0.99,
    metrics: tuple[str, ...] = ("SR", "PR", "NPR"),
    stack_by_class: bool = False,
) -> None:
    edges = agg["edges"]
    centers = (edges[:-1] + edges[1:]) / 2
    counts = agg["count"]
    class_count = agg.get("class_count", {})

    # Decide the right edge of the axis:
    #   * explicit --x-max wins,
    #   * else the smallest size that already captures `coverage` of all frames
    #     (so rare large-bbox outliers don't stretch the axis).
    if x_max is None and counts.sum() > 0:
        cum = np.cumsum(counts) / counts.sum()
        idx = int(np.searchsorted(cum, coverage))
        x_max = float(edges[min(idx + 1, len(edges) - 1)])

    keep = counts >= min_count
    if x_max is not None:
        keep &= (edges[1:] <= x_max + 1e-6)
    if not keep.any():
        print(f"[warn] {ds}: no bins meet min_count={min_count}")
        return
    first = int(np.argmax(keep))
    last = len(keep) - 1 - int(np.argmax(keep[::-1]))
    sl = slice(first, last + 1)
    centers = centers[sl]
    counts = counts[sl]
    edges_sl = edges[first:last + 2]

    if stack_by_class and class_count:
        fig, (ax, ax_cls) = plt.subplots(
            2, 1, figsize=(11, 6.2), sharex=True,
            gridspec_kw={"height_ratios": [2.4, 1.0], "hspace": 0.08},
        )
        ax_bg = None  # no translucent fill on the metric axis in two-panel mode
    else:
        fig, ax = plt.subplots(figsize=(11, 4.8))
        ax_cls = None
        ax_bg = ax.twinx()
        ax_bg.fill_between(centers, counts, step="mid",
                           alpha=0.18, color="#888888", linewidth=0,
                           label="# frames")
        ax_bg.set_ylabel("# frames per bin", color="#555555")
        ax_bg.tick_params(axis="y", colors="#555555")
        ax_bg.set_ylim(0, counts.max() * 1.25 if counts.max() > 0 else 1)

    # Grouped bars on primary axis
    n_metrics = len(metrics)
    width = bin_px / (n_metrics + 1.3)
    offsets = (np.linspace(-width, width, n_metrics)
               if n_metrics > 1 else np.array([0.0]))
    for off, key in zip(offsets, metrics):
        vals = agg[key][sl]
        ax.bar(centers + off, np.nan_to_num(vals, nan=0.0),
               width=width * 0.95, color=METRIC_COLORS[key],
               edgecolor="white", linewidth=0.4,
               label=METRIC_LABELS[key])

    ax.set_xlim(edges_sl[0], edges_sl[-1])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Metric (mean over models)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    # Class-stacked histogram on the bottom panel
    if ax_cls is not None:
        classes = sorted(class_count, key=lambda c: -class_count[c].sum())
        bottoms = np.zeros(len(centers), dtype=np.float64)
        for i, cls in enumerate(classes):
            vals = class_count[cls][sl].astype(float)
            color = CLASS_PALETTE[i % len(CLASS_PALETTE)]
            ax_cls.bar(centers, vals, bottom=bottoms,
                       width=bin_px * 0.95, color=color,
                       edgecolor="white", linewidth=0.3,
                       label=cls)
            bottoms += vals
        ax_cls.set_ylabel("# frames per bin")
        ax_cls.set_xlabel(r"GT bbox size  $\sqrt{\mathrm{area}}$  [px]")
        ax_cls.grid(True, axis="y", linestyle=":", alpha=0.4)
        ax_cls.set_axisbelow(True)
        ax_cls.legend(loc="upper right", fontsize=8,
                      ncol=min(len(classes), 4), framealpha=0.92,
                      title="class")
    else:
        ax.set_xlabel(r"GT bbox size  $\sqrt{\mathrm{area}}$  [px]")

    # Thin ticks so the x-axis stays readable when bins are small.
    tick_ax = ax_cls if ax_cls is not None else ax
    step = max(1, int(np.ceil(len(edges_sl) / 16)))
    tick_edges = edges_sl[::step]
    tick_ax.set_xticks(tick_edges)
    tick_ax.set_xticklabels([f"{int(e)}" for e in tick_edges],
                            rotation=0, fontsize=9)

    # Metric legend (combine with frame-count label if present)
    h1, l1 = ax.get_legend_handles_labels()
    if ax_bg is not None:
        h2, l2 = ax_bg.get_legend_handles_labels()
        h1, l1 = h1 + h2, l1 + l2
    ax.legend(h1, l1, loc="upper right", fontsize=9, framealpha=0.92)

    total = int(agg["count"].sum())
    ax.set_title(
        f"{ds} — mean SOT performance vs GT bbox size "
        f"(bin = {bin_px} px, {agg['n_models']} models, {total:,} frames)",
        fontsize=11,
    )

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=180)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[save] {out_path.with_suffix('.png')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp-root", type=Path, default=EXP_ROOT)
    ap.add_argument("--out-dir", type=Path,
                    default=EXP_ROOT / "analysis" / "perf_vs_bbox_size")
    ap.add_argument("--bin-px", type=int, default=5)
    ap.add_argument("--min-count", type=int, default=30,
                    help="Drop bins with fewer frames from the axis trim")
    ap.add_argument("--x-max", type=float, default=None,
                    help="Optional upper x-axis cap (px)")
    ap.add_argument("--coverage", type=float, default=0.99,
                    help="Auto-trim x-axis to cover this fraction of all frames")
    ap.add_argument("--metrics", type=str, default="SR,PR,NPR",
                    help="Comma-separated subset of {SR,PR,NPR} to plot")
    ap.add_argument("--stack-by-class", action="store_true",
                    help="Add a bottom panel with frame counts stacked by GT class")
    args = ap.parse_args()

    metrics = tuple(m.strip().upper() for m in args.metrics.split(",") if m.strip())
    for m in metrics:
        if m not in ("SR", "PR", "NPR"):
            ap.error(f"unknown metric {m!r}; expected SR/PR/NPR")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading GT bboxes…")
    gt_by_ds = {ds: GT_LOADERS[ds](root) for ds, root in DATA_ROOTS.items()}
    for ds, gt in gt_by_ds.items():
        print(f"  {ds}: {len(gt):,} frame-level GTs")

    cls_by_ds: dict[str, dict[str, str]] = {}
    if args.stack_by_class:
        print("\nLoading class labels…")
        cls_by_ds = {ds: CLS_LOADERS[ds](root) for ds, root in DATA_ROOTS.items()}
        for ds, m in cls_by_ds.items():
            uniq = sorted(set(m.values()))
            print(f"  {ds}: {len(m)} seqs, classes = {uniq}")

    print("\nAggregating per-frame records…")
    agg_by_ds = aggregate(args.exp_root, gt_by_ds, args.bin_px,
                          cls_by_ds=cls_by_ds)

    print("\nPlotting…")
    for ds in ("OOTB", "SatSOT", "SV248S"):
        if ds not in agg_by_ds:
            print(f"[warn] no runs found for {ds}")
            continue
        out_path = args.out_dir / f"{ds.lower()}_perf_vs_bbox_size"
        plot_dataset(ds, agg_by_ds[ds], args.bin_px, out_path,
                     min_count=args.min_count, x_max=args.x_max,
                     coverage=args.coverage,
                     metrics=metrics,
                     stack_by_class=args.stack_by_class)

    # Also dump raw aggregated numbers for downstream use.
    summary = {
        ds: {
            "bin_edges_px": agg["edges"].tolist(),
            "count": agg["count"].tolist(),
            "SR":  agg["SR"].tolist(),
            "PR":  agg["PR"].tolist(),
            "NPR": agg["NPR"].tolist(),
            "SR_std":  agg["SR_std"].tolist(),
            "PR_std":  agg["PR_std"].tolist(),
            "NPR_std": agg["NPR_std"].tolist(),
            "models": agg["models"],
        }
        for ds, agg in agg_by_ds.items()
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[save] {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
