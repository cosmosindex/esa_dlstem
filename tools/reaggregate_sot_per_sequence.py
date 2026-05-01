"""
Re-aggregate SOT results with per-sequence averaging + new PR range + tiny subset.

Re-computes the headline SOT metrics from existing
``per_image_metrics.json`` files written by the SOT eval pipeline, applying
three changes vs the original ``test_metrics.json``:

  1. **Per-sequence aggregation.** Original numbers were per-frame pooled
     (every frame contributes equally → long sequences dominate). We now
     compute each metric per sequence and then arithmetic-mean across
     sequences (matches OTB / LaSOT / GOT-10k convention).

  2. **PR AUC over CLE ∈ [0, 50] px** (was [0, 30]). Aligns with the OTB
     precision-plot range; see ``docs/sot_benchmark_tables.md``.

  3. **Tiny-object subset.** A sequence is *tiny* if its median GT
     ``sqrt(area)`` across annotated frames is < 8 px. We re-aggregate the
     same metrics restricted to tiny sequences and additionally surface
     ``P@5`` (CLE < 5 px) as the primary precision scalar on the subset.

Inputs:  ``<root>/<tracker>/<run>/per_image_metrics.json``.
Outputs: ``<root>/<tracker>/<run>/test_metrics_per_seq.json`` (per run) and
``<root>/analysis/per_seq/{overall,tiny,tiny_subsets}.csv|json`` (top-level).

Usage::

    micromamba run -n esa_dlstem python tools/reaggregate_sot_per_sequence.py \
        --root /work/ziwen/experiments/NeurIPS/SOT_whole_dataset_04_22
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np


# Curve thresholds — see module docstring.
SUCCESS_THRESHOLDS = np.linspace(0.0, 1.0, 21)              # IoU ∈ [0, 1]
PRECISION_THRESHOLDS = np.arange(0, 51, dtype=float)         # CLE ∈ [0, 50] px
NORM_PRECISION_THRESHOLDS = np.linspace(0.0, 0.5, 21)        # nCLE ∈ [0, 0.5]

TINY_SQRT_AREA_THRESH = 8.0   # sequence-level cutoff: median sqrt(area) < 8 px

DATASETS = ("ootb", "satsot", "sv248s")

DATASET_ROOTS = {
    "ootb":   Path("/data/ESA_DLSTEM_2025/data/trafic/OOTB"),
    "satsot": Path("/data/ESA_DLSTEM_2025/data/trafic/SatSOT"),
    "sv248s": Path("/data/ESA_DLSTEM_2025/data/trafic/SV248S"),
}


# ============================================================
# Per-sequence GT median sqrt(area) — defines the tiny subset
# ============================================================

def _polygon_area(coords: np.ndarray) -> float:
    """Shoelace area for a (8,) polygon = 4 corners (x1,y1, …, x4,y4)."""
    xs = coords[0::2]
    ys = coords[1::2]
    return 0.5 * abs(
        xs[0] * (ys[1] - ys[3])
        + xs[1] * (ys[2] - ys[0])
        + xs[2] * (ys[3] - ys[1])
        + xs[3] * (ys[0] - ys[2])
    )


def _ootb_seq_areas(seq_dir: Path) -> list[float]:
    gt_path = seq_dir / "groundtruth.txt"
    areas: list[float] = []
    with open(gt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = re.split(r"[,\t ]+", line)
            try:
                coords = np.array([float(v) for v in vals[:8]], dtype=np.float64)
            except ValueError:
                continue
            if coords.size < 8 or np.isnan(coords).any():
                continue
            a = _polygon_area(coords)
            if a > 0:
                areas.append(a)
    return areas


def _satsot_seq_areas(seq_dir: Path) -> list[float]:
    gt_path = seq_dir / "groundtruth.txt"
    areas: list[float] = []
    with open(gt_path) as f:
        for line in f:
            line = line.strip()
            if not line or "none" in line.lower():
                continue
            vals = re.split(r"[,\t ]+", line)
            try:
                xywh = [float(v) for v in vals[:4]]
            except ValueError:
                continue
            if len(xywh) < 4:
                continue
            w, h = xywh[2], xywh[3]
            if w > 0 and h > 0:
                areas.append(w * h)
    return areas


def _sv248s_seq_areas(rect_path: Path, state_path: Path) -> list[float]:
    rects: list[tuple[float, float]] = []
    with open(rect_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = re.split(r"[,\t ]+", line)
            try:
                w, h = float(vals[2]), float(vals[3])
            except (IndexError, ValueError):
                rects.append((0.0, 0.0))
                continue
            rects.append((w, h))
    states: list[int] = []
    if state_path.exists():
        with open(state_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    states.append(int(line))
                except ValueError:
                    states.append(0)
    else:
        states = [0] * len(rects)
    n = min(len(rects), len(states))
    areas: list[float] = []
    for i in range(n):
        if states[i] == 1:                  # invisible — skip
            continue
        w, h = rects[i]
        if w > 0 and h > 0:
            areas.append(w * h)
    return areas


def compute_tiny_subset(dataset: str) -> tuple[set[str], dict[str, float]]:
    """Return (tiny_video_ids, median_sqrt_area_per_video) for dataset."""
    root = DATASET_ROOTS[dataset]
    if not root.exists():
        raise FileNotFoundError(f"Dataset root missing: {root}")

    medians: dict[str, float] = {}

    if dataset == "ootb":
        for seq_dir in sorted(root.iterdir()):
            if not seq_dir.is_dir() or seq_dir.name == "anno":
                continue
            if not (seq_dir / "groundtruth.txt").exists():
                continue
            areas = _ootb_seq_areas(seq_dir)
            if areas:
                medians[seq_dir.name] = float(np.sqrt(np.median(areas)))

    elif dataset == "satsot":
        for seq_dir in sorted(root.iterdir()):
            if not seq_dir.is_dir():
                continue
            if not (seq_dir / "groundtruth.txt").exists():
                continue
            areas = _satsot_seq_areas(seq_dir)
            if areas:
                medians[seq_dir.name] = float(np.sqrt(np.median(areas)))

    elif dataset == "sv248s":
        for video_dir in sorted(root.iterdir()):
            if not video_dir.is_dir():
                continue
            ann_dir = video_dir / "annotations"
            seq_dir_root = video_dir / "sequences"
            if not ann_dir.is_dir() or not seq_dir_root.is_dir():
                continue
            for seq in sorted(seq_dir_root.iterdir()):
                if not seq.is_dir():
                    continue
                seq_id = seq.name
                rect_path = ann_dir / f"{seq_id}.rect"
                state_path = ann_dir / f"{seq_id}.state"
                if not rect_path.exists():
                    continue
                areas = _sv248s_seq_areas(rect_path, state_path)
                if not areas:
                    continue
                vid_id = f"{video_dir.name}/{seq_id}"
                medians[vid_id] = float(np.sqrt(np.median(areas)))

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    tiny = {vid for vid, m in medians.items() if m < TINY_SQRT_AREA_THRESH}
    return tiny, medians


# ============================================================
# Per-sequence metrics
# ============================================================

def _seq_summary(seq_records: list[dict]) -> dict:
    """Summary metrics for ONE sequence (curves + scalars)."""
    ious = np.array([r["best_iou"] for r in seq_records], dtype=np.float64)
    cd   = np.array([r["center_dist"] for r in seq_records], dtype=np.float64)
    nc   = np.array([r["norm_center_dist"] for r in seq_records], dtype=np.float64)

    success_curve = np.array([float(np.mean(ious >= t)) for t in SUCCESS_THRESHOLDS])
    prec_curve    = np.array([float(np.mean(cd <= t))   for t in PRECISION_THRESHOLDS])
    nprec_curve   = np.array([float(np.mean(nc <= t))   for t in NORM_PRECISION_THRESHOLDS])

    return {
        "n_frames": len(seq_records),
        "success_auc":        float(success_curve.mean()),
        "precision_auc":      float(prec_curve.mean()),
        "norm_precision_auc": float(nprec_curve.mean()),
        "precision_5":        float(np.mean(cd <= 5.0)),
        "precision_20":       float(np.mean(cd <= 20.0)),
        "norm_precision_05":  float(np.mean(nc <= 0.5)),
        "mean_iou":           float(np.mean(ious)),
        "success_curve":      success_curve,
        "precision_curve":    prec_curve,
        "norm_precision_curve": nprec_curve,
    }


def _avg_across_sequences(seq_summaries: list[dict]) -> dict:
    """Equal-weight arithmetic mean across sequences."""
    if not seq_summaries:
        return {}
    n_seq = len(seq_summaries)
    n_frames = sum(s["n_frames"] for s in seq_summaries)

    def _mean(key: str) -> float:
        return float(np.mean([s[key] for s in seq_summaries]))

    success_curve = np.mean([s["success_curve"] for s in seq_summaries], axis=0)
    prec_curve    = np.mean([s["precision_curve"] for s in seq_summaries], axis=0)
    nprec_curve   = np.mean([s["norm_precision_curve"] for s in seq_summaries], axis=0)

    return {
        "n_sequences": n_seq,
        "n_frames":    n_frames,
        "success_auc":        round(_mean("success_auc"), 4),
        "precision_auc":      round(_mean("precision_auc"), 4),
        "norm_precision_auc": round(_mean("norm_precision_auc"), 4),
        "precision_5":        round(_mean("precision_5"), 4),
        "precision_20":       round(_mean("precision_20"), 4),
        "norm_precision_05":  round(_mean("norm_precision_05"), 4),
        "mean_iou":           round(_mean("mean_iou"), 4),
        "success_plot":       {f"{t:.2f}": round(float(v), 4)
                                for t, v in zip(SUCCESS_THRESHOLDS, success_curve)},
        "precision_plot":     {str(int(t)): round(float(v), 4)
                                for t, v in zip(PRECISION_THRESHOLDS, prec_curve)},
        "norm_precision_plot": {f"{t:.3f}": round(float(v), 4)
                                 for t, v in zip(NORM_PRECISION_THRESHOLDS, nprec_curve)},
    }


def aggregate_run(
    per_image_path: Path,
    tiny_video_ids: set[str],
) -> dict:
    """Re-aggregate one run's per_image_metrics.json into per-sequence metrics."""
    with open(per_image_path) as f:
        frames = json.load(f)

    # Group by sequence: video_id → list of records (one record per GT object
    # per frame; SOT is single-target so usually one per frame).
    by_seq: dict[str, list[dict]] = defaultdict(list)
    by_seq_class: dict[str, set[str]] = defaultdict(set)
    by_seq_size: dict[str, str] = {}

    for fr in frames:
        vid = str(fr["video_id"])
        for r in fr.get("sot_records", []):
            by_seq[vid].append(r)
            by_seq_class[vid].add(r.get("gt_class", "unknown"))
            # Use mode/first-seen size key as the sequence's nominal size bucket.
            by_seq_size.setdefault(vid, r.get("gt_size", "unknown"))

    if not by_seq:
        return {}

    # Per-sequence summaries
    seq_summary: dict[str, dict] = {
        vid: _seq_summary(recs) for vid, recs in by_seq.items() if recs
    }

    overall = _avg_across_sequences(list(seq_summary.values()))

    # Tiny subset
    tiny_summaries = [seq_summary[v] for v in seq_summary if v in tiny_video_ids]
    tiny = _avg_across_sequences(tiny_summaries) if tiny_summaries else None

    # Per-category — aggregate over sequences whose dominant class is `name`.
    # A sequence is assigned to a single class (the most-frequent gt_class
    # across its records); ties broken by alphabetical order.
    cat_to_seqs: dict[str, list[dict]] = defaultdict(list)
    for vid, summary in seq_summary.items():
        # Pick dominant class: vote across that sequence's frames.
        votes: dict[str, int] = defaultdict(int)
        for r in by_seq[vid]:
            votes[r.get("gt_class", "unknown")] += 1
        if not votes:
            continue
        dominant = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        cat_to_seqs[dominant].append(summary)
    per_category = {
        name: _avg_across_sequences(seqs) for name, seqs in sorted(cat_to_seqs.items())
    }

    # Per-size (legacy small/large from per-frame gt_size; sequence inherits
    # the first-frame label, mostly stable since GT size doesn't drift much
    # in a single sequence).
    size_to_seqs: dict[str, list[dict]] = defaultdict(list)
    for vid, summary in seq_summary.items():
        size_to_seqs[by_seq_size.get(vid, "unknown")].append(summary)
    per_size = {
        name: _avg_across_sequences(seqs) for name, seqs in sorted(size_to_seqs.items())
    }

    return {
        "aggregation":        "per_sequence",
        "pr_threshold_max":   int(PRECISION_THRESHOLDS[-1]),
        "tiny_sqrt_area_thresh": TINY_SQRT_AREA_THRESH,
        "n_sequences":        overall["n_sequences"],
        "n_frames":           overall["n_frames"],
        "overall":            overall,
        "tiny":               tiny,
        "per_category":       per_category,
        "per_size":           per_size,
    }


# ============================================================
# Run discovery
# ============================================================

DATASET_TAGS = {
    "ootb":   re.compile(r"_ootb_"),
    "satsot": re.compile(r"_satsot_"),
    "sv248s": re.compile(r"_sv248s_"),
}


def detect_dataset(run_name: str) -> str | None:
    for ds, pat in DATASET_TAGS.items():
        if pat.search(run_name):
            return ds
    return None


def discover_runs(root: Path) -> list[tuple[str, str, Path]]:
    """Return list of (tracker, dataset, run_dir) tuples."""
    runs = []
    for tracker_dir in sorted(root.iterdir()):
        if not tracker_dir.is_dir() or tracker_dir.name == "analysis":
            continue
        for run_dir in sorted(tracker_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            if not (run_dir / "per_image_metrics.json").exists():
                continue
            ds = detect_dataset(run_dir.name)
            if ds is None:
                continue
            runs.append((tracker_dir.name, ds, run_dir))
    return runs


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="/work/ziwen/experiments/NeurIPS/SOT_whole_dataset_04_22",
        help="Experiment root containing <tracker>/<run>/ subdirs.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Override output dir (default: <root>/analysis/per_seq).",
    )
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out) if args.out else (root / "analysis" / "per_seq")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Compute tiny subsets per dataset (load GT once per dataset).
    tiny_subsets: dict[str, set[str]] = {}
    medians_per_dataset: dict[str, dict[str, float]] = {}
    print("[step 1/3] computing per-sequence median sqrt(area) for tiny subsets")
    for ds in DATASETS:
        tiny, medians = compute_tiny_subset(ds)
        tiny_subsets[ds] = tiny
        medians_per_dataset[ds] = medians
        print(f"  {ds:8s}: total_seqs={len(medians):4d}  tiny={len(tiny):4d}"
              f"  ({100.0 * len(tiny) / max(len(medians), 1):.1f}%)")

    # Save tiny subset definition for transparency
    with open(out_dir / "tiny_subsets.json", "w") as f:
        json.dump({
            "threshold_sqrt_area_px": TINY_SQRT_AREA_THRESH,
            "criterion": "sequence-level: median sqrt(GT area) < 8 px",
            "datasets": {
                ds: {
                    "total_sequences": len(medians_per_dataset[ds]),
                    "tiny_sequences":  sorted(tiny_subsets[ds]),
                    "tiny_count":      len(tiny_subsets[ds]),
                    "median_sqrt_area_per_sequence": {
                        v: round(m, 3) for v, m in sorted(medians_per_dataset[ds].items())
                    },
                }
                for ds in DATASETS
            },
        }, f, indent=2)

    # 2. Re-aggregate every run.
    print("[step 2/3] re-aggregating runs")
    runs = discover_runs(root)
    overall_rows = []
    tiny_rows = []
    for tracker, ds, run_dir in runs:
        try:
            result = aggregate_run(
                run_dir / "per_image_metrics.json",
                tiny_video_ids=tiny_subsets[ds],
            )
        except Exception as e:
            print(f"  [skip] {tracker}/{run_dir.name}: {e}")
            continue
        if not result:
            continue

        # Save per-run JSON
        with open(run_dir / "test_metrics_per_seq.json", "w") as f:
            json.dump(result, f, indent=2)

        ov = result["overall"]
        overall_rows.append({
            "tracker":     tracker,
            "dataset":     ds,
            "run":         run_dir.name,
            "n_sequences": ov["n_sequences"],
            "n_frames":    ov["n_frames"],
            "SR":          ov["success_auc"],
            "NPR":         ov["norm_precision_auc"],
            "PR":          ov["precision_auc"],
            "P@5":         ov["precision_5"],
            "mean_iou":    ov["mean_iou"],
        })

        if result.get("tiny"):
            tn = result["tiny"]
            tiny_rows.append({
                "tracker":      tracker,
                "dataset":      ds,
                "run":          run_dir.name,
                "n_sequences":  tn["n_sequences"],
                "n_frames":     tn["n_frames"],
                "SR":           tn["success_auc"],
                "NPR":          tn["norm_precision_auc"],
                "PR":           tn["precision_auc"],
                "P@5":          tn["precision_5"],
                "mean_iou":     tn["mean_iou"],
            })

        print(f"  {tracker:10s} {ds:7s}  SR={ov['success_auc']:.3f}"
              f" NPR={ov['norm_precision_auc']:.3f}"
              f" PR={ov['precision_auc']:.3f}"
              f" P@5={ov['precision_5']:.3f}"
              f"  ({ov['n_sequences']} seqs, {ov['n_frames']} frames)")

    # 3. Write top-level CSVs.
    print("[step 3/3] writing CSVs")
    overall_rows.sort(key=lambda r: (r["tracker"], r["dataset"]))
    tiny_rows.sort(key=lambda r: (r["tracker"], r["dataset"]))

    if overall_rows:
        with open(out_dir / "overall.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(overall_rows[0].keys()))
            w.writeheader()
            w.writerows(overall_rows)

    if tiny_rows:
        with open(out_dir / "tiny.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(tiny_rows[0].keys()))
            w.writeheader()
            w.writerows(tiny_rows)

    print(f"\nDone. Outputs under: {out_dir}")
    print(f"  overall.csv      ({len(overall_rows)} rows)")
    print(f"  tiny.csv         ({len(tiny_rows)} rows)")
    print(f"  tiny_subsets.json")
    print(f"  per-run JSONs at <tracker>/<run>/test_metrics_per_seq.json")


if __name__ == "__main__":
    main()
