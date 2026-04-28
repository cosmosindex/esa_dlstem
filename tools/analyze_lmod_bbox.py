"""
Analyze LMOD bounding-box size statistics per category and per split.

LMOD is a multi-class moving-object detection dataset on Jilin-1 satellite
videos. Annotations are Pascal-VOC XML, axis-aligned `(xmin, ymin, xmax, ymax)`,
with 4 categories: car, plane, ship, train.

Splits are 80/10/10 of the *frames* of each sequence (temporal order),
matching the LMODDataset loader.

Stats reported:
  - per coarse category, all splits
  - per coarse category, per split
  - sequence / frame counts
  - dataset totals

Writes a markdown report to `docs/bbox_stats/bbox_stats_report_lmod.md`,
formatted to mirror `bbox_stats_report_satmtb.md`.

Usage:
    micromamba run -n esa_dlstem python tools/analyze_lmod_bbox.py \
        [--root /data/ESA_DLSTEM_2025/data/trafic/LMOD] \
        [--out docs/bbox_stats/bbox_stats_report_lmod.md]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.lmod import LMODDataset

SMALL_THRESH = 32 * 32  # area < 1024 px² → "small"


def collect(root: Path):
    """
    Iterate every (video, frame) of LMOD across all splits and collect
    (w, h) per bbox.

    Returns:
        per_cat:        {cat -> list[(w, h)]}                    # all splits
        per_split_cat:  {split -> {cat -> list[(w, h)]}}
        seq_frame_counts: {(split, cat) -> {"sequences": int, "frames": int}}
    """
    per_cat: dict[str, list[tuple[float, float]]] = defaultdict(list)
    per_split_cat: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    seq_frame_counts: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"sequences": 0, "frames": 0}
    )

    # Load each split separately so we can use the dataset's own splitting logic.
    # LMODDataset reverses class id mapping internally; we just need the raw
    # per-frame XML data, which is cached in `_gt_cache`. The labels in
    # `_load_annotations` are already mapped via class_map; we instead read
    # the raw "name" from the cached XML data.
    for split in ("train", "val", "test"):
        # class_map default is None inside LMODDataset; without one, _map_label
        # returns the raw string. We pass an empty class_map={} to avoid raising.
        ds = LMODDataset(root=root, split=split, mode="detection", class_map={})

        for video in ds.videos:
            seq_name = video.video_id.rsplit("_", 1)[0]  # "Seq1_train" → "Seq1"
            gt_data = ds._gt_cache[seq_name]

            # Track which sequences contain this category in this split
            cats_in_video: set[str] = set()
            n_frames_with_objs = 0

            for fid in video.frame_ids:
                objs = gt_data.get(fid, [])
                if objs:
                    n_frames_with_objs += 1
                for o in objs:
                    cat = o["name"]
                    w = float(o["xmax"] - o["xmin"])
                    h = float(o["ymax"] - o["ymin"])
                    if w <= 0 or h <= 0:
                        continue
                    per_cat[cat].append((w, h))
                    per_split_cat[split][cat].append((w, h))
                    cats_in_video.add(cat)

            for cat in cats_in_video:
                seq_frame_counts[(split, cat)]["sequences"] += 1
                # Frames *with at least one object of any class* — same convention
                # as the SAT-MTB report (per-category sequence counts share the
                # video's full frame count).
                seq_frame_counts[(split, cat)]["frames"] += len(video.frame_ids)

    return per_cat, per_split_cat, seq_frame_counts


def summarize(rows: list[tuple[float, float]]) -> dict:
    if not rows:
        return {
            "total": 0, "small": 0, "small_pct": 0.0,
            "large": 0, "large_pct": 0.0,
            "avg_w": 0.0, "avg_h": 0.0, "avg_area": 0.0,
            "min_area": 0.0, "max_area": 0.0,
        }
    arr = np.array(rows, dtype=np.float64)
    w, h = arr[:, 0], arr[:, 1]
    area = w * h
    n = len(arr)
    small = int((area < SMALL_THRESH).sum())
    large = n - small
    return {
        "total": n,
        "small": small,
        "small_pct": 100.0 * small / n,
        "large": large,
        "large_pct": 100.0 * large / n,
        "avg_w": float(w.mean()),
        "avg_h": float(h.mean()),
        "avg_area": float(area.mean()),
        "min_area": float(area.min()),
        "max_area": float(area.max()),
    }


def fmt_row(dataset, split, cat, s) -> str:
    return (
        f"| {dataset} | {split} | {cat} | {s['total']} | {s['small']} | "
        f"{s['small_pct']:.1f}% | {s['large']} | {s['large_pct']:.1f}% | "
        f"{s['avg_w']:.1f} | {s['avg_h']:.1f} | {s['avg_area']:.0f} | "
        f"{s['min_area']:.0f} | {s['max_area']:.0f} |"
    )


def build_report(per_cat, per_split_cat, seq_frame_counts) -> str:
    lines = []
    a = lines.append

    a("# LMOD — Bounding Box Size Statistics")
    a("")
    a("**Dataset:** LMOD — multi-class moving-object detection on Jilin-1 "
      "satellite videos.")
    a("**Class taxonomy:** 4 categories — `car`, `plane`, `ship`, `train`.")
    a("**Annotation format:** Pascal-VOC XML, axis-aligned `(xmin, ymin, xmax, ymax)`.")
    a("**Split:** 80 / 10 / 10 by frame within each sequence (temporal order); "
      "every sequence contributes to all three splits.")
    a("**Size threshold:** small = area < 32×32 = 1024 px².")
    a("")
    a("Analysis script: [`tools/analyze_lmod_bbox.py`](../../tools/analyze_lmod_bbox.py).")
    a("")

    # ---------- per-category overall ----------
    a("## Per coarse category (all splits)")
    a("")
    a("| dataset | split | category | total_boxes | small (<32²) | small_% | "
      "large (≥32²) | large_% | avg_w | avg_h | avg_area | min_area | max_area |")
    a("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for cat in sorted(per_cat):
        s = summarize(per_cat[cat])
        a(fmt_row("LMOD", "all", cat, s))
    a("")

    # ---------- per-split, per-category ----------
    a("## Per coarse category, per split")
    a("")
    a("| dataset | split | category | total_boxes | small (<32²) | small_% | "
      "large (≥32²) | large_% | avg_w | avg_h | avg_area | min_area | max_area |")
    a("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for split in ("train", "val", "test"):
        cats = per_split_cat.get(split, {})
        for cat in sorted(cats):
            s = summarize(cats[cat])
            a(fmt_row("LMOD", split, cat, s))
    a("")

    # ---------- sequence / frame counts ----------
    a("## Sequence / frame counts")
    a("")
    a("| split | category | sequences | frames |")
    a("| --- | --- | --- | --- |")
    grand_seq = 0
    grand_frames = 0
    for split in ("train", "val", "test"):
        sub_seq = 0
        sub_frames = 0
        cats_in_split = sorted({
            cat for (s, cat) in seq_frame_counts if s == split
        })
        for cat in cats_in_split:
            v = seq_frame_counts[(split, cat)]
            a(f"| {split} | {cat} | {v['sequences']} | {v['frames']} |")
            sub_seq += v["sequences"]
            sub_frames += v["frames"]
        a(f"| {split} | **subtotal** | **{sub_seq}** | **{sub_frames}** |")
        grand_seq += sub_seq
        grand_frames += sub_frames
    a(f"| **all** | **total** | **{grand_seq}** | **{grand_frames}** |")
    a("")
    a("> Sequence / frame counts are aggregated across categories: a sequence "
      "containing both `car` and `plane` is counted under each category (so the "
      "subtotal can exceed the actual number of distinct sequences). Frame "
      "counts are the full split-frame count of each sequence.")
    a("")

    # ---------- dataset total ----------
    a("## Dataset total")
    a("")
    a("| dataset | total_boxes | small (<32²) | small_% | large (≥32²) | "
      "large_% | avg_area |")
    a("| --- | --- | --- | --- | --- | --- | --- |")
    all_rows = [xy for rows in per_cat.values() for xy in rows]
    s = summarize(all_rows)
    a(f"| LMOD | {s['total']} | {s['small']} | {s['small_pct']:.1f}% | "
      f"{s['large']} | {s['large_pct']:.1f}% | {s['avg_area']:.0f} |")
    a("")

    # ---------- notes ----------
    a("## Notes")
    a("")
    a("- LMOD is **multi-object** per frame: `total_boxes` = sum over all "
      "annotated instances in all frames, not number of frames.")
    a("- Annotations are HBB only; no OBB.")
    a("- Splits are *intra-sequence temporal* (first 80 % train, next 10 % val, "
      "last 10 % test) — every sequence appears in all three splits.")
    a("- Categories are taken verbatim from the XML `<name>` field; no remapping.")

    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/ESA_DLSTEM_2025/data/trafic/LMOD")
    ap.add_argument("--out", default="docs/bbox_stats/bbox_stats_report_lmod.md")
    args = ap.parse_args()

    root = Path(args.root)
    per_cat, per_split_cat, seq_frame_counts = collect(root)
    report = build_report(per_cat, per_split_cat, seq_frame_counts)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
