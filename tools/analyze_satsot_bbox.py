"""
Analyze SatSOT bounding-box size statistics per category.

Uses SatSOTDataset(split="no_split") to iterate every (video, frame) and
collects (w, h) per bbox. SatSOT marks frames where the target is absent /
fully occluded with the literal string "none" in `groundtruth.txt`; the
dataset class converts those rows to NaN. We skip those frames.

Writes a markdown report to `docs/bbox_stats_report_satsot.md` following the
same format as `docs/bbox_stats_report_sv248s.md`.

Usage:
    micromamba run -n esa_dlstem python tools/analyze_satsot_bbox.py \
        [--root /data/ESA_DLSTEM_2025/data/trafic/SatSOT] \
        [--out docs/bbox_stats_report_satsot.md]
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.satsot import SatSOTDataset

SMALL_THRESH = 32 * 32  # area < 1024 px² → "small"


def collect(root: Path) -> tuple[dict, dict, dict, dict]:
    """
    Returns:
        per_cat[cat] -> list[(w, h)]
        per_split_cat[split][cat] -> list[(w, h)]
        extreme_breakdown[cat] -> Counter((w_int, h_int))  for max(w,h) <= 2
        absent_counts[cat] -> {"present": n, "absent": n}
    """
    ds = SatSOTDataset(root=root, split="no_split", mode="detection")

    per_cat: dict[str, list[tuple[float, float]]] = defaultdict(list)
    per_split_cat: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    extreme_breakdown: dict[str, Counter] = defaultdict(Counter)
    absent_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"present": 0, "absent": 0}
    )

    for v in ds.videos:
        cat = v.category
        split = v.split
        gt = ds._gt_cache[v.video_id]    # (N, 4) float32 with NaN for absent
        for i in range(len(gt)):
            row = gt[i]
            if np.any(np.isnan(row)):
                absent_counts[cat]["absent"] += 1
                continue
            absent_counts[cat]["present"] += 1
            w, h = float(row[2]), float(row[3])
            per_cat[cat].append((w, h))
            per_split_cat[split][cat].append((w, h))
            if max(w, h) <= 2.0:
                extreme_breakdown[cat][(int(round(w)), int(round(h)))] += 1

    return per_cat, per_split_cat, extreme_breakdown, absent_counts


def summarize(rows: list[tuple[float, float]]) -> dict:
    arr = np.array(rows, dtype=np.float64)
    w, h = arr[:, 0], arr[:, 1]
    area = w * h
    n = len(arr)
    small = int((area < SMALL_THRESH).sum())
    large = n - small
    return {
        "total": n,
        "small": small,
        "small_pct": 100.0 * small / n if n else 0.0,
        "large": large,
        "large_pct": 100.0 * large / n if n else 0.0,
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


def build_report(per_cat, per_split_cat, extreme, absent_counts) -> str:
    lines = []
    a = lines.append

    a("# SatSOT — Bounding Box Size Statistics")
    a("")
    a("**Dataset:** SatSOT — single-object tracking dataset from Jilin-1 satellite video.")
    a("**Annotation format:** per-frame axis-aligned `[x, y, w, h]` in "
      "`<seq>/groundtruth.txt`. Frames where the target is absent or fully "
      "occluded are written as the literal `none` and stored as NaN by the "
      "dataset class.")
    a("**Counting rule:** absent-frame rows (`none` / NaN) are excluded; all "
      "other rows are counted.")
    a("**Size threshold:** small = area < 32×32 = 1024 px².")
    a("**Categories:** `car`, `plane`, `ship`, `train` (inferred from sequence-name prefix).")
    a("**Split:** class-stratified 80/10/10 (seed = 42), with at least 1 sequence "
      "per split per category.")
    a("")
    a("Analysis script: [`tools/analyze_satsot_bbox.py`](../tools/analyze_satsot_bbox.py).")
    a("")

    # ---------- Section 1: per-category overall ----------
    a("## 1. Per-Category (overall, all splits)")
    a("")
    a("| dataset | split | category | total_boxes | small (<32²) | small_% | "
      "large (≥32²) | large_% | avg_w | avg_h | avg_area | min_area | max_area |")
    a("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for cat in sorted(per_cat):
        s = summarize(per_cat[cat])
        a(fmt_row("SatSOT", "all", cat, s))
    a("")

    # ---------- Section 2: per-split, per-category ----------
    a("## 2. Per-Split, Per-Category")
    a("")
    a("| dataset | split | category | total_boxes | small (<32²) | small_% | "
      "large (≥32²) | large_% | avg_w | avg_h | avg_area | min_area | max_area |")
    a("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for split in ("train", "val", "test"):
        cats = per_split_cat.get(split, {})
        for cat in sorted(cats):
            s = summarize(cats[cat])
            a(fmt_row("SatSOT", split, cat, s))
    a("")

    # ---------- Section 3: dataset total ----------
    a("## 3. Dataset Total")
    a("")
    a("| dataset | total_boxes | small (<32²) | small_% | large (≥32²) | large_% | avg_area |")
    a("| --- | --- | --- | --- | --- | --- | --- |")
    all_rows = [xy for rows in per_cat.values() for xy in rows]
    s = summarize(all_rows)
    a(f"| SatSOT | {s['total']} | {s['small']} | {s['small_pct']:.1f}% | "
      f"{s['large']} | {s['large_pct']:.1f}% | {s['avg_area']:.0f} |")
    a("")

    # ---------- Section 4: present / absent frames ----------
    a("## 4. Frame Presence Breakdown")
    a("")
    a("Boxes counted in the tables above come from frames where the target is "
      "present. Absent rows (`none` in `groundtruth.txt`) are excluded.")
    a("")
    a("| category | present (counted) | absent (excluded) | total_frames |")
    a("| --- | --- | --- | --- |")
    for cat in sorted(absent_counts):
        ac = absent_counts[cat]
        a(f"| {cat} | {ac['present']} | {ac['absent']} | {ac['present'] + ac['absent']} |")
    a("")

    # ---------- Section 5: extremely small objects ----------
    a("## 5. Extremely Small Objects (max(w, h) ≤ 2 px)")
    a("")
    a("| category | total | extreme (≤2px) | extreme_% | (w × h) breakdown |")
    a("| --- | --- | --- | --- | --- |")
    for cat in sorted(per_cat):
        total = len(per_cat[cat])
        c = extreme.get(cat, Counter())
        ext = sum(c.values())
        if ext == 0:
            breakdown = "—"
        else:
            breakdown = ", ".join(
                f"({w}×{h}): {n}" for (w, h), n in sorted(c.items())
            )
        pct = 100.0 * ext / total if total else 0.0
        a(f"| {cat} | {total} | {ext} | {pct:.1f}% | {breakdown} |")
    a("")

    # ---------- Notes ----------
    a("## Notes")
    a("")
    a("- SatSOT is a **single-object tracking** dataset: each sequence has exactly "
      "one target, so `total_boxes ≈ #frames` (minus absent frames).")
    a("- Sequence categories are inferred from the directory-name prefix "
      "(`car_03` → `car`, `plane_07` → `plane`, …).")
    a("- Widths/heights are reported as-is from `groundtruth.txt` (integer or "
      "sub-pixel depending on sequence); area is rounded to integers for display.")

    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/ESA_DLSTEM_2025/data/trafic/SatSOT")
    ap.add_argument("--out", default="docs/bbox_stats_report_satsot.md")
    args = ap.parse_args()

    root = Path(args.root)
    per_cat, per_split_cat, extreme, absent_counts = collect(root)
    report = build_report(per_cat, per_split_cat, extreme, absent_counts)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
