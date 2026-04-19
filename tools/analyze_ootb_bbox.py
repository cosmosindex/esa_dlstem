"""
Analyze OOTB bounding-box size statistics per category.

OOTB stores per-frame **oriented bounding boxes** (8 floats = 4 corners) in
`<seq>/groundtruth.txt`. We report two flavours of (w, h):

  * **OBB sides** — `w = min(side1, side2)`, `h = max(side1, side2)`,
    measured along the OBB's own axes. This is what previous reports used
    (`docs/bbox_stats_report_trafic.md`).
  * **AABB** — the axis-aligned bounding box of the OBB corners
    (`(min_x, min_y, max_x, max_y)`), i.e. what an HBB detector actually sees.

Uses OOTBDataset(split="no_split") so the same hybrid 80/10/10 class⊕attr
split used elsewhere in the project is reflected in the per-split tables.

Writes a markdown report to `docs/bbox_stats_report_ootb.md`.

Usage:
    micromamba run -n esa_dlstem python tools/analyze_ootb_bbox.py \
        [--root /data/ESA_DLSTEM_2025/data/trafic/OOTB] \
        [--out docs/bbox_stats_report_ootb.md]
"""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.ootb import OOTBDataset

SMALL_THRESH = 32 * 32  # area < 1024 px² → "small"


def obb_short_long(coords) -> tuple[float, float]:
    """Return (short_side, long_side) of an OBB given as 8 corner floats."""
    x1, y1, x2, y2, x3, y3, x4, y4 = coords
    s1 = math.hypot(x2 - x1, y2 - y1)
    s2 = math.hypot(x3 - x2, y3 - y2)
    return (s1, s2) if s1 <= s2 else (s2, s1)


def obb_to_aabb_wh(coords) -> tuple[float, float]:
    """AABB width/height = (max_x - min_x, max_y - min_y) of the 4 corners."""
    xs = coords[0::2]
    ys = coords[1::2]
    return (max(xs) - min(xs), max(ys) - min(ys))


def collect(root: Path) -> tuple[dict, dict, dict, dict, dict]:
    """
    Returns:
        per_cat[cat] -> list[(w_obb, h_obb)]
        per_split_cat[split][cat] -> list[(w_obb, h_obb)]
        per_cat_aabb[cat] -> list[(w_aabb, h_aabb)]
        extreme_breakdown[cat] -> Counter((w_int, h_int)) for max(w_obb, h_obb) <= 2
        seq_counts[cat] -> {"sequences": n, "frames": n}
    """
    ds = OOTBDataset(root=root, split="no_split", mode="detection")

    per_cat: dict[str, list[tuple[float, float]]] = defaultdict(list)
    per_split_cat: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    per_cat_aabb: dict[str, list[tuple[float, float]]] = defaultdict(list)
    extreme_breakdown: dict[str, Counter] = defaultdict(Counter)
    seq_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"sequences": 0, "frames": 0}
    )

    for v in ds.videos:
        cat = v.category
        split = v.split
        gt = ds._gt_cache[v.video_id]   # (N, 8) float32 — OBB corners
        seq_counts[cat]["sequences"] += 1
        seq_counts[cat]["frames"] += len(gt)
        for i in range(len(gt)):
            coords = gt[i]
            w_obb, h_obb = obb_short_long(coords)
            per_cat[cat].append((w_obb, h_obb))
            per_split_cat[split][cat].append((w_obb, h_obb))
            per_cat_aabb[cat].append(obb_to_aabb_wh(coords))
            if max(w_obb, h_obb) <= 2.0:
                extreme_breakdown[cat][(int(round(w_obb)), int(round(h_obb)))] += 1

    return per_cat, per_split_cat, per_cat_aabb, extreme_breakdown, seq_counts


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


def build_report(per_cat, per_split_cat, per_cat_aabb, extreme, seq_counts) -> str:
    lines = []
    a = lines.append

    a("# OOTB — Bounding Box Size Statistics")
    a("")
    a("**Dataset:** OOTB (Object Tracking Benchmark on satellite video, ISPRS 2024) — "
      "single-object tracking with **oriented bounding boxes** (OBB).")
    a("**Annotation format:** per-frame `[x1, y1, x2, y2, x3, y3, x4, y4]` "
      "(4 OBB corners) in `<seq>/groundtruth.txt`. Categories: `car`, `plane`, `ship`, `train`.")
    a("**Counting rule:** every frame has a valid OBB (no `none` / NaN markers); all rows are counted.")
    a("**Size threshold:** small = area < 32×32 = 1024 px².")
    a("**Split:** hybrid 80/10/10 (seed = 42), iterative-stratification on class ⊕ 12 attribute flags.")
    a("")
    a("Two flavours of `(w, h)` are reported:")
    a("- **OBB sides** (default, sections 1–3): `w = shorter side`, `h = longer side` of the OBB.")
    a("- **AABB** (section 4): width/height of the axis-aligned bounding box that "
      "encloses the OBB — this is what an HBB detector actually sees.")
    a("")
    a("Analysis script: [`tools/analyze_ootb_bbox.py`](../tools/analyze_ootb_bbox.py).")
    a("")

    # ---------- Section 1: per-category overall (OBB sides) ----------
    a("## 1. Per-Category (overall, all splits, OBB sides)")
    a("")
    a("| dataset | split | category | total_boxes | small (<32²) | small_% | "
      "large (≥32²) | large_% | avg_w | avg_h | avg_area | min_area | max_area |")
    a("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for cat in sorted(per_cat):
        s = summarize(per_cat[cat])
        a(fmt_row("OOTB", "all", cat, s))
    a("")

    # ---------- Section 2: per-split, per-category (OBB sides) ----------
    a("## 2. Per-Split, Per-Category (OBB sides)")
    a("")
    a("| dataset | split | category | total_boxes | small (<32²) | small_% | "
      "large (≥32²) | large_% | avg_w | avg_h | avg_area | min_area | max_area |")
    a("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for split in ("train", "val", "test"):
        cats = per_split_cat.get(split, {})
        for cat in sorted(cats):
            s = summarize(cats[cat])
            a(fmt_row("OOTB", split, cat, s))
    a("")

    # ---------- Section 3: dataset total ----------
    a("## 3. Dataset Total (OBB sides)")
    a("")
    a("| dataset | total_boxes | small (<32²) | small_% | large (≥32²) | large_% | avg_area |")
    a("| --- | --- | --- | --- | --- | --- | --- |")
    all_rows = [xy for rows in per_cat.values() for xy in rows]
    s = summarize(all_rows)
    a(f"| OOTB | {s['total']} | {s['small']} | {s['small_pct']:.1f}% | "
      f"{s['large']} | {s['large_pct']:.1f}% | {s['avg_area']:.0f} |")
    a("")

    # ---------- Section 4: AABB stats (per-category) ----------
    a("## 4. Per-Category — AABB (axis-aligned bbox of the OBB)")
    a("")
    a("AABB area is always ≥ OBB area; the gap grows with rotation and aspect ratio "
      "(most pronounced for `train`, which is long & often diagonal).")
    a("")
    a("| dataset | split | category | total_boxes | small (<32²) | small_% | "
      "large (≥32²) | large_% | avg_w | avg_h | avg_area | min_area | max_area |")
    a("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for cat in sorted(per_cat_aabb):
        s = summarize(per_cat_aabb[cat])
        a(fmt_row("OOTB", "all", cat, s))
    a("")

    # ---------- Section 5: sequences & frames per category ----------
    a("## 5. Sequence / Frame Counts per Category")
    a("")
    a("| category | sequences | frames |")
    a("| --- | --- | --- |")
    total_seq = 0
    total_frm = 0
    for cat in sorted(seq_counts):
        sc = seq_counts[cat]
        total_seq += sc["sequences"]
        total_frm += sc["frames"]
        a(f"| {cat} | {sc['sequences']} | {sc['frames']} |")
    a(f"| **total** | **{total_seq}** | **{total_frm}** |")
    a("")

    # ---------- Section 6: extremely small objects ----------
    a("## 6. Extremely Small Objects (max(w, h) ≤ 2 px, OBB sides)")
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
    a("- OOTB is a **single-object tracking** dataset: each sequence has exactly "
      "one target, so `total_boxes = #frames`.")
    a("- Sequence categories are inferred from the directory-name prefix "
      "(`car_3` → `car`, `plane_07` → `plane`, …).")
    a("- OBB-side stats match the OOTB row in `docs/bbox_stats_report_trafic.md`; "
      "AABB stats (section 4) are larger because the rotation-bounding box absorbs "
      "the diagonal.")

    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/ESA_DLSTEM_2025/data/trafic/OOTB")
    ap.add_argument("--out", default="docs/bbox_stats_report_ootb.md")
    args = ap.parse_args()

    root = Path(args.root)
    per_cat, per_split_cat, per_cat_aabb, extreme, seq_counts = collect(root)
    report = build_report(per_cat, per_split_cat, per_cat_aabb, extreme, seq_counts)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
