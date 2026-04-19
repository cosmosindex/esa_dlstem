"""
Analyze SV248S bounding-box size statistics per category.

Uses SV248SDataset(split="no_split") to iterate every (video, frame) and
collects (w, h) per bbox. SV248S provides per-frame state flags:
    0 = NOR (visible)        → counted
    1 = INV (invisible)      → skipped (target has disappeared, rect is stale)
    2 = OCC (occluded)       → counted (still has a valid rect)

Writes a markdown report to `docs/bbox_stats_report_sv248s.md` following the
same format as `docs/bbox_stats_report_trafic.md`.

Usage:
    micromamba run -n esa_dlstem python tools/analyze_sv248s_bbox.py \
        [--root /data/ESA_DLSTEM_2025/data/trafic/SV248S] \
        [--out docs/bbox_stats_report_sv248s.md]
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.sv248s import SV248SDataset

SMALL_THRESH = 32 * 32  # area < 1024 px² → "small"


def collect(root: Path) -> tuple[dict, dict, dict, dict]:
    """
    Returns:
        per_cat[cat] -> list[(w, h)]
        per_split_cat[split][cat] -> list[(w, h)]
        extreme_breakdown[cat] -> Counter((w_int, h_int))  for boxes with max(w,h) <= 2
        state_counts[cat] -> {"visible": n, "occluded": n, "invisible": n}
    """
    ds = SV248SDataset(root=root, split="no_split", mode="detection")

    per_cat: dict[str, list[tuple[float, float]]] = defaultdict(list)
    per_split_cat: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    extreme_breakdown: dict[str, Counter] = defaultdict(Counter)
    state_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"visible": 0, "occluded": 0, "invisible": 0}
    )

    for v in ds.videos:
        cat = v.category
        split = v.split
        rects = ds._rect_cache[v.video_id]          # (N, 4) float x,y,w,h
        states = ds._state_cache[v.video_id]        # (N,) int
        for i in range(len(rects)):
            st = int(states[i])
            if st == 1:
                state_counts[cat]["invisible"] += 1
                continue
            if st == 2:
                state_counts[cat]["occluded"] += 1
            else:
                state_counts[cat]["visible"] += 1
            w, h = float(rects[i, 2]), float(rects[i, 3])
            per_cat[cat].append((w, h))
            per_split_cat[split][cat].append((w, h))
            if max(w, h) <= 2.0:
                extreme_breakdown[cat][(int(round(w)), int(round(h)))] += 1

    return per_cat, per_split_cat, extreme_breakdown, state_counts


def summarize(rows: list[tuple[float, float]]) -> dict:
    arr = np.array(rows, dtype=np.float64)  # (N, 2)
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


def build_report(per_cat, per_split_cat, extreme, state_counts) -> str:
    lines = []
    a = lines.append

    a("# SV248S — Bounding Box Size Statistics")
    a("")
    a("**Dataset:** SV248S SOT — 248 single-target sequences across 6 parent videos.")
    a("**Annotation format:** per-frame axis-aligned `[x, y, w, h]` in `.rect`, "
      "with `.state` flag per frame (0 = visible, 1 = invisible, 2 = occluded).")
    a("**Counting rule:** bboxes on frames flagged `invisible` (state = 1) are "
      "skipped — the target has disappeared and the rect is stale. Visible (state = 0) "
      "and occluded (state = 2) frames are counted.")
    a("**Size threshold:** small = area < 32×32 = 1024 px².")
    a("**Split:** class-stratified 80/10/10 (seed = 42), with tiny classes "
      "(plane, ship) pre-assigned round-robin so every split covers every category.")
    a("")
    a("Analysis script: [`tools/analyze_sv248s_bbox.py`](../tools/analyze_sv248s_bbox.py).")
    a("")

    # ---------- Section 1: per-category overall ----------
    a("## 1. Per-Category (overall, all splits)")
    a("")
    a("| dataset | split | category | total_boxes | small (<32²) | small_% | "
      "large (≥32²) | large_% | avg_w | avg_h | avg_area | min_area | max_area |")
    a("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for cat in sorted(per_cat):
        s = summarize(per_cat[cat])
        a(fmt_row("SV248S", "all", cat, s))
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
            a(fmt_row("SV248S", split, cat, s))
    a("")

    # ---------- Section 3: dataset total ----------
    a("## 3. Dataset Total")
    a("")
    a("| dataset | total_boxes | small (<32²) | small_% | large (≥32²) | large_% | avg_area |")
    a("| --- | --- | --- | --- | --- | --- | --- |")
    all_rows = [xy for rows in per_cat.values() for xy in rows]
    s = summarize(all_rows)
    a(f"| SV248S | {s['total']} | {s['small']} | {s['small_pct']:.1f}% | "
      f"{s['large']} | {s['large_pct']:.1f}% | {s['avg_area']:.0f} |")
    a("")

    # ---------- Section 4: frame state counts ----------
    a("## 4. Frame State Breakdown")
    a("")
    a("Boxes counted in the tables above come from frames with `state ∈ {0, 2}`. "
      "Invisible frames (state = 1) are excluded.")
    a("")
    a("| category | visible (state=0) | occluded (state=2) | invisible (state=1, excluded) | counted_total |")
    a("| --- | --- | --- | --- | --- |")
    for cat in sorted(state_counts):
        sc = state_counts[cat]
        counted = sc["visible"] + sc["occluded"]
        a(f"| {cat} | {sc['visible']} | {sc['occluded']} | {sc['invisible']} | {counted} |")
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
    a("- SV248S is a **single-object tracking** dataset: each sequence has exactly "
      "one target, so `total_boxes ≈ #frames` (minus invisible frames).")
    a("- `car-large` is the SV248S-specific class for buses/trucks and is separate from `car`.")
    a("- Widths and heights are reported **as-is** from `.rect` (which carries sub-pixel "
      "precision), but averages/min/max of `area` are rounded to integers for display.")

    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/ESA_DLSTEM_2025/data/trafic/SV248S")
    ap.add_argument("--out", default="docs/bbox_stats_report_sv248s.md")
    args = ap.parse_args()

    root = Path(args.root)
    per_cat, per_split_cat, extreme, state_counts = collect(root)
    report = build_report(per_cat, per_split_cat, extreme, state_counts)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
