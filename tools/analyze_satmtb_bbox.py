"""
Analyze SAT-MTB bounding-box size statistics per category and per split.

SAT-MTB is multi-task and dual-granularity:
  - 4 coarse classes: airplane, car, ship, train
  - 14 fine classes (defined): WA/NA/RA/FA/CA (airplane),
                                SB/YH/CS/FH/NV/OS (ship),
                                LC/SC (car), TN (train)

Annotation availability by task:
  - det_hbb / det_obb:  airplane, ship, train  (XML <name> is COARSE-only)
  - seg:                 airplane, ship, train  (JSON name = fine, e.g.
                          `wide_bodied_aircraft`; supercategory = coarse)
  - mot:                 airplane, car, ship, train  (CSV class id = coarse)

Practical takeaway: fine-grained labels only exist in `seg/*.json`. HBB and
OBB annotations carry only coarse names in this distribution of the dataset,
so per-fine-class tables are reported for `seg` only.

Splits: official train / test from `data_split.xlsx`; val carved from 30 %
of test (seed=42, stratified by category) — same scheme used elsewhere in
the project.

Stats reported per task:
  - per coarse category, all splits
  - per coarse category, per split
  - per fine category, all splits      (det_hbb / det_obb / seg only)
  - per fine category, per split       (det_hbb / det_obb / seg only)
  - sequence / frame counts
  - dataset totals

Two box shapes coexist:
  - det_hbb / mot / seg: stored as axis-aligned (xmin, ymin, xmax, ymax).
  - det_obb: SAT-MTB OBB XML stores 4 corners; the loader reduces them to the
    enclosing AABB (`max-min` over the 4 xs / ys), so values reported here are
    AABB-of-OBB (what an HBB detector sees on OBB data), not OBB short/long
    sides.

Writes a markdown report to `docs/bbox_stats/bbox_stats_report_satmtb.md`.

Usage:
    micromamba run -n esa_dlstem python tools/analyze_satmtb_bbox.py \
        [--root /data/ESA_DLSTEM_2025/data/trafic/SAT-MTB] \
        [--out docs/bbox_stats/bbox_stats_report_satmtb.md] \
        [--tasks det_hbb det_obb mot seg]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.satmtb import (
    COARSE_CATEGORIES,
    FINE_CATEGORIES,
    FINE_TO_COARSE,
    SATMTBDataset,
)

SMALL_THRESH = 32 * 32  # area < 1024 px² → "small"

# Tasks that carry fine-grained labels (XML <name> / COCO `name`).
_FINE_TASKS = {"det_hbb", "det_obb", "seg"}


def _wh(box: np.ndarray) -> tuple[float, float]:
    """`(xmin, ymin, xmax, ymax)` → `(w, h)`."""
    return float(box[2] - box[0]), float(box[3] - box[1])


def collect_for_task(root: Path, task: str) -> dict:
    """
    Iterate the dataset once for the given task and collect (w, h) lists.

    Reads `class_fine` directly from the dataset's annotation cache so the
    fine-class names appear with whatever string the source files use
    (e.g. `wide_bodied_aircraft` from seg JSONs), without relying on
    `class_map_fine`'s short-code remap.

    Returns:
        {
          "per_cat":            {cat: [(w, h), ...]},
          "per_split_cat":      {split: {cat: [(w, h), ...]}},
          "per_fine":           {fine_name: [(w, h), ...]},
          "per_split_fine":     {split: {fine_name: [(w, h), ...]}},
          "seq_counts":         {cat: {"sequences": n, "frames": n}},
          "split_seq_counts":   {split: {cat: {"sequences": n, "frames": n}}},
          "fine_to_coarse":     {fine_name: coarse_name}  (observed only),
        }
    """
    class_map_coarse = {n: i for i, n in enumerate(COARSE_CATEGORIES)}

    ds = SATMTBDataset(
        root=root,
        split="no_split",
        task=task,
        class_map=class_map_coarse,
        mode="detection",
    )

    per_cat: dict[str, list[tuple[float, float]]] = defaultdict(list)
    per_split_cat: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    per_fine: dict[str, list[tuple[float, float]]] = defaultdict(list)
    per_split_fine: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    seq_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"sequences": 0, "frames": 0}
    )
    split_seq_counts: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"sequences": 0, "frames": 0})
    )
    fine_to_coarse: dict[str, str] = {}

    for v in ds.videos:
        cat = v.category
        split = v.split
        seq_counts[cat]["sequences"] += 1
        seq_counts[cat]["frames"] += v.num_frames
        split_seq_counts[split][cat]["sequences"] += 1
        split_seq_counts[split][cat]["frames"] += v.num_frames

        objs_by_frame = ds._ann_cache.get(v.video_id, {})
        for fid in v.frame_ids:
            for obj in objs_by_frame.get(fid, []):
                coarse_name = obj.get("class")
                if coarse_name not in COARSE_CATEGORIES:
                    continue
                box = obj["box"]
                w = float(box[2] - box[0])
                h = float(box[3] - box[1])
                if w <= 0 or h <= 0:
                    continue
                per_cat[coarse_name].append((w, h))
                per_split_cat[split][coarse_name].append((w, h))

                # In this distribution, only seg JSONs carry true fine labels.
                # HBB/OBB parsers copy the coarse XML <name> into class_fine,
                # so we'd see fake "fine" rows like 'airplane' / 'ship' if we
                # accepted class_fine for those tasks. Restrict to seg.
                if task == "seg":
                    fine_name = obj.get("class_fine")
                    if fine_name:
                        per_fine[fine_name].append((w, h))
                        per_split_fine[split][fine_name].append((w, h))
                        fine_to_coarse.setdefault(fine_name, coarse_name)

    return {
        "per_cat": per_cat,
        "per_split_cat": per_split_cat,
        "per_fine": per_fine,
        "per_split_fine": per_split_fine,
        "seq_counts": seq_counts,
        "split_seq_counts": split_seq_counts,
        "fine_to_coarse": fine_to_coarse,
    }


def summarize(rows: list[tuple[float, float]]) -> dict:
    if not rows:
        return {
            "total": 0, "small": 0, "small_pct": 0.0,
            "large": 0, "large_pct": 0.0,
            "avg_w": 0.0, "avg_h": 0.0,
            "avg_area": 0.0, "min_area": 0.0, "max_area": 0.0,
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


def fmt_row(dataset, split, key, s) -> str:
    return (
        f"| {dataset} | {split} | {key} | {s['total']} | {s['small']} | "
        f"{s['small_pct']:.1f}% | {s['large']} | {s['large_pct']:.1f}% | "
        f"{s['avg_w']:.1f} | {s['avg_h']:.1f} | {s['avg_area']:.0f} | "
        f"{s['min_area']:.0f} | {s['max_area']:.0f} |"
    )


_HEADER = (
    "| dataset | split | category | total_boxes | small (<32²) | small_% | "
    "large (≥32²) | large_% | avg_w | avg_h | avg_area | min_area | max_area |"
)
_SEP = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"


def section_per_cat_overall(task: str, data: dict, lines: list[str]) -> None:
    a = lines.append
    a(f"### {task} — per coarse category (all splits)")
    a("")
    a(_HEADER)
    a(_SEP)
    for cat in sorted(data["per_cat"]):
        s = summarize(data["per_cat"][cat])
        a(fmt_row("SAT-MTB", "all", cat, s))
    a("")


def section_per_cat_per_split(task: str, data: dict, lines: list[str]) -> None:
    a = lines.append
    a(f"### {task} — per coarse category, per split")
    a("")
    a(_HEADER)
    a(_SEP)
    for split in ("train", "val", "test"):
        cats = data["per_split_cat"].get(split, {})
        for cat in sorted(cats):
            s = summarize(cats[cat])
            a(fmt_row("SAT-MTB", split, cat, s))
    a("")


def _fine_sort_key(fname: str, coarse: str) -> tuple[int, str]:
    return (COARSE_CATEGORIES.index(coarse) if coarse in COARSE_CATEGORIES else 999,
            fname)


def section_per_fine_overall(task: str, data: dict, lines: list[str]) -> None:
    if not data["per_fine"]:
        return
    a = lines.append
    a(f"### {task} — per fine class (all splits)")
    a("")
    a(
        "| dataset | split | coarse | fine | total_boxes | small (<32²) | "
        "small_% | large (≥32²) | large_% | avg_w | avg_h | avg_area | "
        "min_area | max_area |"
    )
    a(
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
        "--- | --- | --- | --- |"
    )
    f2c = data["fine_to_coarse"]
    ordered = sorted(
        data["per_fine"].keys(),
        key=lambda fn: _fine_sort_key(fn, f2c.get(fn, "?")),
    )
    for fname in ordered:
        s = summarize(data["per_fine"][fname])
        coarse = f2c.get(fname, "?")
        a(
            f"| SAT-MTB | all | {coarse} | {fname} | {s['total']} | "
            f"{s['small']} | {s['small_pct']:.1f}% | {s['large']} | "
            f"{s['large_pct']:.1f}% | {s['avg_w']:.1f} | {s['avg_h']:.1f} | "
            f"{s['avg_area']:.0f} | {s['min_area']:.0f} | {s['max_area']:.0f} |"
        )
    a("")


def section_per_fine_per_split(task: str, data: dict, lines: list[str]) -> None:
    if not data["per_fine"]:
        return
    a = lines.append
    a(f"### {task} — per fine class, per split")
    a("")
    a(
        "| dataset | split | coarse | fine | total_boxes | small (<32²) | "
        "small_% | large (≥32²) | large_% | avg_w | avg_h | avg_area | "
        "min_area | max_area |"
    )
    a(
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
        "--- | --- | --- | --- |"
    )
    f2c = data["fine_to_coarse"]
    for split in ("train", "val", "test"):
        bucket = data["per_split_fine"].get(split, {})
        ordered = sorted(
            bucket.keys(),
            key=lambda fn: _fine_sort_key(fn, f2c.get(fn, "?")),
        )
        for fname in ordered:
            s = summarize(bucket[fname])
            coarse = f2c.get(fname, "?")
            a(
                f"| SAT-MTB | {split} | {coarse} | {fname} | {s['total']} | "
                f"{s['small']} | {s['small_pct']:.1f}% | {s['large']} | "
                f"{s['large_pct']:.1f}% | {s['avg_w']:.1f} | {s['avg_h']:.1f} | "
                f"{s['avg_area']:.0f} | {s['min_area']:.0f} | {s['max_area']:.0f} |"
            )
    a("")


def section_seq_counts(task: str, data: dict, lines: list[str]) -> None:
    a = lines.append
    a(f"### {task} — sequence / frame counts")
    a("")
    a("| split | category | sequences | frames |")
    a("| --- | --- | --- | --- |")
    grand_seq = grand_frm = 0
    for split in ("train", "val", "test"):
        bucket = data["split_seq_counts"].get(split, {})
        sub_seq = sub_frm = 0
        for cat in sorted(bucket):
            sc = bucket[cat]
            sub_seq += sc["sequences"]
            sub_frm += sc["frames"]
            a(f"| {split} | {cat} | {sc['sequences']} | {sc['frames']} |")
        a(f"| {split} | **subtotal** | **{sub_seq}** | **{sub_frm}** |")
        grand_seq += sub_seq
        grand_frm += sub_frm
    a(f"| **all** | **total** | **{grand_seq}** | **{grand_frm}** |")
    a("")


def section_dataset_total(task: str, data: dict, lines: list[str]) -> None:
    a = lines.append
    all_rows = [xy for rows in data["per_cat"].values() for xy in rows]
    s = summarize(all_rows)
    a(f"### {task} — dataset total")
    a("")
    a(
        "| dataset | total_boxes | small (<32²) | small_% | large (≥32²) | "
        "large_% | avg_area |"
    )
    a("| --- | --- | --- | --- | --- | --- | --- |")
    a(
        f"| SAT-MTB | {s['total']} | {s['small']} | {s['small_pct']:.1f}% | "
        f"{s['large']} | {s['large_pct']:.1f}% | {s['avg_area']:.0f} |"
    )
    a("")


def build_report(by_task: dict[str, dict]) -> str:
    lines: list[str] = []
    a = lines.append

    a("# SAT-MTB — Bounding Box Size Statistics")
    a("")
    a(
        "**Dataset:** SAT-MTB (Multi-Task Benchmark for satellite videos) — "
        "supports detection (HBB / OBB), MOT, and instance segmentation."
    )
    a(
        "**Class taxonomy:** 4 coarse classes (`airplane`, `car`, `ship`, "
        "`train`) and 14 fine classes "
        "(WA/NA/RA/FA/CA, SB/YH/CS/FH/NV/OS, LC/SC, TN)."
    )
    a(
        "**Split:** official `train` / `test` from `data_split.xlsx`; `val` "
        "carved from 30 % of `test` (seed=42, stratified by category)."
    )
    a("**Size threshold:** small = area < 32×32 = 1024 px².")
    a("")
    a("**Annotation availability by task (this distribution):**")
    a("")
    a("| task | airplane | car | ship | train | granularity in source files |")
    a("| --- | :---: | :---: | :---: | :---: | --- |")
    a("| det_hbb | ✓ | ✗ | ✓ | ✓ | coarse only (XML `<name>` ∈ {airplane, ship, train}) |")
    a("| det_obb | ✓ | ✗ | ✓ | ✓ | coarse only (XML `<name>` ∈ {airplane, ship, train}) |")
    a("| mot     | ✓ | ✓ | ✓ | ✓ | coarse only (CSV class id) |")
    a("| seg     | ✓ | ✗ | ✓ | ✓ | **fine** (JSON `name`) + coarse (`supercategory`) |")
    a("")
    a(
        "**Fine-grained labels:** the seg JSONs use names like "
        "`wide_bodied_aircraft`, `speed_boat`, etc. (with underscores). "
        "Per-fine-class tables below appear under `seg` only."
    )
    a("")
    a(
        "**Box shapes:** `det_hbb`, `mot`, `seg` use axis-aligned "
        "`(xmin, ymin, xmax, ymax)`. `det_obb` reduces the 4 OBB corners to "
        "their enclosing AABB (this is what the dataset loader returns), so "
        "its `(w, h)` here is **AABB-of-OBB**, not OBB short/long sides."
    )
    a("")
    a("Analysis script: [`tools/analyze_satmtb_bbox.py`](../../tools/analyze_satmtb_bbox.py).")
    a("")

    for task in ("det_hbb", "det_obb", "mot", "seg"):
        if task not in by_task:
            continue
        data = by_task[task]
        a(f"## Task: `{task}`")
        a("")
        section_per_cat_overall(task, data, lines)
        section_per_cat_per_split(task, data, lines)
        section_per_fine_overall(task, data, lines)
        section_per_fine_per_split(task, data, lines)
        section_seq_counts(task, data, lines)
        section_dataset_total(task, data, lines)

    a("## Notes")
    a("")
    a(
        "- SAT-MTB is **multi-object**: `total_boxes` = sum over all instances "
        "in all frames, not number of frames."
    )
    a(
        "- For `det_obb`, the loader (`SATMTBDataset._parse_det_obb`) projects "
        "the 4 OBB corners to an axis-aligned bbox; that is what trains/evals "
        "an HBB detector. If you need true OBB short/long sides, parse "
        "`<robndbox>` in `det/OBB/<frame>.xml` directly."
    )
    a(
        "- `car` sequences only ship MOT-format annotations, so `car` rows "
        "appear only under `mot`."
    )
    a(
        "- Per-fine-class breakdown only appears under `seg`. HBB/OBB XMLs in "
        "this distribution carry only the coarse name; MOT CSVs carry only "
        "the coarse class id."
    )
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/ESA_DLSTEM_2025/data/trafic/SAT-MTB")
    ap.add_argument(
        "--out", default="docs/bbox_stats/bbox_stats_report_satmtb.md"
    )
    ap.add_argument(
        "--tasks", nargs="+",
        default=["det_hbb", "det_obb", "mot", "seg"],
        choices=["det_hbb", "det_obb", "mot", "seg"],
    )
    args = ap.parse_args()

    root = Path(args.root)
    by_task: dict[str, dict] = {}
    for task in args.tasks:
        print(f"[satmtb-bbox] collecting task={task} ...", flush=True)
        by_task[task] = collect_for_task(root, task)
        n = sum(len(v) for v in by_task[task]["per_cat"].values())
        print(f"  → {n} boxes across {sum(s['sequences'] for s in by_task[task]['seq_counts'].values())} sequences")

    report = build_report(by_task)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
