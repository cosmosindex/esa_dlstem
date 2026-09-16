#!/usr/bin/env python
"""Bring SV248S's tight polygons into the release beside its rectangles.

SV248S annotates each target with a tight contour --- **not** a four-point
oriented box, but a variable-length polygon of 4 to 35 vertices --- and ships it
in a ``.poly`` file next to the ``.rect`` file the release was built from.  Its
own paper calls that contour the reason the dataset can represent objects with
complex outlines, and explains that it deliberately stores the contour rather
than an oriented box so that either can be derived from it.  The first build took only the rectangle, which left the
oriented geometry of 246 of the 395 SOT sequences on the floor and made
in-plane rotation impossible to define anywhere except OOTB.

The two files are row-aligned with the released frames --- one line per frame,
identical count --- so the import is an index join.  It is nonetheless verified
rather than assumed: the axis-aligned box of every imported polygon is compared
against the rectangle already in the release, and a mismatch is reported instead
of being written.

Writes into both views of the annotation, at the fidelity each can hold: the
full contour goes into ``segmentation`` on the COCO annotation, which has no
length constraint, while columns 8..15 (``px1..py4``) of ``groundtruth.txt``
take the minimum-area rotated rectangle of that contour, since those four
columns are fixed-width.  The text file therefore loses contour detail that the
JSON keeps; that is a property of the OTB-style row format, not of the import.

Usage
-----
    python tools/import_sv248s_polygons.py [--dry-run]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

import cv2
import numpy as np

import os as _bootstrap_os, sys as _bootstrap_sys
_bootstrap_sys.path.insert(0, _bootstrap_os.path.dirname(
    _bootstrap_os.path.dirname(_bootstrap_os.path.abspath(__file__))))
from project_paths import DATA_ROOT

#: The released package. ``SPACE_TRACKER_ROOT`` is the name the README and the
#: loading code use; ``SPACE_TRACKER_RELEASE`` is accepted as the older spelling.
RELEASE = Path(os.environ.get("SPACE_TRACKER_ROOT")
               or os.environ.get("SPACE_TRACKER_RELEASE")
               or f"{DATA_ROOT}/release/space_tracker")
TRAFIC = Path(os.environ.get("SPACE_TRACKER_DATA_ROOT", "/data/anon/trafic"))

#: The polygon's own bounding box should reproduce the released rectangle. The
#: sources round to three decimals, so allow a hair more than that.
AABB_TOL_PX = 0.02


def read_poly(path: Path) -> list[list[float] | None]:
    """One contour per frame. Vertex count varies between 4 and 35."""
    out: list[list[float] | None] = []
    for line in path.read_text().strip().split("\n"):
        parts = [p for p in line.replace(",", " ").split() if p]
        ok = len(parts) >= 6 and len(parts) % 2 == 0
        out.append([float(p) for p in parts] if ok else None)
    return out


def min_area_obb(poly: list[float]) -> list[float]:
    """The four corners of the tightest rotated rectangle around a contour."""
    pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
    box = cv2.boxPoints(cv2.minAreaRect(pts))
    return [round(float(c), 3) for c in box.reshape(-1)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    coco_path = RELEASE / "sot" / "annotations" / "space_tracker_sot.json"
    data = json.loads(coco_path.read_text())
    videos = {v["id"]: v for v in data["videos"]}
    images = {i["id"]: i for i in data["images"]}

    ann_by_vf: dict[tuple[int, int], dict] = {}
    for a in data["annotations"]:
        im = images[a["image_id"]]
        ann_by_vf[(im["video_id"], im.get("frame_id", 0))] = a

    stats = collections.Counter()
    mismatches: list[str] = []
    per_seq: dict[str, dict[int, list[float]]] = {}

    for v in data["videos"]:
        if v.get("source_dataset") != "sv248s":
            continue
        parts = (v.get("source_sequence_id") or "").split("/")
        if len(parts) != 3:
            stats["seq_unresolved"] += 1
            continue
        src = TRAFIC / "SV248S" / parts[1] / "annotations" / f"{parts[2]}.poly"
        if not src.exists():
            stats["seq_missing_poly"] += 1
            continue
        polys = read_poly(src)
        if len(polys) != v["n_frames"]:
            stats["seq_length_mismatch"] += 1
            mismatches.append(f"{v['name']}: {len(polys)} poly rows vs "
                              f"{v['n_frames']} released frames")
            continue

        keep: dict[int, list[float]] = {}
        for idx, poly in enumerate(polys):
            frame = idx + 1
            ann = ann_by_vf.get((v["id"], frame))
            if ann is None or poly is None:
                continue          # invisible frame, or a row without geometry
            xs, ys = poly[0::2], poly[1::2]
            x, y = min(xs), min(ys)
            w, h = max(xs) - x, max(ys) - y
            bx, by, bw, bh = ann["bbox"]
            if (abs(x - bx) > AABB_TOL_PX or abs(y - by) > AABB_TOL_PX
                    or abs(w - bw) > AABB_TOL_PX or abs(h - bh) > AABB_TOL_PX):
                stats["box_mismatch"] += 1
                if len(mismatches) < 20:
                    mismatches.append(
                        f"{v['name']} f{frame}: poly aabb "
                        f"({x:.3f},{y:.3f},{w:.3f},{h:.3f}) vs released "
                        f"({bx:.3f},{by:.3f},{bw:.3f},{bh:.3f})")
                continue
            keep[frame] = [round(c, 3) for c in poly]
            stats["boxes_matched"] += 1
        per_seq[v["name"]] = keep
        stats["sequences"] += 1

    print(f"sequences with polygons: {stats['sequences']}")
    print(f"  boxes matched and importable: {stats['boxes_matched']}")
    for k in ("seq_unresolved", "seq_missing_poly", "seq_length_mismatch",
              "box_mismatch"):
        if stats[k]:
            print(f"  {k}: {stats[k]}")
    for m in mismatches[:20]:
        print(f"    {m}")
    if args.dry_run:
        print("\ndry run, nothing written")
        return

    name_of = {v["id"]: v["name"] for v in data["videos"]}
    written = 0
    for (vid, frame), ann in ann_by_vf.items():
        poly = per_seq.get(name_of[vid], {}).get(frame)
        if poly is None:
            continue
        ann["segmentation"] = [poly]
        written += 1
    coco_path.write_text(json.dumps(data, indent=2) + "\n")

    # groundtruth.txt: replace the -1 placeholders in columns 8..15.
    touched = 0
    for v in data["videos"]:
        keep = per_seq.get(v["name"])
        if not keep:
            continue
        gt = RELEASE / "sot" / v["category"] / v["name"] / "groundtruth.txt"
        if not gt.is_file():
            continue
        rows = []
        for line in gt.read_text().strip().split("\n"):
            col = line.split(",")
            poly = keep.get(int(col[0]))
            if poly is not None and len(col) >= 15:
                col[7:15] = [f"{c:.2f}" for c in min_area_obb(poly)]
            rows.append(",".join(col))
        gt.write_text("\n".join(rows) + "\n")
        touched += 1

    print(f"\nwritten: {written} polygons into {coco_path.name}")
    print(f"         {touched} groundtruth.txt files updated")


if __name__ == "__main__":
    main()
