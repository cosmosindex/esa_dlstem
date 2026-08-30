"""Fold SAT-MTB's detection annotations into its MOT ground truth, unattended.

SAT-MTB's MOT CSV keeps only *moving* objects: 83.5% of airplane tracks, 32.0%
of ship and 16.7% of train tracks that the dataset itself annotates in
``det/HBB`` never reach the tracking ground truth (see
``docs/static_annotation/README.md``). Those missing objects are already
annotated per frame, with an ``objectID`` that makes them tracks — recovering
them is a format conversion, not an annotation job, so no human reviews it here.
The interactive tool's job is to *check* the result, one video at a time.

What this does, per SAT-MTB sequence that has ``det/HBB``:

1. Match every detection-XML track against the MOT ground truth, per category,
   at IoU >= 0.5. A det track covered on at least half its frames is already in
   MOT and is dropped as a duplicate.
2. Append every unmatched det track as a new MOT track, with ids above every
   existing one so nothing downstream can confuse recovered with original.
3. Rewrite the geometry of *all* non-car boxes — recovered and original alike —
   from the SAM 3 batch pass (``tools/refine_satmtb_sam3.py``), so one
   annotation standard covers the whole file. Aircraft boxes are 0.618x the
   original area; mixing the two standards would make an IoU metric unable to
   tell "annotated differently" from "detector was wrong".

``car`` is never touched: det XML does not label it, and a parked car at a 4.9
px median is not separable from road texture in a single frame — which is why
this line of work is posed as *moving* object detection at all. Sequences that
mix car with plane/ship keep car movers-only and gain the static plane/ship;
the two protocols are reported per category, never as one number.

Raw dataset files are never written. Output is a parallel tree of corrected MOT
CSVs that a manifest points at through ``gt_path_override``, plus a provenance
JSON per sequence recording which new track id came from which det track.

Usage::

    python tools/merge_det_to_mot.py --out /work/<user>/space_tracker_mot_merged \
        --refined-dir /work/<user>/experiments/satmtb_sam3_refine
    python tools/merge_det_to_mot.py --out ... --dry-run          # report only
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interactive_review.core.batch_refined import load_refined
from interactive_review.core.paths import manifest, satmtb_det_dir
from interactive_review.core.tracks import TrackKey, load_det_tracks, load_mot_frames

#: Experiment/scratch root. Real paths are machine-specific, so they are
#: never written into the repository — set ``WORK_ROOT`` to point at yours.
WORK = Path(os.environ.get("WORK_ROOT", "/work/anon"))

CLS_ID = {"car": 0, "airplane": 1, "ship": 2, "train": 3}
#: Categories a det track may be merged under. car is excluded by policy, not
#: by accident: det XML never labels it, so an unmatched "car" would be a
#: parsing bug rather than a recovered object.
MERGEABLE = ("airplane", "ship", "train")

IOU_THRESH = 0.5
#: Fraction of a det track's frames that must be covered by MOT for it to count
#: as already present. Half, not all: MOT tracks routinely start late or end
#: early on the same object, and requiring full coverage would re-add it.
FRAC_THRESH = 0.5


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.prod(np.clip(a[:, 2:] - a[:, :2], 0, None), axis=1)
    area_b = np.prod(np.clip(b[:, 2:] - b[:, :2], 0, None), axis=1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def coverage_by_mot(det_tracks: dict, mot_frames: dict) -> dict[tuple[str, str], float]:
    """Fraction of each det track's frames that the MOT ground truth also has.

    Matching runs on *raw* geometry from both sides. Comparing a SAM 3-refined
    det box against a raw MOT box would depress IoU by the very 0.618x area
    ratio this merge exists to remove, and silently re-add objects that are
    already annotated.
    """
    per_frame: dict[int, list[tuple[str, str, np.ndarray]]] = defaultdict(list)
    for (category, oid), track in det_tracks.items():
        for fid, box in zip(track.frame_ids, track.boxes):
            per_frame[fid].append((category, oid, box))

    hits: Counter = Counter()
    total: Counter = Counter()
    for fid, items in per_frame.items():
        mot = mot_frames.get(fid, [])
        for category in {c for c, _, _ in items}:
            idx = [i for i, (c, _, _) in enumerate(items) if c == category]
            d = np.stack([items[i][2] for i in idx])
            m = [o.bbox_xyxy.astype(np.float64) for o in mot if o.category == category]
            best = (_iou_matrix(d, np.stack(m)).max(axis=1) if m
                    else np.zeros(len(idx)))
            for j, i in enumerate(idx):
                key = (items[i][0], items[i][1])
                total[key] += 1
                if best[j] >= IOU_THRESH:
                    hits[key] += 1
    return {k: hits[k] / total[k] for k in total}


def merge_sequence(seq, refined_dir: Path | None,
                   fill_dir: Path | None = None) -> tuple[list[str], dict]:
    """Corrected CSV lines for one sequence, plus a provenance/summary record."""
    mot = load_mot_frames(seq.id)
    det = load_det_tracks(seq.id, seq.video_id) if satmtb_det_dir(seq.video_id).is_dir() else {}
    refined = load_refined(str(refined_dir) if refined_dir else None, seq.id)
    # Frames tools/fill_holey_tracks.py added to a recovered track. Kept a layer
    # apart from ``refined`` because these are frames the source annotation never
    # had, not a restatement of frames it did.
    filled = load_refined(str(fill_dir) if fill_dir else None, seq.id)

    rows: list[tuple[int, int, float, float, float, float, int]] = []
    n_regeometried = 0
    for fid, objs in mot.items():
        for o in objs:
            key_id = TrackKey(seq.id, "mot_gt", o.category, str(o.track_id)).id
            box = refined.get(key_id, {}).get(fid)
            if box is None:
                x1, y1, x2, y2 = (float(v) for v in o.bbox_xyxy)
            else:
                n_regeometried += 1
                x1, y1, x2, y2 = box
            rows.append((fid, o.track_id, x1, y1, x2 - x1, y2 - y1, CLS_ID[o.category]))

    coverage = coverage_by_mot(det, mot) if det else {}
    # Sorted so a re-run assigns the same new id to the same det track: the
    # provenance file and any review decision keyed on it must stay valid.
    def _sort_key(k: tuple[str, str]) -> tuple[str, int, str]:
        return (k[0], int(k[1]) if k[1].isdigit() else 1 << 30, k[1])

    next_id = max((r[1] for r in rows), default=0) + 1
    added: list[dict] = []
    skipped: list[dict] = []
    n_added_boxes = 0
    n_added_refined = 0
    n_filled = 0
    for (category, oid) in sorted(coverage, key=_sort_key):
        track = det[(category, oid)]
        record = {
            "det_key": track.key.id, "category": category, "det_object_id": oid,
            "n_boxes": len(track), "frac_frames_in_mot": round(coverage[(category, oid)], 3),
            "subname": track.subname,
            "median_sqrt_area": round(track.median_size, 2),
        }
        if category not in MERGEABLE:
            record["reason"] = f"category {category!r} is out of scope"
            skipped.append(record)
            continue
        if coverage[(category, oid)] >= FRAC_THRESH:
            record["reason"] = "already in MOT"
            skipped.append(record)
            continue

        geometry = {int(f): [float(v) for v in b]
                    for f, b in zip(track.frame_ids, track.boxes)}
        by_sam3 = refined.get(track.key.id, {})
        n_added_refined += sum(1 for f in by_sam3 if f in geometry)
        geometry.update(by_sam3)
        hole_fill = filled.get(track.key.id, {})
        n_filled += len(hole_fill)
        geometry.update(hole_fill)
        for fid in sorted(geometry):
            x1, y1, x2, y2 = geometry[fid]
            rows.append((fid, next_id, x1, y1, x2 - x1, y2 - y1, CLS_ID[category]))
            n_added_boxes += 1
        record["new_track_id"] = next_id
        record["frames"] = [min(geometry), max(geometry)]
        # A static object is present in every frame it spans, so a recovered
        # track with holes is under-annotated rather than intermittent. Flagged
        # here so the video review can fill it instead of having to find it.
        span = max(geometry) - min(geometry) + 1
        record["span"] = span
        record["coverage_in_span"] = round(len(geometry) / span, 3)
        added.append(record)
        next_id += 1

    rows.sort(key=lambda r: (r[0], r[1]))
    lines = [f"{fid},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,{cls},-1,-1,-1"
             for fid, tid, x, y, w, h, cls in rows]

    per_category = Counter(r["category"] for r in added)
    summary = {
        "seq_id": seq.id, "category": seq.category,
        "has_det_xml": bool(det),
        "det_tracks": len(coverage),
        "tracks_added": len(added),
        "tracks_added_airplane": per_category.get("airplane", 0),
        "tracks_added_ship": per_category.get("ship", 0),
        "tracks_added_train": per_category.get("train", 0),
        "tracks_added_holey": sum(1 for r in added if r["coverage_in_span"] < 0.99),
        "tracks_skipped": len(skipped),
        "boxes_before": sum(len(v) for v in mot.values()),
        "boxes_added": n_added_boxes,
        "boxes_added_with_sam3": n_added_refined,
        "boxes_hole_filled": n_filled,
        "boxes_regeometried": n_regeometried,
        "boxes_total": len(rows),
    }
    provenance = {"sequence": seq.id, "gt_path": seq.gt_path,
                  "iou_thresh": IOU_THRESH, "frac_thresh": FRAC_THRESH,
                  "refined_dir": str(refined_dir) if refined_dir else None,
                  "summary": summary, "added": added, "skipped": skipped}
    return lines, provenance


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True,
                    help="output root; mirrors each sequence's gt_path")
    ap.add_argument("--refined-dir", type=Path,
                    default=WORK / "experiments" / "satmtb_sam3_refine",
                    help="output of tools/refine_satmtb_sam3.py; pass '' to keep "
                         "original geometry")
    ap.add_argument("--fill-dir", type=Path,
                    default=WORK / "experiments" / "satmtb_hole_fill",
                    help="output of tools/fill_holey_tracks.py; adds the frames a "
                         "recovered static track skipped")
    ap.add_argument("--dataset", default="satmtb",
                    help="only SAT-MTB has a second annotation source today")
    ap.add_argument("--only", nargs="*", default=None,
                    help="sequence ids to process, for spot checks")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change, write nothing")
    ap.add_argument("--report", type=Path,
                    default=Path(__file__).resolve().parents[1]
                    / "space_tracker" / "data" / "merge_det_to_mot.csv")
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()

    refined_dir = args.refined_dir if args.refined_dir and str(args.refined_dir) else None
    if refined_dir and not refined_dir.is_dir():
        print(f"[warn] {refined_dir} does not exist — keeping original geometry")
        refined_dir = None
    fill_dir = args.fill_dir if args.fill_dir and str(args.fill_dir) else None
    if fill_dir and not fill_dir.is_dir():
        print(f"[warn] {fill_dir} does not exist — recovered tracks keep their holes")
        fill_dir = None

    seqs = [s for s in manifest().sequences if s.dataset == args.dataset]
    if args.only:
        wanted = set(args.only)
        seqs = [s for s in seqs if s.id in wanted]

    summaries: list[dict] = []
    totals: Counter = Counter()
    for seq in seqs:
        lines, provenance = merge_sequence(seq, refined_dir, fill_dir)
        s = provenance["summary"]
        summaries.append(s)
        for k, v in s.items():
            if isinstance(v, int):
                totals[k] += v
        if s["tracks_added"] or s["boxes_regeometried"]:
            print(f"  {seq.id:<22} +{s['tracks_added']:>3} tracks "
                  f"(+{s['boxes_added']:>6} boxes), "
                  f"{s['boxes_regeometried']:>6}/{s['boxes_before']:>6} regeometried, "
                  f"{s['tracks_skipped']:>3} det tracks already in MOT")
            totals["sequences_changed"] += 1
        if args.dry_run:
            continue

        dest = args.out / seq.gt_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("\n".join(lines) + "\n")
        prov_path = args.out / "provenance" / (seq.id.replace("/", "_") + ".json")
        prov_path.parent.mkdir(parents=True, exist_ok=True)
        prov_path.write_text(json.dumps(provenance, indent=1))

    print(f"\n{len(seqs)} sequences, {totals['sequences_changed']} changed")
    print(f"  +{totals['tracks_added']} recovered tracks "
          f"(airplane {totals['tracks_added_airplane']}, "
          f"ship {totals['tracks_added_ship']}, train {totals['tracks_added_train']})")
    print(f"  +{totals['boxes_added']} boxes "
          f"({totals['boxes_added_with_sam3']} with SAM 3 geometry, "
          f"{totals['boxes_hole_filled']} filling holes)")
    print(f"  {totals['boxes_regeometried']} original boxes regeometried")
    print(f"  {totals['boxes_before']} boxes before -> {totals['boxes_total']} after")

    if summaries and not args.no_report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            w.writeheader()
            w.writerows(summaries)
        print(f"\nper-sequence report -> {args.report}")
    if not args.dry_run:
        print(f"corrected ground truth -> {args.out}")


if __name__ == "__main__":
    main()
