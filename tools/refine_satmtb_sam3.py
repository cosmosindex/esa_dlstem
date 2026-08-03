"""Refine every SAT-MTB box with SAM 3, so one annotation standard applies.

Why this exists: SAM 3 returns systematically tighter boxes than SAT-MTB ships
(median 0.52x area on aircraft, 0.93x on ships, measured over sampled tracks).
Adding interactively drawn or recovered tracks without touching the originals
would leave the ground truth internally inconsistent, and an IoU-thresholded
metric cannot distinguish "annotated to a different standard" from "detector
was wrong". Refining everything removes the gap.

**Batched by frame, not by box.** A SAM 3 call costs about the same whatever
its prompt count, so prompting every box in a frame at once turns 231,988 calls
into 33,835. The same trick took the BIRDSAI re-annotation to 36 boxes/second.

``car`` is excluded by default: at a 4.9-6.5 px median it is below what a
promptable segmenter can resolve, and it accounts for 91% of all SAT-MTB boxes.
Pass ``--include-car`` to override.

Sources, selectable with ``--source``:

``mot``     every non-car box in the MOT ground truth (103,274 boxes) — needs
            no human input and can run now
``det``     every detection-XML track, reviewed or not. These are the static
            objects the MOT ground truth drops, so without this pass they are
            the only boxes in the tool still shown at the original standard
``both``    ``mot`` + ``det``

Passes accumulate: a later run merges into the per-sequence JSON rather than
overwriting it, and ``--resume`` skips a sequence only for sources it has
already been run with.

Output is one JSON per sequence under ``--out``, holding
``{track key: {frame id: xyxy}}`` plus guard statistics, which
``interactive_review.export`` and the review UI both consume.

Usage::

    python tools/refine_satmtb_sam3.py --source mot --out /work/<user>/satmtb_sam3
    python tools/refine_satmtb_sam3.py --source mot --out ... --shard 0/2   # GPU 0
    python tools/refine_satmtb_sam3.py --source mot --out ... --shard 1/2   # GPU 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interactive_review.core.paths import manifest, sequence_by_id
from interactive_review.core.render import _read_frame
from interactive_review.core.sam3refine import GROW_MAX, IOU_MIN, SHRINK_MIN, get_tracker
from interactive_review.core.tracks import (TrackKey, load_det_tracks,
                                            load_mot_frames)

NON_CAR = ("airplane", "ship", "train")


def _iou(a, b) -> float:
    lt, rb = np.maximum(a[:2], b[:2]), np.minimum(a[2:], b[2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[0] * wh[1]
    ar = lambda x: max((x[2] - x[0]) * (x[3] - x[1]), 0.0)
    return float(inter / max(ar(a) + ar(b) - inter, 1e-9))


def _accept(orig, refined) -> tuple[bool, str]:
    if refined is None:
        return False, "empty"
    rw, rh = refined[2] - refined[0], refined[3] - refined[1]
    if rw < 1 or rh < 1:
        return False, "empty"
    ao = max((orig[2] - orig[0]) * (orig[3] - orig[1]), 1e-9)
    if _iou(np.asarray(orig, float), np.asarray(refined, float)) < IOU_MIN:
        return False, "iou"
    if rw * rh > GROW_MAX * ao:
        return False, "grow"
    if rw * rh < SHRINK_MIN * ao:
        return False, "shrink"
    return True, ""


def collect_targets(seq_id: str, source: str,
                    include_car: bool) -> dict[int, list[tuple[str, list[float]]]]:
    """-> ``{frame_id: [(track key id, xyxy), ...]}`` for one sequence."""
    seq = sequence_by_id(seq_id)
    keep = ("car",) + NON_CAR if include_car else NON_CAR
    out: dict[int, list[tuple[str, list[float]]]] = defaultdict(list)

    if source in ("mot", "both"):
        for fid, objs in load_mot_frames(seq_id).items():
            for o in objs:
                if o.category not in keep:
                    continue
                key = TrackKey(seq_id, "mot_gt", o.category, str(o.track_id))
                out[fid].append((key.id, [float(v) for v in o.bbox_xyxy]))

    if source in ("det", "both"):
        # Every detection-XML track, review status irrelevant — these carry the
        # static objects, and leaving them unrefined is exactly what makes the
        # review tool show two annotation standards in one frame.
        for (cat, oid), t in load_det_tracks(seq_id, seq.video_id).items():
            if cat not in keep:
                continue
            key = TrackKey(seq_id, "det_xml", cat, oid)
            for fid, box in zip(t.frame_ids, t.boxes):
                out[fid].append((key.id, [float(v) for v in box]))


    return dict(out)


def refine_sequence(seq_id: str, targets: dict[int, list[tuple[str, list[float]]]],
                    ) -> tuple[dict[str, dict[int, list[float]]], dict[str, int]]:
    """Refine every listed box, one SAM 3 call per frame."""
    sam = get_tracker()
    refined: dict[str, dict[int, list[float]]] = defaultdict(dict)
    stats = {"boxes": 0, "refined": 0, "kept_empty": 0, "kept_iou": 0,
             "kept_grow": 0, "kept_shrink": 0, "no_image": 0, "frames": 0}
    ratios: list[float] = []

    for fid in sorted(targets):
        items = targets[fid]
        frame = _read_frame(seq_id, fid)
        if frame is None:
            stats["no_image"] += len(items)
            continue
        H, W = frame.shape[:2]

        boxes, oids, originals = [], [], {}
        for key_id, box in items:
            x1 = float(np.clip(box[0], 0, W - 1)); y1 = float(np.clip(box[1], 0, H - 1))
            x2 = float(np.clip(box[2], 1, W));     y2 = float(np.clip(box[3], 1, H))
            if x2 - x1 < 1 or y2 - y1 < 1:
                stats["kept_empty"] += 1
                continue
            oid = len(boxes)
            boxes.append([x1, y1, x2, y2])
            oids.append(oid)
            originals[oid] = (key_id, [x1, y1, x2, y2])
        if not boxes:
            continue

        stats["boxes"] += len(boxes)
        stats["frames"] += 1
        # Every box in this frame is prompted in one call — the dominant cost
        # is the backbone pass, which is per-frame, not per-prompt.
        sam.init_video([frame])
        sam.add_prompts(0, np.asarray(boxes, np.float32),
                        labels=np.zeros(len(boxes), np.int64), obj_ids=oids)
        outs = sam.propagate()
        sam.reset_state()

        by_oid: dict[int, list[float]] = {}
        if outs and len(outs[0]["boxes"]):
            rb = outs[0]["boxes"].numpy()
            rid = outs[0]["track_ids"].numpy()
            for b, t in zip(rb, rid):
                by_oid[int(t)] = [float(v) for v in b]

        for oid, (key_id, orig) in originals.items():
            new = by_oid.get(oid)
            ok, reason = _accept(orig, new)
            refined[key_id][fid] = new if ok else orig
            if ok:
                stats["refined"] += 1
                ao = max((orig[2] - orig[0]) * (orig[3] - orig[1]), 1e-9)
                ratios.append(((new[2] - new[0]) * (new[3] - new[1])) / ao)
            else:
                stats["kept_" + reason] += 1

    stats["median_area_ratio"] = round(float(np.median(ratios)), 4) if ratios else 0.0
    return dict(refined), stats


def _read_doc(dest: Path) -> dict:
    """Existing per-sequence output, so successive sources accumulate."""
    if not dest.is_file():
        return {"refined": {}, "stats": {}, "sources": []}
    try:
        doc = json.loads(dest.read_text())
    except (OSError, ValueError):
        return {"refined": {}, "stats": {}, "sources": []}
    # Files written before sources were tracked hold one flat stats dict.
    stats = doc.get("stats", {})
    if stats and "boxes" in stats:
        doc["stats"] = {doc.get("source", "mot"): stats}
        doc.setdefault("sources", [doc.get("source", "mot")])
    doc.setdefault("refined", {})
    doc.setdefault("stats", {})
    doc.setdefault("sources", [])
    return doc


def _write_doc(dest: Path, doc: dict, seq_id: str, source: str,
               refined: dict, stats: dict, seconds: float) -> None:
    doc["refined"].update(
        {k: {str(f): [round(v, 2) for v in b] for f, b in fb.items()}
         for k, fb in refined.items()})
    doc["stats"][source] = {**stats, "seconds": round(seconds, 1)}
    doc["sequence"] = seq_id
    if source not in doc["sources"]:
        doc["sources"].append(source)
    dest.write_text(json.dumps(doc, indent=1) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("mot", "det", "both"), default="mot")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--include-car", action="store_true",
                    help="also refine car (median 4.9-6.5 px; off by default)")
    ap.add_argument("--shard", default="0/1",
                    help="'i/n' — process every n-th sequence, for multi-GPU runs")
    ap.add_argument("--limit-frames", type=int, default=None,
                    help="stop each sequence after this many frames (speed probe)")
    ap.add_argument("--resume", action="store_true",
                    help="skip sequences whose output file already exists")
    args = ap.parse_args()

    shard_i, shard_n = (int(v) for v in args.shard.split("/"))
    args.out.mkdir(parents=True, exist_ok=True)
    seqs = [s.id for s in manifest().sequences if s.dataset == "satmtb"]
    seqs = [s for i, s in enumerate(sorted(seqs)) if i % shard_n == shard_i]
    print(f"shard {shard_i}/{shard_n}: {len(seqs)} sequences, source={args.source}, "
          f"car {'included' if args.include_car else 'excluded'}")

    grand = defaultdict(int)
    t_start = time.time()
    for n, seq_id in enumerate(seqs, 1):
        dest = args.out / (seq_id.replace("/", "_") + ".json")
        doc = _read_doc(dest)
        if args.resume and args.source in doc.get("sources", []):
            print(f"[{n}/{len(seqs)}] {seq_id}: source '{args.source}' done already, skipping")
            continue

        targets = collect_targets(seq_id, args.source, args.include_car)
        if args.limit_frames:
            targets = {f: targets[f] for f in sorted(targets)[:args.limit_frames]}
        if not targets:
            print(f"[{n}/{len(seqs)}] {seq_id}: nothing to refine")
            _write_doc(dest, doc, seq_id, args.source, {}, {}, 0.0)
            continue

        t0 = time.time()
        refined, stats = refine_sequence(seq_id, targets)
        dt = time.time() - t0
        _write_doc(dest, doc, seq_id, args.source, refined, stats, dt)

        for k, v in stats.items():
            if k != "median_area_ratio":
                grand[k] += v
        rate = stats["frames"] / dt if dt else 0
        done_frac = n / len(seqs)
        eta = (time.time() - t_start) / max(done_frac, 1e-9) * (1 - done_frac)
        print(f"[{n}/{len(seqs)}] {seq_id}: {stats['boxes']:>6} boxes over "
              f"{stats['frames']:>4} frames in {dt:>6.1f}s ({rate:.2f} fps), "
              f"area {stats['median_area_ratio']:.2f}x, ETA {eta/3600:.1f}h")

    print(f"\ntotal: {grand['refined']:,}/{grand['boxes']:,} boxes refined over "
          f"{grand['frames']:,} frames in {(time.time()-t_start)/3600:.2f} h")
    for k in ("kept_empty", "kept_iou", "kept_grow", "kept_shrink", "no_image"):
        if grand[k]:
            print(f"  {k}: {grand[k]:,}")


if __name__ == "__main__":
    main()
