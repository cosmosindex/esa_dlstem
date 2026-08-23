"""Fill the frames a recovered static track skips, with SAM 3.

``tools/merge_det_to_mot.py`` recovers 842 tracks from SAT-MTB's detection XML.
14 of them do not cover every frame they span — 13 because the XML annotates
only the first and last frame of the object, one (``ship/08`` track 5) because
it has genuine gaps. A static object is present in every frame between its
endpoints, so those gaps are missing annotation, not absence.

**Interpolate, then refine per frame — do not propagate.** Measured over the 14
tracks, net displacement between endpoints is 0.00-1.2 body lengths over spans
up to 326 frames, so a straight line between the two nearest annotated frames
lands the seed on the object. Feeding that seed to the same one-frame SAM 3 pass
that produced the other 398k boxes keeps the result order-independent, free of
propagation drift, and re-runnable; SAM 3 holding a 326-frame clip in its
inference state would risk neither of those things cheaply.

Seeds are only seeds: a filled frame is written only if SAM 3 accepts it under
the usual guards, and the seed is kept otherwise. Because a wrong seed on a 10
px object can make SAM 3 snap onto the wrong thing, every track touched here is
listed in ``needs_verification.json`` for the video review to sign off.

Output mirrors ``tools/refine_satmtb_sam3.py``: one JSON per sequence holding
``{track key: {frame id: xyxy}}``, consumed by ``merge_det_to_mot.py --fill-dir``.

Usage::

    CUDA_VISIBLE_DEVICES=0 python tools/fill_holey_tracks.py \
        --merged /work/<user>/space_tracker_mot_merged \
        --out /work/<user>/experiments/satmtb_hole_fill
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interactive_review.core.paths import sequence_by_id
from interactive_review.core.render import _read_frame
from interactive_review.core.sam3refine import _accept, get_tracker
from interactive_review.core.tracks import TrackKey, load_det_tracks

#: Experiment/scratch root. Real paths are machine-specific, so they are
#: never written into the repository — set ``WORK_ROOT`` to point at yours.
WORK = Path(os.environ.get("WORK_ROOT", "/work/anon"))


def interpolate(frame_ids: list[int], boxes: np.ndarray,
                refined: dict[int, list[float]]) -> dict[int, list[float]]:
    """Seed box for every frame inside the span that the track does not cover.

    Anchors are the refined boxes where they exist, so the seeds are already at
    the SAM 3 standard rather than the looser detection-XML one.
    """
    anchors = np.stack([np.asarray(refined.get(f, b), float)
                        for f, b in zip(frame_ids, boxes)])
    missing = [f for f in range(min(frame_ids), max(frame_ids) + 1)
               if f not in set(frame_ids)]
    if not missing:
        return {}
    known = np.asarray(frame_ids, float)
    return {f: [float(np.interp(f, known, anchors[:, c])) for c in range(4)]
            for f in missing}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merged", type=Path,
                    default=WORK / "space_tracker_mot_merged",
                    help="output of merge_det_to_mot.py; its provenance/ says "
                         "which recovered tracks have holes")
    ap.add_argument("--out", type=Path,
                    default=WORK / "experiments" / "satmtb_hole_fill")
    ap.add_argument("--refined-dir", type=Path,
                    default=WORK / "experiments" / "satmtb_sam3_refine")
    ap.add_argument("--max-coverage", type=float, default=0.99,
                    help="treat a recovered track as holey below this coverage")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the seeds, never load SAM 3")
    args = ap.parse_args()

    from interactive_review.core.batch_refined import load_refined

    # Group by sequence so frames are decoded once and SAM 3 can prompt every
    # box a frame needs in a single call.
    todo: dict[str, list[tuple[TrackKey, dict[int, list[float]]]]] = defaultdict(list)
    for p in sorted((args.merged / "provenance").glob("*.json")):
        doc = json.loads(p.read_text())
        for a in doc["added"]:
            if a.get("coverage_in_span", 1.0) >= args.max_coverage:
                continue
            key = TrackKey.parse(a["det_key"])
            seq = sequence_by_id(key.seq_id)
            track = load_det_tracks(key.seq_id, seq.video_id)[(key.category, key.track_ref)]
            refined = load_refined(str(args.refined_dir), key.seq_id).get(key.id, {})
            seeds = interpolate(track.frame_ids, track.boxes, refined)
            if seeds:
                todo[key.seq_id].append((key, seeds))

    n_tracks = sum(len(v) for v in todo.values())
    n_seeds = sum(len(s) for v in todo.values() for _, s in v)
    print(f"{n_tracks} holey tracks in {len(todo)} sequences, {n_seeds} frames to fill")
    if args.dry_run:
        for seq_id, items in sorted(todo.items()):
            for key, seeds in items:
                print(f"  {key.id:<44} {len(seeds):>4} seeds "
                      f"[{min(seeds)}..{max(seeds)}]")
        return

    sam = get_tracker()
    args.out.mkdir(parents=True, exist_ok=True)
    verify: list[dict] = []
    t0 = time.time()
    done = 0
    for seq_id, items in sorted(todo.items()):
        by_frame: dict[int, list[tuple[str, list[float]]]] = defaultdict(list)
        for key, seeds in items:
            for fid, box in seeds.items():
                by_frame[fid].append((key.id, box))

        filled: dict[str, dict[int, list[float]]] = defaultdict(dict)
        stats = {"seeds": 0, "refined": 0, "kept_empty": 0, "kept_iou": 0,
                 "kept_grow": 0, "kept_shrink": 0, "no_image": 0}
        for fid in sorted(by_frame):
            frame = _read_frame(seq_id, fid)
            if frame is None:
                stats["no_image"] += len(by_frame[fid])
                continue
            H, W = frame.shape[:2]
            prompts, keys = [], []
            for key_id, b in by_frame[fid]:
                box = [float(np.clip(b[0], 0, W - 1)), float(np.clip(b[1], 0, H - 1)),
                       float(np.clip(b[2], 1, W)), float(np.clip(b[3], 1, H))]
                if box[2] - box[0] < 1 or box[3] - box[1] < 1:
                    stats["kept_empty"] += 1
                    continue
                prompts.append(box)
                keys.append(key_id)
            if not prompts:
                continue

            stats["seeds"] += len(prompts)
            sam.init_video([frame])
            sam.add_prompts(0, np.asarray(prompts, np.float32),
                            labels=np.zeros(len(prompts), np.int64),
                            obj_ids=list(range(len(prompts))))
            outs = sam.propagate()
            sam.reset_state()

            out_boxes = (outs[0]["boxes"].numpy() if outs and len(outs[0]["boxes"])
                         else np.zeros((0, 4)))
            # obj_ids are positional here, so a missing object shifts nothing:
            # SAM 3 returns one row per prompt in prompt order.
            for i, (key_id, seed) in enumerate(zip(keys, prompts)):
                new = [float(v) for v in out_boxes[i]] if i < len(out_boxes) else None
                ok, reason = _accept(seed, new)
                filled[key_id][fid] = new if ok else seed
                stats["refined" if ok else "kept_" + reason] += 1
            done += len(prompts)
            if done % 200 < len(prompts):
                rate = done / max(time.time() - t0, 1e-9)
                print(f"    {done}/{n_seeds} frames  {rate:.1f}/s", flush=True)

        doc = {"sequence": seq_id, "source": "hole_fill", "method": "interpolate+sam3",
               "refined": {k: {str(f): v for f, v in sorted(fb.items())}
                           for k, fb in filled.items()},
               "stats": stats}
        (args.out / (seq_id.replace("/", "_") + ".json")).write_text(json.dumps(doc, indent=1))
        for key, seeds in items:
            verify.append({"track": key.id, "sequence": seq_id,
                           "frames_filled": len(filled.get(key.id, {})),
                           "frames_annotated": len(seeds) and
                           len(load_det_tracks(seq_id, sequence_by_id(seq_id).video_id)
                               [(key.category, key.track_ref)].frame_ids),
                           "reason": "interpolated seed — verify in the video review"})
        print(f"  {seq_id:<22} {stats['refined']}/{stats['seeds']} accepted "
              f"({', '.join(f'{k[5:]} {v}' for k, v in stats.items() if k.startswith('kept_') and v) or 'no guard trips'})")

    (args.out / "needs_verification.json").write_text(json.dumps(verify, indent=1))
    print(f"\n{n_seeds} frames filled in {time.time() - t0:.0f}s -> {args.out}")
    print(f"{len(verify)} tracks flagged for review -> {args.out}/needs_verification.json")
    print("re-run merge_det_to_mot.py with --fill-dir to fold these into the ground truth")


if __name__ == "__main__":
    main()
