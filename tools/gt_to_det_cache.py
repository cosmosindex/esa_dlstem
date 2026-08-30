"""
Emit a detection cache whose "detections" are the ground-truth boxes.

The GT-box oracle needs the appearance-aware trackers (BoT-SORT-ReID,
TrackTrack, MASA) as well as the motion-only ones, and those consume a FastReID
embedding per detection. The embedding cache is built by
tools/cache_spacetracker_feats.py from a detection cache, so the shortest honest
route to GT-box embeddings is to hand that script a detection cache that happens
to contain ground truth at score 1.0 -- no second code path, and the crops are
produced by exactly the same code as the detector-fed ones.

Source is the TrackEval workspace behind the published numbers rather than the
dataset, so the boxes are the same ones the published rows were scored against.

Usage:
    python tools/gt_to_det_cache.py \
        --workspace /tmp/hota_car_ws2 --benchmark spacetracker_car_car-test \
        --out /data/.../Detection/gt_dets_cache_st_car/spacetracker_car
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", required=True, type=Path)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dataset-name", default="Space-Tracker-MOT")
    args = ap.parse_args()

    gt_root = args.workspace / "gt" / args.benchmark
    seqs = sorted(d.name for d in gt_root.iterdir() if (d / "gt").is_dir())
    args.out.mkdir(parents=True, exist_ok=True)

    total = 0
    for seq in seqs:
        by_frame: dict[int, list] = collections.defaultdict(list)
        for line in (gt_root / seq / "gt" / "gt.txt").read_text().strip().split("\n"):
            if not line:
                continue
            c = line.split(",")
            f = int(c[0])
            x, y, w, h = (float(v) for v in c[2:6])
            by_frame[f].append([x, y, x + w, y + h])

        # seqLength, not max(frame with a box): a frame with no ground truth is
        # still a frame the tracker must be stepped through, and dropping it
        # would silently shorten the sequence.
        ini = (gt_root / seq / "seqinfo.ini").read_text()
        n = int(next(l.split("=")[1] for l in ini.splitlines()
                     if l.startswith("seqLength")))
        frame_ids = list(range(1, n + 1))
        boxes = [by_frame.get(f, []) for f in frame_ids]
        scores = [[1.0] * len(b) for b in boxes]
        total += sum(len(b) for b in boxes)

        (args.out / f"{seq}.json").write_text(json.dumps({
            "video_id": seq, "dataset": args.dataset_name,
            "num_frames": n, "frame_ids": frame_ids,
            "boxes": boxes, "scores": scores,
        }))
    print(f"{len(seqs)} sequences, {total:,} ground-truth boxes -> {args.out}")


if __name__ == "__main__":
    main()
