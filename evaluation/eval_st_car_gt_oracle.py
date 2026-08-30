"""GT-box association oracle on the Space-Tracker-MOT **car** test split.

Why
---
The car table answers one question and then stops answering questions.  Every
method there consumes the same HiEUM detections, which reach
$\\mathrm{mAP}_{50}=0.212$ on this split, and at that detection quality the
ordering of the trackers says almost nothing: SORT, the oldest method in the
comparison, wins.  That is a finding about the detector, not about association,
and it is the same finding for all seven rows.

This removes the detector.  Every method is handed the ground-truth boxes at
score 1 and runs its own unchanged association, so any difference that remains
is association ability.  The two tables are meant to be read as a pair: the
detector-fed one bounds what is achievable today, the oracle one says how much
of the gap is association's to close.

The output is written into a workspace laid out exactly like the one behind the
published car numbers -- the same ``gt/`` and the same ``seqmaps/``, copied
rather than rebuilt -- so the two are scored by identical code on identical
ground truth and the rows are directly comparable.

A note on ByteTrack.  Its mechanism is a two-tier split on detection score, and
an oracle gives every box score 1, so the low-score tier is empty and ByteTrack
degenerates to single-tier association.  That is a true property of the oracle
rather than a bug, and it is why its oracle row should be read as "ByteTrack's
matching without ByteTrack's idea".

Usage
-----
    python evaluation/eval_st_car_gt_oracle.py \\
        --src-workspace /tmp/hota_car_ws2 --out-workspace /tmp/hota_car_oracle
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import collections
import json
import shutil
import time

import numpy as np
import yaml

from models.trackers import build_tracker

BENCHMARK = "spacetracker_car_car-test"

#: The motion-only trackers, which need nothing but boxes. The appearance-aware
#: ones (botsort_reid, tracktrack) need a FastReID embedding per detection, and
#: on GT boxes that cache has to be built first; they are handled separately.
MOTION_TBD = ("sort", "bytetrack", "ocsort", "botsort")


def read_gt(path: Path) -> dict[int, np.ndarray]:
    """frame -> (N,4) xyxy. The MOTChallenge gt row is x,y,w,h top-left."""
    by_frame: dict[int, list] = collections.defaultdict(list)
    for line in path.read_text().strip().split("\n"):
        if not line:
            continue
        c = line.split(",")
        f = int(c[0])
        x, y, w, h = (float(v) for v in c[2:6])
        by_frame[f].append((x, y, x + w, y + h))
    return {f: np.asarray(v, dtype=np.float32) for f, v in by_frame.items()}


def tracker_kwargs(name: str, config_dir: Path) -> dict:
    """The published car configuration, minus everything score-dependent.

    Score thresholds in those configs are quantile-matched to HiEUM's score
    distribution. Under the oracle every box scores 1, so those thresholds are
    inert by construction; they are dropped rather than left in place at values
    that would silently mean something different here.
    """
    cfg = yaml.safe_load((config_dir / f"{name}_spacetracker_car.yaml").read_text())
    kw = dict(cfg.get("tracker_kwargs") or {})
    for k in list(kw):
        if "thresh" in k and ("track_high" in k or "track_low" in k
                              or "new_track" in k):
            kw.pop(k)
    return kw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-workspace", type=Path, required=True,
                    help="the workspace behind the published car numbers")
    ap.add_argument("--out-workspace", type=Path, required=True)
    ap.add_argument("--trackers", nargs="+", default=list(MOTION_TBD))
    ap.add_argument("--config-dir", type=Path,
                    default=Path("configs/MOT/tracker"))
    args = ap.parse_args()

    src, out = args.src_workspace, args.out_workspace
    gt_root = src / "gt" / BENCHMARK
    if not gt_root.is_dir():
        raise SystemExit(f"no ground truth at {gt_root}")

    # Copy rather than regenerate: identical ground truth is the whole point.
    for sub in ("gt", "seqmaps"):
        dst = out / sub
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src / sub, dst)
    (out / "output").mkdir(parents=True, exist_ok=True)

    seqs = sorted(d.name for d in gt_root.iterdir() if (d / "gt").is_dir())
    print(f"{len(seqs)} sequences, {len(args.trackers)} trackers")

    for name in args.trackers:
        kw = tracker_kwargs(name, args.config_dir)
        data = out / "trackers" / BENCHMARK / f"{name}_oracle" / "data"
        data.mkdir(parents=True, exist_ok=True)
        n_gt = n_out = 0
        t0 = time.perf_counter()

        for seq in seqs:
            tracker = build_tracker(name, **kw)
            tracker.reset()
            by_frame = read_gt(gt_root / seq / "gt" / "gt.txt")
            lines: list[str] = []
            for fid in range(1, max(by_frame) + 1 if by_frame else 1):
                boxes = by_frame.get(fid, np.zeros((0, 4), dtype=np.float32))
                n_gt += len(boxes)
                dets = (np.column_stack([boxes, np.ones(len(boxes), np.float32)])
                        if len(boxes) else np.zeros((0, 5), dtype=np.float32))
                tracks = tracker.update(dets, frame_id=fid)
                if not len(tracks):
                    continue
                tracks = np.asarray(tracks, dtype=np.float32).reshape(-1, 6)
                n_out += len(tracks)
                for x1, y1, x2, y2, sc, tid in tracks:
                    lines.append(f"{fid},{int(tid)},{x1:.2f},{y1:.2f},"
                                 f"{x2 - x1:.2f},{y2 - y1:.2f},1.0,-1,-1,-1")
            (data / f"{seq}.txt").write_text("\n".join(lines) + "\n")

        print(f"  {name:12s} gt_boxes={n_gt:8d} out={n_out:8d} "
              f"recall={n_out / max(n_gt, 1):.3f}  "
              f"{time.perf_counter() - t0:6.1f}s  kwargs={kw}")

    (out / "oracle_config.json").write_text(json.dumps(
        {"benchmark": BENCHMARK, "n_sequences": len(seqs),
         "trackers": args.trackers, "detection": "ground truth, score=1.0",
         "src_workspace": str(src)}, indent=2) + "\n")
    print(f"\nworkspace: {out}")


if __name__ == "__main__":
    main()
