"""
Render a MOT sequence's ground truth with **every** annotated class drawn,
one colour per category.

Unlike `viz_car_track_panel.py` — which loads SAT-MTB with
``categories=["car"]`` so the GT matches HiEUM's car-only detector — this
script applies no class filter at all. It exists to show what a sequence's GT
actually contains: `satmtb/car/18`, for instance, carries 450 non-car boxes
that the car-only view silently drops.

Boxes below ``--ring-below`` px are additionally circled, otherwise a 5 px
satellite car is invisible at native resolution.

Usage::

    python evaluation/viz_mot_gt_allclass.py --dataset satmtb --video car/18
    python evaluation/viz_mot_gt_allclass.py --dataset satmtb --video car/18 \
        --out-name satmtb-car_car-18_raw --ids
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from space_tracker.data_mot import iter_mot_frames
from space_tracker.manifest_mot import MOTManifest
from tools.analyze_size_split import MOT_ROOTS, REPO

#: Experiment/scratch root. Real paths are machine-specific, so they are
#: never written into the repository — set ``WORK_ROOT`` to point at yours.
WORK = Path(os.environ.get("WORK_ROOT", "/work/anon"))

# BGR, one per unified category
COLOURS = {
    "car":      (0, 255, 0),
    "airplane": (0, 165, 255),
    "ship":     (255, 0, 255),
    "train":    (255, 255, 0),
}
FALLBACK = (200, 200, 200)


def _draw_frame(rgb: np.ndarray, objects, seq_id: str, fid: int,
                ring_below: float, show_ids: bool) -> np.ndarray:
    bgr = rgb[..., ::-1].copy()
    counts = Counter(o.category for o in objects)

    for o in objects:
        x1, y1, x2, y2 = (int(round(v)) for v in o.bbox_xyxy)
        col = COLOURS.get(o.category, FALLBACK)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), col, 1, cv2.LINE_AA)
        w, h = x2 - x1, y2 - y1
        if np.sqrt(max(w, 0) * max(h, 0)) < ring_below:
            cv2.circle(bgr, ((x1 + x2) // 2, (y1 + y2) // 2),
                       max(7, int(max(w, h))), col, 1, cv2.LINE_AA)
        if show_ids:
            cv2.putText(bgr, str(o.track_id), (x1, max(9, y1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1, cv2.LINE_AA)

    # header + per-class legend
    pad = 8
    lines = [f"{seq_id}   frame {fid}   GT objects: {len(objects)}"]
    lines += [f"{c}: {counts.get(c, 0)}" for c in COLOURS if counts.get(c)]
    box_h = 20 * len(lines) + pad
    overlay = bgr.copy()
    cv2.rectangle(overlay, (0, 0), (430, box_h), (0, 0, 0), -1)
    bgr = cv2.addWeighted(overlay, 0.45, bgr, 0.55, 0)

    y = 18
    cv2.putText(bgr, lines[0], (pad, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    for line in lines[1:]:
        y += 20
        cat = line.split(":")[0]
        col = COLOURS.get(cat, FALLBACK)
        cv2.rectangle(bgr, (pad, y - 9), (pad + 14, y + 1), col, -1)
        cv2.putText(bgr, line, (pad + 22, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, col, 1, cv2.LINE_AA)
    return bgr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(MOT_ROOTS))
    ap.add_argument("--video", required=True, help="manifest video_id, e.g. car/18")
    ap.add_argument("--out-dir", type=Path,
                    default=WORK / "experiments" / "car_mot_qualitative")
    ap.add_argument("--out-name", default=None,
                    help="basename without extension (default: <dataset>_<video>_gt_allclass)")
    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--ring-below", type=float, default=14.0,
                    help="circle boxes whose sqrt(area) is under this many px")
    ap.add_argument("--ids", action="store_true", help="draw track ids")
    args = ap.parse_args()

    man = MOTManifest.load(REPO / "space_tracker" / "space_tracker_mot.json")
    try:
        seq = next(s for s in man.sequences
                   if s.dataset == args.dataset and s.video_id == args.video)
    except StopIteration:
        sys.exit(f"no sequence {args.dataset}/{args.video} in the MOT manifest")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_name or f"{args.dataset}_{args.video.replace('/', '-')}_gt_allclass"
    tmp = args.out_dir / f"{stem}_mp4v.mp4"
    final = args.out_dir / f"{stem}_h264.mp4"

    writer = None
    totals = Counter()
    n_frames = 0
    for fr in iter_mot_frames(seq, MOT_ROOTS, decode_images=True):
        vis = _draw_frame(fr.image, fr.objects, seq.id, fr.frame_id,
                          args.ring_below, args.ids)
        if writer is None:
            h, w = vis.shape[:2]
            writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"),
                                     args.fps, (w, h))
            print(f"{seq.id}: {w}x{h} @ {args.fps} fps")
        writer.write(vis)
        totals.update(o.category for o in fr.objects)
        n_frames += 1
    if writer is None:
        sys.exit("no frames decoded")
    writer.release()

    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                    str(final)], check=True)
    tmp.unlink()

    print(f"frames: {n_frames}")
    for c, n in totals.most_common():
        print(f"  {c:<10}{n:>8} boxes")
    print(f"wrote {final}  ({final.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
