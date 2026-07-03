#!/usr/bin/env python
"""Build a GT-oracle ``predictions.json`` for the BIRDSAI tracker comparison.

Every frame's ``detections`` are the GROUND-TRUTH boxes (annotations_sam3) with
score = 1.0, in the exact schema ``eval_birdsai_track_sweep.py`` /
``cache_birdsai_feats.py`` expect. Feeding these to the 6 TBD trackers removes
the detector entirely, so the only thing measured is pure association — the
GT-box ORACLE control (mirrors the MOT Exp2 fair-comparison; NOT a headline MOT
number — GT boxes never reach the model in the real benchmark).

Output: <out>/predictions.json  (+ same dir is a normal run dir for the sweep).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from datasets.birdsai_mot import BIRDSAIMOTDataset

BIRDSAI_ROOT = "/data/ESA_DLSTEM_2025/data/wild_animal/BIRDSAI"
CANON = {0: "human", 1: "elephant", 2: "giraffe", 3: "lion", 4: "unknown"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", default="annotations_sam3")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True, help="output dir for predictions.json")
    ap.add_argument("--videos", type=int, default=0,
                    help="limit to first N videos (0 = all; for smoke tests)")
    args = ap.parse_args()

    ds = BIRDSAIMOTDataset(root=BIRDSAI_ROOT, split=args.split, granularity="fine",
                           annotations_dirname=args.annotations,
                           class_map={v: k for k, v in CANON.items()})
    vids = ds.videos[: args.videos] if args.videos else ds.videos

    out = {"model": "gtoracle", "dataset": "BIRDSAI", "split": args.split,
           "img_size": None, "class_names": CANON, "annotations": args.annotations,
           "videos": {}}
    n_box = 0
    for v in vids:
        img_dir = ds._img_dir_cache[v.video_id]
        frames = {}
        for fid in v.frame_ids:
            a = ds._load_annotations(v, fid)
            boxes = np.asarray(a["boxes"], np.float32).reshape(-1, 4)
            labels = np.asarray(a["labels"], np.int64).reshape(-1)
            n_box += len(boxes)
            frames[str(int(fid))] = {
                "image_path": str(img_dir / f"{v.video_id}_{fid:010d}.jpg"),
                "detections": {
                    "boxes": boxes.round(2).tolist(),
                    "scores": [1.0] * len(boxes),
                    "labels": labels.tolist(),
                },
            }
        out["videos"][v.video_id] = {"image_dir": str(img_dir), "frames": frames}

    exp = Path(args.out)
    exp.mkdir(parents=True, exist_ok=True)
    (exp / "predictions.json").write_text(json.dumps(out))
    print(f"GT-oracle predictions: {len(vids)} videos, {n_box} boxes (GT={args.annotations})")
    print(f"wrote {exp / 'predictions.json'}")


if __name__ == "__main__":
    main()
