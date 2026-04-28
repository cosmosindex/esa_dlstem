"""
Cache HiEUM per-video detections to disk for downstream tracker evaluation.

Trackers (SORT / ByteTrack / OC-SORT / BoT-SORT / TGraM) all consume the
same per-frame detection set; rerunning HiEUM once per tracker would
waste ~3 min × 5 trackers × 3 datasets ≈ 45 min. This script runs HiEUM
once per dataset, traversing the test split video-by-video (so the
tracker can later see the whole video as one continuous stream rather
than 20-frame clips), and writes one JSON per video::

    /data/ESA_DLSTEM_2025/experiments/Detection/hieum_dets_cache/<dataset>/<safe_vid>.json

with the schema::

    {
      "video_id":   "test1024/002",
      "dataset":    "RsCarData",
      "num_frames": 326,
      "frame_ids":  [1, 2, ..., 326],
      "boxes":      [ [[x1,y1,x2,y2], ...],   # one list per frame
                      ... ],
      "scores":     [ [s1, s2, ...],
                      ... ],
      "score_floor": 0.05,
      "image_size": [H, W],   # native frame resolution
    }

A low ``--score-floor`` (default 0.05) is applied so each downstream
tracker can pick its own confidence threshold without re-running HiEUM.

Usage::

    python cache_hieum_dets.py --dataset rscardata
    python cache_hieum_dets.py --dataset satmtb
    python cache_hieum_dets.py --dataset sdmcar
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

from models import HiEUMDetector


# Dataset name → (registry key, dataset root, dataset class import path,
#                 extra kwargs dict for the constructor).
_DATASET_TABLE = {
    "rscardata": (
        "RsCarData",
        "/data/ESA_DLSTEM_2025/data/trafic/RsCarData",
        "datasets.rscardata", "RsCarDataset", {},
    ),
    "satmtb": (
        "SAT-MTB",
        "/data/ESA_DLSTEM_2025/data/trafic/SAT-MTB",
        "datasets.satmtb", "SATMTBDataset",
        {"task": "mot", "categories": ["car"]},
    ),
    "sdmcar": (
        "SDM-Car",
        "/data/ESA_DLSTEM_2025/data/trafic/SDM-Car",
        "datasets.sdmcar", "SDMCarDataset", {},
    ),
}


def _load_dataset(name: str, split: str):
    if name not in _DATASET_TABLE:
        raise ValueError(f"unknown dataset {name!r}, choose from {list(_DATASET_TABLE)}")
    _, root, mod_name, cls_name, extra = _DATASET_TABLE[name]
    mod = __import__(mod_name, fromlist=[cls_name])
    cls = getattr(mod, cls_name)
    # We don't need clip slicing — we'll iterate videos directly and load
    # frames manually. Use ``mode="detection"`` so __init__ doesn't bother
    # building a clip index.
    return cls(
        root=root,
        split=split,
        mode="detection",
        class_map={"car": 0},
        **extra,
    )


def _safe_video_id(video_id: str) -> str:
    return video_id.replace("/", "_")


def _process_video(detector: HiEUMDetector, dataset, video) -> dict:
    """Load every frame of one video and run HiEUM on the whole stream."""
    frame_ids = list(video.frame_ids)
    frames_np = [dataset._load_frame(video, fid) for fid in frame_ids]
    H, W = frames_np[0].shape[:2]

    detector.init_video(frames_np)
    preds = detector.propagate()  # auto-chunks via _propagate_long for T>seq_len
    detector.reset_state()

    boxes_per_frame = []
    scores_per_frame = []
    for p in preds:
        b = p["boxes"]
        s = p["scores"]
        if torch.is_tensor(b):
            b = b.cpu().numpy()
        if torch.is_tensor(s):
            s = s.cpu().numpy()
        boxes_per_frame.append([[float(x) for x in row] for row in b])
        scores_per_frame.append([float(v) for v in s])

    return {
        "video_id":  video.video_id,
        "dataset":   video.dataset,
        "num_frames": len(frame_ids),
        "frame_ids":  frame_ids,
        "boxes":      boxes_per_frame,
        "scores":     scores_per_frame,
        "image_size": [int(H), int(W)],
    }


def main():
    parser = argparse.ArgumentParser(description="Cache HiEUM per-video detections")
    parser.add_argument("--dataset", required=True,
                        choices=sorted(_DATASET_TABLE.keys()))
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint",
                        default="/work/ziwen/checkpoints/hieum/model_best.pth")
    parser.add_argument("--output-dir",
                        default="/data/ESA_DLSTEM_2025/experiments/Detection/hieum_dets_cache")
    parser.add_argument("--score-floor", type=float, default=0.05,
                        help="Drop detections below this post-Soft-NMS score "
                             "before caching. Trackers can apply their own "
                             "(higher) threshold downstream.")
    parser.add_argument("--max-dets", type=int, default=128)
    parser.add_argument("--nms-iou", type=float, default=0.1,
                        help="Soft-NMS Nt — matches HiEUM paper protocol.")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")

    dataset_key, _, _, _, _ = _DATASET_TABLE[args.dataset]
    out_dir = Path(args.output_dir) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Caching HiEUM dets → {out_dir}")
    print(f"Dataset: {dataset_key} | split: {args.split}")

    dataset = _load_dataset(args.dataset, split=args.split)
    print(f"Videos: {len(dataset.videos)}")

    detector = HiEUMDetector(
        checkpoint_path=args.checkpoint,
        car_label=0,
        score_thresh=args.score_floor,
        nms_iou=args.nms_iou,
        max_dets=args.max_dets,
    )

    t_total = 0.0
    n_frames_total = 0
    for i, video in enumerate(dataset.videos, 1):
        out_path = out_dir / f"{_safe_video_id(video.video_id)}.json"
        if out_path.exists():
            print(f"  [{i}/{len(dataset.videos)}] {video.video_id} "
                  f"already cached, skipping")
            continue

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        cache = _process_video(detector, dataset, video)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        t_total += dt
        n_frames_total += cache["num_frames"]

        cache["score_floor"] = float(args.score_floor)
        cache["nms_iou"] = float(args.nms_iou)
        cache["max_dets"] = int(args.max_dets)
        cache["checkpoint"] = args.checkpoint

        with open(out_path, "w") as f:
            json.dump(cache, f, separators=(",", ":"))

        n_dets = sum(len(b) for b in cache["boxes"])
        print(f"  [{i}/{len(dataset.videos)}] {video.video_id} "
              f"| {cache['num_frames']} frames | {n_dets} dets | {dt:.1f}s "
              f"({cache['num_frames'] / dt:.1f} fps)")

    print(f"\nDone. {n_frames_total} frames in {t_total:.1f}s "
          f"({n_frames_total / max(t_total, 1e-9):.1f} fps)")


if __name__ == "__main__":
    main()
