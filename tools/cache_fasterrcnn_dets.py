"""Cache FasterRCNN detections (3-class: airplane / ship / train) on the
non-car MOT datasets so the existing tracker pipeline can consume them in
the same way it consumes HiEUM detections.

Output schema (one JSON per video; mirrors HiEUM cache schema with an
extra ``labels`` column for multi-class detection):

    {
      "video_id": "...",
      "dataset":  "...",
      "num_frames": int,
      "frame_ids": [int, ...],
      "boxes":   [ [[x1,y1,x2,y2], ...],  ... ],   # per-frame, xyxy float
      "scores":  [ [float, ...],          ... ],   # per-frame
      "labels":  [ [int, ...],            ... ],   # per-frame, class ids 1=airplane 2=ship 3=train
      "image_size": [W, H],
      "score_floor": float,
      "nms_iou":     float,
      "max_dets":    int,
      "checkpoint":  "...",
      "min_size":    int,
      "max_size":    int,
    }
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import yaml

sys.path.insert(0, "/home/ziwen/code/esa_dlstem")

from datasets.airmot import AIRMOTDataset
from datasets.satmtb import SATMTBDataset
from datasets.viso import VISODataset
from models import FasterRCNNDetector
from lightning_modules import ObjectDetectionModule


_DEFAULT_CKPT = (
    "/work/ziwen/experiments/fasterrcnn_satmtb_hbb_20260430_075421/"
    "checkpoints/best-epoch=5-val_mAP=0.545.ckpt"
)
_DEFAULT_CONFIG = "/home/ziwen/code/esa_dlstem/configs/Detection/fasterrcnn_satmtb_hbb.yaml"
_DEFAULT_OUT_ROOT = "/data/ESA_DLSTEM_2025/experiments/Detection/fasterrcnn_satmtb_hbb_dets_cache"


def _safe_video_id(video_id: str) -> str:
    return video_id.replace("/", "_")


def _build_dataset(name: str):
    """Return (display_name, dataset_obj) — limited to sequences FasterRCNN
    has *not* seen at training time."""
    if name == "satmtb_nocar":
        ds = SATMTBDataset(
            root="/data/ESA_DLSTEM_2025/data/trafic/SAT-MTB",
            split="test",                                            # held-out only
            task="mot",
            categories=["airplane", "ship", "train"],
        )
        return "SAT-MTB", ds
    if name == "viso_nocar":
        ds = VISODataset(
            root="/data/ESA_DLSTEM_2025/data/trafic/VISO",
            split="no_split",
            categories=["plane", "ship", "train"],
        )
        return "VISO", ds
    if name == "airmot":
        ds = AIRMOTDataset(
            root="/data/ESA_DLSTEM_2025/data/trafic/AIR-MOT-100",
            split="no_split",
        )
        return "AIR-MOT-100", ds
    raise ValueError(f"unknown dataset {name!r}")


def _load_frame_rgb(ds, video, frame_id: int) -> np.ndarray:
    """All three datasets expose ``_load_frame`` returning BGR ndarray (cv2)."""
    bgr = ds._load_frame(video, frame_id)
    return bgr[:, :, ::-1]  # -> RGB


def _load_module(ckpt_path: str, config_path: str, min_size: int, max_size: int) -> torch.nn.Module:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model = FasterRCNNDetector(
        num_classes=cfg["num_classes"],
        pretrained=False,
        use_v2=cfg.get("use_v2", False),
        trainable_backbone_layers=cfg.get("trainable_backbone_layers", 3),
        score_thresh=cfg.get("score_thresh", 0.05),
        nms_thresh=cfg.get("nms_thresh", 0.5),
        detections_per_img=cfg.get("detections_per_img", 300),
        enable_tracking=False,
        anchor_sizes=tuple(tuple(s) for s in cfg["anchor_sizes"]),
        anchor_aspect_ratios=tuple(tuple(r) for r in cfg["anchor_aspect_ratios"]),
        rpn_fg_iou_thresh=cfg.get("rpn_fg_iou_thresh"),
        rpn_bg_iou_thresh=cfg.get("rpn_bg_iou_thresh"),
        box_fg_iou_thresh=cfg.get("box_fg_iou_thresh"),
        box_bg_iou_thresh=cfg.get("box_bg_iou_thresh"),
        rpn_pre_nms_top_n_train=cfg.get("rpn_pre_nms_top_n_train"),
        rpn_post_nms_top_n_train=cfg.get("rpn_post_nms_top_n_train"),
        min_size=min_size,
        max_size=max_size,
    )
    module = ObjectDetectionModule.load_from_checkpoint(
        ckpt_path, model=model, has_tracking=False,
    )
    module.eval()
    return module


def _infer_video(module, ds, video, score_floor: float, max_dets: int, device, max_per_image_classes: int = None):
    """Run the detector frame-by-frame and return three parallel lists."""
    boxes_all, scores_all, labels_all = [], [], []
    img_size = None

    for fid in video.frame_ids:
        arr = _load_frame_rgb(ds, video, fid)
        H, W = arr.shape[:2]
        if img_size is None:
            img_size = [W, H]
        tensor = torch.from_numpy(arr.copy()).permute(2, 0, 1).float().div_(255.0).to(device)

        with torch.no_grad():
            out = module.model([tensor])[0]

        b = out["boxes"].detach().cpu().numpy()
        s = out["scores"].detach().cpu().numpy()
        l = out["labels"].detach().cpu().numpy()

        keep = s >= score_floor
        b, s, l = b[keep], s[keep], l[keep]
        if len(s) > max_dets:
            order = np.argsort(-s)[:max_dets]
            b, s, l = b[order], s[order], l[order]

        boxes_all.append(b.tolist())
        scores_all.append(s.tolist())
        labels_all.append(l.tolist())

    return boxes_all, scores_all, labels_all, img_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["satmtb_nocar", "viso_nocar", "airmot"])
    ap.add_argument("--ckpt", default=_DEFAULT_CKPT)
    ap.add_argument("--config", default=_DEFAULT_CONFIG)
    ap.add_argument("--out-root", default=_DEFAULT_OUT_ROOT)
    ap.add_argument("--score-floor", type=float, default=0.05)
    ap.add_argument("--max-dets", type=int, default=128)
    ap.add_argument("--min-size", type=int, default=1024)
    ap.add_argument("--max-size", type=int, default=2304)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None,
                    help="for testing: only process the first N videos per dataset")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}")
    print(f"[setup] ckpt={args.ckpt}")
    print(f"[setup] inference resolution: min={args.min_size} max={args.max_size}")

    module = _load_module(args.ckpt, args.config, args.min_size, args.max_size).to(device)
    print(f"[setup] model loaded")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    overall_start = time.time()

    for ds_name in args.datasets:
        display, ds = _build_dataset(ds_name)
        videos = ds.videos[: args.limit] if args.limit else ds.videos
        ds_out = out_root / ds_name
        ds_out.mkdir(parents=True, exist_ok=True)
        print(f"\n[{display}] {len(videos)} sequences → {ds_out}")
        ds_start = time.time()

        for vi, video in enumerate(videos):
            cache_path = ds_out / f"{_safe_video_id(video.video_id)}.json"
            if cache_path.exists():
                print(f"  [{vi+1:>3d}/{len(videos)}] {video.video_id}  (cached, skip)")
                continue
            t0 = time.time()
            boxes, scores, labels, img_size = _infer_video(
                module, ds, video,
                score_floor=args.score_floor,
                max_dets=args.max_dets,
                device=device,
            )
            payload = {
                "video_id": video.video_id,
                "dataset":  display,
                "num_frames": len(video.frame_ids),
                "frame_ids": list(video.frame_ids),
                "boxes":  boxes,
                "scores": scores,
                "labels": labels,
                "image_size": img_size,
                "score_floor": args.score_floor,
                "nms_iou": None,
                "max_dets": args.max_dets,
                "checkpoint": args.ckpt,
                "min_size": args.min_size,
                "max_size": args.max_size,
            }
            with open(cache_path, "w") as f:
                json.dump(payload, f)
            n_frames = len(video.frame_ids)
            n_dets = sum(len(s) for s in scores)
            elapsed = time.time() - t0
            print(f"  [{vi+1:>3d}/{len(videos)}] {video.video_id:35s} "
                  f"{n_frames:>5d} frames  {n_dets:>6d} dets  "
                  f"{elapsed:5.1f}s  ({n_frames/elapsed:.1f} fps)")
        print(f"[{display}] done in {time.time()-ds_start:.1f}s")

    print(f"\nALL DONE in {(time.time()-overall_start)/60:.1f} min  →  {out_root}")


if __name__ == "__main__":
    main()
