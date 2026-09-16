"""
Build MOTRv2 proposal files (`det_db`) from a trained Faster R-CNN.

MOTRv2 is "Bootstrapping End-to-End MOT by Pretrained Object Detectors": the
model is fed per-frame proposals from an external detector and learns to
propagate track queries on top of them. Officially those proposals come from
YOLOX, at BOTH train and inference time -- never from ground truth. Feeding GT
boxes turns the run into an oracle ablation and silently defeats the machinery
that exists precisely because proposals are noisy (`--fp_ratio`,
`--query_denoise`).

This script replaces the GT-box `det_db` written by `tools/export_motrv2.py`
with real detections, so MOTRv2 is run the way its authors intended, with our
Faster R-CNN standing in for YOLOX.

The image tree written by `export_motrv2.py` is reused untouched -- only the
`det_db` JSONs are regenerated, keyed exactly as MOTRv2 expects:

    { "<split>/<vid>/img1/{t:08d}.txt": ["x,y,w,h,score", ...], ... }

Usage:
    python tools/make_motrv2_det_db.py \
        --ckpt  /path/to/best-epoch=NN-val_mAP=X.ckpt \
        --config configs/Detection/fasterrcnn_spacetracker_mot.yaml \
        --motr-root /path/to/motrv2_st_nocar

Upstream
--------
MOTRv2 -- https://github.com/megvii-research/MOTRv2, commit ``1aac7c3``::

    git clone https://github.com/megvii-research/MOTRv2 && \
        git -C MOTRv2 checkout 1aac7c3

Clone it beside this repository; it is not committed here. MOTRv2 is the one
benchmarked method with no runner of its own under ``evaluation/``: it cannot be
imported alongside this project (its ``datasets`` and ``models`` packages
collide with ours), so it is run from inside its own tree. Its training entry
point is the upstream ``main.py`` driven by ``tools/train.sh``; evaluation is
``eval_motrv2.py``, which reads only the on-disk layout this script writes.

Local changes to the clone, all of them required to run it here:

* ``datasets/dance.py`` -- take the training split directory as an argument
  (``--mot_train_name``) instead of the literal ``DanceTrack/train``; and build
  ``targets['labels']`` with ``dtype=torch.long``. The second is load-bearing: a
  satellite frame can hold zero ground-truth objects, and ``torch.as_tensor([])``
  is float32, which promotes the concatenated per-frame labels to float and
  breaks the matcher's ``cost[:, tgt_ids]`` indexing. DanceTrack never has an
  empty frame, so upstream cannot hit it.
* ``main.py``, ``submit_dance.py``, ``util/tool.py`` -- ``torch.load(...,
  weights_only=False)``. The checkpoints carry an ``argparse.Namespace``, which
  torch 2.6 refuses to unpickle by default.
* ``models/ops/`` -- the CUDA sources of the deformable-attention kernel updated
  for the current toolkit.
* added: ``configs/motrv2_union.args``, ``configs/motrv2_st_nocar.args``,
  ``eval_motrv2.py``, ``eval_motrv2_oracle_assoc.py``. The ``.args`` files are
  read with ``args=$(cat configs/x.args)``, which does not expand shell
  variables, so pass them through ``envsubst`` first; ``$DATA_ROOT`` and
  ``$CHECKPOINT_ROOT`` are the only machine-specific values in them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import SpaceTrackerMOTDataset
from models import FasterRCNNDetector
from project_paths import DATA_ROOT, load_config

# MOTRv2 splits: its "train" tree holds our train split, "eval" holds test.
SPLIT_TO_MODE = {"train": "train", "test": "eval"}

# Must match the `--datasets` name used with tools/export_motrv2.py, since that
# is what the on-disk video directories are named after.
DATASET_TAG = "spacetracker_nocar"


def build_detector(cfg: dict, ckpt_path: Path, device: torch.device):
    model = FasterRCNNDetector(
        num_classes=cfg["num_classes"],
        pretrained=False,
        trainable_backbone_layers=cfg.get("trainable_backbone_layers", 3),
        anchor_sizes=cfg.get("anchor_sizes"),
        anchor_aspect_ratios=cfg.get("anchor_aspect_ratios"),
        score_thresh=cfg.get("score_thresh", 0.05),
        nms_thresh=cfg.get("nms_thresh", 0.5),
        detections_per_img=cfg.get("detections_per_img", 300),
        min_size=cfg.get("min_size", 1024),
        max_size=cfg.get("max_size", 2304),
        use_v2=cfg.get("use_v2", False),
        rpn_fg_iou_thresh=cfg.get("rpn_fg_iou_thresh", 0.7),
        rpn_bg_iou_thresh=cfg.get("rpn_bg_iou_thresh", 0.3),
        box_fg_iou_thresh=cfg.get("box_fg_iou_thresh", 0.5),
        box_bg_iou_thresh=cfg.get("box_bg_iou_thresh", 0.5),
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state.get("state_dict", state)
    # The Lightning module wraps FasterRCNNDetector, which itself wraps the
    # torchvision model, so keys arrive as "model.model.<...>". Strip both
    # levels; a partial strip loads nothing and silently yields a random net.
    prefix = "model.model."
    if any(k.startswith(prefix) for k in sd):
        sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    missing, unexpected = model.model.load_state_dict(sd, strict=False)
    if unexpected:
        # A checkpoint from a different architecture would silently half-load.
        raise RuntimeError(f"unexpected keys in checkpoint: {unexpected[:5]}")
    if missing:
        print(f"[warn] {len(missing)} missing keys (first: {missing[:3]})")
    return model.model.to(device).eval()


@torch.inference_mode()
def run_split(detector, cfg, release_root: Path, split: str, mode: str,
              score_thresh: float, device: torch.device,
              limit_videos: int | None = None) -> dict:
    ds = SpaceTrackerMOTDataset(
        root=release_root, split=split, mode="detection",
        class_map=cfg["class_map"], complete_only=True,
    )
    videos = ds.videos if limit_videos is None else ds.videos[:limit_videos]
    print(f"  {split}: {len(videos)} sequences (of {len(ds.videos)})")

    vid_prefix = f"{DATASET_TAG}__" if mode == "train" else f"{DATASET_TAG}/"
    det_db: dict[str, list[str]] = {}
    n_det = 0
    for vi, video in enumerate(videos, 1):
        vid = video.video_id
        for fid in video.frame_ids:
            img = ds._load_frame(video, fid)
            t = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0).to(device)
            out = detector([t])[0]
            keep = out["scores"] >= score_thresh
            boxes = out["boxes"][keep].cpu().numpy()
            scores = out["scores"][keep].cpu().numpy()
            lines = []
            for (x1, y1, x2, y2), s in zip(boxes, scores):
                w, h = float(x2 - x1), float(y2 - y1)
                if w <= 0 or h <= 0:
                    continue
                lines.append(f"{float(x1):.2f},{float(y1):.2f},{w:.2f},{h:.2f},{float(s):.4f}")
            # export_motrv2.py uses two different layouts and dance.py keys
            # det_db by the exact on-disk path, so both must be reproduced:
            #   train -> train/{dataset}__{sid}/img1/...
            #   eval  -> eval/{dataset}/{sid}/img1/...
            # Getting this wrong raises KeyError on the first lookup.
            det_db[f"{vid_prefix}{vid}/img1/{fid:08d}.txt"] = lines
            n_det += len(lines)
        if vi % 10 == 0 or vi == len(videos):
            print(f"    [{vi}/{len(videos)}] {vid}: {n_det} detections so far")
    return det_db


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--motr-root", required=True)
    ap.add_argument("--release", default=f"{DATA_ROOT}/release/space_tracker")
    ap.add_argument("--score-thresh", type=float, default=0.05,
                    help="Keep the low-scoring tail: MOTRv2 is trained to cope "
                         "with false positives (--fp_ratio) and starving it of "
                         "them changes the task.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-videos", type=int, default=None,
                    help="Smoke-test on the first N sequences of each split.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    detector = build_detector(cfg, Path(args.ckpt), device)
    print(f"Loaded detector from {args.ckpt}")

    motr_root = Path(args.motr_root)
    for split, mode in SPLIT_TO_MODE.items():
        det_db = run_split(detector, cfg, Path(args.release), split, mode,
                           args.score_thresh, device, args.limit_videos)
        # Prefix the keys with the MOTRv2 split directory name.
        det_db = {f"{mode}/{k}": v for k, v in det_db.items()}
        suffix = "_smoke" if args.limit_videos else ""
        out = motr_root / f"det_db_{mode}_frcnn{suffix}.json"
        with open(out, "w") as f:
            json.dump(det_db, f)
        total = sum(len(v) for v in det_db.values())
        print(f"  -> {out}  ({len(det_db)} frames, {total} proposals, "
              f"{total / max(len(det_db), 1):.1f} per frame)")


if __name__ == "__main__":
    main()
