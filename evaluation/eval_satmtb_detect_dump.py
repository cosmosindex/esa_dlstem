#!/usr/bin/env python
"""Dump per-frame detections for the three SAT-MTB HBB detectors.

FasterRCNN / YOLO11l / DINOv3+FCOS are compared on SAT-MTB det/HBB (3 classes
{airplane, ship, train}). Each model is run through the SAME input pipeline it
was *trained* with, because they do not share one:

  * YOLO11l / DINOv3 train on a 1024x1024 square resize (anisotropic; the
    dataloader does it). Their predictions come out in 1024^2 space.
  * FasterRCNN trains on NATIVE-resolution frames -- its config carries no
    ``img_size`` and the dataloader uses ``build_satmot_eval_transform()``
    (no spatial op). torchvision's GeneralizedRCNNTransform then rescales
    aspect-ratio-preserving to ``min_size``/``max_size`` internally and maps
    boxes back, so its predictions are already in native pixels.

Feeding FasterRCNN the 1024^2 square tensor instead (as this script used to)
hands it anisotropically stretched frames it never saw in training -- ~32% of
the test split is non-square, up to 1080x1920 -- and cost it 0.076 mAP.
"Same resolution for everyone" is not the fair protocol when the models were
not trained that way; running each at its own protocol and comparing in a
common output space is.

That common space is NATIVE pixels: square-space predictions are mapped back
via `orig_size`, native-space ones pass through untouched. So:

  * per-object-size buckets use TRUE pixel sizes (a 5 px ship is 5 px), and
  * the dumped boxes are traceable to (video_id, frame_id) in original pixels.

GT is read once from the native loader (no resize round-trip) and every model's
boxes are keyed by (video_id, frame_id), so the two loaders' batch layouts need
not agree. Predictions are kept down to score >= 0.05; downstream plots
threshold at 0.5.

Run (after all three trainings finish):
    EXPERIMENT_ROOT=/work/ziwen/experiments CUDA_VISIBLE_DEVICES=0 \
        micromamba run -n esa_dlstem python evaluation/eval_satmtb_detect_dump.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

from models import FasterRCNNDetector, YOLODetector, DINOv3Detector
from lightning_modules import DetectionDataModule, DataModuleConfig
from transforms import build_eval_transform, build_satmot_eval_transform

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG = 1024
SCORE_KEEP = 0.05  # keep everything above this; plots threshold at 0.5

# Canonical class space shared by the dump + plots (0-indexed, no background).
CANON = {"airplane": 0, "ship": 1, "train": 2}
CANON_MAP = {v: k for k, v in CANON.items()}

EXP_ROOT = os.environ.get("EXPERIMENT_ROOT", "/work/ziwen/experiments")
OUT = Path(EXP_ROOT) / "satmtb_detect_dump" / "satmtb_detect_predictions.json"

CFG = {
    "FasterRCNN": "configs/Detection/fasterrcnn_satmtb_hbb.yaml",
    "YOLO11l":    "configs/Detection/yolo11_satmtb_hbb.yaml",
    "DINOv3":     "configs/Detection/dinov3_satmtb.yaml",
}
# run_name prefixes to glob under EXP_ROOT (newest wins).
RUN_GLOB = {
    "FasterRCNN": "fasterrcnn_satmtb_hbb_2*",
    "YOLO11l":    "yolo11l_satmtb_hbb_manual_2*",
    "DINOv3":     "dinov3_vitb16_satmtb_hbb_2*",
}


def load_yaml(p):
    with open(p) as f:
        return yaml.safe_load(f)


def newest_run(name: str) -> Path:
    cands = sorted(Path(EXP_ROOT).glob(RUN_GLOB[name]))
    if not cands:
        raise FileNotFoundError(f"no run dir for {name} under {EXP_ROOT} ({RUN_GLOB[name]})")
    return cands[-1]


def strip_prefix(sd, prefix="model."):
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def build_square_loader(batch_size=4):
    """1024² square-resize test loader — YOLO11l / DINOv3 training protocol."""
    cfg = load_yaml(CFG["YOLO11l"])  # 0-indexed class map; det_hbb kwargs
    dm_cfg = DataModuleConfig(
        datasets=cfg["datasets"], class_map=CANON,
        batch_size=batch_size, num_workers=0, img_size=(IMG, IMG),
        per_dataset_kwargs=cfg.get("per_dataset_kwargs", {}),
    )
    dm = DetectionDataModule(cfg=dm_cfg,
                             train_transform=build_eval_transform((IMG, IMG)),
                             eval_transform=build_eval_transform((IMG, IMG)))
    dm.setup("test")
    return dm.test_dataloader()


def build_native_loader(batch_size=4):
    """Native-resolution test loader — FasterRCNN training protocol.

    No spatial transform and no ``img_size``: mirrors train_fasterrcnn_satmot.py,
    which lets GeneralizedRCNNTransform do the aspect-preserving resize. Boxes
    (GT and predictions alike) stay in native pixels. Class ids are forced to
    CANON so GT matches the square loader's; the FRCNN head's own 1-indexed
    output is remapped by its label offset.
    """
    cfg = load_yaml(CFG["FasterRCNN"])
    dm_cfg = DataModuleConfig(
        datasets=cfg["datasets"], class_map=CANON,
        batch_size=batch_size, num_workers=0,
        per_dataset_kwargs=cfg.get("per_dataset_kwargs", {}),
    )
    dm = DetectionDataModule(cfg=dm_cfg,
                             train_transform=build_satmot_eval_transform(),
                             eval_transform=build_satmot_eval_transform())
    dm.setup("test")
    return dm.test_dataloader()


def build_frcnn():
    c = load_yaml(CFG["FasterRCNN"])
    m = FasterRCNNDetector(
        num_classes=c["num_classes"], pretrained=False, use_v2=c.get("use_v2", False),
        trainable_backbone_layers=c.get("trainable_backbone_layers", 3),
        score_thresh=SCORE_KEEP, nms_thresh=c.get("nms_thresh", 0.5),
        detections_per_img=c.get("detections_per_img", 300), enable_tracking=False,
        anchor_sizes=tuple(tuple(s) for s in c["anchor_sizes"]),
        anchor_aspect_ratios=tuple(tuple(r) for r in c["anchor_aspect_ratios"]),
        rpn_fg_iou_thresh=c.get("rpn_fg_iou_thresh"), rpn_bg_iou_thresh=c.get("rpn_bg_iou_thresh"),
        box_fg_iou_thresh=c.get("box_fg_iou_thresh"), box_bg_iou_thresh=c.get("box_bg_iou_thresh"),
        min_size=c.get("min_size"), max_size=c.get("max_size"),
    )
    ckpt = next((newest_run("FasterRCNN") / "checkpoints").glob("best-*.ckpt"))
    sd = torch.load(ckpt, map_location="cpu")["state_dict"]
    m.load_state_dict(strip_prefix(sd), strict=True)
    # FRCNN config is 1-indexed {airplane:1,ship:2,train:3}; map to canonical 0-indexed.
    return m, -1, str(ckpt)


def build_yolo():
    c = load_yaml(CFG["YOLO11l"])
    m = YOLODetector(model_name=os.environ.get("YOLO_CKPT") or c.get("model_name", "yolo11l.pt"),
                     num_classes=c["num_classes"], enable_tracking=False,
                     conf_thresh=SCORE_KEEP, iou_thresh=c.get("iou_thresh", 0.5), img_size=IMG)
    ckpt = newest_run("YOLO11l") / "checkpoints" / "best.pt"
    m.load_state_dict(torch.load(ckpt, map_location="cpu")["model"], strict=True)
    return m, 0, str(ckpt)


def build_dinov3():
    c = load_yaml(CFG["DINOv3"])
    m = DINOv3Detector(
        num_classes=c["num_classes"], hf_model_name=c.get("hf_model_name"),
        freeze_backbone=True, head_type=c.get("head_type", "fcos"),
        fcos_num_convs=c.get("fcos_num_convs", 4), fcos_hidden=c.get("fcos_hidden", 256),
        fcos_center_radius=c.get("fcos_center_radius", 1.5), fcos_feat_stride=c.get("fcos_feat_stride"),
        nms_thresh=c.get("nms_thresh", 0.6), max_dets=c.get("max_dets", 300), conf_thresh=SCORE_KEEP,
    )
    ckpt = next((newest_run("DINOv3") / "checkpoints").glob("best-*.ckpt"))
    sd = torch.load(ckpt, map_location="cpu")["state_dict"]
    m.load_state_dict(strip_prefix(sd), strict=False)  # frozen HF backbone re-loaded
    return m, 0, str(ckpt)


BUILDERS = {"FasterRCNN": build_frcnn, "YOLO11l": build_yolo, "DINOv3": build_dinov3}

# Per-model input protocol + autocast dtype, both taken from how each model was
# trained. "native" predictions are already in native pixels; "square" ones are
# in IMG² space and get mapped back. The dtype mirrors each config's
# `precision:` (FRCNN 16-mixed, the other two bf16-mixed).
SPECS = {
    "FasterRCNN": {"space": "native", "amp": torch.float16},
    "YOLO11l":    {"space": "square", "amp": torch.bfloat16},
    "DINOv3":     {"space": "square", "amp": torch.bfloat16},
}


def _to_native(box_xyxy: np.ndarray, orig_hw) -> np.ndarray:
    """Map boxes from IMG×IMG square space back to native (H, W) pixels."""
    if len(box_xyxy) == 0:
        return box_xyxy
    H, W = int(orig_hw[0]), int(orig_hw[1])
    sx, sy = W / IMG, H / IMG
    b = box_xyxy.copy()
    b[:, [0, 2]] *= sx
    b[:, [1, 3]] *= sy
    return b


def _frame_key(t) -> tuple[str, int]:
    return (str(t["video_id"]), int(t["frame_id"]))


@torch.no_grad()
def run_model(name, loader, collect_gt, limit_batches=0):
    """Run one model over its own loader. Returns {(video_id, frame_id): pred}."""
    run_dir = newest_run(name)
    spec = SPECS[name]
    print(f"[{name}] run={run_dir.name}  space={spec['space']}  amp={spec['amp']}", flush=True)
    model, off, ckpt = BUILDERS[name]()
    model.to(DEVICE).eval()
    preds, gts = {}, {}
    for bi, (images, targets) in enumerate(loader):
        if limit_batches and bi >= limit_batches:
            break
        imgs = [im.to(DEVICE) for im in images]
        with torch.autocast(device_type="cuda", dtype=spec["amp"]):
            out = model(imgs)
        for p, t in zip(out, targets):
            oh = t["orig_size"]
            pb = p["boxes"].float().cpu().numpy()
            if spec["space"] == "square":
                pb = _to_native(pb, oh)
            key = _frame_key(t)
            preds[key] = {"boxes": np.round(pb, 2).tolist(),
                          "scores": np.round(p["scores"].float().cpu().numpy(), 4).tolist(),
                          "labels": (p["labels"].long().cpu().numpy() + off).tolist()}
            if collect_gt:
                # Native loader → GT is already in native pixels, no round-trip.
                gts[key] = {"video_id": t["video_id"], "frame_id": int(t["frame_id"]),
                            "orig_hw": [int(oh[0]), int(oh[1])],
                            "gt": {"boxes": np.round(t["boxes"].cpu().numpy(), 2).tolist(),
                                   "labels": t["labels"].cpu().numpy().tolist()}}
    model.to("cpu"); del model; torch.cuda.empty_cache()
    print(f"[{name}] {len(preds)} frames  ckpt={Path(ckpt).name}", flush=True)
    return preds, gts, ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit-batches", type=int, default=0, help="debug: stop after N batches")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    loaders = {"native": build_native_loader(args.batch_size),
               "square": build_square_loader(args.batch_size)}
    out = {"meta": {"dataset": "SAT-MTB", "task": "det_hbb", "split": "test",
                    "square_img": IMG, "box_space": "native_pixels", "class_map": CANON,
                    "score_keep": SCORE_KEEP,
                    "input_protocol": {k: v["space"] for k, v in SPECS.items()}},
           "frames": None, "models": {}, "ckpts": {}}

    # FRCNN runs first and on the native loader, so GT is captured there.
    all_preds, gts = {}, None
    for name in ["FasterRCNN", "YOLO11l", "DINOv3"]:
        preds, g, ckpt = run_model(name, loaders[SPECS[name]["space"]],
                                   collect_gt=gts is None, limit_batches=args.limit_batches)
        all_preds[name] = preds
        out["ckpts"][name] = ckpt
        if gts is None:
            gts = g

    keys = list(gts)  # native-loader order defines frame order
    for name, preds in all_preds.items():
        missing = set(keys) ^ set(preds)
        assert not missing, f"{name}: {len(missing)} frames not aligned with GT"

    out["frames"] = [gts[k] for k in keys]
    out["models"] = {name: [preds[k] for k in keys] for name, preds in all_preds.items()}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"wrote {args.out}  ({len(out['frames'])} frames, {len(out['models'])} models)")


if __name__ == "__main__":
    main()
