#!/usr/bin/env python
"""Dump per-frame detections for the three FireRGBT detectors.

FasterRCNN / YOLO11l / DINOv3+FCOS were evaluated on RGBT-3M (FireRGBT) but only
their aggregate test_metrics.json (small/large buckets) were kept — no per-frame
predictions. This script re-runs inference for all three over the SAME test set
and dumps one predictions JSON so the fine-grained "performance vs object size"
figure (tools/plot_fire_size_trend.py) can be built offline, mirroring the
BIRDSAI size-trend plot.

All boxes/labels are recorded in the canonical 0-indexed class space
{smoke:0, fire:1, person:2} and the 640² eval/model coordinate space (the space
in which mAP was reported). FasterRCNN's 1-indexed labels are remapped (-1).
Predictions are kept down to score >= 0.05; the plot thresholds at 0.5.

Run:
    CUDA_VISIBLE_DEVICES=0 micromamba run -n esa_dlstem \
        python evaluation/eval_fire_detect_dump.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

from models import FasterRCNNDetector, YOLODetector, DINOv3Detector
from lightning_modules import DetectionDataModule, DataModuleConfig
from transforms import build_eval_transform

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG = 640
SCORE_KEEP = 0.05  # keep everything above this; plot thresholds at 0.5
OUT = Path("/work/ziwen/experiments/fire_detect_dump/fire_detect_predictions.json")

# Canonical class space shared by the dump + plot (0-indexed, no background).
CANON = {"smoke": 0, "fire": 1, "person": 2}
CANON_MAP = {v: k for k, v in CANON.items()}

RUNS = {
    "FasterRCNN": "/work/ziwen/experiments/fasterrcnn_fire_20260525_202026",
    "YOLO11l":    "/work/ziwen/experiments/yolo11l_fire_manual_20260611_161635",
    "DINOv3":     "/work/ziwen/experiments/dinov3_vitb16_fire_20260612_122759",
}
CFG = {
    "FasterRCNN": "configs/Detection/fasterrcnn_fire.yaml",
    "YOLO11l":    "configs/Detection/yolo11_fire.yaml",
    "DINOv3":     "configs/Detection/dinov3_fire.yaml",
}


def load_yaml(p):
    with open(p) as f:
        return yaml.safe_load(f)


def build_test_loader():
    """One canonical 0-indexed test loader (images + GT), shuffle=False."""
    cfg = load_yaml(CFG["YOLO11l"])  # 0-indexed class map
    dm_cfg = DataModuleConfig(
        datasets=cfg["datasets"], class_map=CANON,
        batch_size=8, num_workers=0, img_size=(IMG, IMG),
        per_dataset_kwargs=cfg.get("per_dataset_kwargs", {}),
    )
    dm = DetectionDataModule(cfg=dm_cfg, train_transform=build_eval_transform((IMG, IMG)),
                             eval_transform=build_eval_transform((IMG, IMG)))
    dm.setup("test")
    return dm.test_dataloader()


def strip_prefix(sd, prefix="model."):
    out = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
    return out


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
    ckpt = next(Path(RUNS["FasterRCNN"], "checkpoints").glob("best-*.ckpt"))
    sd = torch.load(ckpt, map_location="cpu")["state_dict"]
    m.load_state_dict(strip_prefix(sd), strict=True)
    return m, +(-1)  # label offset: FRCNN 1-indexed -> canonical 0-indexed


def build_dinov3():
    c = load_yaml(CFG["DINOv3"])
    m = DINOv3Detector(
        num_classes=c["num_classes"], hf_model_name=c.get("hf_model_name"),
        freeze_backbone=True, head_type=c.get("head_type", "fcos"),
        fcos_num_convs=c.get("fcos_num_convs", 4), fcos_hidden=c.get("fcos_hidden", 256),
        fcos_center_radius=c.get("fcos_center_radius", 1.5), fcos_feat_stride=c.get("fcos_feat_stride"),
        nms_thresh=c.get("nms_thresh", 0.6), max_dets=c.get("max_dets", 100),
        conf_thresh=SCORE_KEEP,
    )
    ckpt = next(Path(RUNS["DINOv3"], "checkpoints").glob("best-*.ckpt"))
    sd = torch.load(ckpt, map_location="cpu")["state_dict"]
    m.load_state_dict(strip_prefix(sd), strict=False)  # frozen HF backbone may be re-loaded
    return m, 0


def build_yolo():
    c = load_yaml(CFG["YOLO11l"])
    m = YOLODetector(model_name=c.get("model_name", "yolo11l.pt"), num_classes=c["num_classes"],
                     enable_tracking=False, conf_thresh=SCORE_KEEP,
                     iou_thresh=c.get("iou_thresh", 0.5), img_size=IMG)
    ckpt = torch.load(Path(RUNS["YOLO11l"], "checkpoints", "best.pt"), map_location="cpu")
    m.load_state_dict(ckpt["model"], strict=True)
    return m, 0


BUILDERS = {"FasterRCNN": build_frcnn, "YOLO11l": build_yolo, "DINOv3": build_dinov3}


@torch.no_grad()
def run_model(name, loader, collect_gt):
    print(f"[{name}] building + loading checkpoint ...", flush=True)
    model, off = BUILDERS[name]()
    model.to(DEVICE).eval()
    preds_all, gt_all = [], []
    amp = torch.bfloat16
    for images, targets in loader:
        imgs = [im.to(DEVICE) for im in images]
        with torch.autocast(device_type="cuda", dtype=amp):
            preds = model(imgs)
        for p in preds:
            pb = p["boxes"].float().cpu().numpy()
            ps = p["scores"].float().cpu().numpy()
            pl = p["labels"].long().cpu().numpy() + off
            preds_all.append({"boxes": np.round(pb, 2).tolist(),
                              "scores": np.round(ps, 4).tolist(),
                              "labels": pl.tolist()})
        if collect_gt:
            for t in targets:
                gt_all.append({"boxes": np.round(t["boxes"].cpu().numpy(), 2).tolist(),
                               "labels": t["labels"].cpu().numpy().tolist()})
    model.to("cpu")
    del model
    torch.cuda.empty_cache()
    print(f"[{name}] {len(preds_all)} frames", flush=True)
    return preds_all, gt_all


def main():
    loader = build_test_loader()
    out = {"meta": {"dataset": "FireRGBT", "split": "test", "img_space": IMG,
                    "class_map": CANON, "score_keep": SCORE_KEEP},
           "gt": None, "models": {}}
    first = True
    for name in ["FasterRCNN", "YOLO11l", "DINOv3"]:
        preds, gt = run_model(name, loader, collect_gt=first)
        out["models"][name] = preds
        if first:
            out["gt"] = gt
            first = False
        # sanity: same frame count
        assert len(preds) == len(out["gt"]), f"{name} frame count mismatch"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f)
    print(f"wrote {OUT}  ({len(out['gt'])} frames, {len(out['models'])} models)")


if __name__ == "__main__":
    main()
