"""
Re-evaluate the trained FasterRCNN SAT-MTB HBB model at native pixel
scale, fixing the eval-time downscale bug introduced by the original
config's ``max_size: 1333`` (which shrank 1080x1920 inputs by ~31%
and pushed small ships sub-pixel).

This script reuses the same model architecture used at training time
and the same best checkpoint, only overriding the GeneralizedRCNN
transform's ``min_size`` / ``max_size`` so test images pass through at
their native resolution (largest test image is 1080x2152 in
SAT-MTB det_hbb test split, so ``max_size=2304`` covers everything).

Usage:
    python eval_fasterrcnn_satmtb_hbb_native.py \\
        --config configs/Detection/fasterrcnn_satmtb_hbb.yaml \\
        --ckpt /work/.../best-*.ckpt
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import yaml
import lightning as L
from lightning.pytorch.loggers import WandbLogger

from models import FasterRCNNDetector
from lightning_modules import (
    ObjectDetectionModule,
    DetectionDataModule,
    DataModuleConfig,
    DetectionVisualizationCallback,
)
from transforms import build_satmot_eval_transform


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--min-size", type=int, default=1024,
                    help="Override of GeneralizedRCNN transform min_size for eval. "
                         "Original training value: 1024.")
    ap.add_argument("--max-size", type=int, default=2304,
                    help="Override of GeneralizedRCNN transform max_size for eval. "
                         "Original (broken) value: 1333. SAT-MTB det_hbb test "
                         "longest side is 2152 → 2304 is safe.")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.set_float32_matmul_precision("high")

    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out_dir) if args.out_dir else (
        ckpt_path.parent.parent /
        f"reeval_native_min{args.min_size}_max{args.max_size}_{datetime.now():%Y%m%d_%H%M%S}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    dm_cfg = DataModuleConfig(
        datasets=cfg["datasets"],
        class_map=cfg["class_map"],
        batch_size=cfg.get("eval_batch_size", 1),  # native-res images can be big
        num_workers=cfg.get("num_workers", 0),
        per_dataset_kwargs=cfg.get("per_dataset_kwargs", {}),
    )
    dm = DetectionDataModule(cfg=dm_cfg, eval_transform=build_satmot_eval_transform())

    anchor_sizes = tuple(tuple(s) for s in cfg["anchor_sizes"])
    anchor_aspect_ratios = tuple(tuple(r) for r in cfg["anchor_aspect_ratios"])
    model = FasterRCNNDetector(
        num_classes=cfg["num_classes"],
        pretrained=False,
        use_v2=cfg.get("use_v2", False),
        trainable_backbone_layers=cfg.get("trainable_backbone_layers", 3),
        score_thresh=cfg.get("score_thresh", 0.05),
        nms_thresh=cfg.get("nms_thresh", 0.5),
        detections_per_img=cfg.get("detections_per_img", 300),
        enable_tracking=False,
        anchor_sizes=anchor_sizes,
        anchor_aspect_ratios=anchor_aspect_ratios,
        rpn_fg_iou_thresh=cfg.get("rpn_fg_iou_thresh"),
        rpn_bg_iou_thresh=cfg.get("rpn_bg_iou_thresh"),
        box_fg_iou_thresh=cfg.get("box_fg_iou_thresh"),
        box_bg_iou_thresh=cfg.get("box_bg_iou_thresh"),
        rpn_pre_nms_top_n_train=cfg.get("rpn_pre_nms_top_n_train"),
        rpn_post_nms_top_n_train=cfg.get("rpn_post_nms_top_n_train"),
        # === The fix: use native-scale resize bounds for eval ===
        min_size=args.min_size,
        max_size=args.max_size,
    )
    module = ObjectDetectionModule.load_from_checkpoint(
        str(ckpt_path), model=model, has_tracking=False,
    )

    class_names = {v: k for k, v in cfg["class_map"].items() if k != "plane"}

    callbacks = [
        DetectionVisualizationCallback(
            class_names=class_names,
            output_dir=str(out_dir),
            iou_thresh=cfg.get("visualization_iou_thresh", 0.5),
            max_wandb_images=cfg.get("visualization_max_wandb_images", 50),
            score_thresh=cfg.get("visualization_score_thresh", 0.5),
        ),
    ]

    logger = (
        False if args.no_wandb else
        WandbLogger(
            project=cfg.get("wandb_project", "esa-dlstem"),
            entity=cfg.get("wandb_entity", "chengziwen693"),
            name=f"{cfg['run_name']}_reeval_native",
            log_model=False,
        )
    )

    trainer = L.Trainer(
        accelerator="auto", devices=1,
        precision=cfg.get("precision", "16-mixed"),
        logger=logger, callbacks=callbacks,
        default_root_dir=str(out_dir),
    )

    print("=" * 72)
    print(f"FasterRCNN SAT-MTB HBB — native-scale re-eval")
    print(f"  ckpt:     {ckpt_path}")
    print(f"  min_size: {args.min_size} (was 1024)")
    print(f"  max_size: {args.max_size} (was 1333 ← causing eval downscale)")
    print(f"  out:      {out_dir}")
    print("=" * 72)

    trainer.test(module, datamodule=dm)


if __name__ == "__main__":
    main()
