"""
Faster R-CNN (R50-FPN) fine-tuning on the RGBT-3M wildfire dataset (detection).

Single dataset (FireRGBT), 3 classes {smoke, fire, person}. Trains on the
train split, validates on val (video2, carved from the official train side),
tests on the official test split. Apple-to-apple companion to
train_yolo_fire.py — same data, same classes, same 640² input.

Unlike train_fasterrcnn_satmot.py (crop@1024 for tiny satellite objects),
this uses the resize-to-640² transforms so train/val/test object scale stays
identical and matches what YOLO sees. min_size=max_size=640 makes the model's
internal GeneralizedRCNNTransform a no-op.

Usage:
    CUDA_VISIBLE_DEVICES=1 python training_scripts/train_fasterrcnn_fire.py \\
        --config configs/Detection/fasterrcnn_fire.yaml
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from models import FasterRCNNDetector
from lightning_modules import (
    ObjectDetectionModule,
    DetectionDataModule,
    DataModuleConfig,
    DetectionVisualizationCallback,
)
from transforms import build_train_transform, build_eval_transform


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Faster R-CNN FireRGBT detection training")
    parser.add_argument(
        "--config",
        default="configs/Detection/fasterrcnn_fire.yaml",
        help="Path to YAML config",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    torch.set_float32_matmul_precision("high")

    run_name = cfg["run_name"]
    exp_root = cfg.get("experiment_root", "/work/ziwen/experiments")
    experiment_dir = f"{exp_root}/{run_name}_{datetime.now():%Y%m%d_%H%M%S}"

    img_size = cfg.get("img_size", 640)
    img_size_tuple = (img_size, img_size)

    # ------------------------------------------------------------------
    # Data — single dataset (FireRGBT), train/val/test splits, resize to 640².
    # ------------------------------------------------------------------
    dm_cfg = DataModuleConfig(
        datasets=cfg["datasets"],
        class_map=cfg["class_map"],
        batch_size=cfg.get("batch_size", 8),
        num_workers=cfg.get("num_workers", 0),
        img_size=img_size_tuple,
        per_dataset_kwargs=cfg.get("per_dataset_kwargs", {}),
    )
    dm = DetectionDataModule(
        cfg=dm_cfg,
        train_transform=build_train_transform(img_size_tuple),
        eval_transform=build_eval_transform(img_size_tuple),
    )

    # ------------------------------------------------------------------
    # Model — small-object-friendly anchor pyramid + relaxed RPN thresholds.
    # ------------------------------------------------------------------
    anchor_sizes = tuple(tuple(s) for s in cfg["anchor_sizes"])
    anchor_aspect_ratios = tuple(tuple(r) for r in cfg["anchor_aspect_ratios"])

    model = FasterRCNNDetector(
        num_classes=cfg["num_classes"],
        pretrained=cfg.get("pretrained", True),
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
        min_size=cfg.get("min_size"),
        max_size=cfg.get("max_size"),
    )

    module = ObjectDetectionModule(
        model=model,
        has_tracking=False,
        lr=cfg.get("lr", 5e-4),
        weight_decay=cfg.get("weight_decay", 1e-4),
        lr_scheduler=cfg.get("lr_scheduler", "cosine"),
        warmup_epochs=cfg.get("warmup_epochs", 3),
        total_epochs=cfg.get("max_epochs", 30),
    )

    # ------------------------------------------------------------------
    # Logger & callbacks.
    # ------------------------------------------------------------------
    logger = WandbLogger(
        project=cfg.get("wandb_project", "esa-dlstem"),
        entity=cfg.get("wandb_entity", "chengziwen693"),
        name=run_name,
        log_model=False,
    )

    # FasterRCNN class id → name (1-indexed; 0 = background).
    class_names = {v: k for k, v in cfg["class_map"].items()}

    callbacks = [
        ModelCheckpoint(
            dirpath=f"{experiment_dir}/checkpoints",
            monitor=cfg.get("monitor_metric", "val/mAP"),
            mode=cfg.get("monitor_mode", "max"),
            save_top_k=1,
            filename="best-epoch={epoch}-val_mAP={val/mAP:.3f}",
            auto_insert_metric_name=False,
        ),
        EarlyStopping(
            monitor=cfg.get("monitor_metric", "val/mAP"),
            mode=cfg.get("monitor_mode", "max"),
            patience=cfg.get("patience", 8),
        ),
    ]
    # Visualization is opt-out via `skip_visualization: true` (BIRDSAI runs dump
    # per-frame predictions to JSON in a separate eval step instead).
    if not cfg.get("skip_visualization", False):
        callbacks.append(
            DetectionVisualizationCallback(
                class_names=class_names,
                output_dir=experiment_dir,
                iou_thresh=cfg.get("visualization_iou_thresh", 0.5),
                max_wandb_images=cfg.get("visualization_max_wandb_images", 50),
                score_thresh=cfg.get("visualization_score_thresh", 0.5),
            )
        )

    trainer = L.Trainer(
        max_epochs=cfg.get("max_epochs", 30),
        accelerator="auto",
        devices=1,
        precision=cfg.get("precision", "16-mixed"),
        logger=logger,
        callbacks=callbacks,
        default_root_dir=experiment_dir,
        log_every_n_steps=10,
    )

    print("=" * 72)
    print(f"Faster R-CNN FireRGBT detection training: {run_name}")
    print(f"  Classes:  {cfg['class_map']}")
    print(f"  Anchors:  {anchor_sizes}")
    print(f"  imgsz:    {img_size} (min/max={cfg.get('min_size')}/{cfg.get('max_size')})")
    print(f"  Output:   {experiment_dir}")
    print("=" * 72)

    trainer.fit(module, datamodule=dm)
    trainer.test(module, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
