"""
DINOv3 (ViT-B/16) + DETR head fine-tuning on the RGBT-3M wildfire dataset.

The "foundation features + simple head" companion to train_yolo_fire.py and
train_fasterrcnn_fire.py — same data (FireRGBT), same 3 classes
{smoke, fire, person}, same 640² input, same train/val/test splits. The DINOv3
backbone is frozen (LVD-1689M web pretrain); only the lightweight DETR-style
decoder head is trained.

The DINOv3Detector is self-contained: it accepts the SAME [0,1] RGB images and
xyxy-absolute target boxes as FasterRCNN, applying ImageNet normalisation and
the xyxy→cxcywh-norm box conversion internally — so it plugs into the standard
DetectionDataModule / ObjectDetectionModule pipeline unchanged.

Usage (GPU0 is occupied by another training job → use GPU1):
    CUDA_VISIBLE_DEVICES=1 python training_scripts/train_dinov3_fire.py \\
        --config configs/Detection/dinov3_fire.yaml
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

from models import DINOv3Detector
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
    parser = argparse.ArgumentParser(description="DINOv3 FireRGBT detection training")
    parser.add_argument(
        "--config",
        default="configs/Detection/dinov3_fire.yaml",
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
        batch_size=cfg.get("batch_size", 16),
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
    # Model — DINOv3 backbone (frozen) + DETR-style detection head.
    # ------------------------------------------------------------------
    model = DINOv3Detector(
        num_classes=cfg["num_classes"],
        hf_model_name=cfg.get("hf_model_name", "facebook/dinov3-vitb16-pretrain-lvd1689m"),
        freeze_backbone=cfg.get("freeze_backbone", True),
        head_type=cfg.get("head_type", "fcos"),
        # DETR head params (only used when head_type == 'detr')
        num_queries=cfg.get("num_queries", 100),
        num_decoder_layers=cfg.get("num_decoder_layers", 3),
        nhead=cfg.get("nhead", 8),
        # FCOS head params (only used when head_type == 'fcos')
        fcos_num_convs=cfg.get("fcos_num_convs", 4),
        fcos_hidden=cfg.get("fcos_hidden", 256),
        fcos_center_radius=cfg.get("fcos_center_radius", 1.5),
        nms_thresh=cfg.get("nms_thresh", 0.6),
        max_dets=cfg.get("max_dets", 100),
        conf_thresh=cfg.get("conf_thresh", 0.05),
    )

    module = ObjectDetectionModule(
        model=model,
        has_tracking=False,
        optimizer=cfg.get("optimizer", "adamw"),
        lr=cfg.get("lr", 1e-4),
        weight_decay=cfg.get("weight_decay", 1e-4),
        lr_scheduler=cfg.get("lr_scheduler", "cosine"),
        warmup_epochs=cfg.get("warmup_epochs", 5),
        total_epochs=cfg.get("max_epochs", 50),
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

    # 0-indexed class id → name (no background).
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
            patience=cfg.get("patience", 10),
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
        max_epochs=cfg.get("max_epochs", 50),
        accelerator="auto",
        devices=1,
        precision=cfg.get("precision", "bf16-mixed"),
        logger=logger,
        callbacks=callbacks,
        default_root_dir=experiment_dir,
        log_every_n_steps=10,
        gradient_clip_val=cfg.get("gradient_clip_val", None),
    )

    print("=" * 72)
    print(f"DINOv3 FireRGBT detection training: {run_name}")
    print(f"  Backbone: {cfg.get('hf_model_name')} (frozen={cfg.get('freeze_backbone', True)})")
    print(f"  Classes:  {cfg['class_map']}")
    print(f"  imgsz:    {img_size}")
    print(f"  Output:   {experiment_dir}")
    print("=" * 72)

    trainer.fit(module, datamodule=dm)
    trainer.test(module, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
