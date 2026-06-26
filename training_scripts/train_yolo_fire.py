"""
YOLOv11 fine-tuning on the RGBT-3M wildfire dataset (detection).

Single dataset (FireRGBT), 3 classes {smoke, fire, person}. Trains on the
train split, validates on val (video2, carved from the official train side),
tests on the official test split. Apple-to-apple companion to
train_fasterrcnn_fire.py — same data, same classes, same 640² input.

Usage:
    CUDA_VISIBLE_DEVICES=1 python training_scripts/train_yolo_fire.py \\
        --config configs/Detection/yolo11_fire.yaml
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
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from models import YOLODetector
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
    parser = argparse.ArgumentParser(description="YOLOv11 FireRGBT detection training")
    parser.add_argument(
        "--config",
        default="configs/Detection/yolo11_fire.yaml",
        help="Path to YAML config",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    torch.set_float32_matmul_precision("high")

    run_name = cfg["run_name"]
    exp_root = cfg.get("experiment_root", "/work/anon/experiments")
    experiment_dir = f"{exp_root}/{run_name}_{datetime.now():%Y%m%d_%H%M%S}"

    img_size = cfg.get("img_size", 640)
    img_size_tuple = (img_size, img_size)

    # ------------------------------------------------------------------
    # Data — single dataset (FireRGBT), train/val/test splits.
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
    # Model
    # ------------------------------------------------------------------
    model = YOLODetector(
        model_name=cfg.get("model_name", "yolo11l.pt"),
        num_classes=cfg["num_classes"],
        enable_tracking=False,
        conf_thresh=cfg.get("conf_thresh", 0.05),
        iou_thresh=cfg.get("iou_thresh", 0.5),
        img_size=img_size,
    )

    module = ObjectDetectionModule(
        model=model,
        has_tracking=False,
        lr=cfg.get("lr", 1e-3),
        weight_decay=cfg.get("weight_decay", 5e-4),
        lr_scheduler=cfg.get("lr_scheduler", "cosine"),
        warmup_epochs=cfg.get("warmup_epochs", 5),
        total_epochs=cfg.get("max_epochs", 50),
        optimizer=cfg.get("optimizer", "adamw"),
        momentum=cfg.get("momentum", 0.937),
    )

    # ------------------------------------------------------------------
    # Logger & callbacks.
    # ------------------------------------------------------------------
    # WandB for the dashboard + a local CSVLogger so per-epoch train/loss_*
    # and val/mAP are readable on disk (metrics.csv) without scraping the rich
    # progress bar — the only place those numbers showed up before.
    logger = [
        WandbLogger(
            project=cfg.get("wandb_project", "esa-dlstem"),
            entity=cfg.get("wandb_entity", "anonymous"),
            name=run_name,
            log_model=False,
        ),
        CSVLogger(save_dir=experiment_dir, name="metrics"),
    ]

    # YOLO class id → name (0-indexed).
    class_names = {v: k for k, v in cfg["class_map"].items()}

    callbacks = [
        ModelCheckpoint(
            dirpath=f"{experiment_dir}/checkpoints",
            monitor=cfg.get("monitor_metric", "val/mAP"),
            mode=cfg.get("monitor_mode", "max"),
            save_top_k=1,
            filename="best-epoch={epoch}-val_mAP={val/mAP:.3f}",
        ),
        EarlyStopping(
            monitor=cfg.get("monitor_metric", "val/mAP"),
            mode=cfg.get("monitor_mode", "max"),
            patience=cfg.get("patience", 10),
        ),
        DetectionVisualizationCallback(
            class_names=class_names,
            output_dir=experiment_dir,
            iou_thresh=cfg.get("visualization_iou_thresh", 0.5),
            max_wandb_images=cfg.get("visualization_max_wandb_images", 50),
            score_thresh=cfg.get("visualization_score_thresh", 0.5),
        ),
    ]

    trainer = L.Trainer(
        max_epochs=cfg.get("max_epochs", 50),
        accelerator="auto",
        devices=1,
        precision=cfg.get("precision", "16-mixed"),
        gradient_clip_val=cfg.get("gradient_clip_val", 10.0),
        logger=logger,
        callbacks=callbacks,
        default_root_dir=experiment_dir,
        log_every_n_steps=10,
    )

    print("=" * 72)
    print(f"YOLOv11 FireRGBT detection training: {run_name}")
    print(f"  Model:    {cfg.get('model_name')}")
    print(f"  Classes:  {cfg['class_map']}")
    print(f"  imgsz:    {img_size}")
    print(f"  Output:   {experiment_dir}")
    print("=" * 72)

    trainer.fit(module, datamodule=dm)
    trainer.test(module, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
