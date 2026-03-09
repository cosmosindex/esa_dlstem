"""
Training script: YOLO v11 on OOTB dataset.

Usage:
    python train_yolo_ootb.py
"""

from datetime import datetime

import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from models import YOLODetector
from lightning_modules import (
    ObjectDetectionModule,
    DetectionDataModule,
    DataModuleConfig,
    DetectionVisualizationCallback,
)
from transforms import build_train_transform, build_eval_transform

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# YOLO uses 0-indexed class labels (no background class)
CLASS_MAP = {"car": 0, "plane": 1, "ship": 2, "train": 3}
NUM_CLASSES = len(CLASS_MAP)  # 4
CLASS_NAMES = {v: k for k, v in CLASS_MAP.items()}

OOTB_ROOT = "/data/ESA_DLSTEM_2025/data/trafic/OOTB"
IMG_SIZE = (640, 640)

RUN_NAME = "yolo11n_ootb"

# Training hyperparameters
BATCH_SIZE = 8
NUM_WORKERS = 0  # set to 4 with mp.set_start_method("spawn") if needed
MAX_EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 5e-4
WARMUP_EPOCHS = 5


def main():
    torch.set_float32_matmul_precision("high")

    experiment_dir = f"/work/ziwen/experiments/{RUN_NAME}_{datetime.now():%Y%m%d_%H%M%S}"

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    dm_cfg = DataModuleConfig(
        datasets={"OOTB": OOTB_ROOT},
        class_map=CLASS_MAP,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        img_size=IMG_SIZE,
    )

    dm = DetectionDataModule(
        cfg=dm_cfg,
        train_transform=build_train_transform(IMG_SIZE),
        eval_transform=build_eval_transform(IMG_SIZE),
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = YOLODetector(
        model_name="yolo11n.pt",
        num_classes=NUM_CLASSES,
        enable_tracking=True,
        conf_thresh=0.05,
        iou_thresh=0.5,
        img_size=IMG_SIZE[0],
    )

    module = ObjectDetectionModule(
        model=model,
        has_tracking=True,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler="cosine",
        warmup_epochs=WARMUP_EPOCHS,
        total_epochs=MAX_EPOCHS,
    )

    # ------------------------------------------------------------------
    # Logger & callbacks
    # ------------------------------------------------------------------
    logger = WandbLogger(
        project="esa-dlstem",
        entity="chengziwen693",
        name=RUN_NAME,
        log_model=False,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=f"{experiment_dir}/checkpoints",
            monitor="val/AP50",
            mode="max",
            save_top_k=1,
            filename="best-{epoch}-{val/AP50:.3f}",
        ),
        EarlyStopping(
            monitor="val/AP50",
            mode="max",
            patience=10,
        ),
        DetectionVisualizationCallback(
            class_names=CLASS_NAMES,
            output_dir=experiment_dir,
            iou_thresh=0.5,
            max_wandb_images=50,
            score_thresh=0.5,
        ),
    ]

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        gradient_clip_val=10.0,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
    )

    # ------------------------------------------------------------------
    # Train & test
    # ------------------------------------------------------------------
    trainer.fit(module, datamodule=dm)
    trainer.test(module, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
