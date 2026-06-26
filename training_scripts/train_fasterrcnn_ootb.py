"""
Training script: Faster R-CNN on OOTB dataset.

Usage:
    python train_fasterrcnn_ootb.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime

import torch
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# FasterRCNN expects labels 1..C (0 = background), so num_classes = C + 1
CLASS_MAP = {"car": 1, "plane": 2, "ship": 3, "train": 4}
NUM_CLASSES = len(CLASS_MAP) + 1  # 5 (including background)
CLASS_NAMES = {v: k for k, v in CLASS_MAP.items()}  # {1: "car", 2: "plane", ...}

OOTB_ROOT = "/data/ESA_DLSTEM_2025/data/trafic/OOTB"
IMG_SIZE = (640, 640)

RUN_NAME = "fasterrcnn-v2_ootb_trainable_backbone_layers"

# Training hyperparameters
BATCH_SIZE = 8
NUM_WORKERS = 0  # TODO: set back to 4 after debugging
MAX_EPOCHS = 50
LR = 5e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5


def main():
    torch.set_float32_matmul_precision("high")

    experiment_dir = f"/work/anon/experiments/{RUN_NAME}_{datetime.now():%Y%m%d_%H%M%S}"

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
    model = FasterRCNNDetector(
        num_classes=NUM_CLASSES,
        pretrained=True,
        use_v2=True,
        trainable_backbone_layers=2,
        score_thresh=0.05,
        nms_thresh=0.5,
    )

    module = ObjectDetectionModule(
        model=model,
        has_tracking=False,
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
        entity="anonymous",
        name=RUN_NAME,
        log_model=False,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=f"{experiment_dir}/checkpoints",
            monitor="val/AP50",
            mode="max",
            save_top_k=1,
            filename="best-{epoch}-{val_AP50:.3f}",
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
