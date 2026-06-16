"""
Evaluation script: YOLO v11 on BIRDSAI MOT dataset.

Loads the best checkpoint from a previous training run and runs test-set
evaluation only (no training).

Usage:
    python eval_yolo_birdsai_mot.py
"""

# --- repo root on path so top-level modules (transforms, obb_utils, ...) import ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datetime import datetime

import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger

from models import YOLODetector
from lightning_modules import (
    ObjectDetectionModule,
    DetectionDataModule,
    DataModuleConfig,
    DetectionVisualizationCallback,
)
from transforms import build_eval_transform

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CLASS_MAP = {"animal": 0, "human": 1}
NUM_CLASSES = len(CLASS_MAP)  # 2
CLASS_NAMES = {v: k for k, v in CLASS_MAP.items()}

BIRDSAI_ROOT = "/data/ESA_DLSTEM_2025/data/wild_animal/BIRDSAI"
IMG_SIZE = (640, 640)

CHECKPOINT = (
    "/work/ziwen/experiments/Wrong/yolo11n_birdsai_mot_20260312_134348"
    "/checkpoints/best-epoch=2-val_AP50=0.000.ckpt"
)

RUN_NAME = "yolo11n_birdsai_mot_eval"

BATCH_SIZE = 8
NUM_WORKERS = 0


def main():
    torch.set_float32_matmul_precision("high")

    experiment_dir = f"/work/ziwen/experiments/{RUN_NAME}_{datetime.now():%Y%m%d_%H%M%S}"

    # ------------------------------------------------------------------
    # Data (test split only)
    # ------------------------------------------------------------------
    dm_cfg = DataModuleConfig(
        datasets={"BIRDSAI_MOT": BIRDSAI_ROOT},
        class_map=CLASS_MAP,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        img_size=IMG_SIZE,
    )

    dm = DetectionDataModule(
        cfg=dm_cfg,
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
    # Checkpoint was saved after Ultralytics fused Conv+BN; fuse here first
    # so the architecture matches before loading state_dict.
    model.model.fuse()

    module = ObjectDetectionModule.load_from_checkpoint(
        CHECKPOINT,
        model=model,
        has_tracking=True,
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
        DetectionVisualizationCallback(
            class_names=CLASS_NAMES,
            output_dir=experiment_dir,
            iou_thresh=0.5,
            max_wandb_images=50,
            score_thresh=0.5,
        ),
    ]

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        logger=logger,
        callbacks=callbacks,
        default_root_dir=experiment_dir,
    )

    trainer.test(module, datamodule=dm)


if __name__ == "__main__":
    main()
