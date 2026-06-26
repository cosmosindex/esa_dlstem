"""
Evaluation script: SAM2 on BIRDSAI dataset.

Supports two evaluation modes:
  - MOT: uses BIRDSAI_MOT dataset (multi-object CSV annotations)
         Metrics: MOTA, IDF1, TP/FP/FN
  - SOT: uses BIRDSAI dataset (single-object tracking splits)
         Metrics: Success AUC, Precision@20

Each mode runs with the first_frame prompt strategy by default.

Usage:
    python eval_sam2_birdsai.py
"""

# --- repo root on path so top-level modules (transforms, obb_utils, ...) import ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime

import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger

from models import SAM2Tracker
from lightning_modules import (
    SAM2DataModule,
    SAM2DataModuleConfig,
    SAM2EvaluationModule,
    SAM2VisualizationCallback,
    SAM2SOTEvalCallback,
)
from transforms import build_eval_transform

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CLASS_MAP = {"animal": 0, "human": 1}
CLASS_NAMES = {0: "animal", 1: "human"}

BIRDSAI_ROOT = "/data/ESA_DLSTEM_2025/data/wild_animal/BIRDSAI"
IMG_SIZE = (640, 640)

# SAM2 model (downloaded from HuggingFace on first run)
SAM2_MODEL_ID = "facebook/sam2.1-hiera-large"

# Clip parameters
CLIP_LEN = 32
CLIP_STRIDE = 1
BATCH_SIZE = 1          # SAM2 processes one clip at a time (memory-heavy)
NUM_WORKERS = 0

# Re-prompting interval for "every_n" strategy
PROMPT_INTERVAL = 10


def run_mot_evaluation(
    prompt_strategy: str = "first_frame",
    prompt_interval: int = 10,
):
    """Run SAM2 MOT evaluation on BIRDSAI_MOT dataset."""
    torch.set_float32_matmul_precision("high")

    run_name = f"sam2_{prompt_strategy}_birdsai_mot"
    if prompt_strategy == "every_n":
        run_name = f"sam2_every{prompt_interval}_birdsai_mot"

    experiment_dir = f"/work/anon/experiments/{run_name}_{datetime.now():%Y%m%d_%H%M%S}"

    dm = SAM2DataModule(
        cfg=SAM2DataModuleConfig(
            datasets={"BIRDSAI_MOT": BIRDSAI_ROOT},
            class_map=CLASS_MAP,
            clip_len=CLIP_LEN,
            clip_stride=CLIP_STRIDE,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
        ),
        eval_transform=build_eval_transform(IMG_SIZE),
    )

    tracker = SAM2Tracker(model_id=SAM2_MODEL_ID)
    module = SAM2EvaluationModule(
        model=tracker,
        prompt_strategy=prompt_strategy,
        prompt_interval=prompt_interval,
        sot_mode=False,  # MOT evaluation — keep AP/MOTA/IDF1
    )

    logger = WandbLogger(
        project="esa-dlstem",
        entity="anonymous",
        name=run_name,
        log_model=False,
    )

    callbacks = [
        SAM2VisualizationCallback(
            class_names=CLASS_NAMES,
            output_dir=experiment_dir,
            iou_thresh=0.5,
            max_wandb_images=50,
            score_thresh=0.5,
            sot_mode=False,  # MOT mode: TP/FP/FN + MOTA/IDF1
        ),
    ]

    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=callbacks,
        default_root_dir=experiment_dir,
    )

    trainer.test(module, datamodule=dm)


def run_sot_evaluation(
    prompt_strategy: str = "first_frame",
    prompt_interval: int = 10,
):
    """Run SAM2 SOT evaluation on BIRDSAI (tracking splits) dataset."""
    torch.set_float32_matmul_precision("high")

    run_name = f"sam2_{prompt_strategy}_birdsai_sot"
    if prompt_strategy == "every_n":
        run_name = f"sam2_every{prompt_interval}_birdsai_sot"

    experiment_dir = f"/work/anon/experiments/{run_name}_{datetime.now():%Y%m%d_%H%M%S}"

    dm = SAM2DataModule(
        cfg=SAM2DataModuleConfig(
            datasets={"BIRDSAI": BIRDSAI_ROOT},
            class_map=CLASS_MAP,
            clip_len=CLIP_LEN,
            clip_stride=CLIP_STRIDE,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
        ),
        eval_transform=build_eval_transform(IMG_SIZE),
    )

    tracker = SAM2Tracker(model_id=SAM2_MODEL_ID)
    module = SAM2EvaluationModule(
        model=tracker,
        prompt_strategy=prompt_strategy,
        prompt_interval=prompt_interval,
        sot_mode=True,  # SOT evaluation — skip AP/MOTA/IDF1
    )

    logger = WandbLogger(
        project="esa-dlstem",
        entity="anonymous",
        name=run_name,
        log_model=False,
    )

    callbacks = [
        SAM2VisualizationCallback(
            class_names=CLASS_NAMES,
            output_dir=experiment_dir,
            iou_thresh=0.5,
            max_wandb_images=50,
            score_thresh=0.5,
            sot_mode=True,  # SOT mode: Success/Precision per sequence
        ),
        SAM2SOTEvalCallback(
            class_names=CLASS_NAMES,
            output_dir=experiment_dir,
            score_thresh=0.5,
        ),
    ]

    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=callbacks,
        default_root_dir=experiment_dir,
    )

    trainer.test(module, datamodule=dm)


def main():
    # --- MOT evaluation ---
    print("=" * 60)
    print("SAM2 MOT Evaluation on BIRDSAI: first_frame prompt strategy")
    print("=" * 60)
    run_mot_evaluation(prompt_strategy="first_frame")

    # --- SOT evaluation ---
    # print("=" * 60)
    # print("SAM2 SOT Evaluation on BIRDSAI: first_frame prompt strategy")
    # print("=" * 60)
    # run_sot_evaluation(prompt_strategy="first_frame")


if __name__ == "__main__":
    main()
