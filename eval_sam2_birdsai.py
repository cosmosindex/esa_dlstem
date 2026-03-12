"""
Evaluation script: SAM2 on BIRDSAI MOT dataset.

SAM2 receives GT bounding boxes as prompts and propagates all tracks
across the clip.  Multiple objects are tracked simultaneously.

Runs two evaluations:
  1. first_frame — prompt with GT boxes from frame 0 only
  2. every_n    — re-prompt with GT boxes every N frames

Usage:
    python eval_sam2_birdsai.py
"""

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


def run_evaluation(
    prompt_strategy: str,
    prompt_interval: int = 10,
):
    """Run SAM2 evaluation with the given prompt strategy."""
    torch.set_float32_matmul_precision("high")

    run_name = f"sam2_{prompt_strategy}_birdsai_mot"
    if prompt_strategy == "every_n":
        run_name = f"sam2_every{prompt_interval}_birdsai_mot"

    experiment_dir = f"/work/ziwen/experiments/{run_name}_{datetime.now():%Y%m%d_%H%M%S}"

    # ------------------------------------------------------------------
    # Data — use BIRDSAI_MOT (multi-object CSV annotations)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    tracker = SAM2Tracker(model_id=SAM2_MODEL_ID)

    module = SAM2EvaluationModule(
        model=tracker,
        prompt_strategy=prompt_strategy,
        prompt_interval=prompt_interval,
    )

    # ------------------------------------------------------------------
    # Logger
    # ------------------------------------------------------------------
    logger = WandbLogger(
        project="esa-dlstem",
        entity="chengziwen693",
        name=run_name,
        log_model=False,
    )

    # ------------------------------------------------------------------
    # Callbacks — MOT mode (detection TP/FP/FN + tracking metrics)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Trainer (test only)
    # ------------------------------------------------------------------
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=callbacks,
        default_root_dir=experiment_dir,
    )

    trainer.test(module, datamodule=dm)


def main():
    # Experiment 1: first-frame prompt only
    print("=" * 60)
    print("SAM2 MOT Evaluation on BIRDSAI: first_frame prompt strategy")
    print("=" * 60)
    run_evaluation(prompt_strategy="first_frame")

    # Experiment 2: re-prompt every N frames
    # print("=" * 60)
    # print(f"SAM2 MOT Evaluation on BIRDSAI: every_{PROMPT_INTERVAL} prompt strategy")
    # print("=" * 60)
    # run_evaluation(prompt_strategy="every_n", prompt_interval=PROMPT_INTERVAL)


if __name__ == "__main__":
    main()
