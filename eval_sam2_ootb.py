"""
Evaluation script: SAM2 on OOTB dataset.

Runs two evaluations:
  1. first_frame — prompt with GT boxes from frame 0 only
  2. every_n    — re-prompt with GT boxes every N frames

Usage:
    python eval_sam2_ootb.py
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
    SAM2SOTEvalCallback,
)
from transforms import build_eval_transform

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CLASS_MAP = {"car": 0, "plane": 1, "ship": 2, "train": 3}
CLASS_NAMES = {v: k for k, v in CLASS_MAP.items()}

OOTB_ROOT = "/data/ESA_DLSTEM_2025/data/trafic/OOTB"
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

    run_name = f"sam2_{prompt_strategy}_ootb"
    if prompt_strategy == "every_n":
        run_name = f"sam2_every{prompt_interval}_ootb"

    experiment_dir = f"/work/ziwen/experiments/{run_name}_{datetime.now():%Y%m%d_%H%M%S}"

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    dm = SAM2DataModule(
        cfg=SAM2DataModuleConfig(
            datasets={"OOTB": OOTB_ROOT},
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
        sot_mode=True,  # SOT evaluation — skip AP/MOTA/IDF1
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
    # Callbacks
    # ------------------------------------------------------------------
    callbacks = [
        SAM2VisualizationCallback(
            class_names=CLASS_NAMES,
            output_dir=experiment_dir,
            iou_thresh=0.3,
            max_wandb_images=50,
            score_thresh=0.5,
            sot_mode=True,
        ),
        SAM2SOTEvalCallback(
            class_names=CLASS_NAMES,
            output_dir=experiment_dir,
            score_thresh=0.5,
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
    print("SAM2 Evaluation: first_frame prompt strategy")
    print("=" * 60)
    run_evaluation(prompt_strategy="first_frame")

    # Experiment 2: re-prompt every N frames
    # print("=" * 60)
    # print(f"SAM2 Evaluation: every_{PROMPT_INTERVAL} prompt strategy")
    # print("=" * 60)
    # run_evaluation(prompt_strategy="every_n", prompt_interval=PROMPT_INTERVAL)


if __name__ == "__main__":
    main()
