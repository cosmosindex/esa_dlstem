"""
Unified SAM2 evaluation script.

Supports any registered dataset via a YAML config file.
Handles both SOT and MOT evaluation modes, native or resized resolution.

Usage:
    python eval_sam2.py --config configs/SOT/sam2_satsot.yaml
    python eval_sam2.py --config configs/SOT/sam2_ootb.yaml
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import torch
import lightning as L
import yaml
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


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="SAM2 evaluation")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.set_float32_matmul_precision("high")

    # --- Names ---
    dataset_name = cfg["dataset"]
    prompt_strategy = cfg.get("prompt_strategy", "first_frame")
    prompt_interval = cfg.get("prompt_interval", 10)

    run_name = f"sam2_{prompt_strategy}_{dataset_name.lower()}"
    if prompt_strategy == "every_n":
        run_name = f"sam2_every{prompt_interval}_{dataset_name.lower()}"

    exp_root = os.environ.get("EXPERIMENT_ROOT", "/work/ziwen/experiments")
    experiment_dir = f"{exp_root}/{run_name}_{datetime.now():%Y%m%d_%H%M%S}"

    # --- Transform ---
    img_size = cfg.get("img_size")
    eval_transform = build_eval_transform(tuple(img_size)) if img_size else None

    # --- Data ---
    class_map = cfg["class_map"]
    class_names = {v: k for k, v in class_map.items()}

    dm = SAM2DataModule(
        cfg=SAM2DataModuleConfig(
            datasets={dataset_name: cfg["dataset_root"]},
            class_map=class_map,
            clip_len=cfg.get("clip_len", 32),
            clip_stride=cfg.get("clip_stride", 1),
            batch_size=cfg.get("batch_size", 1),
            num_workers=cfg.get("num_workers", 0),
            split=os.environ.get("SOT_SPLIT", cfg.get("split", "test")),
        ),
        eval_transform=eval_transform,
    )

    # --- Model ---
    tracker = SAM2Tracker(model_id=cfg.get("sam2_model_id", "facebook/sam2.1-hiera-large"))

    eval_mode = cfg.get("eval_mode", "sot")
    sot_mode = eval_mode == "sot"

    module = SAM2EvaluationModule(
        model=tracker,
        prompt_strategy=prompt_strategy,
        prompt_interval=prompt_interval,
        sot_mode=sot_mode,
    )

    # --- Logger ---
    logger = WandbLogger(
        project=cfg.get("wandb_project", "esa-dlstem"),
        entity=cfg.get("wandb_entity", "chengziwen693"),
        name=run_name,
        log_model=False,
    )

    # --- Callbacks ---

    callbacks = [
        SAM2VisualizationCallback(
            class_names=class_names,
            output_dir=experiment_dir,
            iou_thresh=cfg.get("iou_thresh", 0.3),
            max_wandb_images=cfg.get("max_wandb_images", 50),
            score_thresh=cfg.get("score_thresh", 0.5),
            sot_mode=sot_mode,
        ),
    ]

    if sot_mode:
        callbacks.append(
            SAM2SOTEvalCallback(
                class_names=class_names,
                output_dir=experiment_dir,
                score_thresh=cfg.get("score_thresh", 0.5),
            )
        )

    # --- Trainer ---
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=callbacks,
        default_root_dir=experiment_dir,
    )

    print("=" * 60)
    print(f"SAM2 Evaluation: {prompt_strategy} | {dataset_name} | "
          f"{'native' if img_size is None else f'{img_size[0]}x{img_size[1]}'}")
    print(f"Output: {experiment_dir}")
    print("=" * 60)

    trainer.test(module, datamodule=dm)


if __name__ == "__main__":
    main()
