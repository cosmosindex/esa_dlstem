"""
Unified SAM3 evaluation script.

Supports any registered dataset via a YAML config file. Mirrors eval_sam2.py
but uses SAM3Tracker + VideoTrackerEvaluationModule.

Usage:
    python eval_sam3.py --config configs/sam3_ootb.yaml
"""

import argparse
from datetime import datetime

import torch
import lightning as L
import yaml
from lightning.pytorch.loggers import WandbLogger

from models import SAM3Tracker, SAM3TextTracker
from lightning_modules import (
    SAM2DataModule,
    SAM2DataModuleConfig,
    VideoTrackerEvaluationModule,
    SAM2VisualizationCallback,
    SAM2SOTEvalCallback,
)
from transforms import build_eval_transform


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="SAM3 evaluation")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.set_float32_matmul_precision("high")

    # --- Names ---
    dataset_name = cfg["dataset"]
    prompt_strategy = cfg.get("prompt_strategy", "first_frame")
    prompt_interval = cfg.get("prompt_interval", 10)

    run_name = f"sam3_{prompt_strategy}_{dataset_name.lower()}"
    if prompt_strategy == "every_n":
        run_name = f"sam3_every{prompt_interval}_{dataset_name.lower()}"

    experiment_dir = f"/work/ziwen/experiments/{run_name}_{datetime.now():%Y%m%d_%H%M%S}"

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
        ),
        eval_transform=eval_transform,
    )

    # --- Model ---
    tracker_type = cfg.get("tracker_type", "box")
    if tracker_type == "text":
        # Class names ordered by their integer id, so label assignment is stable
        class_names_ordered = [
            name for name, _ in sorted(class_map.items(), key=lambda kv: kv[1])
        ]
        tracker = SAM3TextTracker(
            class_names=class_names_ordered,
            label_to_id=class_map,
            checkpoint_path=cfg.get("sam3_checkpoint_path"),
            apply_temporal_disambiguation=cfg.get("apply_temporal_disambiguation", True),
        )
    else:
        tracker = SAM3Tracker(
            checkpoint_path=cfg.get("sam3_checkpoint_path"),
            apply_temporal_disambiguation=cfg.get("apply_temporal_disambiguation", True),
        )

    module = VideoTrackerEvaluationModule(
        model=tracker,
        prompt_strategy=prompt_strategy,
        prompt_interval=prompt_interval,
    )

    # --- Logger ---
    logger = WandbLogger(
        project=cfg.get("wandb_project", "esa-dlstem"),
        entity=cfg.get("wandb_entity", "chengziwen693"),
        name=run_name,
        log_model=False,
    )

    # --- Callbacks ---
    eval_mode = cfg.get("eval_mode", "sot")
    sot_mode = eval_mode == "sot"

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
    print(f"SAM3 Evaluation: {prompt_strategy} | {dataset_name} | "
          f"{'native' if img_size is None else f'{img_size[0]}x{img_size[1]}'}")
    print(f"Output: {experiment_dir}")
    print("=" * 60)

    trainer.test(module, datamodule=dm)


if __name__ == "__main__":
    main()
