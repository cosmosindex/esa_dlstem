"""
Unified GroundingDINO MOT-dataset detection evaluation script.

GroundingDINO is a per-frame text-prompted open-vocabulary detector (no
temporal model). This script reuses the clip-centric VideoTrackerEvaluation
pipeline with ``det_only_mode=True`` so MOT metrics are suppressed and only
detection metrics (AP / AP50 / Precision / Recall / AR_100) are logged.

Usage:
    python eval_grounding_dino_mot.py --config configs/MOT/grounding_dino_airmot.yaml
"""

# --- repo root on path so top-level modules (transforms, obb_utils, ...) import ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import os
from datetime import datetime

import torch
import lightning as L
import yaml
from lightning.pytorch.loggers import WandbLogger

from models import GroundingDINODetector
from lightning_modules import (
    SAM2DataModule,
    SAM2DataModuleConfig,
    VideoTrackerEvaluationModule,
    SAM2VisualizationCallback,
)
from transforms import build_eval_transform


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="GroundingDINO MOT-dataset detection eval")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.set_float32_matmul_precision("high")

    dataset_name = cfg["dataset"]
    run_name = f"grounding_dino_{dataset_name.lower()}"

    exp_root = os.environ.get("EXPERIMENT_ROOT", "/work/ziwen/experiments")
    experiment_dir = f"{exp_root}/{run_name}_{datetime.now():%Y%m%d_%H%M%S}"

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
            split=cfg.get("split", "test"),
            dataset_kwargs=cfg.get("dataset_kwargs", {}),
        ),
        eval_transform=eval_transform,
    )

    # --- Model ---
    class_names_ordered = [name for name, _ in sorted(class_map.items(), key=lambda kv: kv[1])]
    model = GroundingDINODetector(
        config_path=cfg["gdino_config_path"],
        checkpoint_path=cfg["gdino_checkpoint_path"],
        class_names=class_names_ordered,
        label_to_id=class_map,
        box_threshold=cfg.get("box_threshold", 0.35),
        text_threshold=cfg.get("text_threshold", 0.25),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    module = VideoTrackerEvaluationModule(
        model=model,
        prompt_strategy=cfg.get("prompt_strategy", "text"),
        prompt_interval=cfg.get("prompt_interval", 10),
        sot_mode=False,
        det_only_mode=cfg.get("det_only_mode", True),
    )

    logger = WandbLogger(
        project=cfg.get("wandb_project", "esa-dlstem"),
        entity=cfg.get("wandb_entity", "chengziwen693"),
        name=run_name,
        log_model=False,
    )

    callbacks = [
        SAM2VisualizationCallback(
            class_names=class_names,
            output_dir=experiment_dir,
            iou_thresh=cfg.get("iou_thresh", 0.3),
            max_wandb_images=cfg.get("max_wandb_images", 50),
            score_thresh=cfg.get("score_thresh", 0.5),
            sot_mode=False,
        ),
    ]

    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=callbacks,
        default_root_dir=experiment_dir,
    )

    print("=" * 60)
    print(f"GroundingDINO Detection Eval: {dataset_name} | "
          f"{'native' if img_size is None else f'{img_size[0]}x{img_size[1]}'}")
    print(f"Output: {experiment_dir}")
    print("=" * 60)

    trainer.test(module, datamodule=dm)


if __name__ == "__main__":
    main()
