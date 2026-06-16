"""
Unified SiamRPN++ (CVPR 2019) evaluation script.

SiamRPN++ is the classical ResNet-50 Siamese-RPN tracker from SenseTime
(pysot). Runs frame-by-frame with a cached template, produces HBB outputs.
Uses pysot's stock `ModelBuilder` + `SiamRPNTracker` via our thin wrapper.

Usage:
    python eval_siamrpn.py --config configs/SOT/siamrpn_satsot.yaml
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

from models.siamrpn import SiamRPNPPTracker
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
    parser = argparse.ArgumentParser(description="SiamRPN++ evaluation")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.set_float32_matmul_precision("high")

    dataset_name = cfg["dataset"]
    variant = cfg.get("variant", "r50_l234_dwxcorr")
    prompt_strategy = cfg.get("prompt_strategy", "first_frame")
    prompt_interval = cfg.get("prompt_interval", 10)

    run_name = f"siamrpn_{variant}_{prompt_strategy}_{dataset_name.lower()}"
    if prompt_strategy == "every_n":
        run_name = f"siamrpn_{variant}_every{prompt_interval}_{dataset_name.lower()}"

    exp_root = os.environ.get("EXPERIMENT_ROOT", "/work/ziwen/experiments")
    experiment_dir = f"{exp_root}/{run_name}_{datetime.now():%Y%m%d_%H%M%S}"

    img_size = cfg.get("img_size")
    eval_transform = build_eval_transform(tuple(img_size)) if img_size else None

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

    tracker = SiamRPNPPTracker(
        yaml_path=cfg["yaml_path"],
        ckpt_path=cfg["ckpt_path"],
    )

    eval_mode = cfg.get("eval_mode", "sot")
    sot_mode = eval_mode == "sot"

    module = VideoTrackerEvaluationModule(
        model=tracker,
        prompt_strategy=prompt_strategy,
        prompt_interval=prompt_interval,
        sot_mode=sot_mode,
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
            sot_mode=sot_mode,
        ),
    ]

    if sot_mode:
        callbacks.append(
            SAM2SOTEvalCallback(
                class_names=class_names,
                output_dir=experiment_dir,
                score_thresh=cfg.get("score_thresh", 0.5),
                obb_eval_mode=cfg.get("obb_eval_mode", "polygon"),
            )
        )

    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=callbacks,
        default_root_dir=experiment_dir,
    )

    print("=" * 60)
    print(f"SiamRPN++ Evaluation: {variant} | {prompt_strategy} | {dataset_name} | "
          f"{'native' if img_size is None else f'{img_size[0]}x{img_size[1]}'}")
    print(f"Output: {experiment_dir}")
    print("=" * 60)

    trainer.test(module, datamodule=dm)


if __name__ == "__main__":
    main()
