"""
Unified LoRAT (ECCV 2024) evaluation script.

LoRAT is a DINOv2-backbone one-stream SOT tracker with LoRA adapters on the
ViT. Runs frame-by-frame with a cached template, produces HBB outputs. We
drive it directly via our thin `LoRATTracker` wrapper, bypassing the trackit
framework's data pipeline.

Usage:
    python eval_lorat.py --config configs/lorat_satsot.yaml
"""

import argparse
from datetime import datetime

import torch
import lightning as L
import yaml
from lightning.pytorch.loggers import WandbLogger

from models.lorat import LoRATTracker
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
    parser = argparse.ArgumentParser(description="LoRAT evaluation")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.set_float32_matmul_precision("high")

    dataset_name = cfg["dataset"]
    variant = cfg.get("variant", "g-378")
    prompt_strategy = cfg.get("prompt_strategy", "first_frame")
    prompt_interval = cfg.get("prompt_interval", 10)

    run_name = f"lorat_{variant}_{prompt_strategy}_{dataset_name.lower()}"
    if prompt_strategy == "every_n":
        run_name = f"lorat_{variant}_every{prompt_interval}_{dataset_name.lower()}"

    experiment_dir = f"/work/ziwen/experiments/{run_name}_{datetime.now():%Y%m%d_%H%M%S}"

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
        ),
        eval_transform=eval_transform,
    )

    tracker = LoRATTracker(
        ckpt_path=cfg["ckpt_path"],
        config_name=variant,
        method_name=cfg.get("method_name", "LoRAT"),
        lorat_root=cfg.get("lorat_root"),
    )

    module = VideoTrackerEvaluationModule(
        model=tracker,
        prompt_strategy=prompt_strategy,
        prompt_interval=prompt_interval,
    )

    logger = WandbLogger(
        project=cfg.get("wandb_project", "esa-dlstem"),
        entity=cfg.get("wandb_entity", "chengziwen693"),
        name=run_name,
        log_model=False,
    )

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
    print(f"LoRAT Evaluation: {variant} | {prompt_strategy} | {dataset_name} | "
          f"{'native' if img_size is None else f'{img_size[0]}x{img_size[1]}'}")
    print(f"Output: {experiment_dir}")
    print("=" * 60)

    trainer.test(module, datamodule=dm)


if __name__ == "__main__":
    main()
