"""
HiEUM evaluation script.

Runs HiEUM (Moving Object Detection in Satellite Videos, TPAMI 2024) on a
project dataset via the standard clip-centric MOT eval pipeline
(``VideoTrackerEvaluationModule`` in ``det_only_mode`` — HiEUM has no
cross-frame identities).

Usage:
    python eval_hieum.py --config configs/MOT/hieum_sdmcar.yaml
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

from models import HiEUMDetector
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
    parser = argparse.ArgumentParser(description="HiEUM evaluation")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.set_float32_matmul_precision("high")

    dataset_name = cfg["dataset"]
    run_name = f"hieum_{dataset_name.lower()}"

    exp_root = os.environ.get("EXPERIMENT_ROOT", "/work/ziwen/experiments")
    experiment_dir = f"{exp_root}/{run_name}_{datetime.now():%Y%m%d_%H%M%S}"

    img_size = cfg.get("img_size")
    eval_transform = build_eval_transform(tuple(img_size)) if img_size else None

    class_map = cfg["class_map"]
    class_names = {v: k for k, v in class_map.items()}

    # Hard guardrail: HiEUM's checkpoint bakes seq_len into conv_std, so
    # the dataloader's clip_len must match. Catch config drift early.
    seq_len = cfg.get("hieum_seq_len", 20)
    clip_len = cfg.get("clip_len", seq_len)
    if clip_len != seq_len:
        raise ValueError(
            f"clip_len ({clip_len}) must equal hieum_seq_len ({seq_len}) — "
            "HiEUM's pretrained checkpoint requires fixed-length clips."
        )

    dm = SAM2DataModule(
        cfg=SAM2DataModuleConfig(
            datasets={dataset_name: cfg["dataset_root"]},
            class_map=class_map,
            clip_len=clip_len,
            clip_stride=cfg.get("clip_stride", 1),
            batch_size=cfg.get("batch_size", 1),
            num_workers=cfg.get("num_workers", 0),
            split=os.environ.get("SOT_SPLIT", cfg.get("split", "test")),
            dataset_kwargs=cfg.get("dataset_kwargs", {}),
        ),
        eval_transform=eval_transform,
    )

    car_label = class_map["car"]

    model = HiEUMDetector(
        checkpoint_path=cfg["hieum_checkpoint_path"],
        seq_len=seq_len,
        image_size=tuple(cfg.get("hieum_image_size", [1024, 1024])),
        layers=cfg.get("hieum_layers", 3),
        thresh=cfg.get("hieum_thresh", 3.0),
        car_label=car_label,
        score_thresh=cfg.get("score_thresh", 0.2),
        nms_iou=cfg.get("hieum_nms_iou", 0.1),
        max_dets=cfg.get("hieum_max_dets", 128),
    )

    match_metric = cfg.get("match_metric", "centroid")
    centroid_dist_thresh = float(cfg.get("centroid_dist_thresh", 5.0))
    score_sweep = cfg.get("score_sweep")  # list[float] | None

    module = VideoTrackerEvaluationModule(
        model=model,
        prompt_strategy="text",   # bypass GT prompt code path (HiEUM is unsupervised)
        sot_mode=False,
        det_only_mode=True,       # no temporal IDs across clips → suppress MOTA / IDF1 / IDsw
        match_metric=match_metric,
        centroid_dist_thresh=centroid_dist_thresh,
        score_sweep=score_sweep,
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
            score_thresh=cfg.get("score_thresh", 0.2),
            sot_mode=False,
            match_metric=match_metric,
            centroid_dist_thresh=centroid_dist_thresh,
            # Detection-only overlay style: drop ``T#``/``GT#`` suffixes
            # (HiEUM has no temporal IDs across clips), and use thinner
            # boxes + smaller font appropriate for small satellite cars.
            det_only_mode=True,
            font_scale=cfg.get("vis_font_scale", 0.30),
            box_thickness=cfg.get("vis_box_thickness", 1),
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
    print(f"HiEUM Evaluation: {dataset_name} | clip_len={clip_len} | "
          f"checkpoint={cfg['hieum_checkpoint_path']}")
    print(f"Match: {match_metric}"
          + (f" (dist<= {centroid_dist_thresh}px)"
             if match_metric == "centroid"
             else f" (iou>= {cfg.get('iou_thresh', 0.3)})"))
    print(f"Output: {experiment_dir}")
    print("=" * 60)

    trainer.test(module, datamodule=dm)


if __name__ == "__main__":
    main()
