"""
Cross-dataset Faster R-CNN fine-tuning on satellite MOT detection.

Trains on the train splits of LMOD + SAT-MTB + VISO + SDM-Car + AIR-MOT
unified into a single ConcatDataset, with a 4-class taxonomy
{plane, car, ship, train}. Per-frame detection only — no tracking.

The model uses a small-object-friendly multi-scale anchor pyramid and
relaxed RPN/box IoU thresholds; see `configs/Detection/fasterrcnn_satmot.yaml`
and the bbox-stats reports under `docs/bbox_stats/` for the rationale.

Validation during training is **pooled** across all 5 datasets (single mAP).
Per-dataset breakdown is produced post-hoc by `eval_fasterrcnn_satmot.py`
on the resulting checkpoint.

Usage:
    python training_scripts/train_fasterrcnn_satmot.py \
        --config configs/Detection/fasterrcnn_satmot.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime

import torch
import yaml
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from models import FasterRCNNDetector
from lightning_modules import (
    ObjectDetectionModule,
    DetectionDataModule,
    DataModuleConfig,
    DetectionVisualizationCallback,
)
from transforms import (
    build_satmot_train_transform,
    build_satmot_eval_transform,
)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset Faster R-CNN sat-MOT training")
    parser.add_argument(
        "--config",
        default="configs/Detection/fasterrcnn_satmot.yaml",
        help="Path to YAML config",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    torch.set_float32_matmul_precision("high")

    run_name = cfg["run_name"]
    # EXPERIMENT_ROOT env overrides the (anonymised) config path at runtime.
    exp_root = os.environ.get("EXPERIMENT_ROOT") or cfg.get(
        "experiment_root", "/work/anon/experiments")
    experiment_dir = f"{exp_root}/{run_name}_{datetime.now():%Y%m%d_%H%M%S}"

    # ------------------------------------------------------------------
    # Data — ConcatDataset across all 5 datasets, pooled val/test.
    # ------------------------------------------------------------------
    dm_cfg = DataModuleConfig(
        datasets=cfg["datasets"],
        class_map=cfg["class_map"],
        batch_size=cfg.get("batch_size", 4),
        num_workers=cfg.get("num_workers", 0),
        dataset_kwargs=cfg.get("dataset_kwargs", {}),
        per_dataset_kwargs=cfg.get("per_dataset_kwargs", {}),
    )
    dm = DetectionDataModule(
        cfg=dm_cfg,
        train_transform=build_satmot_train_transform(
            crop_size=cfg.get("train_crop_size", 1024),
        ),
        eval_transform=build_satmot_eval_transform(),
    )

    # ------------------------------------------------------------------
    # Model — multi-scale anchor pyramid + relaxed RPN thresholds.
    # ------------------------------------------------------------------
    anchor_sizes = tuple(tuple(s) for s in cfg["anchor_sizes"])
    anchor_aspect_ratios = tuple(tuple(r) for r in cfg["anchor_aspect_ratios"])

    model = FasterRCNNDetector(
        num_classes=cfg["num_classes"],
        pretrained=cfg.get("pretrained", True),
        use_v2=cfg.get("use_v2", True),
        trainable_backbone_layers=cfg.get("trainable_backbone_layers", 3),
        score_thresh=cfg.get("score_thresh", 0.05),
        nms_thresh=cfg.get("nms_thresh", 0.5),
        detections_per_img=cfg.get("detections_per_img", 300),
        enable_tracking=False,
        anchor_sizes=anchor_sizes,
        anchor_aspect_ratios=anchor_aspect_ratios,
        rpn_fg_iou_thresh=cfg.get("rpn_fg_iou_thresh"),
        rpn_bg_iou_thresh=cfg.get("rpn_bg_iou_thresh"),
        box_fg_iou_thresh=cfg.get("box_fg_iou_thresh"),
        box_bg_iou_thresh=cfg.get("box_bg_iou_thresh"),
        rpn_pre_nms_top_n_train=cfg.get("rpn_pre_nms_top_n_train"),
        rpn_post_nms_top_n_train=cfg.get("rpn_post_nms_top_n_train"),
        min_size=cfg.get("min_size"),
        max_size=cfg.get("max_size"),
    )

    module = ObjectDetectionModule(
        model=model,
        has_tracking=False,
        lr=cfg.get("lr", 5e-4),
        weight_decay=cfg.get("weight_decay", 1e-4),
        lr_scheduler=cfg.get("lr_scheduler", "cosine"),
        warmup_epochs=cfg.get("warmup_epochs", 3),
        total_epochs=cfg.get("max_epochs", 30),
    )

    # ------------------------------------------------------------------
    # Logger & callbacks.
    # ------------------------------------------------------------------
    logger = WandbLogger(
        project=cfg.get("wandb_project", "esa-dlstem"),
        # WANDB_ENTITY env overrides the (anonymised) config entity at runtime.
        entity=os.environ.get("WANDB_ENTITY") or cfg.get("wandb_entity", "anonymous"),
        name=run_name,
        log_model=False,
    )

    # FasterRCNN class id → name (1-indexed; 0=background).
    # `plane` and `airplane` both alias to id=1; keep `airplane` as the
    # canonical display name and drop the `plane` alias.
    class_names = {v: k for k, v in cfg["class_map"].items() if k != "plane"}

    callbacks = [
        ModelCheckpoint(
            dirpath=f"{experiment_dir}/checkpoints",
            monitor=cfg.get("monitor_metric", "val/mAP"),
            mode=cfg.get("monitor_mode", "max"),
            save_top_k=1,
            filename="best-epoch={epoch}-val_mAP={val/mAP:.3f}",
            auto_insert_metric_name=False,
        ),
        EarlyStopping(
            monitor=cfg.get("monitor_metric", "val/mAP"),
            mode=cfg.get("monitor_mode", "max"),
            patience=cfg.get("patience", 8),
        ),
    ]
    # Visualization is opt-out via `skip_visualization: true`. When skipped, only
    # the best checkpoint is kept; per-frame predicted boxes are dumped separately
    # by the eval-dump step (no JPEG/W&B image saving).
    if not cfg.get("skip_visualization", False):
        callbacks.append(
            DetectionVisualizationCallback(
                class_names=class_names,
                output_dir=experiment_dir,
                iou_thresh=cfg.get("visualization_iou_thresh", 0.5),
                max_wandb_images=cfg.get("visualization_max_wandb_images", 50),
                score_thresh=cfg.get("visualization_score_thresh", 0.5),
            )
        )

    # ------------------------------------------------------------------
    # Trainer.
    # ------------------------------------------------------------------
    trainer = L.Trainer(
        max_epochs=cfg.get("max_epochs", 30),
        accelerator="auto",
        devices=1,
        precision=cfg.get("precision", "16-mixed"),
        logger=logger,
        callbacks=callbacks,
        default_root_dir=experiment_dir,
        log_every_n_steps=10,
    )

    print("=" * 72)
    print(f"Cross-dataset FasterRCNN sat-MOT training: {run_name}")
    print(f"  Datasets: {list(cfg['datasets'].keys())}")
    print(f"  Output:   {experiment_dir}")
    print(f"  Anchors:  {anchor_sizes}")
    print(f"  Min/Max:  {cfg.get('min_size')} / {cfg.get('max_size')}")
    print("=" * 72)

    trainer.fit(module, datamodule=dm)
    # Pooled test on best checkpoint at the end. Per-dataset breakdown is
    # produced separately by eval_fasterrcnn_satmot.py.
    trainer.test(module, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
