"""
Re-evaluate the trained YOLO11l SAT-MTB HBB model at a larger imgsz so
test images aren't letterboxed to 1024 (which shrinks 1080×1920 inputs
to 1024×576 → cars become ~3 px and ships sub-pixel).

Note: YOLO was *trained* at imgsz=1024, so a much larger eval imgsz
introduces some train/test scale mismatch in the opposite direction.
Empirically YOLO11 anchor-free heads tolerate it, but the cleanest fix
is retraining at imgsz=1920+. We do the cheap re-eval here as a sanity
check first.

Usage:
    python eval_yolo_satmtb_hbb_native.py \\
        --config configs/Detection/yolo11_satmtb_hbb.yaml \\
        --ckpt /work/.../best-*.ckpt --img-size 2048
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import yaml
import lightning as L
from lightning.pytorch.loggers import WandbLogger

from models import YOLODetector
from lightning_modules import (
    ObjectDetectionModule,
    DetectionDataModule,
    DataModuleConfig,
    DetectionVisualizationCallback,
)
from transforms import build_train_transform, build_eval_transform


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--img-size", type=int, default=2048,
                    help="YOLO eval imgsz. Original training value: 1024. "
                         "2048 covers SAT-MTB longest side (2152→2048, 5%% "
                         "downscale).")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.set_float32_matmul_precision("high")

    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out_dir) if args.out_dir else (
        ckpt_path.parent.parent /
        f"reeval_imgsz{args.img_size}_{datetime.now():%Y%m%d_%H%M%S}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    img_size_tuple = (args.img_size, args.img_size)

    dm_cfg = DataModuleConfig(
        datasets=cfg["datasets"],
        class_map=cfg["class_map"],
        batch_size=cfg.get("eval_batch_size", 1),
        num_workers=cfg.get("num_workers", 0),
        img_size=img_size_tuple,
        per_dataset_kwargs=cfg.get("per_dataset_kwargs", {}),
    )
    dm = DetectionDataModule(
        cfg=dm_cfg,
        train_transform=build_train_transform(img_size_tuple),
        eval_transform=build_eval_transform(img_size_tuple),
    )

    model = YOLODetector(
        model_name=cfg.get("model_name", "yolo11n.pt"),
        num_classes=cfg["num_classes"],
        enable_tracking=False,
        conf_thresh=cfg.get("conf_thresh", 0.05),
        iou_thresh=cfg.get("iou_thresh", 0.5),
        img_size=args.img_size,
    )
    # strict=False: ultralytics' YOLO sometimes fuses BN before saving,
    # so the freshly-built (unfused) model has BN running stats that
    # aren't in the checkpoint. Those BN stats are populated during
    # training so they exist on the model but the saved state_dict from
    # ultralytics may drop them. We accept the missing/unexpected keys
    # (verified by inspection: all are BN running_mean/running_var only).
    module = ObjectDetectionModule.load_from_checkpoint(
        str(ckpt_path), model=model, has_tracking=False, strict=False,
    )

    class_names = {v: k for k, v in cfg["class_map"].items()}

    callbacks = [
        DetectionVisualizationCallback(
            class_names=class_names,
            output_dir=str(out_dir),
            iou_thresh=cfg.get("visualization_iou_thresh", 0.5),
            max_wandb_images=cfg.get("visualization_max_wandb_images", 50),
            score_thresh=cfg.get("visualization_score_thresh", 0.5),
        ),
    ]

    logger = (
        False if args.no_wandb else
        WandbLogger(
            project=cfg.get("wandb_project", "esa-dlstem"),
            entity=cfg.get("wandb_entity", "chengziwen693"),
            name=f"{cfg['run_name']}_reeval_imgsz{args.img_size}",
            log_model=False,
        )
    )

    trainer = L.Trainer(
        accelerator="auto", devices=1,
        precision=cfg.get("precision", "16-mixed"),
        gradient_clip_val=cfg.get("gradient_clip_val", 10.0),
        logger=logger, callbacks=callbacks,
        default_root_dir=str(out_dir),
    )

    print("=" * 72)
    print(f"YOLO11l SAT-MTB HBB — native-scale re-eval")
    print(f"  ckpt:    {ckpt_path}")
    print(f"  imgsz:   {args.img_size} (was 1024)")
    print(f"  out:     {out_dir}")
    print("=" * 72)

    trainer.test(module, datamodule=dm)


if __name__ == "__main__":
    main()
