"""
Per-dataset evaluation of the cross-dataset Faster R-CNN satellite-MOT model.

Loads the best checkpoint from a `train_fasterrcnn_satmot.py` run and
evaluates it on each of the 5 satellite MOT datasets separately
(LMOD, SAT-MTB, VISO, SDM-Car, AIR-MOT) and on the pooled aggregate.
Each test pass writes its own `pr_curve.json` / `pr_curve.png` and emits
a per-dataset row to a summary CSV.

Validation-split AP is also reported per-dataset so train/val/test can be
compared at the same granularity.

Usage:
    python eval_fasterrcnn_satmot.py \
        --config configs/Detection/fasterrcnn_satmot.yaml \
        --ckpt   /work/ziwen/experiments/fasterrcnn_satmot_<TS>/checkpoints/best-*.ckpt
"""

from __future__ import annotations

# --- repo root on path so top-level modules (transforms, obb_utils, ...) import ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from datetime import datetime

import torch
import yaml
import lightning as L

from models import FasterRCNNDetector
from lightning_modules import (
    ObjectDetectionModule,
    DetectionDataModule,
    DataModuleConfig,
)
from transforms import build_satmot_eval_transform


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model_from_cfg(cfg: dict) -> FasterRCNNDetector:
    """Identical model architecture to the training run — must match for
    checkpoint loading to succeed."""
    anchor_sizes = tuple(tuple(s) for s in cfg["anchor_sizes"])
    anchor_aspect_ratios = tuple(tuple(r) for r in cfg["anchor_aspect_ratios"])
    return FasterRCNNDetector(
        num_classes=cfg["num_classes"],
        pretrained=False,                       # weights come from checkpoint
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


def evaluate(
    cfg: dict,
    ckpt_path: str,
    datasets_subset: dict[str, str],
    out_dir: Path,
    label: str,
) -> dict:
    """Run val + test eval over the given dataset subset. Returns logged metrics."""
    out_dir.mkdir(parents=True, exist_ok=True)

    dm_cfg = DataModuleConfig(
        datasets=datasets_subset,
        class_map=cfg["class_map"],
        batch_size=cfg.get("batch_size", 4),
        num_workers=cfg.get("num_workers", 0),
        dataset_kwargs=cfg.get("dataset_kwargs", {}),
        per_dataset_kwargs=cfg.get("per_dataset_kwargs", {}),
    )
    dm = DetectionDataModule(cfg=dm_cfg, eval_transform=build_satmot_eval_transform())

    model = build_model_from_cfg(cfg)
    module = ObjectDetectionModule.load_from_checkpoint(
        ckpt_path,
        model=model,
        has_tracking=False,
    )

    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        default_root_dir=str(out_dir),
        precision=cfg.get("precision", "16-mixed"),
    )

    print(f"\n{'=' * 60}\n[eval] {label} — {list(datasets_subset.keys())}\n{'=' * 60}")
    val_metrics  = trainer.validate(module, datamodule=dm)[0]
    test_metrics = trainer.test(module, datamodule=dm)[0]

    return {
        "label": label,
        "datasets": ",".join(datasets_subset.keys()),
        "val_mAP":     val_metrics.get("val/mAP"),
        "val_Precision": val_metrics.get("val/Precision"),
        "val_Recall":  val_metrics.get("val/Recall"),
        "val_F1":      val_metrics.get("val/F1"),
        "test_mAP":    test_metrics.get("test/mAP"),
        "test_AP_overall": test_metrics.get("test/AP_overall"),
        "test_Precision": test_metrics.get("test/Precision"),
        "test_Recall": test_metrics.get("test/Recall"),
        "test_F1":     test_metrics.get("test/F1"),
        "test_fps":    test_metrics.get("test/fps"),
    }


def main():
    parser = argparse.ArgumentParser(description="Per-dataset eval of cross-dataset Faster R-CNN")
    parser.add_argument("--config", default="configs/Detection/fasterrcnn_satmot.yaml")
    parser.add_argument("--ckpt", required=True, help="Path to .ckpt from training run")
    parser.add_argument(
        "--out-root",
        default=None,
        help="Directory for per-dataset PR curves + summary CSV "
             "(default: <ckpt_dir>/../eval_<timestamp>)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.set_float32_matmul_precision("high")

    ckpt_path = Path(args.ckpt)
    if args.out_root:
        out_root = Path(args.out_root)
    else:
        out_root = ckpt_path.parent.parent / f"eval_{datetime.now():%Y%m%d_%H%M%S}"

    rows = []
    # 5 per-dataset passes
    for name, root in cfg["datasets"].items():
        rows.append(evaluate(
            cfg=cfg,
            ckpt_path=str(ckpt_path),
            datasets_subset={name: root},
            out_dir=out_root / name.replace("/", "_"),
            label=name,
        ))
    # Pooled pass
    rows.append(evaluate(
        cfg=cfg,
        ckpt_path=str(ckpt_path),
        datasets_subset=cfg["datasets"],
        out_dir=out_root / "POOLED",
        label="POOLED",
    ))

    # Summary CSV
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "summary.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: (float(v) if hasattr(v, "item") else v) for k, v in r.items()})

    print("\n" + "=" * 60)
    print(f"Per-dataset evaluation summary → {csv_path}")
    print("=" * 60)
    for r in rows:
        print(
            f"  {r['label']:>10s}: "
            f"val_mAP={r['val_mAP']:.4f}  test_mAP={r['test_mAP']:.4f}  "
            f"test_P={r['test_Precision']:.3f}  test_R={r['test_Recall']:.3f}"
        )


if __name__ == "__main__":
    main()
