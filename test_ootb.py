"""
Test script: evaluate models on the OOTB test set.

Loads best checkpoints, runs full test with visualization and metrics export.
Outputs per experiment:
  - visualizations/  (all test images with SOT-mode boxes)
  - per_image_metrics.json
  - test_metrics.json
  - sot_metrics.json  (Success AUC, Precision@20, plots)

Usage:
    python test_ootb.py                          # test both finetuned models
    python test_ootb.py --model fasterrcnn       # test only FasterRCNN
    python test_ootb.py --model yolo             # test only finetuned YOLO
    python test_ootb.py --model yolo_pretrained  # test YOLO with COCO pretrained weights (no finetune)
"""

import argparse
from datetime import datetime
from pathlib import Path

import torch
import lightning as L

from models import FasterRCNNDetector, YOLODetector
from lightning_modules import (
    ObjectDetectionModule,
    DetectionDataModule,
    DataModuleConfig,
    DetectionVisualizationCallback,
    SOTEvalCallback,
)
from transforms import build_eval_transform

# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------
OOTB_ROOT = "/data/ESA_DLSTEM_2025/data/trafic/OOTB"
IMG_SIZE = (640, 640)
BATCH_SIZE = 8
NUM_WORKERS = 0

# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------
MODELS = {
    "fasterrcnn": {
        "class_map": {"car": 1, "plane": 2, "ship": 3, "train": 4},
        "num_classes": 5,
        "checkpoint": (
            "/work/ziwen/experiments/"
            "fasterrcnn-v2_ootb_trainable_backbone_layers_20260309_205156/"
            "checkpoints/best-epoch=3-val/"
            "AP50=0.291.ckpt"
        ),
        "has_tracking": True,
        "build_model": lambda: FasterRCNNDetector(
            num_classes=5,
            pretrained=False,
            use_v2=True,
            trainable_backbone_layers=2,
            score_thresh=0.05,
            nms_thresh=0.5,
        ),
    },
    "yolo": {
        "class_map": {"car": 0, "plane": 1, "ship": 2, "train": 3},
        "num_classes": 4,
        "checkpoint": (
            "/work/ziwen/experiments/"
            "yolo11n_ootb_20260309_190516/"
            "checkpoints/best-epoch=4-val/"
            "AP50=0.237.ckpt"
        ),
        "has_tracking": True,
        "build_model": lambda: YOLODetector(
            model_name="yolo11n.pt",
            num_classes=4,
            enable_tracking=True,
            conf_thresh=0.05,
            iou_thresh=0.5,
            img_size=IMG_SIZE[0],
        ),
    },
    "yolo_pretrained": {
        # COCO pretrained YOLO (80 classes) — map COCO class ids to OOTB ids.
        # COCO: car=2, airplane=4, boat=8, train=6  →  OOTB: car=0, plane=1, ship=2, train=3
        "coco_to_ootb": {2: 0, 4: 1, 8: 2, 6: 3},
        "class_map": {"car": 0, "plane": 1, "ship": 2, "train": 3},
        "num_classes": 80,
        "checkpoint": None,  # no finetune checkpoint
        "has_tracking": True,
        "build_model": lambda: YOLODetector(
            model_name="yolo11n.pt",
            num_classes=80,  # keep COCO head
            enable_tracking=True,
            conf_thresh=0.05,
            iou_thresh=0.5,
            img_size=IMG_SIZE[0],
        ),
    },
}


def test_model(model_name: str):
    cfg = MODELS[model_name]
    ckpt_path = cfg["checkpoint"]

    if ckpt_path is not None and not Path(ckpt_path).exists():
        print(f"[SKIP] Checkpoint not found: {ckpt_path}")
        return

    print(f"\n{'='*60}")
    print(f"Testing {model_name.upper()}")
    print(f"Checkpoint: {ckpt_path or 'pretrained (no finetune)'}")
    print(f"{'='*60}\n")

    torch.set_float32_matmul_precision("high")

    class_map = cfg["class_map"]
    class_names = {v: k for k, v in class_map.items()}

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/work/ziwen/experiments/{model_name}_test_{timestamp}"

    # ------------------------------------------------------------------
    # Data (test split only)
    # ------------------------------------------------------------------
    dm_cfg = DataModuleConfig(
        datasets={"OOTB": OOTB_ROOT},
        class_map=class_map,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        img_size=IMG_SIZE,
    )
    dm = DetectionDataModule(
        cfg=dm_cfg,
        train_transform=build_eval_transform(IMG_SIZE),
        eval_transform=build_eval_transform(IMG_SIZE),
    )

    # ------------------------------------------------------------------
    # Model + load checkpoint
    # ------------------------------------------------------------------
    model = cfg["build_model"]()

    if ckpt_path is not None:
        # Finetuned checkpoint — fuse BN for YOLO to match saved state_dict
        if "yolo" in model_name:
            model.model.fuse()
        module = ObjectDetectionModule.load_from_checkpoint(
            ckpt_path,
            model=model,
            has_tracking=cfg["has_tracking"],
        )
    else:
        # Pretrained (no finetune) — wrap in module directly
        module = ObjectDetectionModule(
            model=model,
            has_tracking=cfg["has_tracking"],
        )

    # For COCO pretrained: remap labels in predictions from COCO ids to OOTB ids
    coco_to_ootb = cfg.get("coco_to_ootb")
    if coco_to_ootb is not None:
        _wrap_with_label_remap(module, coco_to_ootb)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    vis_callback = DetectionVisualizationCallback(
        class_names=class_names,
        output_dir=output_dir,
        iou_thresh=0.3,
        max_wandb_images=50,
        score_thresh=0.5,
        sot_mode=True,
    )

    sot_callback = SOTEvalCallback(
        class_names=class_names,
        output_dir=output_dir,
        score_thresh=0.5,
    )

    # ------------------------------------------------------------------
    # Trainer (test only, no W&B to avoid polluting existing runs)
    # ------------------------------------------------------------------
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        logger=False,
        callbacks=[vis_callback, sot_callback],
    )

    trainer.test(module, datamodule=dm)

    print(f"\n[DONE] {model_name.upper()} results saved to: {output_dir}")
    print(f"  - Visualizations: {output_dir}/visualizations/")
    print(f"  - Per-image metrics: {output_dir}/per_image_metrics.json")
    print(f"  - Test metrics: {output_dir}/test_metrics.json")
    print(f"  - SOT metrics: {output_dir}/sot_metrics.json")


def _wrap_with_label_remap(module: ObjectDetectionModule, coco_to_ootb: dict[int, int]):
    """Monkey-patch the model's forward to remap COCO class ids to OOTB ids.

    Predictions with classes not in coco_to_ootb are discarded.
    """
    import functools

    original_forward = module.model.forward

    @functools.wraps(original_forward)
    def remapped_forward(images, targets=None):
        preds = original_forward(images, targets)
        if module.model.training:
            return preds

        remapped = []
        for pred in preds:
            labels = pred["labels"]
            # Build mask for relevant COCO classes
            keep = torch.zeros(len(labels), dtype=torch.bool, device=labels.device)
            new_labels = labels.clone()
            for coco_id, ootb_id in coco_to_ootb.items():
                mask = labels == coco_id
                keep |= mask
                new_labels[mask] = ootb_id

            remapped.append({
                "boxes": pred["boxes"][keep],
                "scores": pred["scores"][keep],
                "labels": new_labels[keep],
                **{k: v[keep] if isinstance(v, torch.Tensor) and v.shape[0] == len(labels) else v
                   for k, v in pred.items() if k not in ("boxes", "scores", "labels")},
            })
        return remapped

    module.model.forward = remapped_forward


def main():
    parser = argparse.ArgumentParser(description="Test models on OOTB test set")
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()) + ["both"],
        default="both",
        help="Which model to test (default: both = fasterrcnn + yolo finetuned)",
    )
    args = parser.parse_args()

    if args.model == "both":
        models_to_test = ["fasterrcnn", "yolo"]
    else:
        models_to_test = [args.model]

    for name in models_to_test:
        test_model(name)


if __name__ == "__main__":
    main()
