# Experiment Configurations

> Hyperparameters and settings for each experiment.
> All models use the same input resolution and class mapping for fair comparison.

---

## Shared Settings

| Parameter | Value |
|-----------|-------|
| Input resolution | 640 x 640 |
| Class map (FasterRCNN) | car=1, plane=2, ship=3, train=4 (1-indexed, 0=background) |
| Class map (YOLO/DINOv3) | car=0, plane=1, ship=2, train=3 (0-indexed, no background) |
| Optimizer | AdamW |
| Weight decay | 1e-4 |
| LR scheduler | Cosine with linear warmup |
| Warmup epochs | 5 |
| Max epochs | 50 |
| Early stopping | patience=10, monitor=val/AP50 |
| Precision | 16-mixed |
| Logger | Weights & Biases (`esa-dlstem` project) |

---

## Exp 1: Faster R-CNN v2 on OOTB

| Parameter | Value |
|-----------|-------|
| Script | `train_fasterrcnn_ootb.py` |
| Model | `FasterRCNNDetector` (ResNet-50 FPN v2) |
| Pretrained | COCO (torchvision default) |
| Backbone frozen | Yes (`trainable_backbone_layers=0`) |
| Trainable parts | RPN head + ROI head (box predictor) |
| Learning rate | 5e-4 |
| Batch size | 8 |
| Train augmentation | Resize(640,640) + HorizontalFlip(p=0.5) |
| Eval augmentation | Resize(640,640) |
| Dataset | OOTB (train=88, val=10, test=12 videos) |
| score_thresh | 0.05 |
| nms_thresh | 0.5 |
| W&B run name | `fasterrcnn-v2_ootb` |

---

## Exp 2: YOLO v11n on OOTB

| Parameter | Value |
|-----------|-------|
| Script | `train_yolo_ootb.py` |
| Model | `YOLODetector` (YOLOv11 nano) |
| Pretrained | COCO (Ultralytics default) |
| Backbone frozen | No (full model fine-tuned, ~2.6M params) |
| Learning rate | 1e-3 |
| Weight decay | 5e-4 |
| Batch size | 8 |
| Train augmentation | Resize(640,640) + HorizontalFlip(p=0.5) |
| Eval augmentation | Resize(640,640) |
| Dataset | OOTB (train=88, val=10, test=12 videos) |
| conf_thresh | 0.05 |
| iou_thresh (NMS) | 0.5 |
| Tracking | Disabled (detection only) |
| W&B run name | `yolo11n_ootb` |

---

## Exp 3a: SAM2 on OOTB (first-frame prompt)

| Parameter | Value |
|-----------|-------|
| Script | `eval_sam2_ootb.py` |
| Model | `SAM2Tracker` (SAM2.1 Hiera-Large) |
| Pretrained | SA-V (Meta default) |
| Training | None (zero-shot evaluation only) |
| Prompt strategy | First frame GT boxes only |
| Clip length | 32 frames |
| Clip stride | 1 |
| Batch size | 1 |
| Eval augmentation | Resize(640,640) |
| Dataset | OOTB (test=12 videos) |
| W&B run name | `sam2_first_frame_ootb` |

---

## Exp 3b: SAM2 on OOTB (re-prompt every 10 frames)

| Parameter | Value |
|-----------|-------|
| Script | `eval_sam2_ootb.py` |
| Model | `SAM2Tracker` (SAM2.1 Hiera-Large) |
| Pretrained | SA-V (Meta default) |
| Training | None (zero-shot evaluation only) |
| Prompt strategy | GT boxes every 10 frames |
| Clip length | 32 frames |
| Clip stride | 1 |
| Batch size | 1 |
| Eval augmentation | Resize(640,640) |
| Dataset | OOTB (test=12 videos) |
| W&B run name | `sam2_every10_ootb` |
