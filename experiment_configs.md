# Experiment Configurations

> Hyperparameters and settings for each experiment.
> All models use the same input resolution and class mapping for fair comparison.

---

## Shared Settings

| Parameter | Value |
|-----------|-------|
| Input resolution | 640 x 640 |
| Class map | car=1, plane=2, ship=3, train=4 |
| num_classes | 5 (4 categories + background) |
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
