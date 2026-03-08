# Logged Metrics Summary

> All metrics logged by `ObjectDetectionModule` (`lightning_modules/module.py`).
> Metrics are logged via `self.log()` and automatically captured by any Lightning logger (TensorBoard, W&B, CSV, etc.).

---

## Training

Logged every step and/or epoch during `training_step`.

| Key | Type | prog_bar | on_step | on_epoch | Description |
|-----|------|:--------:|:-------:|:--------:|-------------|
| `train/loss` | scalar | Y | Y | Y | Total loss (sum of all loss components) |
| `train/<loss_key>` | scalar | N | Y | N | Individual loss components (e.g. `train/loss_classifier`, `train/loss_box_reg`); keys depend on the model backend |

---

## Validation

Logged at the end of each validation epoch via `on_validation_epoch_end`.

### Detection metrics

| Key | Type | prog_bar | Source | Description |
|-----|------|:--------:|--------|-------------|
| `val/AP50` | scalar | Y | `torchmetrics.MeanAveragePrecision` | COCO-style AP at IoU = 0.50 |
| `val/AP` | scalar | N | `torchmetrics.MeanAveragePrecision` | COCO-style mAP (IoU 0.50 only, since `iou_thresholds=[0.5]`) |
| `val/AR_100` | scalar | N | `torchmetrics.MeanAveragePrecision` | Average Recall with max 100 detections per image |
| `val/Precision` | scalar | N | Custom TP/FP/FN accumulator | TP / (TP + FP) at IoU >= 0.5, greedy matching |
| `val/Recall` | scalar | N | Custom TP/FP/FN accumulator | TP / (TP + FN) at IoU >= 0.5, greedy matching |

### Tracking metrics (only when `has_tracking=True`)

| Key | Type | prog_bar | Source | Description |
|-----|------|:--------:|--------|-------------|
| `val/MOTA` | scalar | Y | Custom accumulator | Multi-Object Tracking Accuracy: 1 - (FP + FN + ID_sw) / num_GT |
| `val/IDF1` | scalar | N | Custom accumulator | ID F1 score: harmonic mean of tracking precision and recall |
| `val/ID_switches` | scalar | N | Custom accumulator | Total number of identity switches across the epoch |

---

## Test

Logged at the end of the test epoch via `on_test_epoch_end`. Includes all validation metrics plus efficiency metrics.

### Detection metrics

| Key | Type | prog_bar | Source | Description |
|-----|------|:--------:|--------|-------------|
| `test/AP50` | scalar | Y | `torchmetrics.MeanAveragePrecision` | COCO-style AP at IoU = 0.50 |
| `test/AP` | scalar | N | `torchmetrics.MeanAveragePrecision` | COCO-style mAP |
| `test/AR_100` | scalar | N | `torchmetrics.MeanAveragePrecision` | Average Recall with max 100 detections per image |
| `test/Precision` | scalar | N | Custom TP/FP/FN accumulator | TP / (TP + FP) at IoU >= 0.5, greedy matching |
| `test/Recall` | scalar | N | Custom TP/FP/FN accumulator | TP / (TP + FN) at IoU >= 0.5, greedy matching |

### Tracking metrics (only when `has_tracking=True`)

| Key | Type | prog_bar | Source | Description |
|-----|------|:--------:|--------|-------------|
| `test/MOTA` | scalar | Y | Custom accumulator | Multi-Object Tracking Accuracy |
| `test/IDF1` | scalar | N | Custom accumulator | ID F1 score |
| `test/ID_switches` | scalar | N | Custom accumulator | Total identity switches |

### Efficiency metrics

| Key | Type | prog_bar | Source | Description |
|-----|------|:--------:|--------|-------------|
| `test/fps` | scalar | Y | `time.perf_counter` + `cuda.synchronize` | Inference throughput (images per second), pure model forward pass |
| `test/total_time_s` | scalar | N | `time.perf_counter` + `cuda.synchronize` | Total inference wall time across all test batches (seconds) |
| `test/model_size_MB` | scalar | N | `parameters + buffers` | Model memory footprint: sum of all parameter and buffer tensors (MB) |

---

## Notes

- **Precision / Recall** are computed with a custom greedy IoU-matching accumulator (not from `torchmetrics.MeanAveragePrecision`, which does not expose these as standalone scalars). The IoU threshold is fixed at 0.5.
- **Tracking metrics** (MOTA, IDF1, ID_switches) are only logged when `has_tracking=True`. They use a separate greedy matching accumulator with configurable IoU threshold (`iou_match_thresh`, default 0.5).
- **FPS timing** uses `torch.cuda.synchronize()` before and after inference to ensure accurate GPU timing. On CPU it falls back to wall-clock time.
- **Model size** counts `parameters + buffers` in MB. This is the in-memory size, not the on-disk checkpoint size (which may differ due to optimizer state, metadata, etc.).
