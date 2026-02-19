"""
YOLO detector / tracker wrapper for fine-tuning via Ultralytics.

YOLO with ByteTrack is end-to-end detect + track, so use this with
ObjectDetectionModule(has_tracking=True).

Note: Ultralytics YOLO has its own trainer (model.train()), but for
integration with PyTorch Lightning we expose the inner nn.Module so that
Lightning controls the training loop. The tracking state (ByteTrack) is
maintained per-sequence and reset between sequences.
"""

import torch
import torch.nn as nn
from typing import Optional


class YOLODetector(nn.Module):
    """
    Ultralytics YOLO wrapper exposing a standard nn.Module interface.

    Supports detection-only mode and detection + ByteTrack tracking mode.

    During training  : delegates to the inner Ultralytics model's loss computation.
    During inference : runs predict (and optionally track) and returns standardised dicts.
    """

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        num_classes: Optional[int] = None,
        enable_tracking: bool = True,
        tracker_config: str = "bytetrack.yaml",
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        img_size: int = 640,
    ):
        """
        Args:
            model_name:      Ultralytics model identifier or path to .pt weights.
            num_classes:     Override number of output classes; if None, keep as-is.
            enable_tracking: If True, ByteTrack assigns track IDs during inference.
            tracker_config:  Tracker config file (used only when enable_tracking=True).
            conf_thresh:     Confidence threshold for predictions.
            iou_thresh:      NMS IoU threshold.
            img_size:        Input image size for inference.
        """
        super().__init__()

        from ultralytics import YOLO

        self.yolo = YOLO(model_name)

        if num_classes is not None:
            # Reinitialise the detection head for a different class count
            self.yolo.model.model[-1].nc = num_classes
            # TODO: rebuild head weights when num_classes differs from pretrained

        self.enable_tracking = enable_tracking
        self.tracker_config = tracker_config
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.img_size = img_size

        # Expose the inner nn.Module so Lightning can call .parameters() on it
        self.model = self.yolo.model

    def forward(
        self,
        images: list[torch.Tensor],
        targets: list[dict] | None = None,
    ):
        """
        Args:
            images:  list of float tensors (C, H, W) or a batched tensor (B, C, H, W).
            targets: list of dicts with 'boxes' and 'labels' for training.
                     If None and model.training, raises an error.

        Returns (train): dict with 'loss' (total) and component losses.
        Returns (eval):  list of dicts, one per image:
            {
                'boxes':     (N, 4) xyxy float,
                'labels':    (N,)   long,
                'scores':    (N,)   float,
                'track_ids': (N,)   long  (only when enable_tracking=True),
            }
        """
        if self.training:
            return self._training_forward(images, targets)
        return self._inference_forward(images)

    def _training_forward(self, images, targets):
        # Stack images into a batch tensor
        if isinstance(images, list):
            batch = torch.stack(images)  # (B, C, H, W)
        else:
            batch = images

        # Ultralytics loss requires a batch dict with 'img' and formatted labels
        # Build the label tensor in Ultralytics format:
        # (batch_idx, cls, xc, yc, w, h) – normalised
        label_rows = []
        for i, t in enumerate(targets):
            boxes_xyxy = t["boxes"]  # (N, 4) xyxy absolute
            labels = t["labels"]     # (N,)
            h, w = batch.shape[2], batch.shape[3]
            # Convert to normalised xywh
            xc = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2 / w
            yc = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2 / h
            bw = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) / w
            bh = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) / h
            batch_idx = torch.full((len(labels),), i, dtype=torch.float32)
            row = torch.stack(
                [batch_idx, labels.float(), xc, yc, bw, bh], dim=1
            )
            label_rows.append(row)

        ul_batch = {
            "img": batch,
            "cls": torch.cat([r[:, 1:2] for r in label_rows]),
            "bboxes": torch.cat([r[:, 2:] for r in label_rows]),
            "batch_idx": torch.cat([r[:, 0] for r in label_rows]),
        }
        loss, loss_items = self.model.loss(ul_batch)
        return {
            "loss": loss,
            "loss_box": loss_items[0],
            "loss_cls": loss_items[1],
            "loss_dfl": loss_items[2],
        }

    def _inference_forward(self, images):
        if self.enable_tracking:
            results = self.yolo.track(
                images,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                imgsz=self.img_size,
                tracker=self.tracker_config,
                persist=True,   # keep tracker state across calls for the same sequence
                verbose=False,
            )
        else:
            results = self.yolo.predict(
                images,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                imgsz=self.img_size,
                verbose=False,
            )

        outputs = []
        for r in results:
            boxes = r.boxes
            out: dict = {
                "boxes":  boxes.xyxy.cpu(),
                "labels": boxes.cls.long().cpu(),
                "scores": boxes.conf.cpu(),
            }
            if self.enable_tracking and boxes.id is not None:
                out["track_ids"] = boxes.id.long().cpu()
            outputs.append(out)
        return outputs

    def reset_tracker(self):
        """Call between sequences to clear ByteTrack state."""
        if hasattr(self.yolo, "predictor") and self.yolo.predictor is not None:
            self.yolo.predictor.trackers = []
