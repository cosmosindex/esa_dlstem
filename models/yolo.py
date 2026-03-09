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
        num_classes: int = 80,
        enable_tracking: bool = True,
        tracker_config: str = "bytetrack.yaml",
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        img_size: int = 640,
    ):
        """
        Args:
            model_name:      Ultralytics model identifier or path to .pt weights.
            num_classes:     Number of target classes. If different from pretrained (80),
                             the classification head (cv3) is rebuilt with random weights.
            enable_tracking: If True, ByteTrack assigns track IDs during inference.
            tracker_config:  Tracker config file (used only when enable_tracking=True).
            conf_thresh:     Confidence threshold for predictions.
            iou_thresh:      NMS IoU threshold.
            img_size:        Input image size for inference.
        """
        super().__init__()

        from ultralytics import YOLO

        yolo = YOLO(model_name)

        # Rebuild classification head if num_classes differs from pretrained
        if num_classes != yolo.model.model[-1].nc:
            self._rebuild_cls_head(yolo, num_classes)

        self.enable_tracking = enable_tracking
        self.tracker_config = tracker_config
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.img_size = img_size

        # Ensure model.args is a proper namespace with loss weights
        # (required by Ultralytics loss function which uses dot-access)
        from ultralytics.cfg import get_cfg
        default_cfg = get_cfg()
        if isinstance(yolo.model.args, dict):
            from ultralytics.utils import IterableSimpleNamespace
            yolo.model.args = IterableSimpleNamespace(**{**vars(default_cfg), **yolo.model.args})
        for key in ("box", "cls", "dfl"):
            if not hasattr(yolo.model.args, key):
                setattr(yolo.model.args, key, getattr(default_cfg, key))

        # Register the inner DetectionModel as a proper submodule (for .parameters())
        self.model = yolo.model

        # Store the YOLO wrapper WITHOUT registering it as an nn.Module child,
        # because YOLO overrides .train() to start Ultralytics' training pipeline.
        object.__setattr__(self, "_yolo", yolo)

        # Ultralytics loads with requires_grad=False; enable for fine-tuning
        for p in self.model.parameters():
            p.requires_grad = True

    @staticmethod
    def _rebuild_cls_head(yolo, nc: int):
        """Replace the classification Conv2d layers in cv3 for a new class count."""
        head = yolo.model.model[-1]
        for scale_branch in head.cv3:
            old_conv = scale_branch[-1]  # Conv2d(..., old_nc, 1x1)
            new_conv = nn.Conv2d(old_conv.in_channels, nc, kernel_size=1)
            nn.init.zeros_(new_conv.bias)
            scale_branch[-1] = new_conv
        head.nc = nc
        head.no = nc + head.reg_max * 4

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
            batch_idx = torch.full((len(labels),), i, dtype=torch.float32, device=batch.device)
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
            "loss": loss.sum(),
            "loss_box": loss_items[0],
            "loss_cls": loss_items[1],
            "loss_dfl": loss_items[2],
        }

    def _inference_forward(self, images):
        # Ultralytics expects numpy HWC uint8 images, not tensors
        import numpy as np
        np_images = [
            (img.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            for img in images
        ]

        if self.enable_tracking:
            results = self._yolo.track(
                np_images,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                imgsz=self.img_size,
                tracker=self.tracker_config,
                persist=True,
                verbose=False,
            )
        else:
            results = self._yolo.predict(
                np_images,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                imgsz=self.img_size,
                verbose=False,
            )

        # Determine device from input images so outputs match targets
        device = images[0].device

        outputs = []
        for r in results:
            boxes = r.boxes
            out: dict = {
                "boxes":  boxes.xyxy.to(device),
                "labels": boxes.cls.long().to(device),
                "scores": boxes.conf.to(device),
            }
            if self.enable_tracking and boxes.id is not None:
                out["track_ids"] = boxes.id.long().to(device)
            outputs.append(out)

        # Ultralytics predictor sets requires_grad=False on parameters;
        # restore so that subsequent training forward passes can compute gradients.
        for p in self.model.parameters():
            p.requires_grad = True

        # Ultralytics inference caches anchors/strides as inference-mode tensors
        # on the Detect head. Reset the cached shape so they are recomputed
        # in normal mode during the next training forward pass.
        head = self.model.model[-1]
        head.shape = None

        return outputs

    def reset_tracker(self):
        """Call between sequences to clear ByteTrack state."""
        if hasattr(self._yolo, "predictor") and self._yolo.predictor is not None:
            self._yolo.predictor.trackers = []
