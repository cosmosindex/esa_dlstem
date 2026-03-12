"""
FasterRCNN detector wrapper for fine-tuning.

Optionally integrates ByteTrack for multi-object tracking at inference time.
"""

from pathlib import Path

import numpy as np
import yaml
import torch
import torch.nn as nn
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class FasterRCNNDetector(nn.Module):
    """
    Faster R-CNN (ResNet-50 FPN backbone) fine-tunable wrapper.

    During training  : forward returns a dict of losses (same as torchvision).
    During inference : forward returns a list of dicts, each with
                       'boxes' (N,4 xyxy), 'labels' (N,), 'scores' (N,),
                       and optionally 'track_ids' (N,) when enable_tracking=True.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        use_v2: bool = False,
        trainable_backbone_layers: int = 3,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        detections_per_img: int = 100,
        enable_tracking: bool = False,
    ):
        super().__init__()

        if use_v2:
            weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
            self.model = fasterrcnn_resnet50_fpn_v2(
                weights=weights,
                trainable_backbone_layers=trainable_backbone_layers,
            )
        else:
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
            self.model = fasterrcnn_resnet50_fpn(
                weights=weights,
                trainable_backbone_layers=trainable_backbone_layers,
            )

        # Override post-processing thresholds
        self.model.roi_heads.score_thresh = score_thresh
        self.model.roi_heads.nms_thresh = nms_thresh
        self.model.roi_heads.detections_per_img = detections_per_img

        # Replace the classification head for the target number of classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # ByteTrack (lazy init on first inference call)
        self.enable_tracking = enable_tracking
        self._tracker = None

    def _init_tracker(self):
        """Lazily initialise ByteTrack from ultralytics."""
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import IterableSimpleNamespace
        import ultralytics

        bt_yaml = Path(ultralytics.__file__).parent / "cfg" / "trackers" / "bytetrack.yaml"
        cfg = IterableSimpleNamespace(**yaml.safe_load(bt_yaml.read_text()))
        self._tracker = BYTETracker(cfg, frame_rate=30)

    def reset_tracker(self):
        """Reset ByteTrack state between video sequences."""
        self._tracker = None

    def forward(self, images: list[torch.Tensor], targets: list[dict] | None = None):
        """
        Args:
            images:  list of float tensors, each (C, H, W), values in [0, 1].
            targets: list of dicts with 'boxes' (N,4 xyxy float) and 'labels' (N, long).
                     Required during training, ignored during eval.

        Returns (train): dict of loss tensors
        Returns (eval): list of dicts, one per image
            [{'boxes': (N,4), 'labels': (N,), 'scores': (N,), ['track_ids': (N,)]}, ...]
        """
        if self.training or not self.enable_tracking:
            return self.model(images, targets)

        # Inference with tracking
        raw_outputs = self.model(images)
        device = images[0].device

        if self._tracker is None:
            self._init_tracker()

        results = []
        for out in raw_outputs:
            boxes = out["boxes"]    # (N, 4) xyxy
            scores = out["scores"]  # (N,)
            labels = out["labels"]  # (N,)

            if len(boxes) == 0:
                out["track_ids"] = torch.tensor([], dtype=torch.long, device=device)
                results.append(out)
                continue

            # ByteTrack expects a Results-like object with .xyxy, .conf, .cls, .xywh
            det = _DetectionResults(
                boxes.detach().cpu().numpy(),
                scores.detach().cpu().numpy(),
                labels.detach().cpu().float().numpy(),
            )
            tracked = self._tracker.update(det)  # (K, 8): x1,y1,x2,y2,id,conf,cls,idx

            if len(tracked) == 0:
                out["track_ids"] = torch.tensor([], dtype=torch.long, device=device)
                out["boxes"] = torch.zeros((0, 4), dtype=torch.float32, device=device)
                out["scores"] = torch.tensor([], dtype=torch.float32, device=device)
                out["labels"] = torch.tensor([], dtype=torch.long, device=device)
            else:
                out["boxes"] = torch.as_tensor(tracked[:, :4], dtype=torch.float32, device=device)
                out["track_ids"] = torch.as_tensor(tracked[:, 4], dtype=torch.long, device=device)
                out["scores"] = torch.as_tensor(tracked[:, 5], dtype=torch.float32, device=device)
                out["labels"] = torch.as_tensor(tracked[:, 6], dtype=torch.long, device=device)

            results.append(out)

        return results


class _DetectionResults:
    """Minimal wrapper matching the interface ByteTrack.update() expects."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy = xyxy.reshape(-1, 4)
        self.conf = conf.reshape(-1)
        self.cls = cls.reshape(-1)
        # xywh: center-x, center-y, width, height
        w = self.xyxy[:, 2] - self.xyxy[:, 0]
        h = self.xyxy[:, 3] - self.xyxy[:, 1]
        cx = self.xyxy[:, 0] + w / 2
        cy = self.xyxy[:, 1] + h / 2
        self.xywh = np.stack([cx, cy, w, h], axis=1)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, idx):
        return _DetectionResults(self.xyxy[idx], self.conf[idx], self.cls[idx])
