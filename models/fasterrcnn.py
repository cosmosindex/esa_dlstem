"""
FasterRCNN detector wrapper for fine-tuning.

Detection only (no built-in tracking). Use ObjectDetectionModule with
has_tracking=False and pair with an external tracker if needed.
"""

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
                       'boxes' (N,4 xyxy), 'labels' (N,), 'scores' (N,).
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

    def forward(self, images: list[torch.Tensor], targets: list[dict] | None = None):
        """
        Args:
            images:  list of float tensors, each (C, H, W), values in [0, 1].
            targets: list of dicts with 'boxes' (N,4 xyxy float) and 'labels' (N, long).
                     Required during training, ignored during eval.

        Returns (train): dict of loss tensors
            {
                'loss_classifier': ...,
                'loss_box_reg': ...,
                'loss_objectness': ...,
                'loss_rpn_box_reg': ...,
            }

        Returns (eval): list of dicts, one per image
            [{'boxes': (N,4), 'labels': (N,), 'scores': (N,)}, ...]
        """
        return self.model(images, targets)
