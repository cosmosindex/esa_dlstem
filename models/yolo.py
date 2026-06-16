"""
YOLO detector / tracker wrapper for fine-tuning via Ultralytics.

YOLO with ByteTrack is end-to-end detect + track, so use this with
ObjectDetectionModule(has_tracking=True).

Note: Ultralytics YOLO has its own trainer (model.train()), but for
integration with PyTorch Lightning we expose the inner nn.Module so that
Lightning controls the training loop. The tracking state (ByteTrack) is
maintained per-sequence and reset between sequences.
"""

import math

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
        """Replace the classification Conv2d layers in cv3 for a new class count.

        The cls bias is **not** zeroed. Ultralytics' ``Detect.bias_init`` sets a
        prior-probability cls bias (``log(5 / nc / (640/stride)**2)`` ≈ -5…-8) so
        the initial per-anchor confidence is ≈0.003. Zeroing the bias instead
        makes every anchor predict p≈0.5 → the BCE cls loss over ~8400 mostly-
        background anchors explodes to ~2000 → gradient explosion → NaN loss,
        and the model never trains (val/mAP stays 0). See yolo_issues.md #12/#14.
        """
        head = yolo.model.model[-1]
        strides = getattr(head, "stride", None)
        for i, scale_branch in enumerate(head.cv3):
            old_conv = scale_branch[-1]  # Conv2d(..., old_nc, 1x1)
            new_conv = nn.Conv2d(old_conv.in_channels, nc, kernel_size=1)
            if strides is not None and i < len(strides):
                s = float(strides[i])
                new_conv.bias.data.fill_(math.log(5 / nc / (640 / s) ** 2))
            else:
                new_conv.bias.data.fill_(-4.6)  # sigmoid ≈ 0.01 fallback
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
        # Ultralytics' v8DetectionLoss returns ``loss * batch_size`` (utils/loss.py:538)
        # to suit its own SGD/nbs-scaled lr. Left un-normalised here the gradients are
        # ``batch_size``× too large, so a normal AdamW lr (1e-3, fine for FasterRCNN on
        # the same module) diverges to NaN once warmup ends — see yolo_issues.md #15.
        # Divide by batch size to get a per-sample mean loss, matching every other model.
        bs = batch.shape[0]
        loss, loss_items = self.model.loss(ul_batch)
        return {
            "loss": loss.sum() / bs,
            "loss_box": loss_items[0],
            "loss_cls": loss_items[1],
            "loss_dfl": loss_items[2],
        }

    def _inference_forward(self, images):
        """Eval-mode detection via a bare module forward + manual NMS.

        The detection path deliberately avoids Ultralytics' high-level
        ``predict()``. That API lazily builds a **cached** predictor
        (``engine/model.py``: ``if not self.predictor``) whose ``setup_model``
        wraps the model in ``AutoBackend(fuse=True, fp16=...)`` — fusing Conv+BN
        (and optionally half-casting) ``self.model`` *in place*. Inside
        Lightning's fit→validate loop that predictor is first built during the
        sanity-check validation, when the rebuilt classification head is still
        random, then frozen and reused every epoch → ``val/mAP`` pinned at 0 and
        the fused model can no longer train. Running ``self.model(x)`` directly
        keeps train and val on the same un-fused weights.

        The tracking path still uses Ultralytics ``track()`` (stateful by
        design); it only runs at test time, so the fit-loop fusion issue above
        does not arise. See ``yolo_issues.md`` #3/#10/#11.
        """
        device = images[0].device

        if self.enable_tracking:
            return self._track_forward(images, device)

        from ultralytics.utils.nms import non_max_suppression

        # images are float [0, 1] CHW (dataset: from_numpy(...).float() / 255).
        batch = torch.stack(images) if isinstance(images, list) else images
        raw = self.model(batch)            # eval mode → (B, 4+nc, num_anchors)
        if isinstance(raw, (list, tuple)):
            raw = raw[0]

        dets = non_max_suppression(
            raw,
            conf_thres=self.conf_thresh,
            iou_thres=self.iou_thresh,
            nc=self.model.model[-1].nc,
            max_det=300,
        )

        outputs = []
        for d in dets:                     # d: (n, 6) → [x1, y1, x2, y2, conf, cls]
            outputs.append({
                "boxes":  d[:, :4].to(device),
                "labels": d[:, 5].long().to(device),
                "scores": d[:, 4].to(device),
            })
        return outputs

    def _track_forward(self, images, device):
        """Stateful multi-object tracking via Ultralytics ``track()``.

        Test-time only (MOT eval). Uses the high-level API on purpose because
        ByteTrack state must persist across frames; the predictor-fusion issue
        documented in ``_inference_forward`` does not apply since there is no
        interleaved training validation here.
        """
        import numpy as np
        np_images = [
            (img.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            for img in images
        ]

        results = self._yolo.track(
            np_images,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=self.img_size,
            tracker=self.tracker_config,
            persist=True,
            verbose=False,
        )

        outputs = []
        for r in results:
            boxes = r.boxes
            out: dict = {
                "boxes":  boxes.xyxy.to(device),
                "labels": boxes.cls.long().to(device),
                "scores": boxes.conf.to(device),
            }
            if boxes.id is not None:
                out["track_ids"] = boxes.id.long().to(device)
            outputs.append(out)

        # track() disables requires_grad + caches inference-mode anchors on the
        # Detect head; restore so a later training forward still works.
        for p in self.model.parameters():
            p.requires_grad = True
        self.model.model[-1].shape = None

        return outputs

    def reset_tracker(self):
        """Call between sequences to clear ByteTrack state."""
        if hasattr(self._yolo, "predictor") and self._yolo.predictor is not None:
            self._yolo.predictor.trackers = []
