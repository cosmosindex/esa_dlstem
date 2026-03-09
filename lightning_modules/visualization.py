"""
DetectionVisualizationCallback
==============================
Lightning Callback that visualises object detection results during testing.

For each test image, draws bounding boxes with colour-coded TP / FP / FN
annotations. Images are:
  1. Saved locally to ``output_dir/visualizations/`` (ALL images)
  2. Logged to W&B ``test/detections`` panel (capped at ``max_wandb_images``)

Colours:
    Green  — TP  (prediction matched a GT box, IoU >= threshold)
    Red    — FP  (prediction with no GT match)
    Blue   — FN  (GT box with no prediction match)
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import lightning as L


# Colour palette (RGB)
_GREEN = (0, 200, 0)     # TP
_RED   = (220, 40, 40)   # FP
_BLUE  = (50, 80, 220)   # FN

_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.45
_THICKNESS  = 1


class DetectionVisualizationCallback(L.Callback):
    """
    Visualise test-set detections as bounding-box overlays.

    Args:
        class_names:       Dict mapping class id → display name, e.g. {1: "car"}.
        output_dir:        Local directory to save all visualizations and metrics.
        iou_thresh:        IoU threshold for TP / FP / FN matching.
        max_wandb_images:  Max images to log to W&B (local saves are unlimited).
        score_thresh:      Only draw predictions with score >= this value.
    """

    def __init__(
        self,
        class_names: dict[int, str],
        output_dir: str | Path = "experiments",
        iou_thresh: float = 0.5,
        max_wandb_images: int = 50,
        score_thresh: float = 0.5,
    ):
        super().__init__()
        self.class_names = class_names
        self.output_dir = Path(output_dir)
        self.iou_thresh = iou_thresh
        self.max_wandb_images = max_wandb_images
        self.score_thresh = score_thresh

        self._wandb_logged = 0
        self._vis_dir: Path | None = None
        self._all_metrics: list[dict] = []

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def on_test_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        self._wandb_logged = 0
        self._all_metrics = []

        # Create output directories
        self._vis_dir = self.output_dir / "visualizations"
        self._vis_dir.mkdir(parents=True, exist_ok=True)

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        images, targets = batch

        pl_module.model.eval()
        with torch.no_grad():
            preds = pl_module.model(images)

        for img_tensor, pred, tgt in zip(images, preds, targets):
            video_id = tgt.get("video_id", f"batch{batch_idx}")
            frame_id = tgt.get("frame_id", 0)

            vis, per_image_metrics = self._draw_detections(img_tensor, pred, tgt)

            # Always save locally
            filename = f"{video_id}_frame{frame_id:04d}.jpg"
            cv2.imwrite(
                str(self._vis_dir / filename),
                cv2.cvtColor(vis, cv2.COLOR_RGB2BGR),
            )

            # Collect per-image metrics
            per_image_metrics["video_id"] = video_id
            per_image_metrics["frame_id"] = frame_id
            self._all_metrics.append(per_image_metrics)

            # Log to W&B (capped)
            if self._wandb_logged < self.max_wandb_images:
                self._log_wandb(trainer, vis, video_id, frame_id)
                self._wandb_logged += 1

    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        # Save per-image metrics
        metrics_path = self.output_dir / "per_image_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self._all_metrics, f, indent=2)

        # Save aggregate metrics from trainer's logged values
        logged = trainer.callback_metrics
        summary = {k: v.item() if isinstance(v, torch.Tensor) else v
                   for k, v in logged.items()}
        summary_path = self.output_dir / "test_metrics.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    # ------------------------------------------------------------------
    # Core drawing logic
    # ------------------------------------------------------------------

    def _draw_detections(
        self,
        img_tensor: torch.Tensor,
        pred: dict,
        tgt: dict,
    ) -> tuple[np.ndarray, dict]:
        """
        Draw TP / FP / FN boxes on the image.

        Returns:
            vis:     np.ndarray (H, W, 3) uint8 RGB image with annotations.
            metrics: dict with per-image TP/FP/FN counts.
        """
        img = (img_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()

        gt_boxes  = tgt["boxes"].cpu().numpy()
        gt_labels = tgt["labels"].cpu().numpy()

        pred_boxes  = pred["boxes"].cpu().numpy()
        pred_scores = pred["scores"].cpu().numpy()
        pred_labels = pred["labels"].cpu().numpy()

        # Filter low-confidence predictions
        keep = pred_scores >= self.score_thresh
        pred_boxes  = pred_boxes[keep]
        pred_scores = pred_scores[keep]
        pred_labels = pred_labels[keep]

        # Match predictions to GT
        tp_mask, fp_mask, fn_mask = self._match(gt_boxes, pred_boxes)

        # Draw FN first (behind) — blue
        for i, is_fn in enumerate(fn_mask):
            if is_fn:
                b = gt_boxes[i].astype(int)
                name = self.class_names.get(int(gt_labels[i]), f"cls{gt_labels[i]}")
                self._draw_box(img, b, _BLUE, f"FN {name}")

        # Draw FP — red
        for i, is_fp in enumerate(fp_mask):
            if is_fp:
                b = pred_boxes[i].astype(int)
                name = self.class_names.get(int(pred_labels[i]), f"cls{pred_labels[i]}")
                self._draw_box(img, b, _RED, f"FP {name} {pred_scores[i]:.2f}")

        # Draw TP — green (on top)
        for i, is_tp in enumerate(tp_mask):
            if is_tp:
                b = pred_boxes[i].astype(int)
                name = self.class_names.get(int(pred_labels[i]), f"cls{pred_labels[i]}")
                self._draw_box(img, b, _GREEN, f"TP {name} {pred_scores[i]:.2f}")

        metrics = {
            "tp": int(tp_mask.sum()),
            "fp": int(fp_mask.sum()),
            "fn": int(fn_mask.sum()),
            "num_preds": int(len(pred_boxes)),
            "num_gt": int(len(gt_boxes)),
        }
        return img, metrics

    # ------------------------------------------------------------------
    # IoU matching
    # ------------------------------------------------------------------

    def _match(
        self,
        gt_boxes: np.ndarray,
        pred_boxes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Greedy IoU matching.

        Returns:
            tp_mask:  bool array (N_pred,)
            fp_mask:  bool array (N_pred,)
            fn_mask:  bool array (M_gt,)
        """
        M = len(gt_boxes)
        N = len(pred_boxes)

        tp_mask = np.zeros(N, dtype=bool)
        fp_mask = np.ones(N, dtype=bool)
        fn_mask = np.ones(M, dtype=bool)

        if M == 0 or N == 0:
            return tp_mask, fp_mask, fn_mask

        iou = self._iou_matrix(gt_boxes, pred_boxes)

        matched_gt: set[int] = set()
        matched_pred: set[int] = set()

        pairs = np.argwhere(iou >= self.iou_thresh)
        if len(pairs) > 0:
            scores = iou[pairs[:, 0], pairs[:, 1]]
            order = scores.argsort()[::-1]
            pairs = pairs[order]

            for r, c in pairs:
                if r in matched_gt or c in matched_pred:
                    continue
                matched_gt.add(r)
                matched_pred.add(c)
                tp_mask[c] = True
                fp_mask[c] = False
                fn_mask[r] = False

        return tp_mask, fp_mask, fn_mask

    @staticmethod
    def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        """Pairwise IoU (M, N) between xyxy boxes."""
        x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
        y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
        x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
        y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
        area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
        union = area_a[:, None] + area_b[None, :] - inter

        return inter / np.clip(union, 1e-6, None)

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_box(img: np.ndarray, box: np.ndarray, color: tuple, label: str):
        """Draw a single bbox with label on the image."""
        x1, y1, x2, y2 = box
        bgr = color[::-1]
        cv2.rectangle(img, (x1, y1), (x2, y2), bgr, 2)

        (tw, th), _ = cv2.getTextSize(label, _FONT, _FONT_SCALE, _THICKNESS)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), bgr, -1)
        cv2.putText(img, label, (x1, y1 - 2), _FONT, _FONT_SCALE, (255, 255, 255), _THICKNESS)

    # ------------------------------------------------------------------
    # W&B logging
    # ------------------------------------------------------------------

    @staticmethod
    def _log_wandb(trainer: L.Trainer, img: np.ndarray, video_id: str, frame_id: int):
        """Log an RGB image to W&B."""
        logger = trainer.logger
        if logger is None:
            return
        if hasattr(logger, "experiment") and hasattr(logger.experiment, "log"):
            import wandb
            caption = f"{video_id}/frame_{frame_id}"
            logger.experiment.log({
                "test/detections": wandb.Image(img, caption=caption),
            })
