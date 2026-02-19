"""
ObjectDetectionModule: a single PyTorch Lightning module that wraps any of
the four model backends (FasterRCNN, YOLO, SAM2, DINOv2).

Design:
  - has_tracking=False  →  ObjectDetector behaviour (per-frame metrics only)
  - has_tracking=True   →  ObjectTracker behaviour  (per-sequence, adds MOTA/IDF1)

Metrics accumulated across val/test steps:
  Detection:  AP@50, Precision, Recall  (via torchmetrics MeanAveragePrecision)
  Tracking:   MOTA, ID-switches, IDF1   (custom accumulators, toggled by has_tracking)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import lightning as L
from torchmetrics.detection import MeanAveragePrecision


class ObjectDetectionModule(L.LightningModule):
    """
    Unified Lightning module for object detection (and optionally tracking).

    Args:
        model:            One of FasterRCNNDetector, YOLODetector, SAM2Tracker,
                          or DINOv3Detector (or any nn.Module with the same interface).
        has_tracking:     If True, the model returns 'track_ids' and tracking metrics
                          (MOTA, IDF1) are computed in addition to detection metrics.
        lr:               Base learning rate.
        weight_decay:     AdamW weight decay.
        lr_scheduler:     One of 'cosine', 'step', or None.
        warmup_epochs:    Linear warmup duration (cosine scheduler only).
        total_epochs:     Total training epochs (cosine scheduler only).
        step_size:        Step scheduler step size (step scheduler only).
        gamma:            Step scheduler decay factor.
        iou_match_thresh: IoU threshold for matching detections to GT in tracking.
    """

    def __init__(
        self,
        model: nn.Module,
        has_tracking: bool = False,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        lr_scheduler: str | None = "cosine",
        warmup_epochs: int = 5,
        total_epochs: int = 50,
        step_size: int = 10,
        gamma: float = 0.1,
        iou_match_thresh: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.has_tracking = has_tracking

        # torchmetrics MAP (COCO-style) – resets automatically at epoch boundaries
        self._val_map  = MeanAveragePrecision(iou_thresholds=[0.5])
        self._test_map = MeanAveragePrecision(iou_thresholds=[0.5])

        # Tracking accumulators (reset at epoch start)
        self._reset_tracking_accumulators()

    # ------------------------------------------------------------------
    # Tracking state helpers
    # ------------------------------------------------------------------

    def _reset_tracking_accumulators(self):
        self._num_gt        = 0
        self._num_tp        = 0   # true positives (matched & correct track ID)
        self._num_id_switch = 0
        self._num_fp        = 0
        self._num_fn        = 0
        # last known GT→pred track-ID assignment (for ID-switch detection)
        self._last_gt_to_pred: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        images, targets = batch  # images: list[Tensor], targets: list[dict]

        loss_dict = self.model(images, targets)

        # Normalise: models return either a raw tensor or a dict
        if isinstance(loss_dict, dict):
            loss = sum(v for v in loss_dict.values() if isinstance(v, torch.Tensor))
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    self.log(f"train/{k}", v, prog_bar=False, on_step=True, on_epoch=False)
        else:
            loss = loss_dict

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    # ------------------------------------------------------------------
    # Validation step
    # ------------------------------------------------------------------

    def validation_step(self, batch: Any, batch_idx: int):
        preds, targets = self._inference_step(batch)

        # Update MAP metric
        self._val_map.update(preds, targets)

        # Tracking metrics (per-frame or per-sequence)
        if self.has_tracking:
            self._update_tracking_accumulators(preds, targets)

    def on_validation_epoch_end(self):
        self._log_map(self._val_map, prefix="val")
        self._val_map.reset()

        if self.has_tracking:
            self._log_tracking("val")
            self._reset_tracking_accumulators()

    # ------------------------------------------------------------------
    # Test step
    # ------------------------------------------------------------------

    def test_step(self, batch: Any, batch_idx: int):
        preds, targets = self._inference_step(batch)
        self._test_map.update(preds, targets)

        if self.has_tracking:
            self._update_tracking_accumulators(preds, targets)

    def on_test_epoch_end(self):
        self._log_map(self._test_map, prefix="test")
        self._test_map.reset()

        if self.has_tracking:
            self._log_tracking("test")
            self._reset_tracking_accumulators()

    # ------------------------------------------------------------------
    # Shared inference helper
    # ------------------------------------------------------------------

    def _inference_step(self, batch):
        """
        Run inference and return (preds, targets) in torchmetrics MAP format.

        torchmetrics MAP expects:
            preds:   list of dict { boxes (N,4 xyxy), scores (N,), labels (N,) }
            targets: list of dict { boxes (M,4 xyxy), labels (M,) }
        """
        images, targets = batch

        with torch.no_grad():
            raw_preds = self.model(images)

        # Normalise prediction format (models may return flat list or nested list)
        preds = self._normalise_preds(raw_preds)

        # Normalise target format to xyxy absolute (FasterRCNN targets already are;
        # DINOv2 targets use normalised cxcywh so the DataLoader must convert)
        return preds, targets

    @staticmethod
    def _normalise_preds(raw) -> list[dict]:
        """Ensure all preds dicts have at minimum boxes / scores / labels."""
        normalised = []
        for p in raw:
            normalised.append({
                "boxes":  p["boxes"],
                "scores": p["scores"],
                "labels": p["labels"],
            })
        return normalised

    # ------------------------------------------------------------------
    # Tracking metric accumulation
    # ------------------------------------------------------------------

    def _update_tracking_accumulators(
        self,
        preds: list[dict],
        targets: list[dict],
    ):
        """
        Update MOTA-style counters from a batch of per-frame predictions.

        Implements greedy IoU matching (Hungarian would be more accurate but
        costlier; can be swapped in later).
        """
        iou_thresh = self.hparams.iou_match_thresh

        for pred, tgt in zip(preds, targets):
            gt_boxes    = tgt["boxes"]    # (M, 4)
            gt_ids      = tgt.get("track_ids", torch.arange(len(gt_boxes)))
            pred_boxes  = pred["boxes"]   # (N, 4)
            pred_ids    = pred.get("track_ids", torch.arange(len(pred_boxes)))

            M = len(gt_boxes)
            N = len(pred_boxes)
            self._num_gt += M

            if M == 0 or N == 0:
                self._num_fn += M
                self._num_fp += N
                continue

            # IoU matrix (M × N)
            iou = self._iou_matrix(gt_boxes, pred_boxes)

            matched_gt  = set()
            matched_pred = set()

            # Greedy matching: highest-IoU pairs first
            rows, cols = (iou >= iou_thresh).nonzero(as_tuple=False).T
            if rows.numel() > 0:
                order = iou[rows, cols].argsort(descending=True)
                rows, cols = rows[order], cols[order]

                for r, c in zip(rows.tolist(), cols.tolist()):
                    if r in matched_gt or c in matched_pred:
                        continue
                    matched_gt.add(r)
                    matched_pred.add(c)

                    gt_id   = int(gt_ids[r])
                    pr_id   = int(pred_ids[c])
                    prev_pr = self._last_gt_to_pred.get(gt_id)

                    if prev_pr is not None and prev_pr != pr_id:
                        self._num_id_switch += 1
                    self._last_gt_to_pred[gt_id] = pr_id
                    self._num_tp += 1

            self._num_fn += M - len(matched_gt)
            self._num_fp += N - len(matched_pred)

    @staticmethod
    def _iou_matrix(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
        """Compute pairwise IoU (M × N) between two sets of xyxy boxes."""
        x1 = torch.max(boxes_a[:, None, 0], boxes_b[None, :, 0])
        y1 = torch.max(boxes_a[:, None, 1], boxes_b[None, :, 1])
        x2 = torch.min(boxes_a[:, None, 2], boxes_b[None, :, 2])
        y2 = torch.min(boxes_a[:, None, 3], boxes_b[None, :, 3])

        inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
        area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
        union = area_a[:, None] + area_b[None, :] - inter

        return inter / union.clamp(min=1e-6)

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log_map(self, metric: MeanAveragePrecision, prefix: str):
        result = metric.compute()
        self.log(f"{prefix}/AP50",    result["map_50"],   prog_bar=True)
        self.log(f"{prefix}/AP",      result["map"],      prog_bar=False)
        self.log(f"{prefix}/AR_100",  result.get("mar_100", torch.tensor(0.0)))

    def _log_tracking(self, prefix: str):
        denom = max(self._num_gt, 1)
        mota  = 1.0 - (self._num_fp + self._num_fn + self._num_id_switch) / denom
        prec  = self._num_tp / max(self._num_tp + self._num_fp, 1)
        rec   = self._num_tp / max(self._num_tp + self._num_fn, 1)
        idf1  = 2 * prec * rec / max(prec + rec, 1e-6)

        self.log(f"{prefix}/MOTA",        torch.tensor(mota),  prog_bar=True)
        self.log(f"{prefix}/IDF1",        torch.tensor(idf1),  prog_bar=False)
        self.log(f"{prefix}/ID_switches", torch.tensor(float(self._num_id_switch)))

    # ------------------------------------------------------------------
    # Optimiser & scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            params,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        sched_name = self.hparams.lr_scheduler
        if sched_name is None:
            return opt

        if sched_name == "cosine":
            sched = torch.optim.lr_scheduler.SequentialLR(
                opt,
                schedulers=[
                    torch.optim.lr_scheduler.LinearLR(
                        opt,
                        start_factor=1e-3,
                        end_factor=1.0,
                        total_iters=self.hparams.warmup_epochs,
                    ),
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        opt,
                        T_max=self.hparams.total_epochs - self.hparams.warmup_epochs,
                    ),
                ],
                milestones=[self.hparams.warmup_epochs],
            )
        elif sched_name == "step":
            sched = torch.optim.lr_scheduler.StepLR(
                opt,
                step_size=self.hparams.step_size,
                gamma=self.hparams.gamma,
            )
        else:
            raise ValueError(f"Unknown lr_scheduler: '{sched_name}'")

        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
