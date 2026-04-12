"""
VideoTrackerEvaluationModule
============================
Lightning module for evaluating prompt-based video object trackers
(SAM2, SAM3, or any tracker exposing the same stateful interface:
``init_video / add_prompts / propagate / reset_state / _empty_output``).

Since these trackers are prompt-based (no training), this module only
implements test_step. It receives VideoClipSample batches from SAM2DataModule
and evaluates tracking and detection quality.

Prompt strategies (applied to the **first clip** of each video only):
    "first_frame"  — GT boxes from frame 0 only; tracker propagates to all others.
    "every_n"      — GT boxes injected every N frames; tests re-prompting benefit.

For subsequent clips of the same video, predictions from the previous clip's
last frame are used as prompts (no GT), ensuring fair comparison with other
models that do not receive GT re-prompting at clip boundaries.
"""

from __future__ import annotations

import time
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import lightning as L
from torchmetrics.detection import MeanAveragePrecision

from datasets.base import VideoClipSample
from obb_utils import obb_iou_matrix


class VideoTrackerEvaluationModule(L.LightningModule):
    """
    Evaluation-only Lightning module for prompt-based video trackers.

    The `model` must expose:
        init_video(frames) / add_prompts(frame_idx, boxes, labels, obj_ids)
        propagate() -> list[dict]  /  reset_state()
        _empty_output() -> dict

    Args:
        model:            Tracker instance (SAM2Tracker, SAM3Tracker, ...).
        prompt_strategy:  "first_frame" or "every_n".
        prompt_interval:  Re-prompt every N frames (only for "every_n").
    """

    def __init__(
        self,
        model: nn.Module,
        prompt_strategy: Literal["first_frame", "every_n"] = "first_frame",
        prompt_interval: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.prompt_strategy = prompt_strategy
        self.prompt_interval = prompt_interval

        # Detection metrics
        self._test_map = MeanAveragePrecision(iou_thresholds=[0.5])
        self._det_tp = 0
        self._det_fp = 0
        self._det_fn = 0

        # Tracking accumulators
        self._num_gt = 0
        self._num_tp = 0
        self._num_fp = 0
        self._num_fn = 0
        self._num_id_switch = 0
        self._last_gt_to_pred: dict[int, int] = {}

        # Timing
        self._test_time_total = 0.0
        self._test_num_frames = 0

        # Cross-clip state: video_id → last frame predictions.
        # Only the first clip of each video receives GT prompts;
        # subsequent clips use these carry-over predictions.
        self._carry_over: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Test step
    # ------------------------------------------------------------------

    def test_step(self, batch: list[VideoClipSample], batch_idx: int):
        results = []
        for clip in batch:
            clip_results = self._evaluate_clip(clip)
            if clip_results is not None:
                results.extend(clip_results)
        return results

    def _evaluate_clip(self, clip: VideoClipSample) -> list[dict] | None:
        """Process one video clip: prompt → propagate → evaluate.

        For the first clip of a video, GT boxes are used as prompts (according
        to the configured strategy).  For subsequent clips, predictions from
        the previous clip's last frame are used instead, so SAM2 must track
        objects across clip boundaries without GT assistance.

        Returns a list of per-frame dicts with images, preds, targets and metadata
        for the visualization callback.
        """
        T = len(clip.frame_ids)
        if T == 0:
            return None

        # Convert frames to numpy uint8 HWC (SAM2 expects this)
        frames_np = [
            (clip.frames[t].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            for t in range(T)
        ]

        # --- Time the SAM2 pipeline ---
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        self.model.init_video(frames_np)

        # Decide prompt source: GT (first clip) or carry-over (subsequent)
        video_id = clip.video_id
        carry = self._carry_over.get(video_id)

        has_prompts = True
        if carry is None:
            # First clip of this video — use GT prompts
            has_prompts = self._add_gt_prompts(clip, T)
        else:
            # Continuation — use predictions from previous clip's last frame
            has_prompts = self._add_carry_over_prompts(carry)

        # Propagate (skip if no prompts — tracker lost all objects)
        if has_prompts:
            preds = self.model.propagate()
        else:
            preds = [self.model._empty_output() for _ in range(T)]
        self.model.reset_state()

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        self._test_time_total += elapsed
        self._test_num_frames += T

        # Store last frame predictions for next clip of this video
        if preds:
            self._carry_over[video_id] = preds[-1]

        # --- Evaluate each frame ---
        has_obb = clip.obb is not None
        frame_results = []
        for t in range(min(T, len(preds))):
            pred = preds[t]
            gt_boxes = clip.boxes[t]
            gt_labels = clip.labels[t]
            gt_track_ids = clip.track_ids[t]
            gt_obb = clip.obb[t] if has_obb else None

            # Ensure pred tensors are on the same device as GT
            device = gt_boxes.device
            for k in ("boxes", "obb", "scores", "labels", "track_ids"):
                if k in pred and isinstance(pred[k], torch.Tensor):
                    pred[k] = pred[k].to(device)

            tgt = {"boxes": gt_boxes, "labels": gt_labels}

            # MAP expects lists of dicts (AABB-based, kept as secondary reference)
            self._test_map.update(
                [{"boxes": pred["boxes"], "scores": pred["scores"], "labels": pred["labels"]}],
                [tgt],
            )

            # Detection TP/FP/FN — use OBB IoU when available
            pred_obb = pred.get("obb")
            self._update_det_accumulators(pred, tgt, pred_obb=pred_obb, gt_obb=gt_obb)

            # Tracking accumulators — use OBB IoU when available
            self._update_tracking_accumulators(
                pred, gt_boxes, gt_track_ids,
                pred_obb=pred_obb, gt_obb=gt_obb,
            )

            # Collect for visualization callback
            target_dict = {"boxes": gt_boxes, "labels": gt_labels}
            if gt_obb is not None:
                target_dict["obb"] = gt_obb
            frame_results.append({
                "image_np": frames_np[t],
                "pred": pred,
                "target": target_dict,
                "video_id": video_id,
                "frame_id": clip.frame_ids[t],
            })

        return frame_results

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def _add_gt_prompts(self, clip: VideoClipSample, T: int) -> bool:
        """Add GT prompts according to the configured strategy (first clip only).

        Returns True if at least one prompt was added, False otherwise
        (e.g. target is invisible on all prompt frames).
        """
        if self.prompt_strategy == "first_frame":
            indices = [0]
        else:  # every_n
            indices = list(range(0, T, self.prompt_interval))

        added = False
        for t in indices:
            boxes_np = clip.boxes[t].cpu().numpy()
            labels_np = clip.labels[t].cpu().numpy()
            obj_ids = clip.track_ids[t].cpu().tolist()
            # Replace -1 track IDs with unique positive IDs
            for i, oid in enumerate(obj_ids):
                if oid < 0:
                    obj_ids[i] = 1000 + i
            if len(boxes_np) > 0:
                self.model.add_prompts(t, boxes_np, labels_np, obj_ids)
                added = True
        return added

    def _add_carry_over_prompts(self, prev_pred: dict) -> bool:
        """Use predictions from the previous clip's last frame as prompts
        for frame 0 of the current clip.  No GT is used.

        Returns True if prompts were added, False if carry-over was empty
        (tracker lost all objects).
        """
        boxes = prev_pred["boxes"]
        labels = prev_pred["labels"]
        track_ids = prev_pred["track_ids"]

        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        if isinstance(track_ids, torch.Tensor):
            track_ids = track_ids.cpu().tolist()

        if len(boxes) == 0:
            return False

        self.model.add_prompts(0, boxes, labels, track_ids)
        return True

    # ------------------------------------------------------------------
    # Metric accumulators
    # ------------------------------------------------------------------

    def _update_det_accumulators(
        self, pred: dict, tgt: dict,
        pred_obb: torch.Tensor | None = None,
        gt_obb: torch.Tensor | None = None,
    ):
        gt_boxes = tgt["boxes"]
        pred_boxes = pred["boxes"]
        M, N = len(gt_boxes), len(pred_boxes)

        if M == 0:
            self._det_fp += N
            return
        if N == 0:
            self._det_fn += M
            return

        # Use OBB IoU when both sides have OBB data
        if pred_obb is not None and gt_obb is not None and len(pred_obb) > 0 and len(gt_obb) > 0:
            iou = obb_iou_matrix(gt_obb, pred_obb)
        else:
            iou = self._iou_matrix(gt_boxes, pred_boxes)

        matched_gt: set[int] = set()
        matched_pred: set[int] = set()
        rows, cols = (iou >= 0.5).nonzero(as_tuple=False).T
        if rows.numel() > 0:
            order = iou[rows, cols].argsort(descending=True)
            rows, cols = rows[order], cols[order]
            for r, c in zip(rows.tolist(), cols.tolist()):
                if r in matched_gt or c in matched_pred:
                    continue
                matched_gt.add(r)
                matched_pred.add(c)

        tp = len(matched_gt)
        self._det_tp += tp
        self._det_fp += N - tp
        self._det_fn += M - tp

    def _update_tracking_accumulators(
        self, pred: dict, gt_boxes: torch.Tensor, gt_track_ids: torch.Tensor,
        pred_obb: torch.Tensor | None = None,
        gt_obb: torch.Tensor | None = None,
    ):
        pred_boxes = pred["boxes"]
        pred_ids = pred.get("track_ids", torch.arange(len(pred_boxes)))

        M = len(gt_boxes)
        N = len(pred_boxes)
        self._num_gt += M

        if M == 0 or N == 0:
            self._num_fn += M
            self._num_fp += N
            return

        # Use OBB IoU when both sides have OBB data
        if pred_obb is not None and gt_obb is not None and len(pred_obb) > 0 and len(gt_obb) > 0:
            iou = obb_iou_matrix(gt_obb, pred_obb)
        else:
            iou = self._iou_matrix(gt_boxes, pred_boxes)

        matched_gt: set[int] = set()
        matched_pred: set[int] = set()

        rows, cols = (iou >= 0.5).nonzero(as_tuple=False).T
        if rows.numel() > 0:
            order = iou[rows, cols].argsort(descending=True)
            rows, cols = rows[order], cols[order]
            for r, c in zip(rows.tolist(), cols.tolist()):
                if r in matched_gt or c in matched_pred:
                    continue
                matched_gt.add(r)
                matched_pred.add(c)

                gt_id = int(gt_track_ids[r])
                pr_id = int(pred_ids[c])
                prev_pr = self._last_gt_to_pred.get(gt_id)
                if prev_pr is not None and prev_pr != pr_id:
                    self._num_id_switch += 1
                self._last_gt_to_pred[gt_id] = pr_id
                self._num_tp += 1

        self._num_fn += M - len(matched_gt)
        self._num_fp += N - len(matched_pred)

    @staticmethod
    def _iou_matrix(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
        """AABB IoU matrix (fallback for non-OBB datasets)."""
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
    # Epoch end: log everything
    # ------------------------------------------------------------------

    def on_test_epoch_end(self):
        # Detection AP
        result = self._test_map.compute()
        self.log("test/AP50", result["map_50"], prog_bar=True)
        self.log("test/AP", result["map"])
        self.log("test/AR_100", result.get("mar_100", torch.tensor(0.0)))
        self._test_map.reset()

        # Detection precision / recall
        prec = self._det_tp / max(self._det_tp + self._det_fp, 1)
        rec = self._det_tp / max(self._det_tp + self._det_fn, 1)
        self.log("test/Precision", torch.tensor(prec))
        self.log("test/Recall", torch.tensor(rec))

        # Tracking metrics
        denom = max(self._num_gt, 1)
        mota = 1.0 - (self._num_fp + self._num_fn + self._num_id_switch) / denom
        t_prec = self._num_tp / max(self._num_tp + self._num_fp, 1)
        t_rec = self._num_tp / max(self._num_tp + self._num_fn, 1)
        idf1 = 2 * t_prec * t_rec / max(t_prec + t_rec, 1e-6)

        self.log("test/MOTA", torch.tensor(mota), prog_bar=True)
        self.log("test/IDF1", torch.tensor(idf1))
        self.log("test/ID_switches", torch.tensor(float(self._num_id_switch)))

        # Speed
        fps = self._test_num_frames / max(self._test_time_total, 1e-9)
        self.log("test/total_time_s", torch.tensor(self._test_time_total))
        self.log("test/fps", torch.tensor(fps), prog_bar=True)

        # Model size
        param_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 ** 2)
        buffer_mb = sum(b.numel() * b.element_size() for b in self.model.buffers()) / (1024 ** 2)
        self.log("test/model_size_MB", torch.tensor(param_mb + buffer_mb))

    # ------------------------------------------------------------------
    # No training — dummy optimizer to satisfy Lightning
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        return None
