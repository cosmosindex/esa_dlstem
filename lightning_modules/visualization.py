"""
Visualization Callbacks
=======================
Lightning Callbacks that visualise object detection results during testing.

DetectionVisualizationCallback — for FasterRCNN / YOLO (per-frame batches)
SAM2VisualizationCallback      — for SAM2 (video clip batches)

Two evaluation modes:
  sot_mode=False (default) — Detection/MOT metrics (TP/FP/FN, MOTA, IDF1)
  sot_mode=True            — SOT metrics (Success AUC, Precision@20)
                              For each GT, only the top-1 same-class prediction
                              is evaluated; all others are ignored.

Colours:
    Green  — TP / good SOT match (IoU >= 0.5)
    Red    — FP / poor SOT match (IoU < 0.5)
    Blue   — FN / GT (SOT mode)
    Orange — TP with ID switch
    Gray   — ignored predictions (SOT mode, not matched to any GT)
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import lightning as L


# Colour palette (RGB)
_GREEN  = (0, 200, 0)       # TP / good match
_RED    = (220, 40, 40)     # FP / poor match
_BLUE   = (50, 80, 220)    # FN / GT
_ORANGE = (255, 165, 0)    # TP with ID switch
_GRAY   = (150, 150, 150)  # ignored (SOT mode)

_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.45
_THICKNESS  = 1

# Size threshold: small < 32×32 = 1024 px²  (matches bbox_stats_report_trafic.md)
_SMALL_AREA_THRESH = 1024

# SOT evaluation thresholds
_SUCCESS_THRESHOLDS  = [i * 0.05 for i in range(21)]    # 0.00 … 1.00
_PRECISION_THRESHOLDS = list(range(0, 51, 1))            # 0 … 50 pixels


def _box_area(box: np.ndarray) -> float:
    return float((box[2] - box[0]) * (box[3] - box[1]))


def _size_key(box: np.ndarray) -> str:
    return "small" if _box_area(box) < _SMALL_AREA_THRESH else "large"


def _center(box: np.ndarray) -> tuple[float, float]:
    return (float(box[0] + box[2]) / 2, float(box[1] + box[3]) / 2)


class DetectionVisualizationCallback(L.Callback):
    """
    Visualise test-set detections as bounding-box overlays.

    Args:
        class_names:       Dict mapping class id → display name, e.g. {1: "car"}.
        output_dir:        Local directory to save all visualizations and metrics.
        iou_thresh:        IoU threshold for TP / FP / FN matching (detection mode).
        max_wandb_images:  Max images to log to W&B (local saves are unlimited).
        score_thresh:      Only draw predictions with score >= this value.
        sot_mode:          If True, use SOT evaluation (Success/Precision plots)
                           instead of detection TP/FP/FN metrics.
    """

    def __init__(
        self,
        class_names: dict[int, str],
        output_dir: str | Path = "experiments",
        iou_thresh: float = 0.5,
        max_wandb_images: int = 50,
        score_thresh: float = 0.5,
        sot_mode: bool = False,
    ):
        super().__init__()
        self.class_names = class_names
        self.output_dir = Path(output_dir)
        self.iou_thresh = iou_thresh
        self.max_wandb_images = max_wandb_images
        self.score_thresh = score_thresh
        self.sot_mode = sot_mode

        self._wandb_logged = 0
        self._vis_dir: Path | None = None
        self._all_metrics: list[dict] = []

        # Per-video tracking state (detection mode only)
        self._video_tracking: dict[str, dict[int, int]] = {}

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def on_test_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        self._wandb_logged = 0
        self._all_metrics = []
        self._video_tracking = {}

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

            vis, per_image_metrics = self._draw_detections(
                img_tensor, pred, tgt, video_id,
            )

            filename = f"{video_id}_frame{frame_id:04d}.jpg"
            cv2.imwrite(
                str(self._vis_dir / filename),
                cv2.cvtColor(vis, cv2.COLOR_RGB2BGR),
            )

            per_image_metrics["video_id"] = video_id
            per_image_metrics["frame_id"] = frame_id
            self._all_metrics.append(per_image_metrics)

            if self._wandb_logged < self.max_wandb_images:
                self._log_wandb(trainer, vis, video_id, frame_id)
                self._wandb_logged += 1

    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        metrics_path = self.output_dir / "per_image_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self._all_metrics, f, indent=2)

    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        logged = trainer.callback_metrics
        summary = {k: v.item() if isinstance(v, torch.Tensor) else v
                   for k, v in logged.items()}

        if self.sot_mode:
            summary.update(self._aggregate_sot_metrics())
        else:
            summary["per_category"] = self._aggregate_det_breakdowns("per_class")
            summary["per_size"] = self._aggregate_det_breakdowns("per_size")

        summary_path = self.output_dir / "test_metrics.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    # ------------------------------------------------------------------
    # Dispatch: detection vs SOT
    # ------------------------------------------------------------------

    def _draw_detections(
        self,
        img_tensor: torch.Tensor,
        pred: dict,
        tgt: dict,
        video_id: str,
    ) -> tuple[np.ndarray, dict]:
        img = (img_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()

        gt_boxes  = tgt["boxes"].cpu().numpy()
        gt_labels = tgt["labels"].cpu().numpy()
        gt_track_ids = self._extract_track_ids(tgt)

        pred_boxes  = pred["boxes"].cpu().numpy()
        pred_scores = pred["scores"].cpu().numpy()
        pred_labels = pred["labels"].cpu().numpy()
        pred_track_ids = self._extract_track_ids(pred)

        # Filter low-confidence predictions
        keep = pred_scores >= self.score_thresh
        pred_boxes  = pred_boxes[keep]
        pred_scores = pred_scores[keep]
        pred_labels = pred_labels[keep]
        if pred_track_ids is not None:
            pred_track_ids = pred_track_ids[keep]

        if self.sot_mode:
            return self._sot_draw(
                img, gt_boxes, gt_labels,
                pred_boxes, pred_scores, pred_labels, pred_track_ids,
            )

        return self._det_draw(
            img, gt_boxes, gt_labels, gt_track_ids,
            pred_boxes, pred_scores, pred_labels, pred_track_ids,
            video_id,
        )

    # ==================================================================
    # DETECTION MODE
    # ==================================================================

    def _det_draw(
        self,
        img: np.ndarray,
        gt_boxes: np.ndarray,
        gt_labels: np.ndarray,
        gt_track_ids: np.ndarray | None,
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray,
        pred_labels: np.ndarray,
        pred_track_ids: np.ndarray | None,
        video_id: str,
    ) -> tuple[np.ndarray, dict]:
        tp_mask, fp_mask, fn_mask, matched_pairs = self._match(gt_boxes, pred_boxes)

        idsw_preds = self._check_id_switches(
            video_id, matched_pairs, gt_track_ids, pred_track_ids,
        )

        self._draw_det_boxes(
            img, gt_boxes, gt_labels, gt_track_ids,
            pred_boxes, pred_scores, pred_labels, pred_track_ids,
            tp_mask, fp_mask, fn_mask, idsw_preds,
        )

        per_class, per_size = self._compute_det_breakdowns(
            gt_boxes, gt_labels, pred_boxes, pred_labels,
            tp_mask, fp_mask, fn_mask, matched_pairs, idsw_preds,
        )

        metrics = {
            "tp": int(tp_mask.sum()),
            "fp": int(fp_mask.sum()),
            "fn": int(fn_mask.sum()),
            "id_switches": len(idsw_preds),
            "num_preds": int(len(pred_boxes)),
            "num_gt": int(len(gt_boxes)),
            "per_class": per_class,
            "per_size": per_size,
        }
        return img, metrics

    def _match(
        self,
        gt_boxes: np.ndarray,
        pred_boxes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
        """Greedy IoU matching."""
        M, N = len(gt_boxes), len(pred_boxes)
        tp_mask = np.zeros(N, dtype=bool)
        fp_mask = np.ones(N, dtype=bool)
        fn_mask = np.ones(M, dtype=bool)
        matched_pairs: list[tuple[int, int]] = []

        if M == 0 or N == 0:
            return tp_mask, fp_mask, fn_mask, matched_pairs

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
                matched_pairs.append((int(r), int(c)))

        return tp_mask, fp_mask, fn_mask, matched_pairs

    def _check_id_switches(
        self,
        video_id: str,
        matched_pairs: list[tuple[int, int]],
        gt_track_ids: np.ndarray | None,
        pred_track_ids: np.ndarray | None,
    ) -> set[int]:
        if gt_track_ids is None or pred_track_ids is None:
            return set()
        if video_id not in self._video_tracking:
            self._video_tracking[video_id] = {}
        tracking = self._video_tracking[video_id]
        idsw_preds: set[int] = set()
        for gt_idx, pred_idx in matched_pairs:
            gt_tid = int(gt_track_ids[gt_idx])
            pred_tid = int(pred_track_ids[pred_idx])
            prev_pred_tid = tracking.get(gt_tid)
            if prev_pred_tid is not None and prev_pred_tid != pred_tid:
                idsw_preds.add(pred_idx)
            tracking[gt_tid] = pred_tid
        return idsw_preds

    def _compute_det_breakdowns(
        self,
        gt_boxes: np.ndarray,
        gt_labels: np.ndarray,
        pred_boxes: np.ndarray,
        pred_labels: np.ndarray,
        tp_mask: np.ndarray,
        fp_mask: np.ndarray,
        fn_mask: np.ndarray,
        matched_pairs: list[tuple[int, int]],
        idsw_preds: set[int],
    ) -> tuple[dict, dict]:
        per_class: dict[str, dict[str, int]] = {}
        per_size: dict[str, dict[str, int]] = {
            "small": {"tp": 0, "fp": 0, "fn": 0, "id_switches": 0, "num_gt": 0},
            "large": {"tp": 0, "fp": 0, "fn": 0, "id_switches": 0, "num_gt": 0},
        }

        def _ensure_cls(name: str):
            if name not in per_class:
                per_class[name] = {"tp": 0, "fp": 0, "fn": 0,
                                   "id_switches": 0, "num_gt": 0}

        for i in range(len(gt_boxes)):
            name = self.class_names.get(int(gt_labels[i]), f"cls{gt_labels[i]}")
            _ensure_cls(name)
            per_class[name]["num_gt"] += 1
            per_size[_size_key(gt_boxes[i])]["num_gt"] += 1

        for gt_idx, pred_idx in matched_pairs:
            name = self.class_names.get(int(gt_labels[gt_idx]), f"cls{gt_labels[gt_idx]}")
            _ensure_cls(name)
            per_class[name]["tp"] += 1
            per_size[_size_key(gt_boxes[gt_idx])]["tp"] += 1
            if pred_idx in idsw_preds:
                per_class[name]["id_switches"] += 1
                per_size[_size_key(gt_boxes[gt_idx])]["id_switches"] += 1

        for i, is_fn in enumerate(fn_mask):
            if is_fn:
                name = self.class_names.get(int(gt_labels[i]), f"cls{gt_labels[i]}")
                _ensure_cls(name)
                per_class[name]["fn"] += 1
                per_size[_size_key(gt_boxes[i])]["fn"] += 1

        for i, is_fp in enumerate(fp_mask):
            if is_fp:
                name = self.class_names.get(int(pred_labels[i]), f"cls{pred_labels[i]}")
                _ensure_cls(name)
                per_class[name]["fp"] += 1
                per_size[_size_key(pred_boxes[i])]["fp"] += 1

        return per_class, per_size

    def _draw_det_boxes(
        self,
        img: np.ndarray,
        gt_boxes: np.ndarray,
        gt_labels: np.ndarray,
        gt_track_ids: np.ndarray | None,
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray,
        pred_labels: np.ndarray,
        pred_track_ids: np.ndarray | None,
        tp_mask: np.ndarray,
        fp_mask: np.ndarray,
        fn_mask: np.ndarray,
        idsw_preds: set[int],
    ):
        for i, is_fn in enumerate(fn_mask):
            if is_fn:
                b = gt_boxes[i].astype(int)
                name = self.class_names.get(int(gt_labels[i]), f"cls{gt_labels[i]}")
                tid = f" GT#{int(gt_track_ids[i])}" if gt_track_ids is not None else ""
                self._draw_box(img, b, _BLUE, f"FN {name}{tid}")

        for i, is_fp in enumerate(fp_mask):
            if is_fp:
                b = pred_boxes[i].astype(int)
                name = self.class_names.get(int(pred_labels[i]), f"cls{pred_labels[i]}")
                tid = f" T#{int(pred_track_ids[i])}" if pred_track_ids is not None else ""
                self._draw_box(img, b, _RED, f"FP {name}{tid} {pred_scores[i]:.2f}")

        for i, is_tp in enumerate(tp_mask):
            if is_tp:
                b = pred_boxes[i].astype(int)
                name = self.class_names.get(int(pred_labels[i]), f"cls{pred_labels[i]}")
                tid = f" T#{int(pred_track_ids[i])}" if pred_track_ids is not None else ""
                if i in idsw_preds:
                    self._draw_box(img, b, _ORANGE, f"IDsw {name}{tid} {pred_scores[i]:.2f}")
                else:
                    self._draw_box(img, b, _GREEN, f"TP {name}{tid} {pred_scores[i]:.2f}")

    def _aggregate_det_breakdowns(self, key: str) -> dict:
        agg: dict[str, dict[str, int]] = {}
        for m in self._all_metrics:
            for name, counts in m.get(key, {}).items():
                if name not in agg:
                    agg[name] = {"tp": 0, "fp": 0, "fn": 0,
                                 "id_switches": 0, "num_gt": 0}
                for k in ("tp", "fp", "fn", "id_switches", "num_gt"):
                    agg[name][k] += counts.get(k, 0)

        for d in agg.values():
            tp, fp, fn = d["tp"], d["fp"], d["fn"]
            idsw, ngt = d["id_switches"], d["num_gt"]
            d["precision"] = round(tp / max(tp + fp, 1), 4)
            d["recall"] = round(tp / max(tp + fn, 1), 4)
            d["MOTA"] = round(1.0 - (fp + fn + idsw) / max(ngt, 1), 4)
            t_prec = tp / max(tp + fp, 1)
            t_rec = tp / max(tp + fn, 1)
            d["IDF1"] = round(2 * t_prec * t_rec / max(t_prec + t_rec, 1e-6), 4)

        return agg

    # ==================================================================
    # SOT MODE
    # ==================================================================

    def _sot_draw(
        self,
        img: np.ndarray,
        gt_boxes: np.ndarray,
        gt_labels: np.ndarray,
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray,
        pred_labels: np.ndarray,
        pred_track_ids: np.ndarray | None,
    ) -> tuple[np.ndarray, dict]:
        """SOT evaluation: for each GT, find top-1 same-class prediction."""
        sot_records: list[dict] = []
        matched_pred_indices: set[int] = set()

        for gi in range(len(gt_boxes)):
            gt_box = gt_boxes[gi]
            gt_cls = int(gt_labels[gi])
            gt_name = self.class_names.get(gt_cls, f"cls{gt_cls}")
            gt_sk = _size_key(gt_box)

            # Filter predictions to same class
            same_class_idx = np.where(pred_labels == gt_cls)[0]

            if len(same_class_idx) == 0:
                # No same-class prediction → miss
                self._draw_box(img, gt_box.astype(int), _BLUE,
                               f"GT {gt_name} [MISS]")
                sot_records.append({
                    "best_iou": 0.0,
                    "center_dist": float("inf"),
                    "norm_center_dist": float("inf"),
                    "gt_class": gt_name,
                    "gt_size": gt_sk,
                })
                continue

            # IoU of GT vs same-class predictions
            sc_boxes = pred_boxes[same_class_idx]
            ious = self._iou_matrix(gt_box[None], sc_boxes)[0]
            best_local = int(ious.argmax())
            best_iou = float(ious[best_local])
            best_pi = int(same_class_idx[best_local])
            matched_pred_indices.add(best_pi)

            best_box = pred_boxes[best_pi]

            # Center distance
            gt_cx, gt_cy = _center(gt_box)
            pr_cx, pr_cy = _center(best_box)
            cdist = float(np.sqrt((gt_cx - pr_cx) ** 2 + (gt_cy - pr_cy) ** 2))

            # Normalised center distance (by GT diagonal)
            gt_w, gt_h = gt_box[2] - gt_box[0], gt_box[3] - gt_box[1]
            gt_diag = float(np.sqrt(gt_w ** 2 + gt_h ** 2))
            ncdist = cdist / max(gt_diag, 1e-6)

            sot_records.append({
                "best_iou": round(best_iou, 4),
                "center_dist": round(cdist, 2),
                "norm_center_dist": round(ncdist, 4),
                "gt_class": gt_name,
                "gt_size": gt_sk,
            })

            # Draw GT
            self._draw_box(img, gt_box.astype(int), _BLUE, f"GT {gt_name}")

            # Draw best prediction
            color = _GREEN if best_iou >= self.iou_thresh else _RED
            score = pred_scores[best_pi]
            tid = ""
            if pred_track_ids is not None:
                tid = f" T#{int(pred_track_ids[best_pi])}"
            self._draw_box(
                img, best_box.astype(int), color,
                f"{gt_name}{tid} IoU={best_iou:.2f} {score:.2f}",
            )

        # Draw unmatched predictions in gray (context only)
        for pi in range(len(pred_boxes)):
            if pi not in matched_pred_indices:
                b = pred_boxes[pi].astype(int)
                name = self.class_names.get(int(pred_labels[pi]), f"cls{pred_labels[pi]}")
                self._draw_box(img, b, _GRAY, f"{name} {pred_scores[pi]:.2f}")

        return img, {"sot_records": sot_records}

    def _aggregate_sot_metrics(self) -> dict:
        """Aggregate per-frame SOT records into Success/Precision plots."""
        all_records: list[dict] = []
        for m in self._all_metrics:
            all_records.extend(m.get("sot_records", []))

        if not all_records:
            return {}

        result = self._sot_summary(all_records)

        # Per-category breakdown
        by_class: dict[str, list[dict]] = {}
        for r in all_records:
            by_class.setdefault(r["gt_class"], []).append(r)
        result["per_category"] = {
            name: self._sot_summary(recs) for name, recs in by_class.items()
        }

        # Per-size breakdown
        by_size: dict[str, list[dict]] = {}
        for r in all_records:
            by_size.setdefault(r["gt_size"], []).append(r)
        result["per_size"] = {
            name: self._sot_summary(recs) for name, recs in by_size.items()
        }

        return result

    @staticmethod
    def _sot_summary(records: list[dict]) -> dict:
        """Compute SOT metrics from a list of per-GT records."""
        n = len(records)
        ious = [r["best_iou"] for r in records]
        cdists = [r["center_dist"] for r in records]
        ncdists = [r["norm_center_dist"] for r in records]

        # Success plot
        success_plot = {
            f"{t:.2f}": round(sum(1 for v in ious if v >= t) / n, 4)
            for t in _SUCCESS_THRESHOLDS
        }
        success_auc = round(sum(success_plot.values()) / len(_SUCCESS_THRESHOLDS), 4)

        # Precision plot (center distance)
        precision_plot = {
            str(d): round(sum(1 for v in cdists if v <= d) / n, 4)
            for d in _PRECISION_THRESHOLDS
        }

        # Normalised precision plot
        norm_precision_plot = {
            f"{t:.2f}": round(sum(1 for v in ncdists if v <= t) / n, 4)
            for t in _SUCCESS_THRESHOLDS  # reuse 0..1 thresholds
        }

        return {
            "n_frames": n,
            "success_auc": success_auc,
            "precision_20": round(sum(1 for v in cdists if v <= 20) / n, 4),
            "norm_precision_50": round(sum(1 for v in ncdists if v <= 0.5) / n, 4),
            "success_plot": success_plot,
            "precision_plot": precision_plot,
            "norm_precision_plot": norm_precision_plot,
        }

    # ==================================================================
    # Shared helpers
    # ==================================================================

    @staticmethod
    def _extract_track_ids(d: dict) -> np.ndarray | None:
        ids = d.get("track_ids")
        if ids is None:
            return None
        if isinstance(ids, torch.Tensor):
            return ids.cpu().numpy()
        return np.asarray(ids)

    @staticmethod
    def _draw_box(img: np.ndarray, box: np.ndarray, color: tuple, label: str):
        x1, y1, x2, y2 = box
        bgr = color[::-1]
        cv2.rectangle(img, (x1, y1), (x2, y2), bgr, 2)
        (tw, th), _ = cv2.getTextSize(label, _FONT, _FONT_SCALE, _THICKNESS)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), bgr, -1)
        cv2.putText(img, label, (x1, y1 - 2), _FONT, _FONT_SCALE,
                    (255, 255, 255), _THICKNESS)

    @staticmethod
    def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
        y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
        x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
        y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
        area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
        union = area_a[:, None] + area_b[None, :] - inter
        return inter / np.clip(union, 1e-6, None)

    @staticmethod
    def _log_wandb(trainer: L.Trainer, img: np.ndarray, video_id: str, frame_id: int):
        logger = trainer.logger
        if logger is None:
            return
        if hasattr(logger, "experiment") and hasattr(logger.experiment, "log"):
            import wandb
            caption = f"{video_id}/frame_{frame_id}"
            logger.experiment.log({
                "test/detections": wandb.Image(img, caption=caption),
            })


# ======================================================================
# SAM2 subclass
# ======================================================================


class SAM2VisualizationCallback(DetectionVisualizationCallback):
    """
    Visualise SAM2 test-set detections.

    Reads per-frame results returned by SAM2EvaluationModule.test_step()
    (list of dicts with 'image_np', 'pred', 'target', 'video_id', 'frame_id').

    Inherits all drawing / matching / W&B / SOT logic from the base class.
    """

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        if outputs is None:
            return

        for frame_result in outputs:
            img = frame_result["image_np"].copy()
            pred = frame_result["pred"]
            tgt = frame_result["target"]
            video_id = frame_result["video_id"]
            frame_id = frame_result["frame_id"]

            vis, per_image_metrics = self._draw_from_numpy(
                img, pred, tgt, video_id,
            )

            filename = f"{video_id}_frame{frame_id:04d}.jpg"
            cv2.imwrite(
                str(self._vis_dir / filename),
                cv2.cvtColor(vis, cv2.COLOR_RGB2BGR),
            )

            per_image_metrics["video_id"] = video_id
            per_image_metrics["frame_id"] = frame_id
            self._all_metrics.append(per_image_metrics)

            if self._wandb_logged < self.max_wandb_images:
                self._log_wandb(trainer, vis, video_id, frame_id)
                self._wandb_logged += 1

    def _draw_from_numpy(
        self,
        img: np.ndarray,
        pred: dict,
        tgt: dict,
        video_id: str,
    ) -> tuple[np.ndarray, dict]:
        """Draw boxes. Input image is already numpy uint8 RGB."""
        gt_boxes = tgt["boxes"].cpu().numpy() if isinstance(tgt["boxes"], torch.Tensor) else tgt["boxes"]
        gt_labels = tgt["labels"].cpu().numpy() if isinstance(tgt["labels"], torch.Tensor) else tgt["labels"]
        gt_track_ids = self._extract_track_ids(tgt)

        pred_boxes = pred["boxes"].cpu().numpy() if isinstance(pred["boxes"], torch.Tensor) else pred["boxes"]
        pred_scores = pred["scores"].cpu().numpy() if isinstance(pred["scores"], torch.Tensor) else pred["scores"]
        pred_labels = pred["labels"].cpu().numpy() if isinstance(pred["labels"], torch.Tensor) else pred["labels"]
        pred_track_ids = self._extract_track_ids(pred)

        keep = pred_scores >= self.score_thresh
        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]
        pred_labels = pred_labels[keep]
        if pred_track_ids is not None:
            pred_track_ids = pred_track_ids[keep]

        if self.sot_mode:
            return self._sot_draw(
                img, gt_boxes, gt_labels,
                pred_boxes, pred_scores, pred_labels, pred_track_ids,
            )

        tp_mask, fp_mask, fn_mask, matched_pairs = self._match(gt_boxes, pred_boxes)

        idsw_preds = self._check_id_switches(
            video_id, matched_pairs, gt_track_ids, pred_track_ids,
        )

        self._draw_det_boxes(
            img, gt_boxes, gt_labels, gt_track_ids,
            pred_boxes, pred_scores, pred_labels, pred_track_ids,
            tp_mask, fp_mask, fn_mask, idsw_preds,
        )

        per_class, per_size = self._compute_det_breakdowns(
            gt_boxes, gt_labels, pred_boxes, pred_labels,
            tp_mask, fp_mask, fn_mask, matched_pairs, idsw_preds,
        )

        return img, {
            "tp": int(tp_mask.sum()),
            "fp": int(fp_mask.sum()),
            "fn": int(fn_mask.sum()),
            "id_switches": len(idsw_preds),
            "num_preds": int(len(pred_boxes)),
            "num_gt": int(len(gt_boxes)),
            "per_class": per_class,
            "per_size": per_size,
        }
