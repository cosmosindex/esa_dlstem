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

import json
import time
from collections import defaultdict
from pathlib import Path
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
        class_hierarchy: dict[int, int] | None = None,
    ):
        """
        ``class_hierarchy``: optional fine-class-id → coarse-class-id map.
        When set, the module assumes **predictions and targets are at the fine
        granularity** and computes a parallel set of coarse-grained metrics
        (mAP, per-class AP, PR curve) by mapping labels through the hierarchy.
        Logged under the ``_coarse`` suffix. Used for SAT-MTB style two-level
        taxonomies (4 coarse + 14 fine classes).
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.has_tracking = has_tracking
        self.class_hierarchy: dict[int, int] | None = class_hierarchy

        # torchmetrics MAP (COCO-style) – resets automatically at epoch boundaries.
        # class_metrics=True so we can log per-class AP alongside the global mAP.
        self._val_map  = MeanAveragePrecision(iou_thresholds=[0.5], class_metrics=True)
        self._test_map = MeanAveragePrecision(iou_thresholds=[0.5], class_metrics=True)

        # Parallel coarse-grained MAP metrics (only when a hierarchy is given).
        if self.class_hierarchy is not None:
            self._val_map_coarse  = MeanAveragePrecision(
                iou_thresholds=[0.5], class_metrics=True,
            )
            self._test_map_coarse = MeanAveragePrecision(
                iou_thresholds=[0.5], class_metrics=True,
            )
            # Separate per-class PR-curve accumulator at the coarse level.
            self._test_pr_preds_coarse: dict[int, list[tuple[float, int]]] = defaultdict(list)
            self._test_pr_gt_count_coarse: dict[int, int] = defaultdict(int)

        # Detection precision/recall accumulators (IoU >= 0.5)
        self._reset_det_accumulators("val")
        self._reset_det_accumulators("test")

        # Tracking accumulators (reset at epoch start)
        self._reset_tracking_accumulators()

        # Test-time timing accumulators
        self._test_time_total = 0.0
        self._test_num_images = 0

        # Per-class PR-curve records for the test split (one entry per pred):
        # {class_label: [(score, is_tp), ...]}; also per-class GT count.
        self._test_pr_preds: dict[int, list[tuple[float, int]]] = defaultdict(list)
        self._test_pr_gt_count: dict[int, int] = defaultdict(int)

        # Class-agnostic (pooled) PR-curve records → Overall AP.
        self._test_overall_pr_preds: list[tuple[float, int]] = []
        self._test_overall_gt_count: int = 0

        # Cached sample for one-shot FLOPs counting on the first test batch.
        self._flop_sample: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Tracking state helpers
    # ------------------------------------------------------------------

    def _reset_det_accumulators(self, prefix: str):
        setattr(self, f"_{prefix}_det_tp", 0)
        setattr(self, f"_{prefix}_det_fp", 0)
        setattr(self, f"_{prefix}_det_fn", 0)

    def _map_labels_to_coarse(self, lbls: torch.Tensor) -> torch.Tensor:
        """Map a fine-grained label tensor to coarse ids via ``class_hierarchy``.

        Unknown fine ids (not present in the mapping) are preserved as-is so
        the caller can decide how to treat them (they usually get filtered
        out by a downstream class_map that only knows coarse ids).
        """
        if self.class_hierarchy is None or lbls.numel() == 0:
            return lbls
        mapped = lbls.clone()
        for fine_id, coarse_id in self.class_hierarchy.items():
            mapped[lbls == fine_id] = coarse_id
        return mapped

    def _coarsen(self, preds: list[dict], targets: list[dict]) -> tuple[list[dict], list[dict]]:
        """Return ``(preds, targets)`` with ``labels`` mapped through the hierarchy."""
        coarse_preds = [
            {**p, "labels": self._map_labels_to_coarse(p["labels"])} for p in preds
        ]
        coarse_targets = [
            {**t, "labels": self._map_labels_to_coarse(t["labels"])} for t in targets
        ]
        return coarse_preds, coarse_targets

    def _update_det_accumulators(self, prefix: str, preds: list[dict], targets: list[dict]):
        """
        Count TP / FP / FN at IoU >= 0.5 using greedy matching.

        On the test split we additionally record:
          * **Per-class** ``(score, is_tp)`` pairs via class-aware matching →
            per-class AP (COCO convention) and per-class PR curves.
          * **Class-agnostic** ``(score, is_tp)`` pairs via label-ignoring
            matching → Overall AP (pool everything as object-vs-nothing) and
            the aggregated PR curve.

        The class-agnostic TP/FP/FN above also drives the legacy P/R/F1
        scalar metrics.
        """
        for pred, tgt in zip(preds, targets):
            gt_boxes   = tgt["boxes"]
            pred_boxes = pred["boxes"]
            M, N = len(gt_boxes), len(pred_boxes)
            pred_scores = pred.get("scores", torch.zeros(N))
            gt_labels   = tgt.get("labels", torch.zeros(M, dtype=torch.long))
            pred_labels = pred.get("labels", torch.zeros(N, dtype=torch.long))

            # ---- Class-agnostic match (drives P/R/F1 + Overall AP) -------
            matched_pred_any: set[int] = set()
            if M == 0:
                setattr(self, f"_{prefix}_det_fp",
                        getattr(self, f"_{prefix}_det_fp") + N)
            elif N == 0:
                setattr(self, f"_{prefix}_det_fn",
                        getattr(self, f"_{prefix}_det_fn") + M)
            else:
                iou = self._iou_matrix(gt_boxes, pred_boxes)
                matched_gt: set[int] = set()
                rows, cols = (iou >= 0.5).nonzero(as_tuple=False).T
                if rows.numel() > 0:
                    order = iou[rows, cols].argsort(descending=True)
                    rows, cols = rows[order], cols[order]
                    for r, c in zip(rows.tolist(), cols.tolist()):
                        if r in matched_gt or c in matched_pred_any:
                            continue
                        matched_gt.add(r)
                        matched_pred_any.add(c)
                tp = len(matched_gt)
                setattr(self, f"_{prefix}_det_tp",
                        getattr(self, f"_{prefix}_det_tp") + tp)
                setattr(self, f"_{prefix}_det_fp",
                        getattr(self, f"_{prefix}_det_fp") + (N - tp))
                setattr(self, f"_{prefix}_det_fn",
                        getattr(self, f"_{prefix}_det_fn") + (M - tp))

            if prefix != "test":
                continue

            # ---- Overall AP pool (class-agnostic) ------------------------
            self._test_overall_gt_count += M
            for i in range(N):
                is_tp = 1 if i in matched_pred_any else 0
                self._test_overall_pr_preds.append((float(pred_scores[i]), is_tp))

            # ---- Per-class PR-curve records ------------------------------
            for c in gt_labels.tolist():
                self._test_pr_gt_count[int(c)] += 1
            if N == 0:
                continue
            # For each class present in predictions, greedy-match within class
            # at IoU >= 0.5. Unmatched preds are FP for that class.
            uniq = torch.unique(pred_labels).tolist()
            for cls in uniq:
                cls = int(cls)
                p_idx = (pred_labels == cls).nonzero(as_tuple=True)[0].tolist()
                g_idx = (gt_labels == cls).nonzero(as_tuple=True)[0].tolist()
                cls_scores = [float(pred_scores[i]) for i in p_idx]
                if not g_idx:
                    for s in cls_scores:
                        self._test_pr_preds[cls].append((s, 0))
                    continue
                cls_iou = self._iou_matrix(gt_boxes[g_idx], pred_boxes[p_idx])
                mg: set[int] = set()
                mp: set[int] = set()
                r, c = (cls_iou >= 0.5).nonzero(as_tuple=False).T
                if r.numel() > 0:
                    order = cls_iou[r, c].argsort(descending=True)
                    r, c = r[order], c[order]
                    for rr, cc in zip(r.tolist(), c.tolist()):
                        if rr in mg or cc in mp:
                            continue
                        mg.add(rr)
                        mp.add(cc)
                for j, s in enumerate(cls_scores):
                    self._test_pr_preds[cls].append((s, 1 if j in mp else 0))

    def _log_precision_recall(self, prefix: str):
        tp = getattr(self, f"_{prefix}_det_tp")
        fp = getattr(self, f"_{prefix}_det_fp")
        fn = getattr(self, f"_{prefix}_det_fn")
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-6)
        self.log(f"{prefix}/Precision", torch.tensor(prec), prog_bar=False)
        self.log(f"{prefix}/Recall",    torch.tensor(rec),  prog_bar=False)
        self.log(f"{prefix}/F1",        torch.tensor(f1),   prog_bar=False)
        self._reset_det_accumulators(prefix)

    def _reset_tracking_accumulators(self):
        self._num_gt        = 0
        self._num_tp        = 0   # true positives (matched & correct track ID)
        self._num_id_switch = 0
        self._num_fp        = 0
        self._num_fn        = 0
        self._iou_sum       = 0.0  # sum of IoU over matched TP pairs (for MOTP)
        # last known GT→pred track-ID assignment (for ID-switch detection)
        self._last_gt_to_pred: dict[int, int] = {}
        # Per-GT-track coverage (for MT / ML): total frames and matched frames.
        self._gt_track_total:   dict[int, int] = {}
        self._gt_track_matched: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        images, targets = batch  # images: list[Tensor], targets: list[dict]
        bs = len(images)

        loss_dict = self.model(images, targets)

        # Normalise: models return either a raw tensor or a dict
        if isinstance(loss_dict, dict):
            loss = loss_dict["loss"] if "loss" in loss_dict else sum(
                v for v in loss_dict.values() if isinstance(v, torch.Tensor)
            )
            for k, v in loss_dict.items():
                if k != "loss" and isinstance(v, torch.Tensor):
                    self.log(f"train/{k}", v, prog_bar=False, on_step=True,
                             on_epoch=False, batch_size=bs)
        else:
            loss = loss_dict

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True,
                 batch_size=bs)
        return loss

    # ------------------------------------------------------------------
    # Validation step
    # ------------------------------------------------------------------

    def validation_step(self, batch: Any, batch_idx: int):
        preds, targets = self._inference_step(batch)

        self._val_map.update(preds, targets)
        self._update_det_accumulators("val", preds, targets)

        if self.class_hierarchy is not None:
            c_preds, c_targets = self._coarsen(preds, targets)
            self._val_map_coarse.update(c_preds, c_targets)

        if self.has_tracking:
            self._update_tracking_accumulators(preds, targets)

    def on_validation_epoch_end(self):
        self._log_map(self._val_map, prefix="val")
        self._val_map.reset()
        if self.class_hierarchy is not None:
            self._log_map(self._val_map_coarse, prefix="val", suffix="_coarse")
            self._val_map_coarse.reset()
        self._log_precision_recall("val")

        if self.has_tracking:
            self._log_tracking("val")
            self._reset_tracking_accumulators()

    # ------------------------------------------------------------------
    # Test step
    # ------------------------------------------------------------------

    def test_step(self, batch: Any, batch_idx: int):
        images, targets = batch
        n_images = len(images)

        # Cache a single-image sample on the first batch for one-shot FLOPs.
        if self._flop_sample is None and n_images > 0:
            self._flop_sample = images[0].detach()

        # Time the inference only (exclude metric bookkeeping)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            raw_preds = self.model(images)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        self._test_time_total += elapsed
        self._test_num_images += n_images

        preds = self._normalise_preds(raw_preds)
        self._test_map.update(preds, targets)
        self._update_det_accumulators("test", preds, targets)

        if self.class_hierarchy is not None:
            c_preds, c_targets = self._coarsen(preds, targets)
            self._test_map_coarse.update(c_preds, c_targets)
            self._update_pr_coarse(c_preds, c_targets)

        if self.has_tracking:
            self._update_tracking_accumulators(preds, targets)

    def on_test_epoch_end(self):
        self._log_map(self._test_map, prefix="test")
        self._test_map.reset()
        if self.class_hierarchy is not None:
            self._log_map(self._test_map_coarse, prefix="test", suffix="_coarse")
            self._test_map_coarse.reset()
        self._log_precision_recall("test")

        # Test-time speed
        fps = self._test_num_images / max(self._test_time_total, 1e-9)
        self.log("test/total_time_s", torch.tensor(self._test_time_total))
        self.log("test/fps", torch.tensor(fps), prog_bar=True)
        self._test_time_total = 0.0
        self._test_num_images = 0

        # Model size (parameters + buffers in MB) + raw parameter count.
        param_mb  = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 ** 2)
        buffer_mb = sum(b.numel() * b.element_size() for b in self.model.buffers()) / (1024 ** 2)
        n_params  = sum(p.numel() for p in self.model.parameters())
        self.log("test/model_size_MB", torch.tensor(param_mb + buffer_mb))
        self.log("test/Params",        torch.tensor(float(n_params)))
        self.log("test/Params_M",      torch.tensor(n_params / 1e6))

        # One-shot FLOPs count on the cached sample.
        flops = self._compute_flops()
        if flops is not None:
            self.log("test/FLOPs",   torch.tensor(float(flops)))
            self.log("test/GFLOPs",  torch.tensor(flops / 1e9))
        self._flop_sample = None

        # PR curves (test split only — written to trainer.default_root_dir).
        self._save_pr_curves()

        if self.has_tracking:
            self._log_tracking("test")
            self._reset_tracking_accumulators()

    # ------------------------------------------------------------------
    # FLOPs + PR-curve helpers
    # ------------------------------------------------------------------

    def _compute_flops(self) -> int | None:
        """
        One-shot FLOPs count using ``torch.utils.flop_counter.FlopCounterMode``.
        Returns None if the counter fails (some backends don't dispatch cleanly
        through the flop counter).
        """
        if self._flop_sample is None:
            return None
        try:
            from torch.utils.flop_counter import FlopCounterMode
        except ImportError:
            return None
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad(), FlopCounterMode(display=False) as fcm:
                self.model([self._flop_sample])
            return int(fcm.get_total_flops())
        except Exception:
            return None
        finally:
            self.model.train(was_training)

    def _save_pr_curves(self) -> None:
        """
        Build PR curves at IoU >= 0.5 from the accumulated ``(score, is_tp)``
        pairs — one curve per class (class-aware matching) and one **Overall**
        curve (class-agnostic pooled matching) — then save ``pr_curve.json``
        and PNG plot under the trainer's ``default_root_dir``. Also logs
        ``test/AP_overall`` (class-agnostic) as a scalar. Safe no-op if no
        data collected.
        """
        def _pr_and_ap(records: list[tuple[float, int]], n_gt: int) -> tuple[dict, float]:
            records_sorted = sorted(records, key=lambda x: x[0], reverse=True)
            tp_cum, fp_cum = 0, 0
            precs, recs = [], []
            for _, is_tp in records_sorted:
                if is_tp:
                    tp_cum += 1
                else:
                    fp_cum += 1
                precs.append(tp_cum / max(tp_cum + fp_cum, 1))
                recs.append(tp_cum / n_gt)
            # AP via 11-point interpolation (PASCAL-style) — cheap and robust.
            ap = 0.0
            for t in [i / 10 for i in range(11)]:
                p_above = [p for p, r in zip(precs, recs) if r >= t]
                ap += max(p_above) if p_above else 0.0
            ap /= 11.0
            scores = [r[0] for r in records_sorted]
            return {
                "n_gt": int(n_gt),
                "n_pred": len(records),
                "AP_11pt": round(float(ap), 4),
                "scores": [round(s, 6) for s in scores],
                "precision": [round(p, 6) for p in precs],
                "recall":    [round(r, 6) for r in recs],
            }, float(ap)

        def _reset_state():
            self._test_pr_preds.clear()
            self._test_pr_gt_count.clear()
            self._test_overall_pr_preds.clear()
            self._test_overall_gt_count = 0
            if self.class_hierarchy is not None:
                self._test_pr_preds_coarse.clear()
                self._test_pr_gt_count_coarse.clear()

        if not self._test_pr_preds and not self._test_overall_pr_preds:
            _reset_state()
            return

        trainer = self.trainer
        out_dir: Path | None = None
        if trainer is not None and trainer.default_root_dir:
            out_dir = Path(trainer.default_root_dir)
        if out_dir is None:
            _reset_state()
            return
        out_dir.mkdir(parents=True, exist_ok=True)

        def _build_payload(
            per_class_records: dict[int, list[tuple[float, int]]],
            per_class_gt: dict[int, int],
            overall_records: list[tuple[float, int]] | None,
            overall_gt: int,
        ) -> tuple[dict, float | None, dict[str, dict]]:
            per_cls: dict[str, dict] = {}
            for cls, records in per_class_records.items():
                n_gt = per_class_gt.get(cls, 0)
                if n_gt == 0 or not records:
                    continue
                curve, _ = _pr_and_ap(records, n_gt)
                per_cls[f"cls{int(cls)}"] = curve

            overall_curve: dict | None = None
            ap_overall: float | None = None
            if overall_records is not None and overall_records and overall_gt > 0:
                overall_curve, ap_overall = _pr_and_ap(overall_records, overall_gt)

            out: dict = {}
            if ap_overall is not None:
                out["AP_overall_11pt"] = round(float(ap_overall), 4)
            if per_cls:
                out["mAP_11pt"] = round(
                    sum(v["AP_11pt"] for v in per_cls.values()) / len(per_cls), 4,
                )
                out["per_class"] = per_cls
            if overall_curve is not None:
                out["overall"] = overall_curve
            return out, ap_overall, per_cls

        def _plot(path: Path, per_cls: dict[str, dict], overall_curve: dict | None,
                  title_suffix: str = "") -> None:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
            except ImportError:
                return
            fig, ax = plt.subplots(figsize=(6, 5))
            if overall_curve is not None:
                ax.plot(overall_curve["recall"], overall_curve["precision"],
                        "k-", linewidth=2,
                        label=f"Overall [AP={overall_curve['AP_11pt']:.3f}]")
            for name, v in sorted(per_cls.items()):
                ax.plot(v["recall"], v["precision"], "--",
                        label=f"{name} [AP={v['AP_11pt']:.3f}]")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            title = f"Precision–Recall (IoU≥0.5){title_suffix}"
            if per_cls:
                m = sum(v["AP_11pt"] for v in per_cls.values()) / len(per_cls)
                title += f" — mAP={m:.3f}"
            ax.set_title(title)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="lower left", fontsize=8)
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)

        # Primary (fine if hierarchy is set; whatever-granularity-the-user-has otherwise)
        payload, ap_overall, per_cls_primary = _build_payload(
            self._test_pr_preds, self._test_pr_gt_count,
            self._test_overall_pr_preds, self._test_overall_gt_count,
        )
        if ap_overall is not None:
            self.log("test/AP_overall", torch.tensor(ap_overall))
        if payload:
            with open(out_dir / "pr_curve.json", "w") as f:
                json.dump(payload, f, indent=2)
            _plot(out_dir / "pr_curve.png", per_cls_primary,
                  payload.get("overall"))

        # Coarse (only when class_hierarchy was provided).
        if self.class_hierarchy is not None and self._test_pr_preds_coarse:
            coarse_payload, _, per_cls_coarse = _build_payload(
                self._test_pr_preds_coarse, self._test_pr_gt_count_coarse,
                None, 0,  # Overall pool is granularity-agnostic — not recomputed
            )
            if coarse_payload:
                with open(out_dir / "pr_curve_coarse.json", "w") as f:
                    json.dump(coarse_payload, f, indent=2)
                _plot(out_dir / "pr_curve_coarse.png",
                      per_cls_coarse, None, title_suffix=" · coarse")

        _reset_state()

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

            # Record each GT track's frame-presence (for MT / ML coverage).
            for gt_id in gt_ids.tolist():
                self._gt_track_total[int(gt_id)] = \
                    self._gt_track_total.get(int(gt_id), 0) + 1

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
                    self._iou_sum += float(iou[r, c])
                    self._gt_track_matched[gt_id] = \
                        self._gt_track_matched.get(gt_id, 0) + 1

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

    def _log_map(
        self,
        metric: MeanAveragePrecision,
        prefix: str,
        suffix: str = "",
    ):
        """
        Log detection AP at IoU=0.5. The underlying ``MeanAveragePrecision`` is
        configured with a single threshold (``iou_thresholds=[0.5]``), so
        ``result["map"]`` is AP@0.5 meaned over classes (= mAP), and
        ``result["map_per_class"]`` gives per-class AP@0.5.

        ``suffix`` lets callers distinguish parallel computations (e.g. the
        coarse-grained channel from ``class_hierarchy`` → ``suffix="_coarse"``).

        Overall AP (class-agnostic pool) is computed separately from the
        PR-curve accumulator and logged in ``_save_pr_curves``.
        """
        result = metric.compute()
        self.log(f"{prefix}/mAP{suffix}", result["map"], prog_bar=(suffix == ""))

        per_class = result.get("map_per_class")
        classes   = result.get("classes")
        if per_class is not None and torch.is_tensor(per_class) and per_class.numel() > 0:
            ids = classes.tolist() if torch.is_tensor(classes) else list(range(per_class.numel()))
            for cls_id, ap in zip(ids, per_class.tolist()):
                self.log(f"{prefix}/AP_per_class{suffix}/cls{int(cls_id)}",
                         torch.tensor(float(ap)))

    def _update_pr_coarse(self, preds: list[dict], targets: list[dict]):
        """Accumulate per-class PR records at the coarse level (test only)."""
        if self.class_hierarchy is None:
            return
        for pred, tgt in zip(preds, targets):
            gt_boxes    = tgt["boxes"]
            pred_boxes  = pred["boxes"]
            M, N = len(gt_boxes), len(pred_boxes)
            pred_scores = pred.get("scores", torch.zeros(N))
            gt_labels   = tgt.get("labels", torch.zeros(M, dtype=torch.long))
            pred_labels = pred.get("labels", torch.zeros(N, dtype=torch.long))

            for c in gt_labels.tolist():
                self._test_pr_gt_count_coarse[int(c)] += 1
            if N == 0:
                continue

            uniq = torch.unique(pred_labels).tolist()
            for cls in uniq:
                cls = int(cls)
                p_idx = (pred_labels == cls).nonzero(as_tuple=True)[0].tolist()
                g_idx = (gt_labels == cls).nonzero(as_tuple=True)[0].tolist()
                cls_scores = [float(pred_scores[i]) for i in p_idx]
                if not g_idx:
                    for s in cls_scores:
                        self._test_pr_preds_coarse[cls].append((s, 0))
                    continue
                cls_iou = self._iou_matrix(gt_boxes[g_idx], pred_boxes[p_idx])
                mg, mp = set(), set()
                r, c = (cls_iou >= 0.5).nonzero(as_tuple=False).T
                if r.numel() > 0:
                    order = cls_iou[r, c].argsort(descending=True)
                    r, c = r[order], c[order]
                    for rr, cc in zip(r.tolist(), c.tolist()):
                        if rr in mg or cc in mp:
                            continue
                        mg.add(rr)
                        mp.add(cc)
                for j, s in enumerate(cls_scores):
                    self._test_pr_preds_coarse[cls].append((s, 1 if j in mp else 0))

    def _log_tracking(self, prefix: str):
        denom = max(self._num_gt, 1)
        mota  = 1.0 - (self._num_fp + self._num_fn + self._num_id_switch) / denom
        motp  = self._iou_sum / max(self._num_tp, 1)
        prec  = self._num_tp / max(self._num_tp + self._num_fp, 1)
        rec   = self._num_tp / max(self._num_tp + self._num_fn, 1)
        idf1  = 2 * prec * rec / max(prec + rec, 1e-6)

        # MT / ML: per-GT-track matched-frame coverage thresholded at 80% / 20%.
        # Expressed as fractions of total GT trajectories (MOTChallenge convention).
        n_tracks = len(self._gt_track_total)
        mt_count = ml_count = 0
        for gt_id, total in self._gt_track_total.items():
            matched = self._gt_track_matched.get(gt_id, 0)
            cov = matched / max(total, 1)
            if cov >= 0.8:
                mt_count += 1
            elif cov <= 0.2:
                ml_count += 1
        mt = mt_count / max(n_tracks, 1)
        ml = ml_count / max(n_tracks, 1)

        self.log(f"{prefix}/MOTA", torch.tensor(mota), prog_bar=True)
        self.log(f"{prefix}/MOTP", torch.tensor(motp), prog_bar=False)
        self.log(f"{prefix}/IDF1", torch.tensor(idf1), prog_bar=False)
        self.log(f"{prefix}/MT",   torch.tensor(mt),   prog_bar=False)
        self.log(f"{prefix}/ML",   torch.tensor(ml),   prog_bar=False)
        self.log(f"{prefix}/FP",   torch.tensor(float(self._num_fp)))
        self.log(f"{prefix}/FN",   torch.tensor(float(self._num_fn)))
        self.log(f"{prefix}/IDs",  torch.tensor(float(self._num_id_switch)))

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
