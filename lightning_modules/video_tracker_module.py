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
    "text"         — No boxes are passed. The tracker runs open-vocabulary
                     detection+tracking from its own internal text prompts
                     (e.g. SAM3TextTracker). Used for MOT-style evaluation.

For subsequent clips of the same video, predictions from the previous clip's
last frame are used as prompts (no GT), ensuring fair comparison with other
models that do not receive GT re-prompting at clip boundaries. Under the
"text" strategy no carry-over is performed — each clip runs detection fresh.
"""

from __future__ import annotations

import time
from typing import Literal, Optional

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
        sot_mode:         If True, skip detection (AP/AR/Precision/Recall) and
                          MOT (MOTA/IDF1/ID_switches) metrics entirely — these
                          are meaningless for single-object tracking (1 GT,
                          1 track per video). SOT-specific metrics
                          (SR/PR/NPR/P@5) are computed by SAM2SOTEvalCallback.
        det_only_mode:    If True, suppress MOT metrics (MOTA/IDF1/ID_switches)
                          and skip text-mode track-ID stitching. Detection
                          metrics (AP/AR/Precision/Recall) are still computed.
                          Use this when the model has no temporal model
                          (e.g. GroundingDINO as a per-frame detector).
                          Mutually exclusive with ``sot_mode``.
    """

    def __init__(
        self,
        model: nn.Module,
        prompt_strategy: Literal["first_frame", "every_n", "text"] = "first_frame",
        prompt_interval: int = 10,
        sot_mode: bool = False,
        det_only_mode: bool = False,
        match_metric: Literal["iou", "centroid"] = "iou",
        centroid_dist_thresh: float = 5.0,
        score_sweep: Optional[list[float]] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        if sot_mode and det_only_mode:
            raise ValueError("sot_mode and det_only_mode are mutually exclusive")
        if match_metric not in ("iou", "centroid"):
            raise ValueError(f"match_metric must be 'iou' or 'centroid', got {match_metric!r}")

        self.model = model
        self.prompt_strategy = prompt_strategy
        self.prompt_interval = prompt_interval
        self.sot_mode = sot_mode
        self.det_only_mode = det_only_mode
        # How TP/FP/FN matching is decided in det + MOT accumulators below.
        # ``"iou"`` keeps the legacy behaviour (IoU >= 0.5 hard-coded);
        # ``"centroid"`` reproduces the SVMOD-paper protocol HiEUM reports
        # under (Euclidean distance between bbox centres <= ``centroid_dist_thresh``,
        # default 5 px). When centroid matching is on we skip the
        # torchmetrics MAP path since AP is inherently IoU-based.
        self.match_metric = match_metric
        self.centroid_dist_thresh = float(centroid_dist_thresh)

        # Optional score-threshold sweep for SVMOD-style reporting.
        # When set, the same forward pass is matched against GT under
        # multiple confidence cutoffs; we log per-threshold P/R/F1 plus
        # the best F1 in test_metrics.json. HiEUM's paper sweeps
        # [0.1, 0.15, 0.2, 0.25, 0.3, 0.32, 0.34, 0.35] and reports the
        # peak F1, **macro-averaged over the 7 test sequences** (see
        # ``eval_func_final`` in HiEUM's repo). Both micro (pooled
        # TP/FP/FN over all frames) and macro (per-seq P/R/F1 then
        # arithmetic mean) are logged so users can compare apples to
        # apples with whichever protocol a paper reports.
        self.score_sweep = sorted(set(float(s) for s in score_sweep)) if score_sweep else None
        self._sweep_acc: dict[float, dict[str, int]] = (
            {s: {"tp": 0, "fp": 0, "fn": 0} for s in self.score_sweep}
            if self.score_sweep else {}
        )
        # Per-sequence per-threshold accumulators for macro averaging.
        # Keyed by ``(video_id, threshold) → {"tp", "fp", "fn"}``.
        self._sweep_seq_acc: dict[tuple[str, float], dict[str, int]] = {}
        # Populated by ``_build_sweep_summary`` at epoch end. The
        # visualization callback dumps this to ``sweep_results.json``.
        self.sweep_summary: Optional[dict] = None

        # Detection / MOT metrics — not instantiated in sot_mode.
        if not sot_mode:
            # MAP is IoU-based; only meaningful under match_metric="iou".
            self._test_map = (
                MeanAveragePrecision(iou_thresholds=[0.5])
                if match_metric == "iou" else None
            )
            self._det_tp = 0
            self._det_fp = 0
            self._det_fn = 0

            # MOT-only accumulators — skipped in det_only_mode.
            if not det_only_mode:
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

        # Text-mode track-ID stitching state (MOT, prompt_strategy="text").
        # SAM3 text-tracker's track_ids restart at 1 on every clip. We stitch
        # IDs across clip boundaries externally via IoU matching:
        #   - match this clip's frame-0 boxes against the previous clip's
        #     last-frame boxes (global IDs)
        #   - matched → reuse previous global ID
        #   - unmatched → allocate a fresh video-global ID
        # Kept separate from `_carry_over` since no box is ever sent back to
        # the model (model stays fully open-vocab, no GT leakage).
        self._text_id_state: dict[str, dict] = {}

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

        video_id = clip.video_id

        if self.prompt_strategy == "text":
            # Open-vocabulary MOT: tracker uses text prompts, not GT boxes.
            # Forward the clip's dominant category if the tracker accepts it,
            # so datasets like AIR-MOT drive SAM3 with a single noun phrase
            # per video instead of looping over every class.
            if hasattr(self.model, "set_text_prompt"):
                # Drive SAM3 with the clip's dominant category as a single noun
                # phrase only when it names a class the tracker knows (e.g.
                # AIR-MOT "airplane"). Datasets whose videos mix classes set
                # category="mixed" (BIRDSAI MOT) — that isn't a prompt, so pass
                # None and let the tracker loop over every class prompt instead.
                cat = getattr(clip, "category", "") or None
                known = set(getattr(self.model, "class_names", []) or [])
                self.model.set_text_prompt(cat if cat in known else None)
            preds = self.model.propagate()
            # Cross-clip ID stitching only matters for MOT — skip in det-only.
            if not self.det_only_mode:
                self._stitch_text_track_ids(preds, video_id)
        else:
            # Decide prompt source: GT (first clip) or carry-over (subsequent)
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
        # (skipped for text strategy — each clip re-runs detection from scratch)
        if preds and self.prompt_strategy != "text":
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

            pred_obb = pred.get("obb")

            # Skip detection / MOT accumulation entirely in SOT mode — those
            # metrics are not meaningful for single-object tracking and the
            # SOT-specific metrics come from SAM2SOTEvalCallback.
            if not self.sot_mode:
                # MAP expects lists of dicts (AABB-based, kept as secondary
                # reference). Skipped under centroid matching since AP is
                # inherently IoU-based and would mix two protocols.
                if self._test_map is not None:
                    self._test_map.update(
                        [{"boxes": pred["boxes"], "scores": pred["scores"], "labels": pred["labels"]}],
                        [tgt],
                    )

                # Detection TP/FP/FN — use OBB IoU when available
                self._update_det_accumulators(pred, tgt, pred_obb=pred_obb, gt_obb=gt_obb)

                # Score-threshold sweep — same matcher, different score cutoffs.
                if self.score_sweep:
                    self._update_score_sweep(
                        pred, tgt, video_id=video_id,
                        pred_obb=pred_obb, gt_obb=gt_obb,
                    )

                # Tracking accumulators — skipped in det_only_mode.
                if not self.det_only_mode:
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

    def _build_match_score(
        self,
        gt_boxes: torch.Tensor,
        pred_boxes: torch.Tensor,
        pred_obb: torch.Tensor | None,
        gt_obb: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Build the (score, accept-mask) matrices for greedy matching.

        ``score`` is the value sorted on (descending for IoU, ascending for
        centroid-distance); ``accept`` is a bool matrix of pairs that meet
        the threshold. Returns ``(score, accept, descending)`` so the
        caller can pick sort direction.
        """
        if self.match_metric == "centroid":
            dist = self._centroid_dist_matrix(gt_boxes, pred_boxes)
            return dist, dist <= self.centroid_dist_thresh, False

        if pred_obb is not None and gt_obb is not None and len(pred_obb) > 0 and len(gt_obb) > 0:
            iou = obb_iou_matrix(gt_obb, pred_obb)
        else:
            iou = self._iou_matrix(gt_boxes, pred_boxes)
        return iou, iou >= 0.5, True

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

        score, accept, descending = self._build_match_score(
            gt_boxes, pred_boxes, pred_obb, gt_obb,
        )

        matched_gt: set[int] = set()
        matched_pred: set[int] = set()
        rows, cols = accept.nonzero(as_tuple=False).T
        if rows.numel() > 0:
            order = score[rows, cols].argsort(descending=descending)
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

    def _update_score_sweep(
        self, pred: dict, tgt: dict,
        video_id: str,
        pred_obb: torch.Tensor | None = None,
        gt_obb: torch.Tensor | None = None,
    ):
        """Per-frame accumulator that re-runs matching at every cutoff in
        ``self.score_sweep``. Reuses the same matcher
        (``_build_match_score``) so the 5 px centroid criterion is honored.

        Maintains both **global** counts (``self._sweep_acc``) for micro-
        averaged P/R/F1 *and* **per-video** counts (``self._sweep_seq_acc``)
        so we can compute the macro-average HiEUM's paper reports.

        Cheap because: K ≤ 128 preds and dozens-to-hundreds of GT per
        frame; matching is O(M·N).
        """
        gt_boxes = tgt["boxes"]
        all_boxes = pred["boxes"]
        all_scores = pred["scores"]
        M = len(gt_boxes)
        N_full = len(all_boxes)

        for thr in self.score_sweep:
            keep = all_scores >= thr
            pred_boxes = all_boxes[keep]
            sub_pred_obb = pred_obb[keep] if (pred_obb is not None and N_full > 0) else None
            N = len(pred_boxes)

            acc = self._sweep_acc[thr]
            seq_acc = self._sweep_seq_acc.setdefault(
                (video_id, thr), {"tp": 0, "fp": 0, "fn": 0},
            )
            if M == 0:
                acc["fp"] += N
                seq_acc["fp"] += N
                continue
            if N == 0:
                acc["fn"] += M
                seq_acc["fn"] += M
                continue

            score, accept, descending = self._build_match_score(
                gt_boxes, pred_boxes, sub_pred_obb, gt_obb,
            )
            matched_gt: set[int] = set()
            matched_pred: set[int] = set()
            rows, cols = accept.nonzero(as_tuple=False).T
            if rows.numel() > 0:
                order = score[rows, cols].argsort(descending=descending)
                rows, cols = rows[order], cols[order]
                for r, c in zip(rows.tolist(), cols.tolist()):
                    if r in matched_gt or c in matched_pred:
                        continue
                    matched_gt.add(r)
                    matched_pred.add(c)

            tp = len(matched_gt)
            acc["tp"] += tp
            acc["fp"] += N - tp
            acc["fn"] += M - tp
            seq_acc["tp"] += tp
            seq_acc["fp"] += N - tp
            seq_acc["fn"] += M - tp

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

        score, accept, descending = self._build_match_score(
            gt_boxes, pred_boxes, pred_obb, gt_obb,
        )

        matched_gt: set[int] = set()
        matched_pred: set[int] = set()

        rows, cols = accept.nonzero(as_tuple=False).T
        if rows.numel() > 0:
            order = score[rows, cols].argsort(descending=descending)
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

    def _build_sweep_summary(self) -> dict:
        """Aggregate the per-threshold sweep into a compact, serialisable dict.

        Layout::

            {
              "thresholds": [0.10, 0.15, ...],
              "micro":  {"Pr": [...], "Re": [...], "F1": [...]},
              "macro":  {"Pr": [...], "Re": [...], "F1": [...]},
              "best_micro": {"score_thresh": 0.35, "Precision": ..., "Recall": ..., "F1": ...},
              "best_macro": {"score_thresh": 0.35, "Precision": ..., "Recall": ..., "F1": ...},
              "per_video_macro": {video_id: {"score_thresh": <best>, "Pr":..., "Re":..., "F1":...}},
            }
        """
        thrs = list(self.score_sweep)
        video_ids = sorted({vid for (vid, _) in self._sweep_seq_acc.keys()})

        micro = {"Pr": [], "Re": [], "F1": []}
        macro = {"Pr": [], "Re": [], "F1": []}
        best_mi = {"score_thresh": thrs[0], "Precision": -1.0, "Recall": -1.0, "F1": -1.0}
        best_ma = {"score_thresh": thrs[0], "Precision": -1.0, "Recall": -1.0, "F1": -1.0}

        for thr in thrs:
            a = self._sweep_acc[thr]
            p_mi = a["tp"] / max(a["tp"] + a["fp"], 1)
            r_mi = a["tp"] / max(a["tp"] + a["fn"], 1)
            f_mi = 2 * p_mi * r_mi / max(p_mi + r_mi, 1e-9)
            micro["Pr"].append(p_mi); micro["Re"].append(r_mi); micro["F1"].append(f_mi)

            seq_p, seq_r, seq_f = [], [], []
            for vid in video_ids:
                sa = self._sweep_seq_acc.get((vid, thr))
                if sa is None or (sa["tp"] + sa["fp"] + sa["fn"]) == 0:
                    continue
                sp = sa["tp"] / max(sa["tp"] + sa["fp"], 1)
                sr = sa["tp"] / max(sa["tp"] + sa["fn"], 1)
                sf = 2 * sp * sr / max(sp + sr, 1e-9)
                seq_p.append(sp); seq_r.append(sr); seq_f.append(sf)
            p_ma = sum(seq_p) / max(len(seq_p), 1)
            r_ma = sum(seq_r) / max(len(seq_r), 1)
            f_ma = sum(seq_f) / max(len(seq_f), 1)
            macro["Pr"].append(p_ma); macro["Re"].append(r_ma); macro["F1"].append(f_ma)

            if f_mi > best_mi["F1"]:
                best_mi = {"score_thresh": thr, "Precision": p_mi, "Recall": r_mi, "F1": f_mi}
            if f_ma > best_ma["F1"]:
                best_ma = {"score_thresh": thr, "Precision": p_ma, "Recall": r_ma, "F1": f_ma}

        # Per-video best (under macro convention's per-video numbers at best_macro thr)
        per_video_macro = {}
        for vid in video_ids:
            sa = self._sweep_seq_acc.get((vid, best_ma["score_thresh"]))
            if sa is None:
                continue
            sp = sa["tp"] / max(sa["tp"] + sa["fp"], 1)
            sr = sa["tp"] / max(sa["tp"] + sa["fn"], 1)
            sf = 2 * sp * sr / max(sp + sr, 1e-9)
            per_video_macro[vid] = {
                "score_thresh": best_ma["score_thresh"],
                "Precision": sp, "Recall": sr, "F1": sf,
                "tp": sa["tp"], "fp": sa["fp"], "fn": sa["fn"],
            }

        return {
            "thresholds": thrs,
            "micro": micro,
            "macro": macro,
            "best_micro": best_mi,
            "best_macro": best_ma,
            "per_video_macro": per_video_macro,
        }

    @staticmethod
    def _centroid_dist_matrix(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
        """Pairwise Euclidean distance between bbox centres of two xyxy sets."""
        if len(boxes_a) == 0 or len(boxes_b) == 0:
            return torch.zeros((len(boxes_a), len(boxes_b)),
                               device=boxes_a.device, dtype=boxes_a.dtype)
        cx_a = (boxes_a[:, 0] + boxes_a[:, 2]) * 0.5
        cy_a = (boxes_a[:, 1] + boxes_a[:, 3]) * 0.5
        cx_b = (boxes_b[:, 0] + boxes_b[:, 2]) * 0.5
        cy_b = (boxes_b[:, 1] + boxes_b[:, 3]) * 0.5
        dx = cx_a[:, None] - cx_b[None, :]
        dy = cy_a[:, None] - cy_b[None, :]
        return torch.sqrt(dx * dx + dy * dy)

    # ------------------------------------------------------------------
    # Text-mode cross-clip track ID stitching (MOT open-vocabulary)
    # ------------------------------------------------------------------

    def _stitch_text_track_ids(self, preds: list[dict], video_id: str):
        """Remap clip-local track_ids to video-global ids via IoU matching.

        The SAM3 text-tracker restarts obj_ids at 1 for every clip. To keep
        IDs continuous across clip boundaries of the same video, we match
        the current clip's frame-0 predictions against the previous clip's
        last-frame predictions (already in video-global id space) by IoU.
        Matched detections reuse the previous global id; unmatched ones
        receive fresh video-global ids allocated from a per-video counter.

        No model state is changed — we only rewrite the ``track_ids`` field
        of every frame's pred dict in place.
        """
        if not preds:
            return

        state = self._text_id_state.setdefault(video_id, {
            "next_global": 1,
            "prev_last": None,  # {"boxes": Tensor[N,4], "track_ids": Tensor[N]}
        })

        # Greedy IoU match: first non-empty frame of this clip → previous last
        local_to_global: dict[int, int] = {}
        prev = state["prev_last"]
        first = next((p for p in preds if len(p.get("boxes", [])) > 0), None)
        if (prev is not None
                and first is not None
                and len(prev["boxes"]) > 0):
            iou = self._iou_matrix(prev["boxes"], first["boxes"])
            rows, cols = (iou >= 0.3).nonzero(as_tuple=False).T
            if rows.numel() > 0:
                order = iou[rows, cols].argsort(descending=True)
                rows, cols = rows[order], cols[order]
                matched_prev: set[int] = set()
                matched_cur: set[int] = set()
                for r, c in zip(rows.tolist(), cols.tolist()):
                    if r in matched_prev or c in matched_cur:
                        continue
                    matched_prev.add(r)
                    matched_cur.add(c)
                    local_id = int(first["track_ids"][c])
                    global_id = int(prev["track_ids"][r])
                    # Only bind if this local id hasn't been claimed already
                    local_to_global.setdefault(local_id, global_id)

        # Rewrite track_ids for every frame in the clip using the mapping
        # (unknown locals get a fresh video-global id, cached for later frames).
        for frame_pred in preds:
            tids = frame_pred.get("track_ids")
            if tids is None or len(tids) == 0:
                continue
            new_ids = []
            for lid in tids.tolist():
                lid = int(lid)
                if lid in local_to_global:
                    new_ids.append(local_to_global[lid])
                else:
                    gid = state["next_global"]
                    state["next_global"] += 1
                    local_to_global[lid] = gid
                    new_ids.append(gid)
            frame_pred["track_ids"] = torch.tensor(
                new_ids, dtype=tids.dtype, device=tids.device,
            )

        # Cache this clip's last non-empty frame for the next clip's boundary.
        for frame_pred in reversed(preds):
            if len(frame_pred.get("boxes", [])) > 0:
                state["prev_last"] = {
                    "boxes": frame_pred["boxes"].detach().clone(),
                    "track_ids": frame_pred["track_ids"].detach().clone(),
                }
                break

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
        # Detection + MOT metrics — skipped in SOT mode (see __init__ docstring).
        if not self.sot_mode:
            # Detection AP — IoU-based; only meaningful under match_metric="iou".
            if self._test_map is not None:
                result = self._test_map.compute()
                self.log("test/AP50", result["map_50"], prog_bar=True)
                self.log("test/AP", result["map"])
                self.log("test/AR_100", result.get("mar_100", torch.tensor(0.0)))
                self._test_map.reset()

            # Detection precision / recall — F1 is added so SVMOD-style
            # papers (HiEUM and friends) can report directly off this run.
            prec = self._det_tp / max(self._det_tp + self._det_fp, 1)
            rec = self._det_tp / max(self._det_tp + self._det_fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            self.log("test/Precision", torch.tensor(prec), prog_bar=True)
            self.log("test/Recall", torch.tensor(rec), prog_bar=True)
            self.log("test/F1", torch.tensor(f1), prog_bar=True)

            # Score-threshold sweep — paper protocol (HiEUM Table III).
            # We compute both aggregations internally:
            #   micro: pool TP/FP/FN over all frames, then one P/R/F1.
            #   macro: per-sequence P/R/F1, then arithmetic mean — this
            #          is HiEUM's ``eval_func_final`` convention, so the
            #          paper-comparable number lives here.
            #
            # Only the *best macro* operating point is logged via
            # ``self.log`` (so ``test_metrics.json`` stays compact); the
            # full per-threshold breakdown is exposed via
            # ``self.sweep_summary`` for the visualization callback to
            # write into a separate ``sweep_results.json`` file.
            if self.score_sweep:
                self.sweep_summary = self._build_sweep_summary()
                best = self.sweep_summary["best_macro"]
                self.log("test/best_F1", torch.tensor(best["F1"]), prog_bar=True)
                self.log("test/best_Precision", torch.tensor(best["Precision"]))
                self.log("test/best_Recall", torch.tensor(best["Recall"]))
                self.log("test/best_score_thresh", torch.tensor(best["score_thresh"]))

            # Tracking metrics — skipped in det_only_mode.
            if not self.det_only_mode:
                denom = max(self._num_gt, 1)
                mota = 1.0 - (self._num_fp + self._num_fn + self._num_id_switch) / denom
                t_prec = self._num_tp / max(self._num_tp + self._num_fp, 1)
                t_rec = self._num_tp / max(self._num_tp + self._num_fn, 1)
                idf1 = 2 * t_prec * t_rec / max(t_prec + t_rec, 1e-6)

                self.log("test/MOTA", torch.tensor(mota), prog_bar=True)
                self.log("test/IDF1", torch.tensor(idf1))
                self.log("test/ID_switches", torch.tensor(float(self._num_id_switch)))

        # Speed (always logged)
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
