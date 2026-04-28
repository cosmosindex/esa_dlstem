"""
HiEUM moving-object detector wrapper.

HiEUM (Xiao et al., TPAMI 2024) is a sparse 3D-conv CenterNet that takes a
clip of T=20 consecutive 1024x1024 frames and emits per-frame point
detections of moving objects. The pretrained checkpoint is single-class
(car), trained on RsCarData.

This wrapper plugs HiEUM into the project's clip-centric MOT eval pipeline
(``VideoTrackerEvaluationModule``) by exposing the same stateful API as
``GroundingDINODetector`` (no GT prompts, det-only mode):

    set_text_prompt(text)        # no-op (single class)
    init_video(frames)           # cache + resize + buffer
    add_prompts(...)             # no-op
    propagate() -> list[dict]    # per-frame {boxes,scores,labels,track_ids,obb}
    reset_state()                # release buffer

Use with ``det_only_mode=True`` on ``VideoTrackerEvaluationModule`` since
HiEUM emits no cross-frame identities — track_ids inside one clip are just
``arange(K)+1`` per frame, suitable for AP/Precision/Recall but not MOTA.

Implementation notes
--------------------
* HiEUM's checkpoint bakes T=20 into ``conv_std``; we therefore enforce
  ``clip_len=20`` in the configs. Short clips at video tails are padded by
  repeating the last frame; padded outputs are discarded.
* Frames are resized (uniform scale + center-pad) to 1024x1024 to match the
  training distribution; predicted boxes are mapped back to the original
  resolution before scoring.
* The repo is imported via path-insert from
  ``Moving-object-detection-in-satellite-videos-HiEUM/`` to avoid copying
  any of its code into this tree.
* Image normalisation uses RsCarData mean/std (0.49965 / 0.08255 across
  all 3 channels) and BGR ordering, matching the upstream pipeline.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HIEUM_REPO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Moving-object-detection-in-satellite-videos-HiEUM",
)
if _HIEUM_REPO not in sys.path:
    sys.path.insert(0, _HIEUM_REPO)

import spconv.pytorch as _spconv  # noqa: E402
from spconv.core import ConvAlgo as _ConvAlgo  # noqa: E402
from lib.models.spconv_centerDet_minus import sp_centerDet_minus  # noqa: E402
from lib.models.stNet import load_model  # noqa: E402
from lib.utils1.decode import ctdet_decode  # noqa: E402


# RsCarData normalisation (single value broadcast across RGB channels)
_HIEUM_MEAN = np.array([0.49965, 0.49965, 0.49965], dtype=np.float32)
_HIEUM_STD = np.array([0.08255, 0.08255, 0.08255], dtype=np.float32)


def _xyxy_to_obb(boxes_xyxy: torch.Tensor) -> torch.Tensor:
    """Expand xyxy boxes into 8-coord AABB corners for OBB-aware eval."""
    if len(boxes_xyxy) == 0:
        return torch.zeros((0, 8), dtype=torch.float32)
    x1, y1, x2, y2 = boxes_xyxy.unbind(-1)
    return torch.stack([x1, y1, x2, y1, x2, y2, x1, y2], dim=-1).float()


def _soft_nms_linear(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_thresh: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear Soft-NMS as used in the HiEUM eval pipeline.

    For each detection in descending-score order, **decay** (not delete)
    overlapping detections by ``score *= (1 - iou)`` whenever
    ``iou > iou_thresh``. Returns boxes in original order with their
    decayed scores. Caller is expected to apply a final score filter.

    Operates on small N (≤K=128 per frame) so a Python loop is fine.
    Reference: Bodla et al. 2017, "Soft-NMS — Improving Object Detection
    With One Line of Code". Mirrors the HiEUM pipeline's ``soft_nms`` call
    with ``Nt=0.1, method=1`` (linear).
    """
    if len(boxes) == 0:
        return boxes, scores

    boxes = boxes.detach().cpu().float()
    scores = scores.detach().cpu().float().clone()
    N = boxes.shape[0]

    # Iterate in descending order, decay the others.
    order = scores.argsort(descending=True).tolist()
    settled: list[int] = []
    while order:
        i = order.pop(0)
        settled.append(i)
        if not order:
            break
        # IoU between detection i and remaining candidates
        x1 = torch.max(boxes[i, 0], boxes[order, 0])
        y1 = torch.max(boxes[i, 1], boxes[order, 1])
        x2 = torch.min(boxes[i, 2], boxes[order, 2])
        y2 = torch.min(boxes[i, 3], boxes[order, 3])
        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_j = (boxes[order, 2] - boxes[order, 0]) * (boxes[order, 3] - boxes[order, 1])
        iou = inter / (area_i + area_j - inter).clamp(min=1e-9)

        # Linear decay where iou > Nt, no change otherwise
        decay = torch.where(iou > iou_thresh, 1.0 - iou, torch.ones_like(iou))
        scores[order] = scores[order] * decay

        # Re-sort the remaining by their (possibly decayed) scores
        order_t = torch.tensor(order, dtype=torch.long)
        new_order = order_t[scores[order_t].argsort(descending=True)].tolist()
        order = new_order

    return boxes, scores


class HiEUMDetector(nn.Module):
    """HiEUM wrapped as a clip-centric moving-object detector.

    Args:
        checkpoint_path: Path to HiEUM ``model_best.pth``.
        seq_len:         Frames per clip baked into the checkpoint (20).
        image_size:      Spatial size HiEUM was trained on (1024, 1024).
        layers:          Backbone depth — must match the checkpoint (3).
        thresh:          Background-subtraction threshold multiplier
                         (``mean + thresh*std``); upstream default 3.
        car_label:       Integer class id assigned to every detection.
                         Set this to whatever ``class_map["car"]`` is in
                         the dataset config (e.g. 0 for the car-only YAMLs
                         used here). HiEUM is single-class internally.
        score_thresh:    Drop detections below this confidence after Soft-NMS
                         decay. For paper-protocol score-threshold sweeps,
                         set this to a low floor (e.g. 0.05) so every
                         candidate above the lowest sweep value survives.
        nms_iou:         IoU threshold (Nt) for **linear Soft-NMS** —
                         overlapping detections are *decayed* by
                         ``score *= (1 - iou)`` rather than deleted.
                         Matches HiEUM's ``soft_nms(Nt=0.1, method=1)``.
        max_dets:        Top-K output points per frame from CenterNet decode.
        device:          "cuda" or "cpu".
    """

    def __init__(
        self,
        checkpoint_path: str,
        seq_len: int = 20,
        image_size: tuple[int, int] = (1024, 1024),
        layers: int = 3,
        thresh: float = 3.0,
        car_label: int = 0,
        score_thresh: float = 0.2,
        nms_iou: float = 0.1,
        max_dets: int = 128,
        device: str = "cuda",
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.car_label = int(car_label)
        self.score_thresh = float(score_thresh)
        self.nms_iou = float(nms_iou)
        self.max_dets = int(max_dets)
        self.device_str = device

        heads = {"hm": 1, "wh": 2, "reg": 2}
        self.model = sp_centerDet_minus(
            heads,
            image_size=list(self.image_size),
            img_num=self.seq_len,
            layers=int(layers),
            thresh=float(thresh),
        )
        self.model = load_model(self.model, checkpoint_path)
        # spconv 2.3.x ships GEMM cubins compiled for sm_52..sm_90 in `spconv`
        # but the matching cumm wheel only ships sm_52, so the implicit_gemm
        # tuner cannot find a kernel on Ada (sm_89). Switch every sparse conv
        # to the Native algo — slower but Python/PyTorch only and works on
        # any CUDA arch torch supports.
        for layer in self.model.modules():
            if isinstance(layer, (_spconv.SubMConv3d,
                                  _spconv.SparseConv3d,
                                  _spconv.SparseInverseConv3d)):
                layer.algo = _ConvAlgo.Native
        self.model.to(device).eval()

        # Per-clip state
        self._frames: list[np.ndarray] | None = None
        self._H: int | None = None
        self._W: int | None = None

    # ------------------------------------------------------------------
    # Stateful video-level API (clip-centric pipeline compatibility)
    # ------------------------------------------------------------------

    def set_text_prompt(self, text: Optional[str]):
        """No-op. HiEUM is single-class (car) — no text steering."""
        return

    def init_video(self, frames: list[np.ndarray]):
        """Cache the clip's frames (RGB uint8 HWC) for later inference."""
        if not frames:
            raise ValueError("HiEUMDetector.init_video: empty frame list")
        self._frames = frames
        self._H, self._W = frames[0].shape[:2]

    def add_prompts(self, frame_idx, boxes, labels=None, obj_ids=None):
        """No-op — moving-object detection takes no GT prompts."""
        return

    @torch.no_grad()
    def propagate(self) -> list[dict]:
        """Run HiEUM on the buffered clip; return one dict per real frame.

        Pads short clips up to ``seq_len`` by repeating the last frame
        (HiEUM's checkpoint requires exactly ``seq_len`` frames per call);
        padded outputs are discarded before returning.
        """
        if not self._frames:
            return []

        T = len(self._frames)
        H_orig, W_orig = self._H, self._W
        H_in, W_in = self.image_size
        device = self.device_str

        # --- Pad clip up to seq_len ---
        T_in = self.seq_len
        if T == T_in:
            padded = list(self._frames)
        elif T < T_in:
            padded = list(self._frames) + [self._frames[-1]] * (T_in - T)
        else:
            # Caller asked for a longer clip than HiEUM was trained on.
            # We process in non-overlapping chunks of seq_len and stitch
            # detections back together. The rare tail (< seq_len) is
            # handled by padding with the last frame and trimming.
            return self._propagate_long(T)

        # --- Resize + BGR conversion + normalisation ---
        rgb_resized, scale, pad_x, pad_y = self._resize_keep_ratio_batch(
            padded, H_in, W_in
        )                                                           # [T, H, W, 3] uint8 RGB
        bgr = rgb_resized[..., ::-1].astype(np.float32) / 255.0     # to BGR float
        normed = (bgr - _HIEUM_MEAN) / _HIEUM_STD                   # [T, H, W, 3]
        gray = rgb_resized.astype(np.float32).mean(
            axis=-1, keepdims=True,
        )                                                           # [T, H, W, 1] float32

        # HiEUM forward expects [B=1, C, T, H, W] for both rgb (3) and gray (1)
        inp = torch.from_numpy(
            np.ascontiguousarray(normed.transpose(3, 0, 1, 2))[None]
        ).to(device, non_blocking=True)
        inp_gray = torch.from_numpy(
            np.ascontiguousarray(gray.transpose(3, 0, 1, 2))[None]
        ).to(device, non_blocking=True)

        # --- Forward ---
        # spconv raises ValueError("tensor must not empty") when the
        # background-subtraction threshold (mean + k*std) leaves zero
        # active voxels — happens on very static SDM-Car clips. Treat
        # it as "no movers detected" and emit empty preds for the chunk
        # rather than killing the whole video.
        try:
            out = self.model({"input": inp, "input_gray": inp_gray})[-1]
        except (ValueError, RuntimeError) as exc:
            msg = str(exc)
            if "must not empty" in msg or "tensor must not empty" in msg:
                return [self._empty_output() for _ in range(T)]
            raise
        hm = out["hm"]   # [1, 1, T_in, H_in, W_in]
        wh = out["wh"]   # [1, 2, T_in, H_in, W_in]
        reg = out["reg"] # [1, 2, T_in, H_in, W_in]

        # ctdet_decode expects [batch, cat, H, W] — reuse T as batch dim.
        hm_t = hm[0].permute(1, 0, 2, 3)        # [T_in, 1, H_in, W_in]
        wh_t = wh[0].permute(1, 0, 2, 3)        # [T_in, 2, H_in, W_in]
        reg_t = reg[0].permute(1, 0, 2, 3)      # [T_in, 2, H_in, W_in]

        dets = ctdet_decode(hm_t, wh_t, reg=reg_t, K=self.max_dets)  # [T_in, K, 6]
        # dets columns: [x1, y1, x2, y2, score, cls=0] in resized-frame coords.

        # --- Decode per real frame, undo letterbox, NMS, threshold ---
        outputs: list[dict] = []
        for t in range(T):
            frame_dets = dets[t]                              # [K, 6]
            scores = frame_dets[:, 4]
            keep = scores >= self.score_thresh
            if not keep.any():
                outputs.append(self._empty_output())
                continue
            frame_dets = frame_dets[keep]
            boxes_resized = frame_dets[:, :4]                 # [N, 4] xyxy
            scores = frame_dets[:, 4]

            # Undo letterbox: subtract pad, divide by scale.
            boxes_orig = boxes_resized.clone()
            boxes_orig[:, 0::2] = (boxes_orig[:, 0::2] - pad_x) / scale
            boxes_orig[:, 1::2] = (boxes_orig[:, 1::2] - pad_y) / scale
            boxes_orig[:, 0::2].clamp_(0, W_orig)
            boxes_orig[:, 1::2].clamp_(0, H_orig)

            # Linear Soft-NMS — same protocol as HiEUM's official pipeline.
            boxes_orig, scores = _soft_nms_linear(boxes_orig, scores, self.nms_iou)

            # Apply final score floor *after* soft decay.
            keep = scores >= self.score_thresh
            boxes_orig = boxes_orig[keep]
            scores = scores[keep]

            if len(boxes_orig) == 0:
                outputs.append(self._empty_output())
                continue

            labels = torch.full(
                (len(boxes_orig),), self.car_label, dtype=torch.long, device=boxes_orig.device,
            )
            track_ids = torch.arange(
                1, len(boxes_orig) + 1, dtype=torch.long, device=boxes_orig.device,
            )
            outputs.append({
                "boxes":     boxes_orig.float().cpu(),
                "obb":       _xyxy_to_obb(boxes_orig.float().cpu()),
                "labels":    labels.cpu(),
                "scores":    scores.float().cpu(),
                "track_ids": track_ids.cpu(),
            })
        return outputs

    def reset_state(self):
        self._frames = None
        self._H = None
        self._W = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resize_keep_ratio_batch(
        frames: list[np.ndarray], H_in: int, W_in: int,
    ) -> tuple[np.ndarray, float, float, float]:
        """Letterbox a batch of HxWx3 RGB uint8 frames into (H_in, W_in).

        Returns the padded batch, the uniform scale factor, and the pad
        offsets ``(pad_x, pad_y)`` (top-left corner of the original
        content inside the padded canvas).
        """
        H_orig, W_orig = frames[0].shape[:2]
        scale = min(H_in / H_orig, W_in / W_orig)
        new_h = int(round(H_orig * scale))
        new_w = int(round(W_orig * scale))
        pad_y = (H_in - new_h) / 2.0
        pad_x = (W_in - new_w) / 2.0
        top = int(pad_y)
        left = int(pad_x)

        # cv2 is the lightest dependency already in use across the project.
        import cv2  # local import to avoid hard dep at module top
        out = np.zeros((len(frames), H_in, W_in, 3), dtype=np.uint8)
        for i, f in enumerate(frames):
            resized = cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_AREA)
            out[i, top:top + new_h, left:left + new_w] = resized
        return out, scale, pad_x, pad_y

    def _propagate_long(self, T: int) -> list[dict]:
        """Run HiEUM in non-overlapping seq_len chunks, then concat.

        Caller is ``propagate``; we save / restore the frame buffer so each
        chunk reuses the standard code path.
        """
        all_frames = self._frames
        H, W = self._H, self._W
        outputs: list[dict] = []
        for start in range(0, T, self.seq_len):
            self._frames = all_frames[start:start + self.seq_len]
            self._H, self._W = H, W
            outputs.extend(self.propagate())
        self._frames = all_frames
        self._H, self._W = H, W
        return outputs

    @staticmethod
    def _empty_output() -> dict:
        return {
            "boxes":     torch.zeros((0, 4), dtype=torch.float32),
            "obb":       torch.zeros((0, 8), dtype=torch.float32),
            "labels":    torch.zeros(0, dtype=torch.long),
            "scores":    torch.zeros(0, dtype=torch.float32),
            "track_ids": torch.zeros(0, dtype=torch.long),
        }
