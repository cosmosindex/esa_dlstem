"""
GroundingDINO open-vocabulary detector wrapper.

GroundingDINO has no temporal model — it is a pure single-image text-prompted
detector. To plug it into the existing clip-centric MOT eval pipeline
(``VideoTrackerEvaluationModule``), this wrapper exposes the standard
stateful API (``init_video / add_prompts / propagate / reset_state``) but
internally just loops over the clip's frames and runs one forward pass per
frame.

Use with ``det_only_mode=True`` on ``VideoTrackerEvaluationModule`` so that
MOT metrics (MOTA / IDF1 / ID_switches) are not logged — they are not
meaningful for a per-frame detector.

Per-clip operating mode
-----------------------
* The wrapper accepts a per-clip text prompt via :meth:`set_text_prompt`,
  matching ``SAM3TextTracker``'s contract. If set, the prompt becomes the
  caption (a single noun phrase like ``"car"``); ``track_ids`` for matching
  detections will all share that class label.
* If no per-clip prompt is set (or it is empty), the caption falls back to
  ``". ".join(class_names)`` — GroundingDINO's standard multi-class prompt
  format. Each detection's class id comes from substring-matching the
  predicted phrase against ``class_names``.

Track IDs
---------
``track_ids`` are assigned ``arange(K)+1`` per frame, i.e. unique within a
frame but with no temporal correspondence. This is correct for detection-only
evaluation; the existing ID-stitching logic in ``VideoTrackerEvaluationModule``
is bypassed when ``det_only_mode=True``.

Integration quirks
------------------
* ``import torch`` MUST happen before ``import groundingdino._C`` so the
  loader picks up libtorch. We do ``import torch`` at module top.
* GroundingDINO's ``predict()`` returns boxes in normalized cxcywh; we
  convert to absolute xyxy using the source frame's H, W.
* The image preprocessing pipeline (RandomResize 800 short side, max 1333,
  ImageNet normalization) is applied per frame — not vectorized over a
  batch. This is the upstream's reference path.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch  # noqa: F401  — must be imported before groundingdino._C
import torch.nn as nn
from PIL import Image
from torchvision.ops import box_convert

import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model, predict


def _xyxy_to_obb(boxes_xyxy: torch.Tensor) -> torch.Tensor:
    """Expand xyxy boxes into 8-coord AABB corners (x1,y1, x2,y1, x2,y2, x1,y2).

    GroundingDINO is an HBB detector; the AABB corner form lets the OBB-aware
    eval path consume its outputs uniformly with OBB-capable trackers.
    """
    if len(boxes_xyxy) == 0:
        return torch.zeros((0, 8), dtype=torch.float32)
    x1, y1, x2, y2 = boxes_xyxy.unbind(-1)
    return torch.stack([x1, y1, x2, y1, x2, y2, x1, y2], dim=-1).float()


class GroundingDINODetector(nn.Module):
    """GroundingDINO wrapped as a clip-friendly detection-only "tracker".

    Args:
        config_path:    Path to the GroundingDINO model config .py
                        (e.g. ``GroundingDINO_SwinT_OGC.py``).
        checkpoint_path: Path to the .pth weights file.
        class_names:    Ordered list of class names. The integer class id of
                        each detection is the index of the matched name in
                        this list.
        label_to_id:    Map from class name → integer label id, used to fill
                        the ``labels`` field of each per-frame output dict.
                        Must match the dataset's class_map.
        box_threshold:  Box confidence threshold (GroundingDINO default 0.35).
        text_threshold: Per-token text-match threshold for phrase decoding
                        (GroundingDINO default 0.25).
        device:         "cuda" or "cpu". Model is moved here at construction.
    """

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        class_names: list[str],
        label_to_id: dict[str, int],
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        device: str = "cuda",
    ):
        super().__init__()
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.class_names = list(class_names)
        self.label_to_id = dict(label_to_id)
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.device_str = device

        self.model = load_model(config_path, checkpoint_path, device=device)
        self.model.to(device).eval()

        self._transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # Per-clip state
        self._frames: list[np.ndarray] | None = None
        self._video_h: int | None = None
        self._video_w: int | None = None
        self._current_text: Optional[str] = None

    # ------------------------------------------------------------------
    # Stateful video-level API (clip-centric pipeline compatibility)
    # ------------------------------------------------------------------

    def set_text_prompt(self, text: Optional[str]):
        """Set per-clip text prompt. If empty/None, fall back to all classes."""
        self._current_text = text if text else None

    def init_video(self, frames: list[np.ndarray]):
        """Cache the clip's frames for per-frame inference in ``propagate``."""
        self._frames = frames
        self._video_h, self._video_w = frames[0].shape[:2]

    def add_prompts(self, frame_idx, boxes, labels=None, obj_ids=None):
        """No-op. GroundingDINO is open-vocab text-prompted; no GT boxes ever
        reach the model. Kept for interface compatibility."""
        return

    @torch.no_grad()
    def propagate(self) -> list[dict]:
        """Run GroundingDINO once per frame; return per-frame detection dicts."""
        if not self._frames:
            return []

        if self._current_text:
            caption = self._current_text
            classes_for_match = [self._current_text]
        else:
            caption = ". ".join(self.class_names)
            classes_for_match = self.class_names

        out: list[dict] = []
        H, W = self._video_h, self._video_w
        for frame_rgb in self._frames:
            out.append(self._infer_frame(frame_rgb, caption, classes_for_match, H, W))
        return out

    def reset_state(self):
        self._frames = None
        self._video_h = None
        self._video_w = None
        self._current_text = None

    # ------------------------------------------------------------------
    # Per-frame inference
    # ------------------------------------------------------------------

    def _infer_frame(
        self,
        frame_rgb: np.ndarray,
        caption: str,
        classes_for_match: list[str],
        H: int,
        W: int,
    ) -> dict:
        pil = Image.fromarray(frame_rgb)
        image_t, _ = self._transform(pil, None)

        boxes_cxcywh, scores, phrases = predict(
            model=self.model,
            image=image_t,
            caption=caption,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device_str,
        )

        if len(boxes_cxcywh) == 0:
            return self._empty_output()

        # Map normalized cxcywh → absolute xyxy
        scale = torch.tensor([W, H, W, H], dtype=boxes_cxcywh.dtype)
        boxes_xyxy = box_convert(boxes_cxcywh * scale, in_fmt="cxcywh", out_fmt="xyxy")

        # Phrase → class id. Skip detections whose phrase doesn't match any
        # known class; this prevents spurious labels from hurting AP.
        labels_list, keep = [], []
        for i, phrase in enumerate(phrases):
            phrase_l = phrase.lower().strip()
            matched_id = None
            for cls_name in classes_for_match:
                if cls_name.lower() in phrase_l:
                    matched_id = self.label_to_id.get(cls_name, -1)
                    break
            if matched_id is None or matched_id < 0:
                continue
            labels_list.append(matched_id)
            keep.append(i)

        if not keep:
            return self._empty_output()

        keep_idx = torch.tensor(keep, dtype=torch.long)
        boxes_xyxy = boxes_xyxy[keep_idx].float().contiguous()
        scores = scores[keep_idx].float().contiguous()
        labels_t = torch.tensor(labels_list, dtype=torch.long)
        # Per-frame unique IDs; no temporal correspondence (det-only mode).
        track_ids = torch.arange(1, len(boxes_xyxy) + 1, dtype=torch.long)

        return {
            "boxes":     boxes_xyxy,
            "obb":       _xyxy_to_obb(boxes_xyxy),
            "labels":    labels_t,
            "scores":    scores,
            "track_ids": track_ids,
        }

    @staticmethod
    def _empty_output() -> dict:
        return {
            "boxes":     torch.zeros((0, 4), dtype=torch.float32),
            "obb":       torch.zeros((0, 8), dtype=torch.float32),
            "labels":    torch.zeros(0, dtype=torch.long),
            "scores":    torch.zeros(0, dtype=torch.float32),
            "track_ids": torch.zeros(0, dtype=torch.long),
        }
