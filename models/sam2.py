"""
SAM2 tracker wrapper.

SAM2 is a video object segmentation model. It is not a classical detector:
  - It requires first-frame bounding-box or point prompts.
  - It then propagates segmentation masks across the video.
  - Masks are converted to axis-aligned bounding boxes for evaluation.

Typical usage in evaluation:
    tracker = SAM2Tracker("facebook/sam2.1-hiera-large")
    tracker.init_video(frames)                       # list of H×W×3 np arrays
    tracker.add_prompts(frame_idx=0, boxes=..., labels=..., obj_ids=...)
    outputs = tracker.propagate()                    # list of per-frame dicts
    tracker.reset_state()
"""

import os
import tempfile

import cv2
import numpy as np
import torch
import torch.nn as nn

from obb_utils import mask_to_obb, mask_to_aabb


class SAM2Tracker(nn.Module):
    """
    SAM2 video predictor wrapped as an nn.Module.

    Operates on entire video sequences using a stateful API:
        init_video → add_prompts (one or more frames) → propagate → reset_state.

    Args:
        model_id:   HuggingFace model ID (e.g. "facebook/sam2.1-hiera-large").
                    Used with SAM2VideoPredictor.from_pretrained().
        cfg_path:   Path to SAM2 config YAML (alternative to model_id).
        ckpt_path:  Path to SAM2 checkpoint (used with cfg_path).
    """

    def __init__(
        self,
        model_id: str | None = None,
        cfg_path: str | None = None,
        ckpt_path: str | None = None,
    ):
        super().__init__()

        if model_id is None and (cfg_path is None or ckpt_path is None):
            raise ValueError("Provide either model_id or both cfg_path and ckpt_path.")

        self.model_id = model_id
        self.cfg_path = cfg_path
        self.ckpt_path = ckpt_path

        self.predictor = self._build_predictor()

        # Internal video state (reset between sequences)
        self._inference_state = None
        self._tmp_dir: str | None = None
        # Mapping from SAM2 obj_id → class label (set during add_prompts)
        self._obj_id_to_label: dict[int, int] = {}

    def _build_predictor(self):
        """Build the SAM2 video predictor from HuggingFace or local weights."""
        from sam2.sam2_video_predictor import SAM2VideoPredictor

        if self.model_id is not None:
            return SAM2VideoPredictor.from_pretrained(self.model_id)
        else:
            from sam2.build_sam import build_sam2_video_predictor
            return build_sam2_video_predictor(self.cfg_path, self.ckpt_path)

    # ------------------------------------------------------------------
    # Stateful video-level API
    # ------------------------------------------------------------------

    def init_video(self, frames: list[np.ndarray]):
        """
        Initialise SAM2 video state from a list of frames.

        Args:
            frames: list of H×W×3 uint8 numpy arrays (RGB).
        """
        # SAM2 expects a directory of JPEG frames
        self._tmp_dir = tempfile.mkdtemp(prefix="sam2_")
        for i, f in enumerate(frames):
            path = os.path.join(self._tmp_dir, f"{i:05d}.jpg")
            cv2.imwrite(path, cv2.cvtColor(f, cv2.COLOR_RGB2BGR))

        self._inference_state = self.predictor.init_state(
            video_path=self._tmp_dir,
            offload_video_to_cpu=True,
        )
        self._obj_id_to_label.clear()

    def add_prompts(
        self,
        frame_idx: int,
        boxes: np.ndarray,
        labels: np.ndarray | None = None,
        obj_ids: list[int] | None = None,
    ):
        """
        Add bounding-box prompts for objects on a given frame.

        Args:
            frame_idx: Index of the frame (0-based within the clip).
            boxes:     (N, 4) xyxy absolute-pixel boxes.
            labels:    (N,) int class labels. Stored for output assignment.
            obj_ids:   Optional list of N integer object IDs; defaults to 1…N.
        """
        if len(boxes) == 0:
            return

        if obj_ids is None:
            obj_ids = list(range(1, len(boxes) + 1))

        if labels is not None:
            for oid, lbl in zip(obj_ids, labels):
                self._obj_id_to_label[int(oid)] = int(lbl)

        for obj_id, box in zip(obj_ids, boxes):
            self.predictor.add_new_points_or_box(
                inference_state=self._inference_state,
                frame_idx=frame_idx,
                obj_id=int(obj_id),
                box=np.asarray(box, dtype=np.float32),
            )

    def propagate(self) -> list[dict]:
        """
        Propagate masks through the video after prompts have been added.

        Returns:
            List of per-frame dicts (ordered by frame index):
            {
                'boxes':     (N, 4) xyxy float tensor,
                'obb':       (N, 8) OBB corner float tensor (from minAreaRect),
                'labels':    (N,)   long  tensor,
                'scores':    (N,)   float tensor (mask area ratio as proxy),
                'track_ids': (N,)   long  tensor,
            }
        """
        frame_outputs: dict[int, dict] = {}

        for frame_idx, obj_ids, mask_logits in self.predictor.propagate_in_video(
            self._inference_state
        ):
            # mask_logits: (N_objects, 1, H, W) float — threshold at 0
            masks = (mask_logits > 0.0).cpu()

            boxes_list, obb_list, scores_list, ids_list, labels_list = [], [], [], [], []
            for obj_id, mask in zip(obj_ids, masks):
                mask_2d = mask[0].numpy()  # (H, W) bool
                # Tight AABB from the mask itself (best for HBB GT datasets)
                box_xyxy = mask_to_aabb(mask_2d)
                if box_xyxy is None:
                    continue
                # OBB from cv2.minAreaRect (best for OBB GT datasets)
                obb_8 = mask_to_obb(mask_2d)
                if obb_8 is None:
                    continue

                # Score: fraction of mask area relative to OBB area (crude proxy)
                mask_area = float(mask_2d.sum())
                obb_area = float(cv2.contourArea(obb_8.reshape(4, 2)))
                score = mask_area / max(obb_area, 1.0)

                boxes_list.append(torch.from_numpy(box_xyxy))
                obb_list.append(torch.from_numpy(obb_8))
                scores_list.append(min(score, 1.0))
                ids_list.append(int(obj_id))
                labels_list.append(self._obj_id_to_label.get(int(obj_id), 0))

            if boxes_list:
                frame_outputs[frame_idx] = {
                    "boxes":     torch.stack(boxes_list),
                    "obb":       torch.stack(obb_list),
                    "labels":    torch.tensor(labels_list, dtype=torch.long),
                    "scores":    torch.tensor(scores_list, dtype=torch.float32),
                    "track_ids": torch.tensor(ids_list, dtype=torch.long),
                }
            else:
                frame_outputs[frame_idx] = self._empty_output()

        return [frame_outputs.get(i, self._empty_output())
                for i in range(max(frame_outputs.keys()) + 1)] if frame_outputs else []

    def reset_state(self):
        """Reset video state and clean up temporary files."""
        if self._inference_state is not None:
            self.predictor.reset_state(self._inference_state)
            self._inference_state = None
        self._obj_id_to_label.clear()

        # Clean up temp JPEG directory
        if self._tmp_dir is not None:
            import shutil
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            self._tmp_dir = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_output() -> dict:
        return {
            "boxes":     torch.zeros((0, 4), dtype=torch.float32),
            "obb":       torch.zeros((0, 8), dtype=torch.float32),
            "labels":    torch.zeros(0, dtype=torch.long),
            "scores":    torch.zeros(0, dtype=torch.float32),
            "track_ids": torch.zeros(0, dtype=torch.long),
        }
