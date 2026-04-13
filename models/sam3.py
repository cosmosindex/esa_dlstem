"""
SAM3 tracker wrapper.

Thin wrapper around the SAM 3 video tracker that exposes the same stateful
interface as SAM2Tracker, so it can be used interchangeably with
`VideoTrackerEvaluationModule`:

    init_video → add_prompts → propagate → reset_state

Internally it uses the `tracker` sub-model of SAM 3 (which is API-compatible
with SAM 2's `SAM2VideoPredictor`), sharing the detector backbone.

Notes
-----
* SAM 3's `add_new_points_or_box` expects **relative coordinates in [0, 1]**
  (with `rel_coordinates=True`). We convert xyxy pixel boxes accordingly.
* `init_state` only accepts a JPEG folder or video path, so we dump frames
  to a temporary directory (same trick as SAM2Tracker).
* `propagate_in_video` yields 5-tuples in SAM3
  `(frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores)`,
  whereas SAM2 yielded only `(frame_idx, obj_ids, mask_logits)`. We use
  `video_res_masks` (full-resolution mask logits) and `obj_scores` for the
  per-object confidence.
* The first call to `propagate_in_video` needs `propagate_preflight=True`.
"""

import os
import tempfile

import cv2
import numpy as np
import torch
import torch.nn as nn

from obb_utils import mask_to_obb, mask_to_aabb


class SAM3Tracker(nn.Module):
    """
    SAM3 video tracker wrapped as an nn.Module, mirroring `SAM2Tracker`'s API.

    Args:
        checkpoint_path:        Optional path to a local SAM 3 checkpoint.
                                If None, the model is fetched from Hugging Face
                                (requires `hf auth login` and SAM 3 access).
        apply_temporal_disambiguation:
                                Forwarded to `build_sam3_video_model`.
                                True enables the temporal disambiguation
                                heuristics used in the official SAM 3 release.
        offload_video_to_cpu:   Offload decoded frames to CPU RAM to save GPU.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        apply_temporal_disambiguation: bool = True,
        offload_video_to_cpu: bool = True,
    ):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.apply_temporal_disambiguation = apply_temporal_disambiguation
        self.offload_video_to_cpu = offload_video_to_cpu

        self.predictor = self._build_predictor()

        # Internal video state (reset between sequences)
        self._inference_state = None
        self._tmp_dir: str | None = None
        self._video_h: int | None = None
        self._video_w: int | None = None
        # Mapping from obj_id → class label (set during add_prompts)
        self._obj_id_to_label: dict[int, int] = {}
        # Earliest frame index that received a prompt — used as propagation start
        self._min_prompt_frame: int | None = None
        # Highest frame index seen in init_video
        self._num_frames: int = 0

    def _build_predictor(self):
        """Build the SAM3 video tracker, sharing the detector backbone."""
        # setuptools >= 81 dropped `pkg_resources`, but sam3.model_builder still
        # imports it at module top to locate the BPE asset. Inject a minimal
        # shim before importing so we don't need to patch sam3 or downgrade
        # setuptools.
        self._ensure_pkg_resources_shim()

        # Locate the BPE asset path explicitly (also used by the shim fallback).
        import sam3 as _sam3_pkg
        from pathlib import Path
        sam3_pkg_dir = Path(_sam3_pkg.__path__[0])
        bpe_path = str(sam3_pkg_dir / "assets" / "bpe_simple_vocab_16e6.txt.gz")

        from sam3.model_builder import build_sam3_video_model

        sam3_model = build_sam3_video_model(
            checkpoint_path=self.checkpoint_path,
            bpe_path=bpe_path,
            apply_temporal_disambiguation=self.apply_temporal_disambiguation,
        )
        tracker = sam3_model.tracker
        # The tracker re-uses the detector's backbone for feature extraction
        tracker.backbone = sam3_model.detector.backbone
        tracker.eval()
        return tracker

    @staticmethod
    def _ensure_pkg_resources_shim():
        """Inject a minimal `pkg_resources` replacement if it's missing.

        setuptools >= 81 removed `pkg_resources`. sam3.model_builder still
        uses `pkg_resources.resource_filename(pkg, path)` to locate asset
        files, so we provide a small shim backed by `importlib`.
        """
        import sys
        try:
            import pkg_resources  # noqa: F401
            return
        except ModuleNotFoundError:
            pass

        import types
        import importlib
        from pathlib import Path

        shim = types.ModuleType("pkg_resources")

        def resource_filename(package: str, resource: str) -> str:
            mod = importlib.import_module(package)
            base = Path(mod.__path__[0])
            return str(base / resource)

        shim.resource_filename = resource_filename
        sys.modules["pkg_resources"] = shim

    # ------------------------------------------------------------------
    # Stateful video-level API
    # ------------------------------------------------------------------

    def init_video(self, frames: list[np.ndarray]):
        """
        Initialise SAM3 video state from a list of frames.

        Args:
            frames: list of H×W×3 uint8 numpy arrays (RGB).
        """
        # SAM 3 expects a directory of JPEG frames or a video path
        self._tmp_dir = tempfile.mkdtemp(prefix="sam3_")
        for i, f in enumerate(frames):
            path = os.path.join(self._tmp_dir, f"{i:05d}.jpg")
            cv2.imwrite(path, cv2.cvtColor(f, cv2.COLOR_RGB2BGR))

        self._video_h, self._video_w = frames[0].shape[:2]
        self._num_frames = len(frames)

        self._inference_state = self.predictor.init_state(
            video_path=self._tmp_dir,
            offload_video_to_cpu=self.offload_video_to_cpu,
        )
        self._obj_id_to_label.clear()
        self._min_prompt_frame = None

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

        W, H = self._video_w, self._video_h
        for obj_id, box in zip(obj_ids, boxes):
            x1, y1, x2, y2 = [float(v) for v in box]
            rel_box = torch.tensor(
                [x1 / W, y1 / H, x2 / W, y2 / H], dtype=torch.float32
            )
            self.predictor.add_new_points_or_box(
                inference_state=self._inference_state,
                frame_idx=frame_idx,
                obj_id=int(obj_id),
                box=rel_box,
                rel_coordinates=True,
                clear_old_points=True,
            )

        # Track the earliest prompt frame (propagation starts there)
        if self._min_prompt_frame is None or frame_idx < self._min_prompt_frame:
            self._min_prompt_frame = frame_idx

    def propagate(self) -> list[dict]:
        """
        Propagate masks through the video after prompts have been added.

        Runs forward (and, if prompts are placed past frame 0, also backward)
        from the earliest prompted frame.

        Returns:
            List of per-frame dicts (ordered by frame index):
            {
                'boxes':     (N, 4) xyxy float tensor,
                'obb':       (N, 8) OBB corner float tensor (from minAreaRect),
                'labels':    (N,)   long  tensor,
                'scores':    (N,)   float tensor (object score sigmoid),
                'track_ids': (N,)   long  tensor,
            }
        """
        if self._min_prompt_frame is None:
            return [self._empty_output() for _ in range(self._num_frames)]

        frame_outputs: dict[int, dict] = {}

        # Forward pass from the earliest prompt frame
        for it in self.predictor.propagate_in_video(
            self._inference_state,
            start_frame_idx=self._min_prompt_frame,
            max_frame_num_to_track=None,
            reverse=False,
            propagate_preflight=True,
            tqdm_disable=True,
        ):
            self._collect_frame(it, frame_outputs)

        # Backward pass if prompts aren't already at frame 0
        if self._min_prompt_frame > 0:
            for it in self.predictor.propagate_in_video(
                self._inference_state,
                start_frame_idx=self._min_prompt_frame,
                max_frame_num_to_track=None,
                reverse=True,
                propagate_preflight=False,
                tqdm_disable=True,
            ):
                self._collect_frame(it, frame_outputs)

        max_idx = max(frame_outputs.keys()) if frame_outputs else -1
        total = max(max_idx + 1, self._num_frames)
        return [frame_outputs.get(i, self._empty_output()) for i in range(total)]

    def _collect_frame(self, iteration_out, frame_outputs: dict):
        """Parse one `(frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores)` tuple."""
        frame_idx, obj_ids, _low_res_masks, video_res_masks, obj_scores = iteration_out

        # video_res_masks: (N_obj, 1, H, W) float — threshold at 0 (logit)
        masks = (video_res_masks > 0.0).cpu()
        # obj_scores: (N_obj, 1) logit; sigmoid → confidence
        scores_tensor = torch.sigmoid(obj_scores.float().cpu()).flatten()

        boxes_list, obb_list, scores_list, ids_list, labels_list = [], [], [], [], []
        for i, obj_id in enumerate(obj_ids):
            mask_2d = masks[i, 0].numpy()  # (H, W) bool
            if not mask_2d.any():
                continue
            # Tight AABB from the mask itself (best for HBB GT datasets)
            box_xyxy = mask_to_aabb(mask_2d)
            if box_xyxy is None:
                continue
            # OBB from cv2.minAreaRect (best for OBB GT datasets)
            obb_8 = mask_to_obb(mask_2d)
            if obb_8 is None:
                continue

            boxes_list.append(torch.from_numpy(box_xyxy))
            obb_list.append(torch.from_numpy(obb_8))
            scores_list.append(float(scores_tensor[i]))
            ids_list.append(int(obj_id))
            labels_list.append(self._obj_id_to_label.get(int(obj_id), 0))

        if boxes_list:
            frame_outputs[int(frame_idx)] = {
                "boxes":     torch.stack(boxes_list),
                "obb":       torch.stack(obb_list),
                "labels":    torch.tensor(labels_list, dtype=torch.long),
                "scores":    torch.tensor(scores_list, dtype=torch.float32),
                "track_ids": torch.tensor(ids_list, dtype=torch.long),
            }
        else:
            frame_outputs[int(frame_idx)] = self._empty_output()

    def reset_state(self):
        """Reset video state and clean up temporary files."""
        if self._inference_state is not None:
            try:
                self.predictor.clear_all_points_in_video(self._inference_state)
            except Exception:
                pass
            self._inference_state = None
        self._obj_id_to_label.clear()
        self._min_prompt_frame = None
        self._num_frames = 0

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
