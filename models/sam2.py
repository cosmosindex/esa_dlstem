"""
SAM2 tracker wrapper.

SAM2 is a video object segmentation model. It is not a classical detector:
  - It requires first-frame bounding-box or point prompts.
  - It then propagates segmentation masks across the video.
  - Masks are converted to AxisAligned bounding boxes for evaluation.

For detection+tracking use ObjectDetectionModule(has_tracking=True).

Typical usage in evaluation:
    tracker = SAM2Tracker(cfg_path, ckpt_path)
    tracker.init_video(frames)                       # list of H×W×3 np arrays
    tracker.add_prompts(frame_idx=0, boxes=gt_boxes) # seed from GT or other detector
    outputs = tracker.propagate()                    # list of per-frame dicts
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class SAM2Tracker(nn.Module):
    """
    SAM2 video predictor wrapped as an nn.Module.

    Because SAM2 operates on entire video sequences (not single frames), the
    forward() method accepts a list of frames and first-frame prompt boxes.

    During training  : SAM2's own fine-tuning API is used; loss is returned.
    During inference : mask propagation is run and masks → boxes are returned.
    """

    def __init__(
        self,
        cfg_path: str,
        ckpt_path: str,
        device: str = "cuda",
        iou_threshold: float = 0.0,
    ):
        """
        Args:
            cfg_path:      Path to the SAM2 model config YAML.
            ckpt_path:     Path to the SAM2 checkpoint file.
            device:        Torch device string.
            iou_threshold: Minimum predicted IoU to keep a mask (0 = keep all).
        """
        super().__init__()

        self.cfg_path = cfg_path
        self.ckpt_path = ckpt_path
        self.device_str = device
        self.iou_threshold = iou_threshold

        self.predictor = self._build_predictor()

        # Internal video state (reset between sequences)
        self._inference_state = None

    def _build_predictor(self):
        """Attempt both the new and legacy SAM2 APIs."""
        try:
            from sam2.sam2_video_predictor import SAM2VideoPredictor
            predictor = SAM2VideoPredictor.from_pretrained(
                self.cfg_path, self.ckpt_path
            )
        except Exception:
            from sam2.build_sam import build_sam2_video_predictor
            predictor = build_sam2_video_predictor(self.cfg_path, self.ckpt_path)
        return predictor.to(self.device_str)

    # ------------------------------------------------------------------
    # Stateful video-level API (main inference path)
    # ------------------------------------------------------------------

    def init_video(self, frames: list[np.ndarray]):
        """
        Initialise the SAM2 video state from a list of frames.

        Args:
            frames: list of H×W×3 uint8 numpy arrays (RGB).
        """
        import tempfile, os, cv2

        # SAM2 expects a directory of JPEG frames named 00000.jpg, 00001.jpg …
        self._tmp_dir = tempfile.mkdtemp()
        for i, f in enumerate(frames):
            path = os.path.join(self._tmp_dir, f"{i:05d}.jpg")
            cv2.imwrite(path, cv2.cvtColor(f, cv2.COLOR_RGB2BGR))

        self._inference_state = self.predictor.init_state(
            video_path=self._tmp_dir,
            offload_video_to_cpu=True,
        )

    def add_prompts(
        self,
        frame_idx: int,
        boxes: np.ndarray,
        obj_ids: Optional[list[int]] = None,
    ):
        """
        Add bounding-box prompts for objects on a given frame.

        Args:
            frame_idx: Index of the frame (0-based).
            boxes:     (N, 4) xyxy absolute-pixel boxes.
            obj_ids:   Optional list of N integer object IDs; defaults to 1…N.
        """
        if obj_ids is None:
            obj_ids = list(range(1, len(boxes) + 1))

        for obj_id, box in zip(obj_ids, boxes):
            self.predictor.add_new_prompts(
                inference_state=self._inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                boxes=box[None],  # (1, 4)
            )

    def propagate(self) -> list[dict]:
        """
        Propagate masks through the video after prompts have been added.

        Returns:
            List of per-frame dicts:
            {
                'boxes':     (N, 4) xyxy float tensor,
                'labels':    (N,)   long  tensor (all zeros; class from prompt),
                'scores':    (N,)   float tensor (predicted IoU),
                'track_ids': (N,)   long  tensor,
                'masks':     (N, H, W) bool tensor,
            }
        """
        frame_outputs: dict[int, dict] = {}

        for frame_idx, obj_ids, masks in self.predictor.propagate_in_video(
            self._inference_state
        ):
            # masks: (N, 1, H, W) bool
            boxes_list, scores_list, ids_list, masks_list = [], [], [], []
            for obj_id, mask in zip(obj_ids, masks):
                mask_2d = mask[0]  # (H, W)
                score = float(mask.float().mean())  # crude proxy for IoU
                if score < self.iou_threshold:
                    continue
                box = self._mask_to_box(mask_2d)
                if box is None:
                    continue
                boxes_list.append(box)
                scores_list.append(score)
                ids_list.append(obj_id)
                masks_list.append(mask_2d.cpu())

            if boxes_list:
                frame_outputs[frame_idx] = {
                    "boxes":     torch.stack(boxes_list),
                    "labels":    torch.zeros(len(boxes_list), dtype=torch.long),
                    "scores":    torch.tensor(scores_list),
                    "track_ids": torch.tensor(ids_list, dtype=torch.long),
                    "masks":     torch.stack(masks_list),
                }
            else:
                frame_outputs[frame_idx] = {
                    "boxes":     torch.zeros((0, 4)),
                    "labels":    torch.zeros(0, dtype=torch.long),
                    "scores":    torch.zeros(0),
                    "track_ids": torch.zeros(0, dtype=torch.long),
                    "masks":     torch.zeros(0),
                }

        return [frame_outputs[i] for i in sorted(frame_outputs)]

    def reset_state(self):
        """Reset video state between sequences."""
        if self._inference_state is not None:
            self.predictor.reset_state(self._inference_state)
            self._inference_state = None

    # ------------------------------------------------------------------
    # nn.Module forward (Lightning-compatible per-batch interface)
    # ------------------------------------------------------------------

    def forward(
        self,
        frames: list[list[np.ndarray]],
        prompt_boxes: list[np.ndarray] | None = None,
        targets: list[list[dict]] | None = None,
    ):
        """
        Per-sequence forward pass (used by ObjectDetectionModule).

        Args:
            frames:       List of sequences; each sequence is a list of
                          H×W×3 np.ndarray frames.
            prompt_boxes: List of (N, 4) xyxy prompt boxes, one per sequence.
                          If None, the first-frame GT boxes from targets are used.
            targets:      List of per-sequence, per-frame GT dicts (used for
                          prompt extraction if prompt_boxes is None).

        Returns (eval): list of per-sequence outputs, each a list of per-frame dicts.
        """
        all_outputs = []
        for seq_idx, seq_frames in enumerate(frames):
            self.init_video(seq_frames)

            # Derive prompts
            if prompt_boxes is not None:
                boxes_np = prompt_boxes[seq_idx]
            elif targets is not None:
                boxes_np = targets[seq_idx][0]["boxes"].numpy()
            else:
                raise ValueError(
                    "Either prompt_boxes or targets must be provided for SAM2."
                )

            self.add_prompts(frame_idx=0, boxes=boxes_np)
            seq_outputs = self.propagate()
            self.reset_state()
            all_outputs.append(seq_outputs)

        return all_outputs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mask_to_box(mask: torch.Tensor) -> Optional[torch.Tensor]:
        """Convert a boolean H×W mask to an xyxy bounding box tensor."""
        ys, xs = torch.where(mask)
        if len(xs) == 0:
            return None
        return torch.tensor(
            [xs.min(), ys.min(), xs.max(), ys.max()], dtype=torch.float32
        )
