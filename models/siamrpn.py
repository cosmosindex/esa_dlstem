"""
SiamRPN++ tracker wrapper (pysot implementation).

SiamRPN++ (CVPR 2019) is a classical Siamese-RPN tracker with a ResNet-50
backbone, multi-layer feature aggregation (layers 2/3/4), and depthwise
cross-correlation. This wrapper uses the SenseTime pysot repo's
`ModelBuilder` + `SiamRPNTracker` directly — those classes already expose a
clean per-frame `init(img, bbox) / track(img)` API, so there is almost no
glue needed.

Source: `/home/anon/code/esa_dlstem/pysot` (SenseTime pysot).

Integration quirks
------------------
* pysot uses a global `cfg` (yacs CfgNode) — we call `cfg.merge_from_file()`
  once at build time. Building two different SiamRPN variants in the same
  process would clobber cfg; not a concern for single-eval scripts.
* `SiamRPNTracker.init(img, bbox)` expects **BGR** uint8 images and **xywh**
  bboxes. Our pipeline passes **RGB** uint8 frames and **xyxy** boxes, so we
  convert both.
* The checkpoint we have (`siamrpn_r50_l234_dwxcorr.pth`) comes from a fork
  with per-layer adjust widths [128, 256, 512] instead of the official
  uniform [256, 256, 256]. We ship a compatible pysot-native yaml at
  `configs/SOT/pysot_siamrpn_r50_l234_dwxcorr.yaml` that encodes the right widths.
* HBB-only tracker (no masks). `obb` is the 8-corner form of the AABB as a
  compatibility stub.

Usage mirrors OSTrackTracker:
    tracker = SiamRPNTracker_Wrapper(
        yaml_path="configs/SOT/pysot_siamrpn_r50_l234_dwxcorr.yaml",
        ckpt_path="/work/anon/checkpoints/siamrpn/siamrpn_r50_l234_dwxcorr.pth",
    )
    tracker.init_video(frames)
    tracker.add_prompts(frame_idx=0, boxes=..., labels=..., obj_ids=...)
    outputs = tracker.propagate()
    tracker.reset_state()
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn


_PYSOT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pysot"))


def _activate_pysot_root():
    if _PYSOT_ROOT not in sys.path:
        sys.path.insert(0, _PYSOT_ROOT)


class SiamRPNPPTracker(nn.Module):
    """
    SiamRPN++ (pysot) single-object tracker wrapped as an nn.Module,
    mirroring OSTrackTracker's API.

    Args:
        yaml_path:  Absolute path to a pysot-native experiment yaml.
        ckpt_path:  Absolute path to the .pth weight file.
        device:     Torch device string ("cuda" or "cpu").
    """

    def __init__(
        self,
        yaml_path: str,
        ckpt_path: str,
        device: str = "cuda",
    ):
        super().__init__()
        if not os.path.isfile(yaml_path):
            raise FileNotFoundError(f"SiamRPN++ yaml not found: {yaml_path}")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"SiamRPN++ checkpoint not found: {ckpt_path}")

        self.yaml_path = yaml_path
        self.ckpt_path = ckpt_path
        self.device = torch.device(device)

        self._predictor, self._model = self._build_predictor()

        self._frames: list[np.ndarray] | None = None
        self._prompt_frame: int | None = None
        self._prompt_label: int = 0

    def _build_predictor(self):
        _activate_pysot_root()

        from pysot.core.config import cfg
        from pysot.models.model_builder import ModelBuilder
        from pysot.tracker.siamrpn_tracker import SiamRPNTracker

        cfg.merge_from_file(self.yaml_path)
        cfg.CUDA = self.device.type == "cuda"

        model = ModelBuilder()
        state_dict = torch.load(
            self.ckpt_path, map_location="cpu", weights_only=False
        )
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise RuntimeError(
                f"Unexpected keys in SiamRPN++ checkpoint: {unexpected[:5]}..."
            )
        if missing:
            # A few BN num_batches_tracked buffers are commonly absent — tolerable.
            print(f"[SiamRPN++] {len(missing)} missing keys "
                  f"(first: {missing[:3]})")

        model.to(self.device).eval()
        tracker = SiamRPNTracker(model)
        return tracker, model

    # ------------------------------------------------------------------
    # Stateful video-level API (mirrors OSTrackTracker)
    # ------------------------------------------------------------------

    def init_video(self, frames: list[np.ndarray]):
        """Cache H×W×3 uint8 RGB frames for this sequence."""
        self._frames = [np.ascontiguousarray(f) for f in frames]
        self._prompt_frame = None
        self._prompt_label = 0

    def add_prompts(
        self,
        frame_idx: int,
        boxes: np.ndarray,
        labels: np.ndarray | None = None,
        obj_ids: list[int] | None = None,
    ):
        """Initialise the template from a single xyxy bbox (N=1 only)."""
        if len(boxes) == 0:
            return
        if self._frames is None:
            raise RuntimeError("init_video() must be called before add_prompts().")
        if frame_idx < 0 or frame_idx >= len(self._frames):
            raise IndexError(f"prompt frame_idx {frame_idx} out of range")

        if labels is not None and len(labels) > 0:
            self._prompt_label = int(labels[0])

        x1, y1, x2, y2 = [float(v) for v in boxes[0]]
        init_bbox_xywh = [x1, y1, x2 - x1, y2 - y1]

        bgr = self._rgb_to_bgr(self._frames[frame_idx])
        self._predictor.init(bgr, init_bbox_xywh)
        self._prompt_frame = frame_idx

    def propagate(self) -> list[dict]:
        """
        Propagate the template forward through the video.

        SiamRPN++ has no backward pass, so frames before the prompt frame are
        filled with empty outputs.
        """
        if self._frames is None or self._prompt_frame is None:
            return []

        n = len(self._frames)
        outputs: list[dict] = [self._empty_output() for _ in range(n)]

        p = self._prompt_frame
        # Echo the init bbox on the prompt frame — `self._predictor` stores it
        # as (center_pos, size) after init(), not as a bbox field, so we
        # reconstruct.
        init_xywh = self._current_state_xywh()
        outputs[p] = self._bbox_to_output(init_xywh, score=1.0)

        with torch.inference_mode():
            for i in range(p + 1, n):
                bgr = self._rgb_to_bgr(self._frames[i])
                out = self._predictor.track(bgr)
                xywh = out["bbox"]
                score = float(out.get("best_score", 1.0))
                outputs[i] = self._bbox_to_output(xywh, score=score)

        return outputs

    def reset_state(self):
        """Clear per-sequence frame cache. The underlying model is reused."""
        self._frames = None
        self._prompt_frame = None
        self._prompt_label = 0

    # ------------------------------------------------------------------

    def _current_state_xywh(self) -> list[float]:
        cx, cy = self._predictor.center_pos.tolist()
        w, h = self._predictor.size.tolist()
        return [cx - w / 2.0, cy - h / 2.0, w, h]

    @staticmethod
    def _rgb_to_bgr(frame_rgb: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(frame_rgb[..., ::-1])

    def _bbox_to_output(self, xywh, score: float = 1.0) -> dict:
        x, y, w, h = [float(v) for v in xywh]
        x2, y2 = x + w, y + h
        box_xyxy = torch.tensor([[x, y, x2, y2]], dtype=torch.float32)
        obb = torch.tensor(
            [[x, y, x2, y, x2, y2, x, y2]], dtype=torch.float32
        )
        return {
            "boxes":     box_xyxy,
            "obb":       obb,
            "labels":    torch.tensor([self._prompt_label], dtype=torch.long),
            "scores":    torch.tensor([score], dtype=torch.float32),
            "track_ids": torch.tensor([1], dtype=torch.long),
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
