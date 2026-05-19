"""
ODTrack tracker wrapper.

ODTrack (AAAI 2024) is a classical Siamese-transformer single-object tracker.
Unlike the SAM family, it works frame-by-frame: given an initial bbox on one
frame it encodes a template, then processes each subsequent frame's search
region and regresses a new HBB. It does NOT produce segmentation masks.

Source: `/home/ziwen/code/esa_dlstem/ODTrack` (cloned from yangchris11/ODTrack).

Integration quirks
------------------
* The repo uses absolute imports rooted at `lib.*`, so `ODTrack/` must be on
  sys.path. We prepend it on first build.
* `lib.test.tracker.basetracker` unconditionally `import visdom` at module
  top (via `lib.vis.visdom_cus`). We inject a stub `visdom` into sys.modules
  before importing, so we don't need to install visdom.
* `lib.test.parameter.odtrack.parameters()` goes through `env_settings()`,
  which hard-codes checkpoint paths under `save_dir/checkpoints/train/...`.
  We bypass it and build a `TrackerParams` directly from our config.
* Single-object only. SOT datasets are single-object, so that's fine.
* Outputs HBB; `obb` field is the 8-corner form of the AABB as a compatibility
  stub. Use AABB-based metrics; OBB metrics on OOTB will be degenerate.

Usage mirrors SAM2Tracker:
    tracker = ODTrackTracker(yaml_path=..., ckpt_path=...)
    tracker.init_video(frames)
    tracker.add_prompts(frame_idx=0, boxes=..., labels=..., obj_ids=...)
    outputs = tracker.propagate()
    tracker.reset_state()
"""

import os
import sys
import types

import numpy as np
import torch
import torch.nn as nn


_ODTRACK_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "ODTrack")
)


def _install_compat_stubs():
    """Inject stubs for modules ODTrack imports that aren't available here.

    * `visdom` / `visdom.server` — ODTrack's basetracker imports visdom at top,
      but we never trigger debug mode so a no-op stub is enough.
    * `torch._six` — removed in PyTorch 2.x; ODTrack's data loader imports
      `string_classes` and `int_classes` from it at module top even though
      the loader itself is never used during tracker inference.
    """
    if "visdom" not in sys.modules:
        stub = types.ModuleType("visdom")

        class _NoopVisdom:
            def __init__(self, *a, **kw):
                pass

            def __getattr__(self, name):
                return lambda *a, **kw: None

        stub.Visdom = _NoopVisdom
        sys.modules["visdom"] = stub
        sys.modules["visdom.server"] = types.ModuleType("visdom.server")

    if "torch._six" not in sys.modules:
        six_stub = types.ModuleType("torch._six")
        six_stub.string_classes = (str,)
        six_stub.int_classes = (int,)
        sys.modules["torch._six"] = six_stub

    if "jpeg4py" not in sys.modules:
        jpeg_stub = types.ModuleType("jpeg4py")

        class _JPEG:
            def __init__(self, *a, **kw):
                raise RuntimeError("jpeg4py stub — not usable for real decoding")

        jpeg_stub.JPEG = _JPEG
        sys.modules["jpeg4py"] = jpeg_stub


def _activate_odtrack_root():
    """Put ODTrack's project root on sys.path so `lib.*` is importable.

    Purges any cached foreign ``lib`` modules (OSTrack/HiEUM also expose
    top-level ``lib`` packages) so the fresh import binds to ODTrack's.
    """
    _install_compat_stubs()
    for key in [k for k in sys.modules if k == "lib" or k.startswith("lib.")]:
        del sys.modules[key]
    if _ODTRACK_ROOT in sys.path:
        sys.path.remove(_ODTRACK_ROOT)
    sys.path.insert(0, _ODTRACK_ROOT)


class ODTrackTracker(nn.Module):
    """
    ODTrack single-object tracker wrapped as an nn.Module, mirroring SAM2Tracker's API.

    Args:
        yaml_path:    Absolute path to an ODTrack experiment yaml
                      (e.g. ODTrack/experiments/odtrack/baseline_large.yaml).
        ckpt_path:    Absolute path to the .pth.tar checkpoint.
        device:       Torch device string ("cuda" or "cpu").
    """

    def __init__(
        self,
        yaml_path: str,
        ckpt_path: str,
        device: str = "cuda",
    ):
        super().__init__()
        if not os.path.isfile(yaml_path):
            raise FileNotFoundError(f"ODTrack yaml not found: {yaml_path}")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"ODTrack checkpoint not found: {ckpt_path}")

        self.yaml_path = yaml_path
        self.ckpt_path = ckpt_path
        self.device = device

        self.predictor = self._build_predictor()

        # Per-sequence state
        self._frames: list[np.ndarray] | None = None
        self._prompt_frame: int | None = None
        self._prompt_label: int = 0

    def _build_predictor(self):
        _activate_odtrack_root()

        from lib.config.odtrack.config import cfg, update_config_from_file
        from lib.test.utils import TrackerParams
        from lib.test.tracker.odtrack import ODTrack as _ODTrackImpl

        update_config_from_file(self.yaml_path)

        params = TrackerParams()
        params.cfg = cfg
        params.template_factor = cfg.TEST.TEMPLATE_FACTOR
        params.template_size = cfg.TEST.TEMPLATE_SIZE
        params.search_factor = cfg.TEST.SEARCH_FACTOR
        params.search_size = cfg.TEST.SEARCH_SIZE
        params.checkpoint = self.ckpt_path
        params.save_all_boxes = False
        params.debug = 0

        tracker = _ODTrackImpl(params)
        # `_ODTrackImpl.__init__` hard-codes .cuda(); nothing to do for CPU.
        return tracker

    # ------------------------------------------------------------------
    # Stateful video-level API (mirrors SAM2Tracker)
    # ------------------------------------------------------------------

    def init_video(self, frames: list[np.ndarray]):
        """Cache the frames for this sequence. Frames are H×W×3 uint8 RGB."""
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
        """
        Initialise the ODTrack template from a single xyxy bbox.

        ODTrack is single-object: only the first box is used if multiple
        are provided.
        """
        if len(boxes) == 0:
            return
        if frame_idx < 0 or frame_idx >= len(self._frames):
            raise IndexError(f"prompt frame_idx {frame_idx} out of range")

        x1, y1, x2, y2 = [float(v) for v in boxes[0]]
        init_bbox_xywh = [x1, y1, x2 - x1, y2 - y1]

        if labels is not None and len(labels) > 0:
            self._prompt_label = int(labels[0])

        self.predictor.initialize(
            self._frames[frame_idx],
            {"init_bbox": init_bbox_xywh},
        )
        self._prompt_frame = frame_idx

    def propagate(self) -> list[dict]:
        """
        Propagate the template forward through the video.

        ODTrack has no backward pass, so frames before the prompt frame are
        filled with empty outputs.

        Returns: list of per-frame dicts, same schema as SAM2Tracker.
        """
        if self._frames is None or self._prompt_frame is None:
            return []

        n = len(self._frames)
        outputs: list[dict] = [self._empty_output() for _ in range(n)]

        # Prompt frame: use the init bbox itself
        p = self._prompt_frame
        init_xywh = self.predictor.state  # [x, y, w, h] set by initialize()
        outputs[p] = self._bbox_to_output(init_xywh)

        # Forward from prompt+1 to end
        for i in range(p + 1, n):
            out = self.predictor.track(self._frames[i])
            xywh = out["target_bbox"]
            outputs[i] = self._bbox_to_output(xywh)

        return outputs

    def reset_state(self):
        """Clear per-sequence caches. The underlying model is reused across sequences."""
        self._frames = None
        self._prompt_frame = None
        self._prompt_label = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _bbox_to_output(self, xywh) -> dict:
        x, y, w, h = [float(v) for v in xywh]
        x2, y2 = x + w, y + h
        box_xyxy = torch.tensor([[x, y, x2, y2]], dtype=torch.float32)
        # OBB as the 4 corners of the AABB, flattened (x1,y1,x2,y1,x2,y2,x1,y2).
        obb = torch.tensor(
            [[x, y, x2, y, x2, y2, x, y2]], dtype=torch.float32
        )
        return {
            "boxes":     box_xyxy,
            "obb":       obb,
            "labels":    torch.tensor([self._prompt_label], dtype=torch.long),
            "scores":    torch.tensor([1.0], dtype=torch.float32),
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
