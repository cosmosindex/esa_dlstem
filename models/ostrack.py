"""
OSTrack tracker wrapper.

OSTrack (ECCV 2022) is the direct predecessor to ODTrack — same lab, same
`lib/` layout, near-identical inference API. One-stream Siamese transformer
single-object tracker with optional Candidate Elimination (CE) module.

Source: `/home/anon/code/esa_dlstem/OSTrack`.

This wrapper mirrors `ODTrackTracker` almost exactly; see that file for the
rationale behind each compat stub. Differences from ODTrack:

* `OSTrack.__init__(params, dataset_name)` takes an extra `dataset_name`
  positional arg. It's accepted but never referenced inside the class —
  we pass an empty string.
* No multi-template memory: template is encoded once in `initialize()` and
  reused for every subsequent frame.

Usage mirrors SAM2Tracker / ODTrackTracker:
    tracker = OSTrackTracker(yaml_path=..., ckpt_path=...)
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


_OSTRACK_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "OSTrack")
)


def _install_compat_stubs():
    """Inject stubs for modules OSTrack imports that aren't available here.

    Same set as the ODTrack wrapper — both repos share the same `lib/train/`
    transitively-imported dependencies:
      * `visdom` / `visdom.server` — basetracker imports it at module top.
      * `torch._six` — removed in PyTorch 2.x; `lib/train/data/loader.py`
        imports `string_classes` and `int_classes` from it.
      * `jpeg4py` — optional fast JPEG decoder; imported by `image_loader.py`.
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


def _activate_ostrack_root():
    """Put OSTrack's project root on sys.path so `lib.*` is importable.

    Warning: both OSTrack and ODTrack expose a top-level `lib` package, and
    so does HiEUM (auto-imported by ``models/__init__.py``). Once Python
    caches a foreign ``lib`` module the OSTrack import resolves there
    instead. We purge any cached ``lib`` / ``lib.*`` entries and prepend
    OSTrack's root so the fresh import binds to OSTrack's package.
    """
    _install_compat_stubs()
    for key in [k for k in sys.modules if k == "lib" or k.startswith("lib.")]:
        del sys.modules[key]
    if _OSTRACK_ROOT in sys.path:
        sys.path.remove(_OSTRACK_ROOT)
    sys.path.insert(0, _OSTRACK_ROOT)


class OSTrackTracker(nn.Module):
    """
    OSTrack single-object tracker wrapped as an nn.Module, mirroring SAM2Tracker's API.

    Args:
        yaml_path:    Absolute path to an OSTrack experiment yaml
                      (e.g. OSTrack/experiments/ostrack/vitb_384_mae_ce_32x4_ep300.yaml).
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
            raise FileNotFoundError(f"OSTrack yaml not found: {yaml_path}")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"OSTrack checkpoint not found: {ckpt_path}")

        self.yaml_path = yaml_path
        self.ckpt_path = ckpt_path
        self.device = device

        self.predictor = self._build_predictor()

        self._frames: list[np.ndarray] | None = None
        self._prompt_frame: int | None = None
        self._prompt_label: int = 0

    def _build_predictor(self):
        _activate_ostrack_root()

        from lib.config.ostrack.config import cfg, update_config_from_file
        from lib.test.utils import TrackerParams
        from lib.test.tracker.ostrack import OSTrack as _OSTrackImpl

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

        # dataset_name is accepted but unused inside OSTrack.__init__ — pass "".
        tracker = _OSTrackImpl(params, dataset_name="")
        return tracker

    # ------------------------------------------------------------------
    # Stateful video-level API (mirrors SAM2Tracker / ODTrackTracker)
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
        """Initialise the OSTrack template from a single xyxy bbox (N=1 only)."""
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

        OSTrack has no backward pass, so frames before the prompt frame are
        filled with empty outputs.
        """
        if self._frames is None or self._prompt_frame is None:
            return []

        n = len(self._frames)
        outputs: list[dict] = [self._empty_output() for _ in range(n)]

        p = self._prompt_frame
        init_xywh = self.predictor.state
        outputs[p] = self._bbox_to_output(init_xywh)

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

    def _bbox_to_output(self, xywh) -> dict:
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
