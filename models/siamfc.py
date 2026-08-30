"""
SiamFC tracker wrapper (huanglianghua/siamfc-pytorch implementation).

SiamFC (ECCV 2016 workshops) is the original fully-convolutional Siamese
tracker: an AlexNet backbone embeds an exemplar patch and a search patch, a
cross-correlation head produces a response map, and the peak of that map --
after a cosine window and a 3-scale pyramid search -- is the new target
centre. It has no region proposals, no online update and no appearance
memory, which makes it the cheapest tracker in this benchmark and the natural
"can a 2016 tracker still be the right operating point?" reference for
on-board deployment.

Source: `<repo_root>/siamfc-pytorch` (huanglianghua/siamfc-pytorch).

Integration quirks
------------------
* **Third-party re-implementation, GOT-10k-trained.** The original SiamFC is
  MatConvNet/MATLAB. This is the widely-used clean PyTorch port, and its
  released `siamfc_alexnet_e50.pth` is trained on GOT-10k, whereas the paper
  trained on ILSVRC-VID. Numbers here are therefore NOT the paper's numbers
  and must be footnoted as a re-implementation, the same way SiamRPN++'s
  community-fork checkpoint is.
* **got10k dependency is stubbed.** `siamfc/siamfc.py` imports
  `got10k.trackers.Tracker` purely to inherit a base class that provides a
  name and an `is_deterministic` flag; the whole got10k toolkit is not
  otherwise used at inference. We inject a minimal stub instead of adding a
  dependency (mirrors `models/ostrack.py::_install_compat_stubs`).
* **Colour order is RGB.** Upstream reads frames with
  `ops.read_image(..., cv2.COLOR_BGR2RGB)`, so the tracker expects RGB uint8 --
  same as our pipeline hands us. No conversion (unlike pysot/SiamRPN++).
* **Boxes are 1-indexed xywh (OTB convention).** `init()` expects
  `[x, y, w, h]` with 1-based top-left, and `update()` returns the same. Our
  pipeline speaks 0-indexed xyxy, so we shift by 1 on the way in and out. We
  keep the repo-wide `x2 = x1 + w` convention (rather than OTB's inclusive
  `x1 + w - 1`) so SiamFC's boxes stay directly comparable with every other
  tracker's here.
* **No confidence score.** `update()` returns only a box; the response peak
  is not exposed and is un-normalised across frames anyway. We emit a
  constant `score = 1.0`, which means the global `tau = 0.5` prediction filter
  is a no-op for SiamFC. This is a real cross-tracker calibration asymmetry
  and is reported rather than hidden -- SiamFC never declares "no valid box".
* **Model is built on `cuda:0` by upstream.** `TrackerSiamFC.__init__` picks
  the device itself; we move the net afterwards so an explicit `device=` (or
  a `CUDA_VISIBLE_DEVICES`-pinned GPU) is honoured.
* HBB-only tracker (no masks). `obb` is the 8-corner form of the AABB as a
  compatibility stub.

Usage mirrors OSTrackTracker:
    tracker = SiamFCTracker(
        ckpt_path=".../checkpoints/siamfc/siamfc_alexnet_e50.pth",
    )
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


_SIAMFC_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "siamfc-pytorch")
)


def _install_compat_stubs():
    """
    Provide the single got10k symbol `siamfc/siamfc.py` imports at module load.

    `TrackerSiamFC` subclasses `got10k.trackers.Tracker` only to record a name
    and a determinism flag; none of the toolkit's dataset/eval machinery is
    touched during inference. Stubbing it keeps got10k (and its own transitive
    deps) out of the environment.
    """
    if "got10k.trackers" in sys.modules:
        return

    class _Tracker(object):
        def __init__(self, name, is_deterministic=False):
            self.name = name
            self.is_deterministic = is_deterministic

    got10k = sys.modules.setdefault("got10k", types.ModuleType("got10k"))
    trackers_mod = types.ModuleType("got10k.trackers")
    trackers_mod.Tracker = _Tracker
    got10k.trackers = trackers_mod
    sys.modules["got10k.trackers"] = trackers_mod


def _activate_siamfc_root():
    if _SIAMFC_ROOT not in sys.path:
        sys.path.insert(0, _SIAMFC_ROOT)


class SiamFCTracker(nn.Module):
    """
    SiamFC single-object tracker wrapped as an nn.Module, mirroring
    OSTrackTracker's API.

    Args:
        ckpt_path: Absolute path to `siamfc_alexnet_e50.pth`.
        device:    Torch device string ("cuda" or "cpu").
    """

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
    ):
        super().__init__()
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"SiamFC checkpoint not found: {ckpt_path}")

        self.ckpt_path = ckpt_path
        self.device = torch.device(device)

        self._predictor = self._build_predictor()

        self._frames: list[np.ndarray] | None = None
        self._prompt_frame: int | None = None
        self._prompt_label: int = 0
        self._init_xywh: list[float] | None = None

    def _build_predictor(self):
        _activate_siamfc_root()
        _install_compat_stubs()

        from siamfc import TrackerSiamFC

        # Upstream loads the state dict itself and hard-selects cuda:0.
        predictor = TrackerSiamFC(net_path=self.ckpt_path)

        predictor.device = self.device
        predictor.cuda = self.device.type == "cuda"
        predictor.net.to(self.device).eval()
        return predictor

    # ------------------------------------------------------------------
    # Stateful video-level API (mirrors OSTrackTracker)
    # ------------------------------------------------------------------

    def init_video(self, frames: list[np.ndarray]):
        """Cache H×W×3 uint8 RGB frames for this sequence."""
        self._frames = [np.ascontiguousarray(f) for f in frames]
        self._prompt_frame = None
        self._prompt_label = 0
        self._init_xywh = None

    def add_prompts(
        self,
        frame_idx: int,
        boxes: np.ndarray,
        labels: np.ndarray | None = None,
        obj_ids: list[int] | None = None,
    ):
        """Initialise the exemplar from a single xyxy bbox (N=1 only)."""
        if len(boxes) == 0:
            return
        if self._frames is None:
            raise RuntimeError("init_video() must be called before add_prompts().")
        if frame_idx < 0 or frame_idx >= len(self._frames):
            raise IndexError(f"prompt frame_idx {frame_idx} out of range")

        if labels is not None and len(labels) > 0:
            self._prompt_label = int(labels[0])

        x1, y1, x2, y2 = [float(v) for v in boxes[0]]
        w, h = x2 - x1, y2 - y1

        # Our xyxy is 0-indexed; upstream wants 1-indexed top-left xywh.
        self._predictor.init(self._frames[frame_idx], [x1 + 1.0, y1 + 1.0, w, h])
        self._prompt_frame = frame_idx
        self._init_xywh = [x1, y1, w, h]

    def propagate(self) -> list[dict]:
        """
        Propagate the exemplar forward through the video.

        SiamFC is causal, so frames before the prompt frame are filled with
        empty outputs.
        """
        if self._frames is None or self._prompt_frame is None:
            return []

        n = len(self._frames)
        outputs: list[dict] = [self._empty_output() for _ in range(n)]

        p = self._prompt_frame
        outputs[p] = self._bbox_to_output(self._init_xywh)

        with torch.inference_mode():
            for i in range(p + 1, n):
                box_1idx = self._predictor.update(self._frames[i])
                outputs[i] = self._bbox_to_output(self._to_0indexed(box_1idx))

        return outputs

    def reset_state(self):
        """Clear per-sequence frame cache. The underlying model is reused."""
        self._frames = None
        self._prompt_frame = None
        self._prompt_label = 0
        self._init_xywh = None

    # ------------------------------------------------------------------

    @staticmethod
    def _to_0indexed(box_1idx) -> list[float]:
        """Upstream 1-indexed top-left xywh → our 0-indexed xywh."""
        x, y, w, h = [float(v) for v in box_1idx]
        return [x - 1.0, y - 1.0, w, h]

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
