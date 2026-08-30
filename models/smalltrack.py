"""
SmallTrack tracker wrapper (xyl-507/SmallTrack implementation).

SmallTrack (TGRS 2023) is a UAV small-object tracker built on SiamBAN: a
ResNet-50 whose pooling layers are replaced by a discrete wavelet transform
(WPL, to keep high-frequency detail that stride-2 pooling destroys) plus a
graph-enhanced classification head (GEM/GAL) that refines the classification
map. It is included here as a *tiny-object-specialised* reference point: it
was designed for targets an order of magnitude larger than ours, and the
question this benchmark asks is whether that specialisation survives the drop
to sub-8-pixel spaceborne targets.

Source: `<repo_root>/SmallTrack` (xyl-507/SmallTrack).

Integration quirks
------------------
* **Package-name collision with the vendored `siamban/`.** SmallTrack ships
  its own modified `siamban` package (DWT backbone + GAL head) under its repo
  root, and this project separately vendors upstream `hqucv/siamban`. Both
  claim the module name `siamban`. `_activate_smalltrack_root()` therefore
  puts SmallTrack's root at the FRONT of sys.path and evicts any already
  imported `siamban*` modules, mirroring the OSTrack/ODTrack `lib` fix in
  `models/ostrack.py`. Never import both in one process.
* **Weights are per-evaluation-dataset.** The authors release five
  checkpoints, one per benchmark they report (DTB70, UAV20L/UAVDT, VisDrone,
  LaTOT), each with its own WINDOW_INFLUENCE / PENALTY_K / LR triple baked
  into the config comments. We use the **LaTOT** checkpoint and the LaTOT
  hyper-parameters, since LaTOT is by far the smallest-target benchmark of the
  five and therefore the configuration most favourable to SmallTrack on our
  data. Report this choice: it is what makes "SmallTrack does not transfer" a
  statement about the domain rather than about tuning.
* `SiamBANTracker.init(img, bbox)` expects **BGR** uint8 images and **xywh**
  bboxes, exactly like pysot's SiamRPNTracker. Our pipeline passes RGB uint8
  and xyxy, so we convert both (identical to `models/siamrpn.py`).
* `cfg.BACKBONE.PRETRAINED` is only read by the upstream `tools/train.py`;
  the inference path never touches it, so the missing `resnet50.model` file
  is irrelevant here.
* `track()` returns a genuine `best_score`, so unlike SiamFC the global
  tau = 0.5 prediction filter is meaningful for this tracker.
* HBB-only tracker (no masks). `obb` is the 8-corner form of the AABB as a
  compatibility stub.

Usage mirrors SiamRPNPPTracker:
    tracker = SmallTrackTracker(
        yaml_path="configs/SOT/smalltrack_r50_l234_latot.yaml",
        ckpt_path=".../checkpoints/smalltrack/SmallTrack-LaTOT.pth",
    )
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn


_SMALLTRACK_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "SmallTrack")
)


def _activate_smalltrack_root():
    """
    Make SmallTrack's `siamban` the one that wins the import.

    The project also vendors upstream `hqucv/siamban` at the repo root, so a
    bare `import siamban` is ambiguous. We evict any `siamban*` modules that
    were imported from elsewhere and prepend SmallTrack's root.
    """
    for name in [m for m in sys.modules if m == "siamban" or m.startswith("siamban.")]:
        mod = sys.modules[name]
        origin = getattr(mod, "__file__", None) or ""
        if not origin.startswith(_SMALLTRACK_ROOT):
            del sys.modules[name]

    while _SMALLTRACK_ROOT in sys.path:
        sys.path.remove(_SMALLTRACK_ROOT)
    sys.path.insert(0, _SMALLTRACK_ROOT)


class SmallTrackTracker(nn.Module):
    """
    SmallTrack single-object tracker wrapped as an nn.Module, mirroring
    SiamRPNPPTracker's API.

    Args:
        yaml_path:  Absolute path to a siamban-native experiment yaml.
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
            raise FileNotFoundError(f"SmallTrack yaml not found: {yaml_path}")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"SmallTrack checkpoint not found: {ckpt_path}")

        self.yaml_path = yaml_path
        self.ckpt_path = ckpt_path
        self.device = torch.device(device)

        self._predictor, self._model = self._build_predictor()

        self._frames: list[np.ndarray] | None = None
        self._prompt_frame: int | None = None
        self._prompt_label: int = 0

    def _build_predictor(self):
        _activate_smalltrack_root()

        from siamban.core.config import cfg
        from siamban.models.model_builder import ModelBuilder
        from siamban.tracker.siamban_tracker import SiamBANTracker

        cfg.merge_from_file(self.yaml_path)
        cfg.CUDA = self.device.type == "cuda"

        model = ModelBuilder()
        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        # Upstream training wrapped the model in DataParallel for some runs.
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise RuntimeError(
                f"Unexpected keys in SmallTrack checkpoint: {unexpected[:5]}..."
            )
        if missing:
            # GAL carries SynchronizedBatchNorm buffers that are absent from the
            # released checkpoint; anything beyond that is a real mismatch.
            hard = [k for k in missing if not k.startswith("gal.")]
            if hard:
                raise RuntimeError(
                    f"Missing non-GAL keys in SmallTrack checkpoint: {hard[:5]}..."
                )
            print(f"[SmallTrack] {len(missing)} missing GAL buffer keys "
                  f"(first: {missing[:3]})")

        model.to(self.device).eval()
        tracker = SiamBANTracker(model)
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

        SmallTrack is causal, so frames before the prompt frame are filled with
        empty outputs.
        """
        if self._frames is None or self._prompt_frame is None:
            return []

        n = len(self._frames)
        outputs: list[dict] = [self._empty_output() for _ in range(n)]

        p = self._prompt_frame
        outputs[p] = self._bbox_to_output(self._current_state_xywh(), score=1.0)

        with torch.inference_mode():
            for i in range(p + 1, n):
                bgr = self._rgb_to_bgr(self._frames[i])
                out = self._predictor.track(bgr)
                outputs[i] = self._bbox_to_output(
                    out["bbox"], score=float(out.get("best_score", 1.0))
                )

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
