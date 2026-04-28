"""
BoT-SORT wrapper (Aharon et al., arXiv 2206.14651).

We disable ReID — appearance features are unreliable for the 5–15 px
satellite cars that HiEUM detects, and the upstream FastReID dependency
is heavyweight (PyTorch model + OpenCV-CPU build of FastReID). Camera
motion compensation (GMC) can stay enabled or disabled per dataset:
``cmc_method="none"`` for fixed-camera satellite footage,
``"sparseOptFlow"`` for moving platforms.

Upstream's import chain pulls ``from fast_reid.fast_reid_interfece import
FastReIDInterface`` unconditionally even when ``with_reid=False``, so we
inject a stub module into ``sys.modules`` before importing.
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np

# ---- inject a permissive stub fast_reid -----------------------------
# Upstream BoT-SORT does ``from fast_reid.fast_reid_interfece import
# FastReIDInterface`` unconditionally at module load. Both this wrapper
# (with_reid=False) and the sibling ``botsort_reid.py`` rely on this
# stub being permissive — botsort_reid replaces ``self.tracker.encoder``
# with a cache-lookup stub *after* construction, so the stub's
# ``__init__`` must succeed. ``inference`` is never reached: with_reid
# False short-circuits before it, with_reid True replaces the encoder
# before it.
if "fast_reid" not in sys.modules:
    fr_root = types.ModuleType("fast_reid")
    fr_iface = types.ModuleType("fast_reid.fast_reid_interfece")

    class _StubFastReID:  # noqa: D401 — stub
        def __init__(self, *a, **kw):
            self.config_path = a[0] if a else kw.get("config_path", "")
            self.weights_path = a[1] if len(a) > 1 else kw.get("weights_path", "")
            self.device = a[2] if len(a) > 2 else kw.get("device", "cpu")

        def inference(self, image, detections):
            raise RuntimeError(
                "FastReIDInterface.inference must not run — encoder must "
                "be replaced before use, or with_reid must be False.")

    fr_iface.FastReIDInterface = _StubFastReID
    sys.modules["fast_reid"] = fr_root
    sys.modules["fast_reid.fast_reid_interfece"] = fr_iface

# Make BoT-SORT importable: it uses ``from tracker import matching`` etc.
# Pointing sys.path at the repo root lets that resolve.
_BOTSORT_REPO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "BoT-SORT",
)
if _BOTSORT_REPO not in sys.path:
    sys.path.insert(0, _BOTSORT_REPO)

from tracker.bot_sort import BoTSORT  # noqa: E402


class _Args:
    """Minimal args namespace consumed by BoTSORT."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


class BoTSortTracker:
    """BoT-SORT — Kalman + GMC + (no ReID).

    Args:
        track_high_thresh: First-stage detection threshold. Default 0.6.
                           Lower for HiEUM whose Soft-NMS scores cluster
                           at 0.3–0.5.
        track_low_thresh:  Floor for detections still considered for the
                           low-conf second stage. Default 0.1.
        new_track_thresh:  Score above which a previously-unseen detection
                           seeds a new track. Default 0.7. Lower for
                           sparse satellite scenes.
        track_buffer:      Frames a lost track is kept around. Default 30.
        match_thresh:      First-stage IoU acceptance threshold. Default 0.8.
        proximity_thresh:  IoU floor used in the appearance-fusion path
                           (irrelevant when with_reid=False). Default 0.5.
        appearance_thresh: Appearance distance floor (irrelevant when
                           with_reid=False). Default 0.25.
        cmc_method:        Camera-motion-comp method:
                           ``"none" | "orb" | "sift" | "ecc" | "sparseOptFlow"``.
                           Use ``"none"`` for fixed-camera satellite video
                           (default here).
        mot20:             MOT20 mode flag (disables in-image clipping).
                           Default False.
        frame_rate:        Used for buffer scaling. Default 30.
    """

    def __init__(
        self,
        track_high_thresh: float = 0.6,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.7,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        proximity_thresh: float = 0.5,
        appearance_thresh: float = 0.25,
        cmc_method: str = "none",
        mot20: bool = False,
        frame_rate: int = 30,
    ):
        self._args = _Args(
            track_high_thresh=float(track_high_thresh),
            track_low_thresh=float(track_low_thresh),
            new_track_thresh=float(new_track_thresh),
            track_buffer=int(track_buffer),
            match_thresh=float(match_thresh),
            proximity_thresh=float(proximity_thresh),
            appearance_thresh=float(appearance_thresh),
            with_reid=False,
            cmc_method=cmc_method,
            mot20=bool(mot20),
            name="hieum_botsort",
            ablation=False,
            fast_reid_config=None,
            fast_reid_weights=None,
            device="cpu",
        )
        self._frame_rate = int(frame_rate)
        self.tracker = BoTSORT(self._args, frame_rate=self._frame_rate)

    def reset(self):
        # Recreate the tracker so internal track-id counters / lists clear.
        self.tracker = BoTSORT(self._args, frame_rate=self._frame_rate)

    def update(self, dets: np.ndarray, frame_id: int | None = None,
               img: np.ndarray | None = None) -> np.ndarray:
        """Advance one frame.

        Args:
            dets: [N, 5] xyxy + score in original-image coords.
            img:  Frame image — only consulted by GMC when
                  ``cmc_method != "none"``. ``None`` is fine for fixed
                  cameras / cmc_method='none'.
        Returns:
            [M, 6] xyxy + score + track_id.
        """
        if dets is None or len(dets) == 0:
            dets = np.zeros((0, 5), dtype=np.float32)
        else:
            dets = np.asarray(dets, dtype=np.float32)

        # Upstream branches on ``output_results.shape[1]``:
        #   == 5  → treat as [xyxy, score]  (single foreground class)
        #   else → ``score = output[:,4] * output[:,5]`` (obj × cls)
        # We pass [N, 5] so the score isn't multiplied by an extra column.
        # GMC needs an image when cmc_method != 'none'. Provide a 1x1
        # stub if caller didn't pass one and CMC is disabled.
        if img is None:
            img = np.zeros((1, 1, 3), dtype=np.uint8)

        online = self.tracker.update(dets, img)
        if not online:
            return np.zeros((0, 6), dtype=np.float32)

        out_rows: list[np.ndarray] = []
        for st in online:
            tlbr = st.tlbr
            out_rows.append(np.array([
                tlbr[0], tlbr[1], tlbr[2], tlbr[3], float(st.score), float(st.track_id),
            ], dtype=np.float32))
        return np.stack(out_rows, axis=0)
