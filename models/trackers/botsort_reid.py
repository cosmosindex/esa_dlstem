"""
BoT-SORT-ReID wrapper — same upstream BoT-SORT as ``botsort.py``, but with
``with_reid=True`` and a fake encoder that returns pre-computed FastReID
features instead of running FastReID inline.

Why a stub encoder: upstream ``BoTSORT.update(output_results, img)`` is
hard-wired to call ``self.encoder.inference(img, dets)`` whenever
``with_reid=True``; it has no entry point for "use these features I
already have." Replacing ``self.encoder`` with an object that
implements ``.inference(img, dets) -> [M, D]`` and looks features up
from a per-frame cache is the minimum-touch way to avoid a 2048-D
FastReID forward pass for every frame.

Feature lookup is by exact float-equality against the boxes registered
for the current frame. Both the boxes passed to ``BoTSORT.update`` and
the cached features come from the same numpy array, so BoT-SORT's
boolean-mask filtering preserves the values bit-for-bit — no FP noise
is introduced and ``np.where`` finds an exact match.
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np

# ---- inject a permissive stub fast_reid -------------------------------
# Upstream imports ``from fast_reid.fast_reid_interfece import
# FastReIDInterface`` at module load. ``BoTSortTracker`` (the
# ``with_reid=False`` sibling) installs a *strict* stub that raises on
# construction. With ``with_reid=True`` we *must* let construction
# succeed, then replace ``self.tracker.encoder`` with the cache stub.
# Force-overwrite to ensure our permissive stub is the one in effect
# regardless of import order.
class _PermissiveStubFastReID:  # noqa: D401
    """Permissive placeholder — instantiation is allowed, but the
    encoder is overwritten with :class:`_CachedFeatureEncoder` before
    ``inference`` ever runs."""
    def __init__(self, *a, **kw):
        self.config_path = a[0] if a else kw.get("config_path", "")
        self.weights_path = a[1] if len(a) > 1 else kw.get("weights_path", "")
        self.device = a[2] if len(a) > 2 else kw.get("device", "cpu")

    def inference(self, image, detections):
        raise RuntimeError(
            "FastReIDInterface.inference should never run — encoder "
            "must be replaced by _CachedFeatureEncoder before use.")


_fr_root = types.ModuleType("fast_reid")
_fr_iface = types.ModuleType("fast_reid.fast_reid_interfece")
_fr_iface.FastReIDInterface = _PermissiveStubFastReID
sys.modules["fast_reid"] = _fr_root
sys.modules["fast_reid.fast_reid_interfece"] = _fr_iface

_BOTSORT_REPO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "BoT-SORT",
)
if _BOTSORT_REPO not in sys.path:
    sys.path.insert(0, _BOTSORT_REPO)

from tracker.bot_sort import BoTSORT  # noqa: E402


class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _CachedFeatureEncoder:
    """Stand-in for ``FastReIDInterface``.

    The wrapper sets ``self.boxes`` and ``self.feats`` per frame (the
    full pre-filter set). When BoT-SORT calls ``inference(img, dets)``
    it has already applied ``score > track_high_thresh``; we look up
    each filtered row's feature by exact-equality match against
    ``self.boxes``.
    """
    def __init__(self):
        self.boxes: np.ndarray = np.zeros((0, 4), dtype=np.float32)
        self.feats: np.ndarray = np.zeros((0, 2048), dtype=np.float32)

    def set_frame(self, boxes: np.ndarray, feats: np.ndarray):
        self.boxes = np.asarray(boxes, dtype=np.float32)
        self.feats = np.asarray(feats, dtype=np.float32)

    def inference(self, image, detections) -> np.ndarray:
        if len(detections) == 0:
            return np.zeros((0, self.feats.shape[1]), dtype=np.float32)
        det = np.asarray(detections, dtype=np.float32)
        if len(self.boxes) == 0:
            return np.zeros((len(det), self.feats.shape[1]), dtype=np.float32)
        # Exact-equality match against the registered boxes. With
        # numpy-slice provenance this is guaranteed bit-for-bit.
        out = np.zeros((len(det), self.feats.shape[1]), dtype=np.float32)
        used = np.zeros(len(self.boxes), dtype=bool)
        for i, row in enumerate(det):
            mask = np.all(self.boxes == row, axis=1) & ~used
            idxs = np.where(mask)[0]
            if len(idxs) == 0:
                # Tolerate the rare case where BoT-SORT trims a box
                # (shouldn't happen in current upstream code, but safe).
                continue
            j = int(idxs[0])
            used[j] = True
            out[i] = self.feats[j]
        return out


class BoTSortReIDTracker:
    """BoT-SORT with appearance features pulled from a precomputed cache.

    Same hyperparameters as :class:`BoTSortTracker`, but ``with_reid``
    is forced to True and the FastReID forward pass is replaced by a
    cache lookup.

    Args mirror ``BoTSortTracker``; defaults retuned for satellite cars
    (HiEUM scores cluster 0.3-0.5).
    """
    def __init__(
        self,
        feat_dim: int = 2048,
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
        self.feat_dim = int(feat_dim)
        self._args = _Args(
            track_high_thresh=float(track_high_thresh),
            track_low_thresh=float(track_low_thresh),
            new_track_thresh=float(new_track_thresh),
            track_buffer=int(track_buffer),
            match_thresh=float(match_thresh),
            proximity_thresh=float(proximity_thresh),
            appearance_thresh=float(appearance_thresh),
            with_reid=True,
            cmc_method=cmc_method,
            mot20=bool(mot20),
            name="hieum_botsort_reid",
            ablation=False,
            fast_reid_config="",
            fast_reid_weights="",
            device="cpu",
        )
        self._frame_rate = int(frame_rate)
        self._build_tracker()

    def _build_tracker(self):
        self.tracker = BoTSORT(self._args, frame_rate=self._frame_rate)
        # Replace the FastReID encoder with the cache-lookup stub.
        self._encoder = _CachedFeatureEncoder()
        self.tracker.encoder = self._encoder

    def reset(self, vid_name: str = ""):
        self._build_tracker()

    def update_with_feats(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        feats: np.ndarray,
        frame_id: int | None = None,
        img: np.ndarray | None = None,
    ) -> np.ndarray:
        """Advance one frame using cached features.

        Args:
            boxes:  ``[N, 4]`` xyxy in original image coords.
            scores: ``[N]`` HiEUM detection scores (post Soft-NMS).
            feats:  ``[N, D]`` L2-normalized FastReID features
                    aligned with ``boxes``.
            img:    Frame image — only consulted by GMC when
                    ``cmc_method != "none"``. ``None`` is fine when
                    cmc_method='none'.

        Returns:
            ``[M, 6]`` xyxy + score + track_id for each active track.
        """
        boxes = np.asarray(boxes, dtype=np.float32) if boxes is not None else np.zeros((0, 4), np.float32)
        scores = np.asarray(scores, dtype=np.float32) if scores is not None else np.zeros(0, np.float32)
        feats = np.asarray(feats, dtype=np.float32) if feats is not None else np.zeros((0, self.feat_dim), np.float32)

        # Register feats for this frame BEFORE update. BoT-SORT's
        # internal score-filter selects a subset of `boxes`; the encoder
        # stub finds those rows by exact-eq match.
        self._encoder.set_frame(boxes, feats)

        if len(boxes):
            dets = np.concatenate(
                [boxes, scores[:, None]], axis=1
            ).astype(np.float32)
        else:
            dets = np.zeros((0, 5), dtype=np.float32)

        if img is None:
            img = np.zeros((1, 1, 3), dtype=np.uint8)

        online = self.tracker.update(dets, img)
        if not online:
            return np.zeros((0, 6), dtype=np.float32)

        rows: list[np.ndarray] = []
        for st in online:
            tlbr = st.tlbr
            rows.append(np.array([
                tlbr[0], tlbr[1], tlbr[2], tlbr[3], float(st.score), float(st.track_id),
            ], dtype=np.float32))
        return np.stack(rows, axis=0)
