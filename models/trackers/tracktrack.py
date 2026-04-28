"""
TrackTrack wrapper (Shim et al., CVPR 2025) — https://github.com/kamkyu94/TrackTrack

Upstream `Tracker.update(dets, dets_95)` consumes per-detection arrays of
shape `[N, 6+D]` where columns are `[x1, y1, x2, y2, score, _, *feat_D]`.
`detection[6:]` is the L2-normalized appearance embedding, required —
TrackTrack's iterative_assignment cost is `0.5 * iou + 0.5 * cos`.

Two upstream behaviors are stubbed for satellite videos:

1. **CMC** (`trackers.cmc.CMC`) reads pre-computed GMC homographies from
   `trackers/cmc/GMC-<vid_name>.txt`. Those exist for MOT17/MOT20/DanceTrack
   only, so we monkey-patch `__init__` to a no-op and `get_warp_matrix` to
   return identity. (Future: pre-compute via OpenCV ECC for our datasets.)

2. **Aspect-ratio filter** in upstream `run.py` drops boxes with
   `w/h > 1.6` only when `'MOT' in data_path`. We set `args.data_path = ""`
   so this and the `'Dance' in data_path` velocity-zero branch in
   `Track.predict` are both bypassed.
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np

# Add the upstream code dirs to sys.path. Names contain spaces + dots,
# but Python handles those fine in sys.path.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TT_TRACKER_DIR = os.path.join(_PROJECT_ROOT, "TrackTrack", "3. Tracker")
_TT_FASTREID_DIR = os.path.join(_PROJECT_ROOT, "TrackTrack", "2. FastReID")
for _d in (_TT_TRACKER_DIR, _TT_FASTREID_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

# Patch CMC before importing Tracker (Tracker -> trackers.cmc.* via star-import).
from trackers import cmc as _cmc_mod  # noqa: E402

def _cmc_init_stub(self, vid_name):
    self.vid_name = vid_name

def _cmc_warp_identity(self):
    return np.eye(2, 3, dtype=np.float64)

_cmc_mod.CMC.__init__ = _cmc_init_stub
_cmc_mod.CMC.get_warp_matrix = _cmc_warp_identity

from trackers.tracker import Tracker as _UpstreamTracker  # noqa: E402


class _Args:
    """Mimics argparse.Namespace consumed by `Tracker(args, vid_name)`.

    Holds the set of attributes actually read by upstream code paths
    (Tracker / Track / iterative_assignment).
    """
    def __init__(
        self,
        max_time_lost: int = 30,
        tai_thr: float = 0.55,
        init_thr: float = 0.60,
        det_thr: float = 0.50,
        match_thr: float = 0.70,
        penalty_p: float = 0.20,
        penalty_q: float = 0.40,
        reduce_step: float = 0.05,
        min_len: int = 3,
        min_box_area: float = 100.0,
    ):
        self.max_time_lost = int(max_time_lost)
        self.tai_thr = float(tai_thr)
        self.init_thr = float(init_thr)
        self.det_thr = float(det_thr)
        self.match_thr = float(match_thr)
        self.penalty_p = float(penalty_p)
        self.penalty_q = float(penalty_q)
        self.reduce_step = float(reduce_step)
        self.min_len = int(min_len)
        self.min_box_area = float(min_box_area)
        # Empty so the 'MOT' / 'Dance' string checks in upstream run.py
        # / track.py both fall through to the default branch.
        self.data_path = ""


class TrackTrackTracker:
    """TrackTrack — Focusing on Tracks for Online MOT (CVPR 2025).

    Args:
        feat_dim:        Length of the appearance embedding fed via
                         `dets[:, 6:]`. Default 2048 (FastReID SBS-S50).
        det_thr / init_thr / match_thr / tai_thr:
                         Confidence and IoU thresholds. Defaults are
                         dataset-agnostic — set per-config for satellite
                         car footage (typically lower than MOT17).
        penalty_p / penalty_q / reduce_step / min_len / min_box_area:
                         Standard TrackTrack hyperparameters.
        max_time_lost:   Frames a lost track is retained. Default 30.

    Notes:
        - This wrapper does not honour the existing
          `update(dets[N,5], frame_id)` interface used by SORT/ByteTrack/
          OC-SORT/BoT-SORT — TrackTrack needs both a second detection set
          (`dets_95`) and per-detection appearance features. The dedicated
          driver `eval_tracktrack.py` calls `update_with_feats` instead.
        - `vid_name` is required at construction — passed to upstream
          `CMC(vid_name)` (stubbed) and used as a key for any future
          pre-computed-GMC pathway.
    """

    def __init__(
        self,
        feat_dim: int = 2048,
        det_thr: float = 0.50,
        init_thr: float = 0.60,
        match_thr: float = 0.70,
        tai_thr: float = 0.55,
        penalty_p: float = 0.20,
        penalty_q: float = 0.40,
        reduce_step: float = 0.05,
        min_len: int = 3,
        min_box_area: float = 100.0,
        max_time_lost: int = 30,
    ):
        self.feat_dim = int(feat_dim)
        self.args = _Args(
            max_time_lost=max_time_lost,
            tai_thr=tai_thr,
            init_thr=init_thr,
            det_thr=det_thr,
            match_thr=match_thr,
            penalty_p=penalty_p,
            penalty_q=penalty_q,
            reduce_step=reduce_step,
            min_len=min_len,
            min_box_area=min_box_area,
        )
        self._tracker: _UpstreamTracker | None = None
        self._vid_name: str = ""

    def reset(self, vid_name: str = ""):
        """Reset state for a new video. ``vid_name`` is forwarded to CMC."""
        self._vid_name = vid_name or "default"
        self._tracker = _UpstreamTracker(self.args, self._vid_name)

    def update_with_feats(
        self,
        dets: np.ndarray,
        dets_95: np.ndarray | None = None,
    ) -> np.ndarray:
        """Advance one frame.

        Args:
            dets:    Float array of shape ``[N, 6+D]`` with columns
                     ``[x1, y1, x2, y2, score, _pad, *feat_D]``. Boxes
                     in original-image xyxy coordinates. Pass an empty
                     ``[0, 6+D]`` array to call ``update_without_detections``.
            dets_95: Optional second detection set (looser NMS). If None,
                     the same array is used as both — `find_deleted_detections`
                     then returns an empty pool, so the rescue path is
                     a no-op.

        Returns:
            ``[M, 6]`` float array: ``[x1, y1, x2, y2, score, track_id]``.
            Filtering applied: only `state == Tracked` tracks with
            ``track_id > 0`` and ``area > min_box_area``.
        """
        if self._tracker is None:
            self.reset()

        if dets is None or len(dets) == 0:
            tracks = self._tracker.update_without_detections()
        else:
            dets = np.ascontiguousarray(dets, dtype=np.float64)
            if dets.shape[1] < 6 + self.feat_dim:
                raise ValueError(
                    f"dets must have shape [N, 6+{self.feat_dim}], got "
                    f"shape {dets.shape}"
                )
            if dets_95 is None or len(dets_95) == 0:
                dets_95 = dets
            else:
                dets_95 = np.ascontiguousarray(dets_95, dtype=np.float64)
            tracks = self._tracker.update(dets, dets_95)

        if not tracks:
            return np.zeros((0, 6), dtype=np.float32)

        rows: list[np.ndarray] = []
        min_area = self.args.min_box_area
        for t in tracks:
            x1, y1, w, h = t.x1y1wh
            if t.track_id <= 0 or w * h <= min_area:
                continue
            rows.append(np.array(
                [x1, y1, x1 + w, y1 + h, float(t.score), float(t.track_id)],
                dtype=np.float32,
            ))
        if not rows:
            return np.zeros((0, 6), dtype=np.float32)
        return np.stack(rows, axis=0)
