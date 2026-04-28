"""
SORT — Simple Online and Realtime Tracking (Bewley et al., ICIP 2016).

Pure motion-model tracker: per-track constant-velocity Kalman filter on
``(cx, cy, area, aspect)``, Hungarian-IoU assignment, track lifecycle
governed by ``min_hits`` and ``max_age``.

Why this implementation in-tree
-------------------------------
Original SORT lives in ``abewley/sort`` but ships a single 300 LOC
script tied to a CLI entrypoint and a specific bbox file format. We need
the Tracker class only, so we re-implement the core 150 LOC here. Same
algorithm, same defaults, much cleaner integration with our cache /
eval pipeline.

References
----------
Bewley, A., Ge, Z., Ott, L., Ramos, F., & Upcroft, B. (2016). Simple
online and realtime tracking. ICIP 2016. arXiv:1602.00763.
"""

from __future__ import annotations

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Bbox helpers
# ---------------------------------------------------------------------------

def _xyxy_to_z(box: np.ndarray) -> np.ndarray:
    """xyxy → (cx, cy, area, aspect) column vector for the Kalman state."""
    w = box[2] - box[0]
    h = box[3] - box[1]
    cx = box[0] + w / 2.0
    cy = box[1] + h / 2.0
    s = w * h            # area
    r = w / max(h, 1e-6) # aspect ratio
    return np.array([cx, cy, s, r], dtype=np.float32).reshape(4, 1)


def _z_to_xyxy(z: np.ndarray) -> np.ndarray:
    """(cx, cy, area, aspect) → xyxy. Tolerates shape (4,) or (4, 1)."""
    z = np.asarray(z).reshape(-1)
    cx, cy, s, r = float(z[0]), float(z[1]), float(z[2]), float(z[3])
    s = max(s, 1e-6)
    r = max(r, 1e-6)
    w = np.sqrt(s * r)
    h = s / w
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0],
                    dtype=np.float32)


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two xyxy sets, shape [N, M]."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


# ---------------------------------------------------------------------------
# Per-track Kalman state
# ---------------------------------------------------------------------------

class _KalmanBoxTracker:
    """Tracks one bbox using a 7-D constant-velocity Kalman filter.

    State: ``[cx, cy, s, r, dcx, dcy, ds]``. Aspect ratio ``r`` is treated
    as constant (no velocity term) — same as the original SORT.
    """

    _next_id = 1

    def __init__(self, det_box: np.ndarray):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # Constant-velocity transition for cx,cy,s; r is held constant.
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)
        # Observation: we measure (cx, cy, s, r) directly.
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=np.float32)
        # Measurement noise — area / aspect are noisier than pixel coords.
        self.kf.R[2:, 2:] *= 10.0
        # Process noise — give the velocity components large variance.
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = _xyxy_to_z(det_box)

        self.id = _KalmanBoxTracker._next_id
        _KalmanBoxTracker._next_id += 1

        self.time_since_update = 0
        self.hit_streak = 1   # increment to >= min_hits before output starts
        self.age = 0

    @classmethod
    def reset_id_counter(cls):
        cls._next_id = 1

    def predict(self) -> np.ndarray:
        # Guard against negative area produced by extrapolated velocity.
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return _z_to_xyxy(self.kf.x[:4])

    def update(self, det_box: np.ndarray):
        self.time_since_update = 0
        self.hit_streak += 1
        self.kf.update(_xyxy_to_z(det_box))

    def state(self) -> np.ndarray:
        return _z_to_xyxy(self.kf.x[:4])


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class SORTTracker:
    """SORT — Bewley 2016 defaults.

    Args:
        max_age:        Tracks lost for more than this many frames are
                        deleted. Original paper default: 1.
        min_hits:       Tracks must hit this many consecutive frames before
                        their boxes are returned by ``update``. Default 3.
        iou_threshold:  Hungarian rejection threshold; pairs with IoU below
                        this are not allowed to match. Default 0.3.
        score_thresh:   Drop detections below this score before matching
                        (let downstream pick the operating point). Default 0.

    For tiny satellite cars (5–15 px), the original 0.3 IoU threshold is
    too strict — most predictions vs predicted-state pairs land at IoU
    0.05–0.20. Override via the constructor or per-dataset YAML.
    """

    def __init__(
        self,
        max_age: int = 1,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        score_thresh: float = 0.0,
    ):
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.iou_threshold = float(iou_threshold)
        self.score_thresh = float(score_thresh)

        self.tracks: list[_KalmanBoxTracker] = []
        self._frame_count = 0

    def reset(self):
        """Clear all tracks and global ID counter — call between videos."""
        self.tracks = []
        self._frame_count = 0
        _KalmanBoxTracker.reset_id_counter()

    def update(self, dets: np.ndarray, frame_id: int | None = None) -> np.ndarray:
        """Advance one frame.

        Args:
            dets: [N, 5] (x1, y1, x2, y2, score). May be empty.
            frame_id: optional, unused — only kept for interface parity
                      with trackers that need absolute timestamps.

        Returns: [M, 6] (x1, y1, x2, y2, score, track_id) — one row per
            currently active confirmed track.
        """
        self._frame_count += 1
        if dets is None or len(dets) == 0:
            dets = np.zeros((0, 5), dtype=np.float32)
        else:
            dets = np.asarray(dets, dtype=np.float32)
            if self.score_thresh > 0:
                dets = dets[dets[:, 4] >= self.score_thresh]

        # 1) Predict every track forward.
        predicted = np.zeros((len(self.tracks), 4), dtype=np.float32)
        invalid = []
        for i, trk in enumerate(self.tracks):
            box = trk.predict()
            if np.any(np.isnan(box)):
                invalid.append(i)
            else:
                predicted[i] = box
        for i in reversed(invalid):
            self.tracks.pop(i)
            predicted = np.delete(predicted, i, axis=0)

        # 2) Hungarian-IoU matching between predicted boxes and detections.
        det_boxes = dets[:, :4] if len(dets) else np.zeros((0, 4), dtype=np.float32)
        matches, unmatched_dets, unmatched_trks = self._associate(
            det_boxes, predicted, self.iou_threshold,
        )

        # 3) Update matched tracks with their assigned detections.
        for d_idx, t_idx in matches:
            self.tracks[t_idx].update(det_boxes[d_idx])

        # 4) Spawn new tracks for unmatched detections, remembering each
        #    new track's seed score so the first frame's output carries
        #    the detection confidence rather than zero.
        det_score_for_track: dict[int, float] = {}
        for d_idx, t_idx in matches:
            det_score_for_track[t_idx] = float(dets[d_idx, 4])
        for d_idx in unmatched_dets:
            new_idx = len(self.tracks)
            self.tracks.append(_KalmanBoxTracker(det_boxes[d_idx]))
            det_score_for_track[new_idx] = float(dets[d_idx, 4])

        # 5) Build output: confirmed tracks (hit_streak >= min_hits OR
        #    we're still inside the warm-up window) that hit on this frame.
        out_rows: list[np.ndarray] = []
        for i, trk in enumerate(self.tracks):
            if trk.time_since_update > 0:
                continue
            if not (trk.hit_streak >= self.min_hits
                    or self._frame_count <= self.min_hits):
                continue
            box = trk.state()
            score = det_score_for_track.get(i, 0.0)
            out_rows.append(np.array([
                box[0], box[1], box[2], box[3], score, trk.id,
            ], dtype=np.float32))

        # 6) Garbage-collect dead tracks.
        self.tracks = [
            t for t in self.tracks if t.time_since_update <= self.max_age
        ]

        if not out_rows:
            return np.zeros((0, 6), dtype=np.float32)
        return np.stack(out_rows, axis=0)

    @staticmethod
    def _associate(
        dets: np.ndarray, trks: np.ndarray, iou_thr: float,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if len(dets) == 0:
            return [], [], list(range(len(trks)))
        if len(trks) == 0:
            return [], list(range(len(dets))), []

        iou = _iou_matrix(dets, trks)
        # Hungarian wants a cost matrix to minimise → use -IoU.
        row_ind, col_ind = linear_sum_assignment(-iou)

        matches: list[tuple[int, int]] = []
        matched_d: set[int] = set()
        matched_t: set[int] = set()
        for r, c in zip(row_ind.tolist(), col_ind.tolist()):
            if iou[r, c] < iou_thr:
                continue
            matches.append((r, c))
            matched_d.add(r)
            matched_t.add(c)

        unmatched_dets = [i for i in range(len(dets)) if i not in matched_d]
        unmatched_trks = [i for i in range(len(trks)) if i not in matched_t]
        return matches, unmatched_dets, unmatched_trks
