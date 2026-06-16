"""
OC-SORT wrapper (Cao et al., CVPR 2023).

Wraps the upstream ``OC_SORT/trackers/ocsort_tracker/ocsort.py`` so it
exposes the same ``update(dets, frame_id) -> [M, 6]`` interface as
``SORTTracker``. We use the upstream's ``update_public`` entry point
because it takes raw ``dets`` directly without the scale-ratio juggling
of the regular ``update`` method.
"""

from __future__ import annotations

import os
import sys

import numpy as np


_OCSORT_REPO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "OC_SORT", "trackers",
)
if _OCSORT_REPO not in sys.path:
    sys.path.insert(0, _OCSORT_REPO)

from ocsort_tracker.ocsort import OCSort  # noqa: E402


class OCSortTracker:
    """Thin wrapper around upstream OCSort.

    Args:
        det_thresh:    Score floor for detections to be tracked. Default
                       0.3 (paper default for MOT17). Lower it for sparse
                       satellite scenes.
        max_age:       Frames a track can survive without a hit. Default
                       30 (paper default).
        min_hits:      Hits required before a track is reported. Default 3.
        iou_threshold: IoU floor for matching. Default 0.3.
        delta_t:       OC observation re-anchoring window (paper default 3).
        asso_func:     Association cost: ``"iou" | "giou" | "ciou" |
                       "diou" | "ct_dist"``. Default ``"iou"``.
        inertia:       OC inertia weight. Default 0.2.
        use_byte:      Apply the ByteTrack second-stage low-score rescue.
                       Default False.
    """

    def __init__(
        self,
        det_thresh: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        delta_t: int = 3,
        asso_func: str = "iou",
        inertia: float = 0.2,
        use_byte: bool = False,
    ):
        # Stash the original kwargs as strings/numbers so ``reset`` can
        # rebuild the tracker. Upstream's OCSort overwrites ``asso_func``
        # with a function reference on construction, so we can't read it
        # back from the instance.
        self._init_kwargs = dict(
            det_thresh=float(det_thresh),
            max_age=int(max_age),
            min_hits=int(min_hits),
            iou_threshold=float(iou_threshold),
            delta_t=int(delta_t),
            asso_func=asso_func,
            inertia=float(inertia),
            use_byte=bool(use_byte),
        )
        self.tracker = OCSort(**self._init_kwargs)

    def reset(self):
        # Rebuild from the original (string-form) kwargs so the
        # ASSO_FUNCS lookup keeps working.
        self.tracker = OCSort(**self._init_kwargs)

    def update(self, dets: np.ndarray, frame_id: int | None = None) -> np.ndarray:
        """Advance one frame.

        Args:
            dets: [N, 5] xyxy + score.
        Returns:
            [M, 6] xyxy + score + track_id.
        """
        if dets is None or len(dets) == 0:
            dets = np.zeros((0, 5), dtype=np.float32)
        else:
            dets = np.asarray(dets, dtype=np.float32)

        boxes = dets[:, :4] if len(dets) else np.zeros((0, 4), dtype=np.float32)
        scores = dets[:, 4] if len(dets) else np.zeros((0,), dtype=np.float32)
        cats = np.zeros_like(scores)  # single class

        # ``update_public`` returns [M, 6] = [x1, y1, x2, y2, track_id, cat_id].
        out = self.tracker.update_public(boxes, cats, scores)
        if out is None or len(out) == 0:
            return np.zeros((0, 6), dtype=np.float32)
        out = np.asarray(out, dtype=np.float32)
        # Re-attach the per-track score by IoU-matching against the input
        # detections (upstream drops the score column from update_public).
        out_score = self._lookup_scores(out[:, :4], boxes, scores)
        result = np.column_stack([
            out[:, :4],            # xyxy
            out_score,             # score
            out[:, 4].astype(np.float32),   # track_id
        ])
        return self._dedup_by_id(result)

    @staticmethod
    def _dedup_by_id(tracks: np.ndarray) -> np.ndarray:
        """Keep one row per track id in a frame (highest score; ties -> first).

        Upstream ``update_public``'s observation-centric recovery can emit the
        same track id twice in one frame (its matched detection *and* a
        re-anchored last-observation box). A track cannot occupy two boxes in a
        timestep, so collapse them — otherwise TrackEval rejects the sequence
        ("predicts the same ID more than once in a single timestep") and scores
        it NaN.
        """
        if len(tracks) <= 1:
            return tracks
        order = np.argsort(-tracks[:, 4], kind="stable")  # score desc, stable
        srt = tracks[order]
        _, first = np.unique(srt[:, 5], return_index=True)
        return srt[np.sort(first)]

    @staticmethod
    def _lookup_scores(track_boxes, det_boxes, det_scores):
        if len(track_boxes) == 0 or len(det_boxes) == 0:
            return np.zeros(len(track_boxes), dtype=np.float32)
        x1 = np.maximum(track_boxes[:, None, 0], det_boxes[None, :, 0])
        y1 = np.maximum(track_boxes[:, None, 1], det_boxes[None, :, 1])
        x2 = np.minimum(track_boxes[:, None, 2], det_boxes[None, :, 2])
        y2 = np.minimum(track_boxes[:, None, 3], det_boxes[None, :, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area_t = (track_boxes[:, 2] - track_boxes[:, 0]) * (track_boxes[:, 3] - track_boxes[:, 1])
        area_d = (det_boxes[:, 2] - det_boxes[:, 0]) * (det_boxes[:, 3] - det_boxes[:, 1])
        union = area_t[:, None] + area_d[None, :] - inter
        iou = inter / np.maximum(union, 1e-9)
        best = iou.argmax(axis=1)
        return det_scores[best].astype(np.float32)
