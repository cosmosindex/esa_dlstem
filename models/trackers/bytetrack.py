"""
ByteTrack wrapper (Zhang et al., ECCV 2022).

Wraps the upstream ``ByteTrack/yolox/tracker/byte_tracker.py`` so it
exposes the same ``update(dets, frame_id) -> [M, 6]`` interface as
``SORTTracker`` and ``OCSortTracker``.

Upstream's ``update(output_results, img_info, img_size)`` accepts the
detection set in xyxy + score form and rescales by ``img_info / img_size``
internally. We pass ``img_info == img_size`` so no rescaling happens —
HiEUM already produces detections in original-image coordinates.
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np

_BYTETRACK_REPO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "ByteTrack",
)
if _BYTETRACK_REPO not in sys.path:
    sys.path.insert(0, _BYTETRACK_REPO)

from yolox.tracker.byte_tracker import BYTETracker  # noqa: E402


class _Args:
    """Minimal argparse-like namespace consumed by upstream BYTETracker."""
    def __init__(self, track_thresh, track_buffer, match_thresh, mot20):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.mot20 = mot20


class ByteTracker:
    """ByteTrack — high/low score two-stage matching.

    Args:
        track_thresh:   Score threshold above which a detection enters
                        the high-confidence first stage (matched first).
                        Default 0.5 — lower (e.g. 0.30) for HiEUM, whose
                        decayed Soft-NMS scores cluster around 0.3–0.5.
        track_buffer:   Frames a lost track is kept around. Default 30.
        match_thresh:   IoU acceptance threshold for first-stage match.
                        Default 0.8.
        mot20:          MOT20 mode disables the in-image-bound clipping.
                        Default False.
        frame_rate:     Used by upstream to scale the lost-track buffer.
                        Default 30.
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        mot20: bool = False,
        frame_rate: int = 30,
    ):
        self._args = _Args(
            track_thresh=float(track_thresh),
            track_buffer=int(track_buffer),
            match_thresh=float(match_thresh),
            mot20=bool(mot20),
        )
        self._frame_rate = int(frame_rate)
        self.tracker = BYTETracker(self._args, frame_rate=self._frame_rate)

    def reset(self):
        # BYTETracker stores running state in instance lists + a frame
        # counter; rebuilding is the cleanest reset path.
        self.tracker = BYTETracker(self._args, frame_rate=self._frame_rate)

    def update(self, dets: np.ndarray, frame_id: int | None = None) -> np.ndarray:
        """Advance one frame.

        Args:
            dets: [N, 5] xyxy + score in original-image coords.
        Returns:
            [M, 6] xyxy + score + track_id.
        """
        if dets is None or len(dets) == 0:
            dets = np.zeros((0, 5), dtype=np.float32)
        else:
            dets = np.asarray(dets, dtype=np.float32)

        # Pass img_info == img_size so the internal scale factor is 1.0
        # — HiEUM already produces boxes in the original frame.
        # Upstream needs a 2-tuple: it reads index [0] and [1].
        H, W = 1, 1   # placeholder; the actual ratio = img_size/img_info
        online = self.tracker.update(dets, (H, W), (H, W))
        if not online:
            return np.zeros((0, 6), dtype=np.float32)

        out_rows: list[np.ndarray] = []
        for st in online:
            tlbr = st.tlbr  # x1, y1, x2, y2
            out_rows.append(np.array([
                tlbr[0], tlbr[1], tlbr[2], tlbr[3], float(st.score), float(st.track_id),
            ], dtype=np.float32))
        return np.stack(out_rows, axis=0)
