"""
Online MOT trackers consuming HiEUM detections.

Common interface (all trackers expose the same minimal API)::

    class Tracker:
        def update(self, dets: np.ndarray, frame_id: int) -> np.ndarray:
            '''
            dets: [N, 5] xyxy + score, post-NMS detections for this frame.
            returns: [M, 6] xyxy + score + track_id (int), one row per
                     active track at this frame.
            '''

The eval driver (``eval_tracker.py``) iterates a video's frames in order,
calls ``update`` once per frame, and accumulates the (track_id, box, frame)
triples to compute MOTA / IDF1 / IDsw / HOTA.
"""

from .sort import SORTTracker
from .ocsort import OCSortTracker
from .bytetrack import ByteTracker
from .botsort import BoTSortTracker
from .botsort_reid import BoTSortReIDTracker
from .tracktrack import TrackTrackTracker


def build_tracker(name: str, **kwargs):
    """Factory used by ``eval_tracker.py`` and the per-dataset YAMLs.

    TrackTrack and BoT-SORT-ReID use a different update signature (need
    per-detection appearance features) and are consumed by
    ``eval_tracktrack.py`` / ``eval_botsort_reid.py`` rather than
    ``eval_tracker.py``.
    """
    name = name.lower()
    if name == "sort":
        return SORTTracker(**kwargs)
    if name == "ocsort":
        return OCSortTracker(**kwargs)
    if name == "bytetrack":
        return ByteTracker(**kwargs)
    if name == "botsort":
        return BoTSortTracker(**kwargs)
    if name == "botsort_reid":
        return BoTSortReIDTracker(**kwargs)
    if name == "tracktrack":
        return TrackTrackTracker(**kwargs)
    raise ValueError(f"unknown tracker {name!r}; choose from "
                     "sort/ocsort/bytetrack/botsort/botsort_reid/tracktrack")


__all__ = [
    "SORTTracker",
    "OCSortTracker",
    "ByteTracker",
    "BoTSortTracker",
    "BoTSortReIDTracker",
    "TrackTrackTracker",
    "build_tracker",
]
