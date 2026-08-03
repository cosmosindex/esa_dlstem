"""Track-level data access for the annotation review tool.

A *review item* is always one track, never one box: a track's category is
constant by construction and its presence/absence in the MOT ground truth is a
per-track property, so 300k boxes collapse to ~1k human decisions.

Two annotation sources are read side by side:

``det_xml``
    SAT-MTB's per-frame ``det/HBB`` PASCAL-VOC files. These carry ``objectID``,
    so they already form tracks, and they include the *static* objects the MOT
    ground truth drops.
``mot_gt``
    The MOT CSV that the benchmark actually scores against.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from space_tracker.data_mot import MOTObject, _parse_gt

from .paths import MOT_ROOTS, satmtb_det_dir


@dataclass(frozen=True)
class TrackKey:
    """Identifies one reviewable track."""

    seq_id: str
    source: str          # "det_xml" | "mot_gt"
    category: str        # label as written by that source
    track_ref: str       # objectID (det_xml) or track_id (mot_gt)

    @property
    def id(self) -> str:
        return f"{self.seq_id}#{self.source}:{self.category}:{self.track_ref}"

    @classmethod
    def parse(cls, key_id: str) -> "TrackKey":
        seq_id, rest = key_id.split("#", 1)
        source, category, track_ref = rest.split(":", 2)
        return cls(seq_id, source, category, track_ref)


@dataclass
class Track:
    """One track's boxes, ordered by frame."""

    key: TrackKey
    frame_ids: list[int]
    boxes: np.ndarray           # (N, 4) xyxy float
    #: SAT-MTB fine-grained class, majority vote over frames. Detection XML only.
    subname: str = ""
    subname_agreement: float = 0.0

    def __len__(self) -> int:
        return len(self.frame_ids)

    def box_at(self, frame_id: int) -> np.ndarray | None:
        try:
            return self.boxes[self.frame_ids.index(frame_id)]
        except ValueError:
            return None

    @property
    def median_size(self) -> float:
        wh = np.clip(self.boxes[:, 2:] - self.boxes[:, :2], 0, None)
        return float(np.median(np.sqrt(wh[:, 0] * wh[:, 1])))

    def sample_frames(self, n: int) -> list[int]:
        """Evenly spaced frame ids spanning the track, endpoints included."""
        if len(self) <= n:
            return list(self.frame_ids)
        idx = np.linspace(0, len(self) - 1, n).round().astype(int)
        return [self.frame_ids[i] for i in dict.fromkeys(idx.tolist())]


# ---------------------------------------------------------------------------
# detection XML
# ---------------------------------------------------------------------------

def _parse_hbb_xml(path: Path) -> list[tuple[str, str, str, np.ndarray]]:
    """-> [(category, objectID, subname, xyxy), ...] for one frame.

    ``subname`` is SAT-MTB's fine-grained class (``speed_boat``, ``yacht``,
    ``narrow_bodied_aircraft``, ...). Present in 141 of the 142 non-car
    sequences; a strong cross-check when judging whether a coarse label is
    right, since "det XML calls this a yacht" settles a ship/airplane dispute
    that "det XML calls this a ship" only restates.
    """
    out = []
    for obj in ET.parse(path).findall("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue
        out.append((
            (obj.findtext("name") or "").strip(),
            (obj.findtext("objectID") or "").strip(),
            (obj.findtext("subname") or "").strip(),
            np.array([
                float(bb.findtext("xmin", "0")), float(bb.findtext("ymin", "0")),
                float(bb.findtext("xmax", "0")), float(bb.findtext("ymax", "0")),
            ], dtype=np.float64),
        ))
    return out


@lru_cache(maxsize=4)
def load_det_tracks(seq_id: str, video_id: str) -> dict[tuple[str, str], Track]:
    """All detection-XML tracks of one SAT-MTB sequence, keyed by (category, objectID).

    Cached because the review UI walks a whole sequence one track at a time;
    re-parsing a few hundred XML files per click would dominate latency.
    """
    hbb_dir = satmtb_det_dir(video_id)
    if not hbb_dir.is_dir():
        return {}

    acc: dict[tuple[str, str], list[tuple[int, np.ndarray]]] = defaultdict(list)
    subnames: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for xml_path in sorted(hbb_dir.glob("*.xml")):
        try:
            fid = int(xml_path.stem)
        except ValueError:
            continue
        for category, oid, subname, box in _parse_hbb_xml(xml_path):
            acc[(category, oid)].append((fid, box))
            if subname:
                subnames[(category, oid)][subname] += 1

    tracks = {}
    for (category, oid), obs in acc.items():
        obs.sort(key=lambda t: t[0])
        # A track should carry one fine label; take the majority so a single
        # inconsistent frame does not decide it.
        votes = subnames[(category, oid)]
        tracks[(category, oid)] = Track(
            key=TrackKey(seq_id, "det_xml", category, oid),
            frame_ids=[f for f, _ in obs],
            boxes=np.stack([b for _, b in obs]),
            subname=votes.most_common(1)[0][0] if votes else "",
            subname_agreement=(votes.most_common(1)[0][1] / sum(votes.values())
                               if votes else 0.0),
        )
    return tracks


# ---------------------------------------------------------------------------
# MOT ground truth
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def load_mot_frames(seq_id: str) -> dict[int, list[MOTObject]]:
    """MOT ground truth of one sequence as ``{frame_id: [MOTObject, ...]}``."""
    from .paths import sequence_by_id

    seq = sequence_by_id(seq_id)
    return _parse_gt(seq, MOT_ROOTS[seq.dataset])


@lru_cache(maxsize=4)
def load_mot_tracks(seq_id: str) -> dict[tuple[str, int], Track]:
    """All MOT ground-truth tracks of one sequence, keyed by (category, track_id)."""
    acc: dict[tuple[str, int], list[tuple[int, np.ndarray]]] = defaultdict(list)
    for fid, objs in load_mot_frames(seq_id).items():
        for o in objs:
            acc[(o.category, o.track_id)].append((fid, o.bbox_xyxy.astype(np.float64)))

    tracks = {}
    for (category, tid), obs in acc.items():
        obs.sort(key=lambda t: t[0])
        tracks[(category, tid)] = Track(
            key=TrackKey(seq_id, "mot_gt", category, str(tid)),
            frame_ids=[f for f, _ in obs],
            boxes=np.stack([b for _, b in obs]),
        )
    return tracks


def load_track(key: TrackKey) -> Track | None:
    """Fetch a single track by key, from whichever source it came from."""
    from .paths import sequence_by_id

    if key.source == "det_xml":
        seq = sequence_by_id(key.seq_id)
        return load_det_tracks(key.seq_id, seq.video_id).get((key.category, key.track_ref))
    if key.source == "mot_gt":
        return load_mot_tracks(key.seq_id).get((key.category, int(key.track_ref)))
    raise ValueError(f"unknown track source {key.source!r}")
