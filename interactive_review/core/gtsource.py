"""The ground truth a video review actually looks at, for every dataset.

The track-level tool read two annotation sources side by side and asked a human
to reconcile them. That reconciliation is now a batch job
(``tools/merge_det_to_mot.py``), so what the reviewer faces is a *single*
completed ground truth per sequence — and the question changes from "should this
object be in the ground truth" to "is this box right".

Three layers, lowest first:

``raw``
    The dataset's own MOT ground truth, parsed by ``space_tracker.data_mot``.
    This is all there is for AIR-MOT, VISO, SDM-Car and RsCarData.
``merged``
    ``tools/merge_det_to_mot.py`` output: SAT-MTB with its static objects
    restored and all non-car geometry at the SAM 3 standard. Present as a
    parallel tree mirroring each sequence's ``gt_path``, so it loads through the
    same parser with a different root.
``reviewed``
    Per-frame corrections a human made in the UI, plus tracks drawn by hand.
    Always wins.

Provenance travels with every box, because the reviewer's attention should go
where the annotation is newest: a box recovered from detection XML and one that
shipped with the dataset deserve different scrutiny, and after they are merged
into one file nothing else can tell them apart.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from space_tracker.data_mot import _parse_gt

from .paths import MOT_ROOTS, sequence_by_id

#: Root of ``tools/merge_det_to_mot.py`` output. Unset -> raw ground truth only.
MERGED_ROOT = Path(os.environ.get("SPACE_TRACKER_MERGED",
                                  "/work/anon/space_tracker_mot_merged"))

#: Where a box came from. Drives colour in every view.
ORIGINAL = "original"      # shipped with the dataset
RECOVERED = "recovered"    # restored from detection XML by the merge
FILLED = "filled"          # a frame the merge interpolated + SAM 3 refined
REVIEWED = "reviewed"      # corrected by hand in this tool
DRAWN = "drawn"            # annotated from nothing, in this tool


@dataclass
class Obj:
    """One ground-truth box on one frame, with where it came from."""

    track_id: int
    category: str
    box: np.ndarray            # (4,) xyxy float
    provenance: str = ORIGINAL

    @property
    def key(self) -> str:
        """Stable id for a track within its sequence: ``"airplane:12"``."""
        return f"{self.category}:{self.track_id}"

    @property
    def size(self) -> float:
        w, h = max(self.box[2] - self.box[0], 0.0), max(self.box[3] - self.box[1], 0.0)
        return float(np.sqrt(w * h))


def merged_available(seq_id: str) -> bool:
    seq = sequence_by_id(seq_id)
    return (MERGED_ROOT / seq.gt_path).is_file()


@lru_cache(maxsize=8)
def _provenance(seq_id: str) -> dict[int, str]:
    """``{track id: provenance}`` for the tracks the merge added.

    Read from the merge's own sidecar rather than re-derived: which ids are new
    is a fact about how the file was built, and recomputing it from box
    positions would be a guess.
    """
    path = MERGED_ROOT / "provenance" / (seq_id.replace("/", "_") + ".json")
    if not path.is_file():
        return {}
    doc = json.loads(path.read_text())
    return {a["new_track_id"]: RECOVERED for a in doc.get("added", [])
            if "new_track_id" in a}


@lru_cache(maxsize=8)
def _filled_frames(seq_id: str) -> set[tuple[int, int]]:
    """``(track id, frame id)`` pairs the hole filler produced, if it ran.

    Needs the merge sidecar to translate the filler's det-XML track keys into
    the MOT track ids the merged file uses.
    """
    fill_dir = Path(os.environ.get("SPACE_TRACKER_FILL",
                                   "/work/anon/experiments/satmtb_hole_fill"))
    path = fill_dir / (seq_id.replace("/", "_") + ".json")
    prov = MERGED_ROOT / "provenance" / (seq_id.replace("/", "_") + ".json")
    if not (path.is_file() and prov.is_file()):
        return set()
    by_det = {a["det_key"]: a["new_track_id"]
              for a in json.loads(prov.read_text()).get("added", [])
              if "new_track_id" in a}
    out = set()
    for det_key, frames in json.loads(path.read_text()).get("refined", {}).items():
        tid = by_det.get(det_key)
        if tid is not None:
            out.update((tid, int(f)) for f in frames)
    return out


@lru_cache(maxsize=8)
def load_frames(seq_id: str) -> dict[int, list[Obj]]:
    """Completed ground truth of one sequence as ``{frame id: [Obj, ...]}``.

    Merged where a merged file exists, raw otherwise. Human corrections are
    *not* applied here — they are a session-lifetime layer that
    :func:`frame_objects` adds, so this stays cacheable.
    """
    seq = sequence_by_id(seq_id)
    root = MERGED_ROOT if merged_available(seq_id) else MOT_ROOTS[seq.dataset]
    prov = _provenance(seq_id) if root is MERGED_ROOT else {}
    filled = _filled_frames(seq_id) if root is MERGED_ROOT else set()

    out: dict[int, list[Obj]] = {}
    for fid, objs in _parse_gt(seq, root).items():
        out[fid] = [Obj(o.track_id, o.category, o.bbox_xyxy.astype(np.float64),
                        FILLED if (o.track_id, fid) in filled
                        else prov.get(o.track_id, ORIGINAL))
                    for o in objs]
    return out


def frame_objects(seq_id: str, frame_id: int, decisions=None) -> list[Obj]:
    """Objects on one frame with human corrections and additions applied."""
    objs = [Obj(o.track_id, o.category, o.box.copy(), o.provenance)
            for o in load_frames(seq_id).get(frame_id, [])]
    if decisions is None:
        return objs

    fixes = decisions.boxes(seq_id)
    for o in objs:
        box = fixes.get(o.key, {}).get(frame_id)
        if box is not None:
            o.box = np.asarray(box, float)
            o.provenance = REVIEWED
    for key, frames in decisions.drawn(seq_id).items():
        box = frames.get(frame_id)
        if box is None:
            continue
        category, tid = key.split(":", 1)
        objs.append(Obj(int(tid), category, np.asarray(box, float), DRAWN))
    return objs


@lru_cache(maxsize=4)
def raw_geometry(seq_id: str) -> dict[str, dict[int, list[float]]]:
    """The boxes as they were *before* the SAM 3 pass, keyed like :attr:`Obj.key`.

    Exists so the two annotation standards can be compared in place. Whether
    SAM 3's tighter boxes are an improvement or a regression is a judgement
    about the imagery, and it cannot be made from either version alone —
    measured over SAT-MTB the refined/original area ratio is 0.96 for ship but
    0.66 for airplane, so the answer is plausibly different per category.

    Empty for a sequence with no merged file: there, what is loaded *is* raw.
    """
    if not merged_available(seq_id):
        return {}
    from .tracks import load_det_tracks

    seq = sequence_by_id(seq_id)
    out: dict[str, dict[int, list[float]]] = {}
    for fid, objs in _parse_gt(seq, MOT_ROOTS[seq.dataset]).items():
        for o in objs:
            out.setdefault(f"{o.category}:{o.track_id}", {})[fid] = \
                [float(v) for v in o.bbox_xyxy]

    path = MERGED_ROOT / "provenance" / (seq_id.replace("/", "_") + ".json")
    if path.is_file():
        det = load_det_tracks(seq_id, seq.video_id)
        for a in json.loads(path.read_text()).get("added", []):
            track = det.get((a["category"], a["det_object_id"]))
            if track is None or "new_track_id" not in a:
                continue
            key = f"{a['category']}:{a['new_track_id']}"
            out[key] = {int(f): [float(v) for v in b]
                        for f, b in zip(track.frame_ids, track.boxes)}
    return out


def area_ratio(seq_id: str, key: str, decisions=None) -> float | None:
    """Median area of a track's current boxes over its pre-SAM 3 boxes."""
    raw = raw_geometry(seq_id).get(key)
    if not raw:
        return None
    fixes = decisions.boxes(seq_id).get(key, {}) if decisions else {}
    ratios = []
    for fid, objs in load_frames(seq_id).items():
        for o in objs:
            if o.key != key or fid not in raw:
                continue
            b = fixes.get(fid, o.box)
            r = raw[fid]
            denom = max((r[2] - r[0]) * (r[3] - r[1]), 1e-9)
            ratios.append(max((b[2] - b[0]) * (b[3] - b[1]), 0.0) / denom)
    return float(np.median(ratios)) if ratios else None


def track_frames(seq_id: str, key: str) -> list[int]:
    """Frames one track appears on, ascending."""
    return sorted(fid for fid, objs in load_frames(seq_id).items()
                  if any(o.key == key for o in objs))


def sequence_summary(seq_id: str) -> dict:
    """Counts a reviewer needs before opening a sequence."""
    frames = load_frames(seq_id)
    per_provenance: dict[str, int] = {}
    tracks: dict[str, str] = {}
    sizes = []
    for objs in frames.values():
        for o in objs:
            per_provenance[o.provenance] = per_provenance.get(o.provenance, 0) + 1
            tracks[o.key] = o.provenance
            sizes.append(o.size)
    return {
        "boxes": sum(len(v) for v in frames.values()),
        "tracks": len(tracks),
        "tracks_recovered": sum(1 for v in tracks.values() if v == RECOVERED),
        "by_provenance": per_provenance,
        "median_size": float(np.median(sizes)) if sizes else 0.0,
        "merged": merged_available(seq_id),
    }
