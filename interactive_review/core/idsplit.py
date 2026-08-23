"""Separate two objects that a dataset filed under one track id.

SDM-Car writes some frames twice for one ``(class, track id)`` with *different*
boxes: 2,347 such rows across 49 sequences, the two centres a median of 10 px
and as much as 1,654 px apart. That is not a duplicated row — it is two objects
sharing an identity, which is the one thing a track id exists to prevent. An
association metric scoring against it is being asked to merge two cars.

Keeping the first box and dropping the second, which is what plain
de-duplication does, throws away a real object's annotation on every one of
those frames. So the identity is split instead: the boxes on each frame are
assigned to as many slots as there are objects, slot 0 keeps the dataset's id,
and every further slot gets a fresh one.

**Assignment is by continuity, not by row order.** The order two rows appear in
on a frame carries no meaning — the same object can be written first on one
frame and second on the next — so slots are matched to boxes by nearest centre
to where each slot last was, greedily, closest pair first. A box too far from
every open slot starts its own, since two objects that were one id may also
simply not overlap in time.

New ids start above the sequence's own maximum, so nothing collides with an id
the dataset already uses and an unsplit track keeps the id it always had.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

#: How far a box may sit from a slot's last known position, in units of its own
#: size, and still be that slot continuing. Generous: the objects here are a few
#: pixels and the gap between the two identities is a median of 10 px, so the
#: test only has to beat "this is obviously the other one".
CONTINUITY_SIZES = 6.0
#: A slot unseen for this many frames is closed; a box arriving later starts a
#: new one rather than teleporting an old identity across the gap.
SLOT_MAX_GAP = 12


def _centre(b) -> np.ndarray:
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], float)


def _size(b) -> float:
    return max(float(np.sqrt(max(b[2] - b[0], 0) * max(b[3] - b[1], 0))), 1.0)


def split_shared_id(frames: dict[int, list[list[float]]]) -> list[dict[int, list[float]]]:
    """``{frame: [box, ...]}`` for one id → one ``{frame: box}`` per object.

    The first returned track is the one that keeps the dataset's id: it is the
    slot opened first, which is the one present on the earliest frame.
    """
    slots: list[dict[int, list[float]]] = []
    last: list[tuple[int, np.ndarray, float]] = []   # (frame, centre, size)

    for fid in sorted(frames):
        boxes = frames[fid]
        open_slots = [i for i, (f, _, _) in enumerate(last) if fid - f <= SLOT_MAX_GAP]
        pairs = []
        for bi, b in enumerate(boxes):
            c, s = _centre(b), _size(b)
            for si in open_slots:
                d = float(np.linalg.norm(c - last[si][1]))
                if d <= CONTINUITY_SIZES * max(s, last[si][2]):
                    pairs.append((d, bi, si))
        pairs.sort()
        taken_b: set[int] = set()
        taken_s: set[int] = set()
        for _d, bi, si in pairs:
            if bi in taken_b or si in taken_s:
                continue
            taken_b.add(bi); taken_s.add(si)
            slots[si][fid] = boxes[bi]
            last[si] = (fid, _centre(boxes[bi]), _size(boxes[bi]))
        for bi, b in enumerate(boxes):
            if bi in taken_b:
                continue
            slots.append({fid: b})
            last.append((fid, _centre(b), _size(b)))
    return slots


def split_reused_ids(tracks: dict[tuple, dict[int, list[list[float]]]]
                     ) -> tuple[dict[tuple, dict[int, list[float]]], dict]:
    """Split every identity in ``tracks`` that carries two objects at once.

    ``tracks`` maps ``(category, track_id)`` to ``{frame: [box, ...]}``. Returns
    the flattened one-box-per-frame tracks plus a report of what was split.
    """
    next_id = max((tid for _c, tid in tracks), default=0) + 1
    out: dict[tuple, dict[int, list[float]]] = {}
    report = {"ids_split": 0, "tracks_created": 0, "frames_recovered": 0,
              "detail": []}

    for (cat, tid), frames in sorted(tracks.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if all(len(v) == 1 for v in frames.values()):
            out[(cat, tid)] = {f: v[0] for f, v in frames.items()}
            continue
        parts = split_shared_id(frames)
        if len(parts) <= 1:
            out[(cat, tid)] = {f: v[0] for f, v in frames.items()}
            continue
        extra = sum(len(p) for p in parts) - len(frames)
        report["ids_split"] += 1
        report["tracks_created"] += len(parts) - 1
        report["frames_recovered"] += extra
        out[(cat, tid)] = parts[0]
        for p in parts[1:]:
            out[(cat, next_id)] = p
            next_id += 1
        report["detail"].append(
            f"{cat}:{tid} carried {len(parts)} objects — "
            f"{', '.join(str(len(p)) + 'f' for p in parts)}; "
            f"{len(parts)-1} new id(s), {extra} frames recovered")
    return out, report
