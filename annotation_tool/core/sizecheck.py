"""Boxes a hand-drawn track has no business carrying, by size alone.

SAM 3 segments what is under the prompt. When the prompt lands a pixel off a
5 px car — onto its shadow, the kerb, the truck beside it — the mask leaks into
the surrounding structure and the box comes back tens of pixels wide. The track
survives, because the next frame re-anchors on the object; what is left is a
handful of frames carrying a box that is not the object, and occasionally a
whole track that never found one.

Both are worse in ground truth than a gap would be. A missing frame costs a
tracker one undetected object; a 148 px box on a 5 px car matches nothing, so it
counts as a miss *and* a false positive, and any size-conditioned result reads
it as a large object that every method fails on. So the oversized frames are
dropped rather than resized: a box invented to replace one that was wrong is
still not the object, and it would be indistinguishable from an annotation.

Two scales, because they are different failures:

*the frame*
    A box more than :data:`OVERSIZE_FRAME` times its own track's median. Within
    one track the object's size is nearly constant — a car does not quadruple in
    area between frames — so this is measured against the track itself and needs
    no outside reference.

*the track*
    A track whose median exceeds :data:`OVERSIZE_TRACK` times the size of the
    boxes the dataset itself annotates. Here the track's own median is the thing
    that is wrong, so the reference has to come from outside it. Marked deleted,
    not erased: "this is too big to be a car" is a judgement about the imagery
    and the reviewer may disagree, whereas a single frame 20x its neighbours is
    not a judgement call.
"""

from __future__ import annotations

import numpy as np

from .gtsource import load_frames

#: A single box, as a multiple of its own track's median size.
OVERSIZE_FRAME = 2.0
#: A whole track's median, as a multiple of the median box the dataset annotates.
OVERSIZE_TRACK = 2.0
#: Below this many frames a track is not worth keeping once the bad frames go.
MIN_FRAMES = 2


def _sqrt_area(box) -> float:
    x1, y1, x2, y2 = (float(v) for v in box)
    return float(np.sqrt(max(x2 - x1, 0.0) * max(y2 - y1, 0.0)))


def dataset_median_px(seq_id: str) -> float:
    """Median ``sqrt(w * h)`` of the boxes the dataset ships for this sequence.

    The reference for "too big to be one of these objects". Taken from the
    dataset rather than from the drawn layer because the drawn layer is what is
    on trial — on a sequence where half the sweep leaked, its own median is
    inflated by exactly what this is meant to catch.
    """
    sizes = [_sqrt_area(o.box) for objs in load_frames(seq_id).values() for o in objs]
    return float(np.median(sizes)) if sizes else 0.0


def find_oversize(seq_id: str, decisions,
                  frame_mult: float = OVERSIZE_FRAME,
                  track_mult: float = OVERSIZE_TRACK) -> dict:
    """What the size rules would remove. Changes nothing.

    Returns ``{"ref_px", "tracks": [...], "frames": [...]}`` where ``tracks``
    are ``(key, median_px, ratio)`` and ``frames`` are
    ``(key, [frame ids], [sizes], track_median_px)``. A track flagged whole is
    not also listed for its frames — deleting it settles them.
    """
    ref = dataset_median_px(seq_id)
    drawn = decisions.drawn(seq_id)
    deleted = decisions.deleted(seq_id)

    tracks, frames = [], []
    for key in sorted(drawn, key=lambda k: (k.split(":")[0], int(k.split(":")[1]))):
        if key in deleted:
            continue
        fb = drawn[key]
        fids = sorted(fb)
        sizes = np.array([_sqrt_area(fb[f]) for f in fids])
        if not len(sizes):
            continue
        med = float(np.median(sizes))
        if ref > 0 and med > track_mult * ref:
            tracks.append((key, med, med / ref))
            continue
        bad = sizes > frame_mult * max(med, 1e-6)
        if bad.any():
            frames.append((key, [fids[i] for i in np.flatnonzero(bad)],
                           [float(s) for s in sizes[bad]], med))
    return {"ref_px": ref, "tracks": tracks, "frames": frames}


def strike_oversize(seq_id: str, decisions,
                    frame_mult: float = OVERSIZE_FRAME,
                    track_mult: float = OVERSIZE_TRACK) -> dict:
    """Apply :func:`find_oversize`. Saves, and returns what it did.

    ``{"ref_px", "tracks_deleted", "frames_dropped", "tracks_emptied", "detail"}``.
    A track left under :data:`MIN_FRAMES` frames is marked deleted too — what
    remains of it is not a track, and leaving a two-frame stub in the ground
    truth is its own kind of wrong annotation.
    """
    plan = find_oversize(seq_id, decisions, frame_mult, track_mult)
    detail = []

    for key, med, ratio in plan["tracks"]:
        decisions.set_deleted(seq_id, key, True)
        detail.append(f"{key} deleted — median {med:.1f} px is {ratio:.1f}x the "
                      f"{plan['ref_px']:.1f} px this dataset annotates")

    dropped = emptied = 0
    for key, fids, sizes, med in plan["frames"]:
        kept = {f: b for f, b in decisions.drawn(seq_id)[key].items()
                if f not in set(fids)}
        dropped += len(fids)
        if len(kept) < MIN_FRAMES:
            decisions.set_deleted(seq_id, key, True)
            emptied += 1
            detail.append(f"{key} deleted — only {len(kept)} frames survive the "
                          f"size check")
            continue
        decisions.set_drawn(seq_id, key, kept)
        detail.append(f"{key}: dropped {len(fids)} frames up to "
                      f"{max(sizes):.0f} px against a {med:.1f} px track "
                      f"(frames {fids[0]}..{fids[-1]})")

    if plan["tracks"] or plan["frames"]:
        decisions.save()
    return {"ref_px": plan["ref_px"], "tracks_deleted": len(plan["tracks"]),
            "frames_dropped": dropped, "tracks_emptied": emptied, "detail": detail}
