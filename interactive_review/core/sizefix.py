"""Boxes that cannot be annotation, whatever they were meant to be.

Two shapes of broken box survive into the ground truth, and neither is a
judgement about whether a box is *tight* — that question needs the imagery.
These are the ones that are invalid on their own terms.

*degenerate*
    Zero area. ``rscardata/train/023`` ships 45 of them, ``train/027`` 41 —
    present in the dataset's own MOT file, before any review touched it. Written
    out they become ``w=0.00,h=0.00`` rows, which every consumer either crashes
    on or silently drops.

*implausible*
    Present but impossible: an area many times the track's own median, on an
    object whose real size is fixed. ``sdmcar/train/50-1`` carries ``ship:9021``
    at a 13.5 px median with one frame at 1213 px, where a SAM 3 mask took the
    whole harbour.

**Why these are dropped and not resized.** Resizing was tried and measured. For
each corrected frame the track's own uncorrected frames were made into an
appearance template, and the box before and after was scored against it across
1,477 comparable frames: 47.8% improved, 43.2% got worse, mean NCC +0.030. It is
a coin flip, and the samples show why — a box that has leaked over the whole
frame has a *centre* in the middle of the frame, not on the object, so shrinking
it about that centre lands on water. A degenerate box has no centre at all.

Dropping is the conservative move, and for ground truth it is strictly the
better one. An unannotated frame costs a detector nothing: the object is simply
not scored there. A wrongly annotated frame costs it twice — a false positive
where the box is, and a miss where the object is. The same reasoning is already
why :mod:`.sizecheck` deletes rather than repairs.

Applied at the export boundary rather than written into the review document:
these are defects in the sources, not decisions anyone made, and the dataset
files stay untouched while the count travels in the manifest.
"""

from __future__ import annotations

import numpy as np

#: Below this sqrt(area) a box has no extent worth writing down.
DEGENERATE_PX = 0.5
#: Multiple of a track's own median sqrt(area) beyond which a box is taken to be
#: a failed mask rather than the object. 4x in sqrt-area is 16x in area — far
#: outside anything a rigid object viewed from orbit does, and outside the
#: aspect swing of a ship under turn.
IMPLAUSIBLE_RATIO = 4.0
#: A track needs this many sound boxes before its median means anything.
MIN_SOUND = 8


def sqrt_area(box) -> float:
    return float(np.sqrt(max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0)))


def invalid_frames(track: dict[int, list[float]]) -> dict[int, str]:
    """``{frame: reason}`` for boxes of one track that cannot stand.

    ``track`` is ``{frame id: xyxy}`` as the export will write it. A track with
    too few sound boxes is only checked for degeneracy — there is no reliable
    scale to call anything implausible against.
    """
    out: dict[int, str] = {}
    sizes = {f: sqrt_area(b) for f, b in track.items()}
    sound = [s for s in sizes.values() if s > DEGENERATE_PX]
    for f, s in sizes.items():
        if s <= DEGENERATE_PX:
            out[f] = "degenerate"
    if len(sound) < MIN_SOUND:
        return out
    med = float(np.median(sound))
    if med <= 0:
        return out
    for f, s in sizes.items():
        if f not in out and s / med >= IMPLAUSIBLE_RATIO:
            out[f] = "implausible"
    return out
