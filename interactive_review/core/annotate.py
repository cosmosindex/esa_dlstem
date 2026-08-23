"""Interactive annotation of objects that no source annotation contains.

The review modes both start from boxes that already exist. This one does not:
a human points at something SAM 3 has never been told about, SAM 3 segments it,
the mask's tight AABB becomes the box, and that box is propagated through the
sequence to become a track.

It covers what the detection XML cannot. Detection XML exists for 139 of 237
SAT-MTB sequences and never labels ``car``; nothing at all covers the other
datasets. Where there is no source annotation to review, this is the only way
to add one.

Prompting follows SAM's own interaction model:

positive click
    "this pixel is the object" — usually one click is enough for a vessel or an
    aircraft against uniform ground
negative click
    "this pixel is not the object" — pushes the mask off a wake, a shadow, or a
    neighbouring object it leaked into
box
    two clicks marking opposite corners, when a click lands ambiguously
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .paths import sequence_by_id
from .render import _read_frame
from .sam3refine import get_tracker, sam3_autocast


@dataclass
class Draft:
    """Prompts and results for one object being annotated, before it is saved."""

    seq_id: str
    category: str = "airplane"
    anchor_frame: int = 0
    #: Positive/negative clicks as (x, y) in full-resolution pixels.
    points: list[tuple[float, float]] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    #: Corner clicks pending a second click, then a finished box.
    box_corners: list[tuple[float, float]] = field(default_factory=list)
    box: list[float] | None = None
    #: Result of segmenting the anchor frame.
    seed_box: list[float] | None = None
    seed_score: float = 0.0
    #: Result of propagating: {frame_id: xyxy}.
    boxes: dict[int, list[float]] = field(default_factory=dict)

    def add_point(self, x: float, y: float, positive: bool = True) -> None:
        self.points.append((float(x), float(y)))
        self.labels.append(1 if positive else 0)

    def add_corner(self, x: float, y: float) -> bool:
        """Collect a box corner. Returns True once a box is complete."""
        self.box_corners.append((float(x), float(y)))
        if len(self.box_corners) < 2:
            return False
        (x0, y0), (x1, y1) = self.box_corners[-2], self.box_corners[-1]
        self.box = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
        self.box_corners = []
        return True

    def undo(self) -> None:
        """Remove the most recent prompt, whatever kind it was."""
        if self.box_corners:
            self.box_corners.pop()
        elif self.box is not None:
            self.box = None
        elif self.points:
            self.points.pop()
            self.labels.pop()

    def clear_prompts(self) -> None:
        self.points.clear()
        self.labels.clear()
        self.box_corners.clear()
        self.box = None

    def has_prompt(self) -> bool:
        return bool(self.points) or self.box is not None

    def summary(self) -> str:
        bits = []
        pos = sum(1 for l in self.labels if l == 1)
        neg = len(self.labels) - pos
        if pos:
            bits.append(f"{pos} positive click{'s' * (pos > 1)}")
        if neg:
            bits.append(f"{neg} negative click{'s' * (neg > 1)}")
        if self.box:
            bits.append("1 box")
        if self.box_corners:
            bits.append("waiting for the opposite corner")
        return ", ".join(bits) if bits else "no prompts yet"


def _prompt_predictor(sam, frame_idx: int, draft: Draft, W: int, H: int,
                      obj_id: int = 0) -> None:
    """Feed a draft's clicks and box to SAM 3 for one frame.

    Coordinates are normalised here because the predictor takes relative
    coordinates; passing pixels through silently produces an empty mask.
    """
    points = labels = None
    if draft.points:
        points = torch.tensor(
            [[x / W, y / H] for x, y in draft.points], dtype=torch.float32)
        labels = torch.tensor(draft.labels, dtype=torch.int32)
    box = None
    if draft.box is not None:
        x1, y1, x2, y2 = draft.box
        box = torch.tensor([x1 / W, y1 / H, x2 / W, y2 / H], dtype=torch.float32)

    sam.predictor.add_new_points_or_box(
        inference_state=sam._inference_state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        points=points,
        labels=labels,
        box=box,
        rel_coordinates=True,
        normalize_coords=False,
        clear_old_points=True,
    )
    if sam._min_prompt_frame is None or frame_idx < sam._min_prompt_frame:
        sam._min_prompt_frame = frame_idx


def segment_frame(draft: Draft) -> tuple[list[float] | None, float, np.ndarray | None]:
    """Segment the anchor frame from the draft's prompts.

    Returns ``(tight AABB, object score, boolean mask)``. The mask is returned
    alongside the box because a box alone cannot tell a reviewer *why* it is
    wrong — whether the mask leaked into a wake or missed a wing decides
    whether the fix is a negative click or a different prompt entirely.
    """
    if not draft.has_prompt():
        return None, 0.0, None
    frame = _read_frame(draft.seq_id, draft.anchor_frame)
    if frame is None:
        return None, 0.0, None
    H, W = frame.shape[:2]

    sam = get_tracker()
    mask, score = None, 0.0
    with sam3_autocast():
        sam.init_video([frame])
        _prompt_predictor(sam, 0, draft, W, H)

        # Step the predictor directly rather than via propagate(), which
        # discards masks and returns boxes only.
        try:
            for frame_idx, _obj_ids, _low, video_res_masks, obj_scores in \
                    sam.predictor.propagate_in_video(
                        sam._inference_state, start_frame_idx=0,
                        max_frame_num_to_track=0, reverse=False,
                        propagate_preflight=True, tqdm_disable=True):
                if frame_idx != 0 or not len(video_res_masks):
                    continue
                mask = (video_res_masks[0, 0] > 0.0).cpu().numpy()
                score = float(torch.sigmoid(obj_scores.float().cpu()).flatten()[0])
                break
        finally:
            sam.reset_state()

    if mask is None or not mask.any():
        return None, 0.0, None
    ys, xs = np.nonzero(mask)
    box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
    if box[2] - box[0] < 1 or box[3] - box[1] < 1:
        return None, 0.0, None
    return box, score, mask


# --- propagation ------------------------------------------------------------

#: Frames per propagation chunk. Every chunk boundary re-prompts SAM 3 with its
#: own last box, so a long sequence is anchored many times instead of once.
CHUNK = 48
#: Frames the next chunk reaches back over, so an object whose box went missing
#: right at the boundary can still be re-seeded from a few frames earlier.
OVERLAP = 8
#: Ground-relative displacement over one chunk, below which an object is not
#: moving any more, and above which it was. A tracker that latches onto the
#: background produces a box that follows the *camera* — screen-static and
#: ground-static — while the object drives out of it, and re-prompting from that
#: box welds the mistake in place for the rest of the video. Measured on
#: satmtb/airplane/40, where a car ran 31, 36, 25 px per 40 frames and then
#: 5, 4, 4, 4, 4 once the tracker had lost it.
FROZEN_PX = 6.0
MOVING_PX = 15.0
#: Object score below which a re-seed is not worth making. Measured on
#: satmtb/airplane/06: while the tracker still holds a car the score sits at
#: 0.84-0.998, and the frame where it genuinely loses one is not a low score but
#: an absent box. So this is a backstop, not the main test.
RESEED_MIN_SCORE = 0.5
#: How far a re-seed box may depart, in area, from the object's own running
#: median before it is resized to fit. It is *not* rejected: on a 10 px car the
#: mask breathes — frame 48 of that sequence came back at 0.24x area with a
#: score of 0.84, and the old code went on to track the car another 40 frames.
#: The centre of such a box is right and its extent is noise, so the centre is
#: kept and the extent replaced. Rejecting instead ended the track at 48.
RESEED_SIZE_LO, RESEED_SIZE_HI = 0.5, 2.0


def _area(b) -> float:
    return max((b[2] - b[0]) * (b[3] - b[1]), 1e-9)


def _stabilise(box, ref_area: float):
    """Keep a re-seed box's centre, pull its size back towards ``ref_area``."""
    r = _area(box) / max(ref_area, 1e-9)
    if RESEED_SIZE_LO <= r <= RESEED_SIZE_HI:
        return list(box)
    k = float(np.sqrt(min(max(r, RESEED_SIZE_LO), RESEED_SIZE_HI) / r))
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    hw, hh = (box[2] - box[0]) / 2 * k, (box[3] - box[1]) / 2 * k
    return [cx - hw, cy - hh, cx + hw, cy + hh]


def _read_window(seq_id: str, fids: list[int]):
    frames, kept = [], []
    for fid in fids:
        img = _read_frame(seq_id, fid)
        if img is None:
            continue
        frames.append(img)
        kept.append(fid)
    return frames, kept


def _run_chunk(sam, frames, prompts: dict[int, tuple[int, list[float]]],
               prompt_fn=None):
    """Propagate one chunk.

    ``prompts`` maps obj_id -> (local frame index, xyxy box). ``prompt_fn(sam,
    local_idx)`` replaces the box prompt for obj_id 0, which is how a draft's
    clicks seed the first chunk. Returns ``{obj_id: {local_idx: (box, score)}}``.
    """
    with sam3_autocast():
        sam.init_video(frames)
        if prompt_fn is not None:
            local, _ = prompts[0]
            prompt_fn(sam, local)
        else:
            # Grouped by frame: add_prompts takes one frame at a time, and
            # objects re-seeded from different frames are the normal case once
            # they start dropping out at different points.
            by_frame: dict[int, list[tuple[int, list[float]]]] = {}
            for oid, (local, box) in prompts.items():
                by_frame.setdefault(local, []).append((oid, box))
            for local, items in sorted(by_frame.items()):
                ids = [oid for oid, _ in items]
                sam.add_prompts(local,
                                np.asarray([b for _, b in items], np.float32),
                                labels=np.zeros(len(ids), np.int64), obj_ids=ids)
        outs = sam.propagate()
        sam.reset_state()

    got: dict[int, dict[int, tuple[list[float], float]]] = {}
    for i, o in enumerate(outs):
        if o is None or not len(o["boxes"]):
            continue
        for b, oid, sc in zip(o["boxes"].numpy(), o["track_ids"].tolist(),
                              o["scores"].tolist()):
            box = [float(v) for v in b]
            if box[2] - box[0] < 1 or box[3] - box[1] < 1:
                continue
            got.setdefault(int(oid), {})[i] = (box, float(sc))
    return got


def _ground_step(seq_id: str, f0: int, f1: int, box0, box1) -> float | None:
    """How far a box moved between two frames, minus what the ground did."""
    from .motion import _local_shift

    c0 = np.array([(box0[0] + box0[2]) / 2, (box0[1] + box0[3]) / 2])
    c1 = np.array([(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2])
    half = max(6.0 * max(box0[2] - box0[0], box0[3] - box0[1]), 48.0)
    shift = _local_shift(seq_id, f0, f1, c0[0], c0[1], half)
    if shift is None:
        return None
    return float(np.linalg.norm((c1 - c0) - shift))


def propagate_boxes(seq_id: str, anchor_frame: int,
                    seeds: dict[int, list[float]], *, prompt_fn=None,
                    chunk: int = CHUNK, min_score: float = 0.0,
                    frozen: dict | None = None, backward: bool = True,
                    progress=None) -> dict[int, dict[int, list[float]]]:
    """Track boxes across the whole sequence, re-prompting every ``chunk`` frames.

    A single propagation loses a 10 px car partway: the memory bank fills with
    frames where the object is a handful of pixels, the mask comes back empty,
    and nothing re-establishes it — which is why hand-drawn tracks on
    ``satmtb/airplane/06`` stop at frames 77, 88 and 103 in a 178-frame video.
    Cutting the run into chunks and re-prompting each one with the previous
    chunk's own last box gives the tracker a fresh anchor every ``chunk``
    frames instead of once.

    Re-seeding from a prediction can equally well entrench a drift, so a seed
    whose area jumped is dropped and that object's track simply ends there. A
    short true track is worth more than a long one that wandered onto a roof.

    The window is the whole sequence — the old 200-frame cap silently truncated
    every video longer than that (``satmtb/airplane/14`` is 243 frames, and its
    drawn track stops dead at 200). Memory stays bounded because only one chunk
    is resident at a time.

    ``backward`` also runs from the anchor towards frame 1. Right for a sweep,
    whose anchor frame is wherever the sweep happened to be run and says nothing
    about when the object entered the video. Wrong for a reviewer's draft: they
    anchor on the frame where they *found* the object, so frames before it are
    the ones where it is not visible, or not there at all, and propagating into
    them invents boxes on whatever the tracker latches onto instead.

    ``frozen``, if given, is filled with ``{obj_id: frame_id}`` for each track
    cut short because it stopped moving over the ground.

    Returns ``{obj_id: {frame_id: xyxy}}``.
    """
    seq = sequence_by_id(seq_id)
    base = seq.frame_index_base
    all_fids = list(range(base, base + seq.n_frames))
    if anchor_frame not in all_fids or not seeds:
        return {}
    a = all_fids.index(anchor_frame)

    sam = get_tracker()
    tracks: dict[int, dict[int, list[float]]] = {oid: {} for oid in seeds}
    #: Per-object ground displacement per chunk, and where each track was cut.
    moved: dict[int, list[float]] = {}
    frozen_at: dict[int, int] = {}
    n_chunks_est = max(1, 2 * len(all_fids) / max(chunk - OVERLAP, 1))
    done = 0

    for reverse in ((True, False) if backward else (False,)):
        live = {oid: (anchor_frame, list(box)) for oid, box in seeds.items()}
        first = True
        while live:
            edge = min(f for f, _ in live.values()) if reverse \
                else max(f for f, _ in live.values())
            i = all_fids.index(edge)
            sl = all_fids[max(0, i - chunk + 1):i + 1] if reverse \
                else all_fids[i:i + chunk]
            frames, kept = _read_window(seq_id, sl)
            prompts = {oid: (kept.index(f), box) for oid, (f, box) in live.items()
                       if f in kept}
            if not frames or not prompts:
                break

            if progress is not None:
                progress(min(done / n_chunks_est, 0.98),
                         f"SAM 3 {'backward' if reverse else 'forward'} "
                         f"from frame {edge} ({len(prompts)} object"
                         f"{'s' * (len(prompts) > 1)})")
            got = _run_chunk(sam, frames, prompts,
                             prompt_fn=prompt_fn if (first and prompt_fn) else None)
            done += 1
            first = False

            nxt: dict[int, tuple[int, list[float]]] = {}
            for oid, (pfid, _pbox) in live.items():
                per = got.get(oid, {})
                # Did this object move over the *ground* during this chunk? One
                # that was moving and has stopped dead is either parked — and
                # `car` is movers-only here — or lost to the background, and in
                # both cases ending the track beats emitting boxes for it.
                froze = False
                if len(per) >= 2:
                    lo_li, hi_li = min(per), max(per)
                    ga = _ground_step(seq_id, kept[lo_li], kept[hi_li],
                                      per[lo_li][0], per[hi_li][0])
                    if ga is not None:
                        hist = moved.setdefault(oid, [])
                        if (hist and ga < FROZEN_PX
                                and float(np.median(hist)) > MOVING_PX):
                            froze = True
                        else:
                            hist.append(ga)
                if froze:
                    # This chunk's boxes go too: they are the frozen ones, and a
                    # wrong box in ground truth costs more than a missing one.
                    frozen_at.setdefault(oid, kept[min(per)])
                    continue
                for li, (box, _sc) in per.items():
                    tracks.setdefault(oid, {})[kept[li]] = box
                # Re-seed from the box furthest along in this chunk's direction.
                order = range(len(kept)) if reverse else range(len(kept) - 1, -1, -1)
                pick = next((li for li in order if li in per), None)
                if pick is None:
                    continue                       # no box anywhere — object is gone
                fid_new, (box, sc) = kept[pick], per[pick]
                if (fid_new >= pfid) if reverse else (fid_new <= pfid):
                    continue                       # no progress — this object is done
                if sc < max(min_score, RESEED_MIN_SCORE):
                    continue
                seen = list(tracks.get(oid, {}).values())
                ref = float(np.median([_area(b) for b in seen])) if seen else _area(box)
                nxt[oid] = (fid_new, _stabilise(box, ref))
            live = nxt

    if progress is not None:
        progress(1.0, "done")
    if frozen is not None:
        frozen.update(frozen_at)
    return {oid: fb for oid, fb in tracks.items() if fb}


def propagate_draft(draft: Draft, max_frames: int | None = None,
                    progress=None, frozen: dict | None = None,
                    backward: bool = False) -> dict[int, list[float]]:
    """Track the drafted object across the sequence from its anchor frame.

    ``max_frames`` is accepted for compatibility and ignored: the propagation is
    chunked now, so covering the whole video costs no more memory than covering
    200 frames of it did.

    The track starts at the frame the reviewer segmented and runs forward. That
    frame is a statement — *this is where I can see the object* — and treating
    it as the middle of a track rather than its start writes boxes over every
    frame before it, on an object the reviewer never looked at. ``backward``
    turns the reverse pass back on for the case where the object really was
    there earlier and simply unannotated.

    ``frozen``, if given, receives ``{0: frame}`` when the run was cut short
    because the object stopped moving over the ground. That is a normal outcome
    and a recoverable one — re-anchoring on a frame where the object is moving
    usually covers the whole sequence — but only if the reviewer is told it
    happened, so the caller is given the frame rather than a short track and no
    explanation.
    """
    if not draft.has_prompt():
        return {}
    seed = draft.seed_box
    if seed is None:
        # Segment was skipped; the clicks alone still seed the first chunk, and
        # a placeholder area only has to be plausible enough for the re-seed
        # guard on the *second* chunk.
        pts = np.asarray(draft.points, np.float32) if draft.points else None
        if draft.box is not None:
            seed = list(draft.box)
        elif pts is not None:
            cx, cy = pts[-1]
            seed = [cx - 8, cy - 8, cx + 8, cy + 8]
        else:
            return {}

    W = H = None
    frame = _read_frame(draft.seq_id, draft.anchor_frame)
    if frame is None:
        return {}
    H, W = frame.shape[:2]

    out = propagate_boxes(
        draft.seq_id, draft.anchor_frame, {0: seed},
        prompt_fn=lambda sam, local: _prompt_predictor(sam, local, draft, W, H),
        progress=progress, frozen=frozen, backward=backward)
    return out.get(0, {})


def save_draft(draft: Draft, decisions) -> str | None:
    """Store a propagated draft as a new hand-drawn track. Returns its key.

    The track goes straight into the sequence's review document — there is no
    lower layer for it to fall back to, which is exactly what distinguishes an
    annotated object from a corrected one.
    """
    if not draft.boxes:
        return None
    key = decisions.new_drawn_key(draft.seq_id, draft.category)
    decisions.set_drawn(draft.seq_id, key, draft.boxes)
    decisions.save()
    return key
