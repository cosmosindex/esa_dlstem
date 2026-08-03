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
from .sam3refine import get_tracker


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
    sam.init_video([frame])
    _prompt_predictor(sam, 0, draft, W, H)

    # Step the predictor directly rather than via propagate(), which discards
    # masks and returns boxes only.
    mask, score = None, 0.0
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


def propagate_draft(draft: Draft, max_frames: int = 200,
                    progress=None) -> dict[int, list[float]]:
    """Track the drafted object across the sequence from its anchor frame.

    The prompts are placed on the anchor and SAM 3 runs both forward and
    backward, so an object first noticed mid-sequence still gets its earlier
    frames. ``max_frames`` bounds the decoded window — full sequences at
    1600x1200 are large and the whole clip lives in the inference state.
    """
    if not draft.has_prompt():
        return {}
    seq = sequence_by_id(draft.seq_id)
    base = seq.frame_index_base
    lo = max(base, draft.anchor_frame - max_frames // 2)
    hi = min(base + seq.n_frames - 1, lo + max_frames - 1)
    window = list(range(lo, hi + 1))

    if progress is not None:
        progress(0.0, f"decoding {len(window)} frames")
    frames, kept = [], []
    for fid in window:
        img = _read_frame(draft.seq_id, fid)
        if img is None:
            continue
        frames.append(img)
        kept.append(fid)
    if not frames or draft.anchor_frame not in kept:
        return {}

    H, W = frames[0].shape[:2]
    if progress is not None:
        progress(0.3, f"SAM 3 propagating across {len(frames)} frames")
    sam = get_tracker()
    sam.init_video(frames)
    _prompt_predictor(sam, kept.index(draft.anchor_frame), draft, W, H)
    outs = sam.propagate()
    sam.reset_state()

    boxes: dict[int, list[float]] = {}
    for i, fid in enumerate(kept):
        o = outs[i] if i < len(outs) else None
        if o is None or not len(o["boxes"]):
            continue
        b = [float(v) for v in o["boxes"].numpy()[0]]
        if b[2] - b[0] >= 1 and b[3] - b[1] >= 1:
            boxes[fid] = b
    if progress is not None:
        progress(1.0, "done")
    return boxes


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
