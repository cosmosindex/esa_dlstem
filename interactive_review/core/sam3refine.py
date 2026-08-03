"""SAM 3 box refinement and propagation for the review tool.

Two operations, both driven by boxes that already exist:

``per_frame``
    Each frame is run as an independent one-frame video: the existing box is a
    prompt, SAM 3 returns a mask, and the mask's tight AABB replaces the box.
    Isolating frames this way prevents cross-object contamination and makes the
    result order-independent. This is the same procedure that re-annotated
    165k BIRDSAI boxes.

``propagate``
    One prompted frame, then SAM 3 tracks the object through the clip. Used to
    fill holes — 14 of the 545 static-review tracks have gaps inside their own
    span, one of them 324 frames wide — and to label objects that have no
    detection XML at all.

Refinements are never adopted blindly. A refined box is rejected, and the
original kept, when the mask is empty, drifts off the object, or changes area
implausibly. On BIRDSAI these guards fired on 0 of 165,042 boxes, so a guard
trip here is a signal worth looking at, not routine noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .paths import frame_path, sequence_by_id
from .render import _read_frame
from .tracks import Track

# Guard thresholds, matching the BIRDSAI re-annotation.
IOU_MIN = 0.1
GROW_MAX = 4.0
SHRINK_MIN = 0.05

_tracker = None


def get_tracker():
    """Build SAM 3 once, on first use — importing this module must not take a GPU."""
    global _tracker
    if _tracker is None:
        from models.sam3 import SAM3Tracker
        _tracker = SAM3Tracker()
    return _tracker


def unload_tracker() -> None:
    global _tracker
    _tracker = None
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:                                       # noqa: BLE001
        pass


@dataclass
class RefineResult:
    """Refined boxes for one track, plus why each frame ended up as it did."""

    boxes: dict[int, list[float]] = field(default_factory=dict)      # fid -> xyxy
    accepted: dict[int, bool] = field(default_factory=dict)
    reasons: dict[int, str] = field(default_factory=dict)
    stats: dict[str, int] = field(default_factory=dict)
    mode: str = "per_frame"

    @property
    def n_accepted(self) -> int:
        return sum(1 for v in self.accepted.values() if v)

    def area_ratio(self, track: Track) -> float:
        """Median area of refined boxes over original, accepted frames only."""
        ratios = []
        for fid, ok in self.accepted.items():
            if not ok:
                continue
            orig = track.box_at(fid)
            new = self.boxes.get(fid)
            if orig is None or new is None:
                continue
            ao = max((orig[2] - orig[0]) * (orig[3] - orig[1]), 1e-9)
            ratios.append(((new[2] - new[0]) * (new[3] - new[1])) / ao)
        return float(np.median(ratios)) if ratios else 0.0

    def summary(self, track: Track) -> str:
        s = self.stats
        kept = " ".join(f"{k[5:]} {v}" for k, v in sorted(s.items())
                        if k.startswith("kept_") and v)
        return (f"SAM 3 `{self.mode}`: **{self.n_accepted}/{s.get('boxes', 0)}** frames "
                f"refined, median area {self.area_ratio(track):.2f}× original"
                + (f" &nbsp;·&nbsp; kept original: {kept}" if kept else ""))


def _iou(a, b) -> float:
    lt = np.maximum(a[:2], b[:2])
    rb = np.minimum(a[2:], b[2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[0] * wh[1]
    area = lambda x: max((x[2] - x[0]) * (x[3] - x[1]), 0.0)
    return float(inter / max(area(a) + area(b) - inter, 1e-9))


def _accept(orig, refined) -> tuple[bool, str]:
    if refined is None:
        return False, "empty"
    rw, rh = refined[2] - refined[0], refined[3] - refined[1]
    if rw < 1 or rh < 1:
        return False, "empty"
    ao = max((orig[2] - orig[0]) * (orig[3] - orig[1]), 1e-9)
    ar = rw * rh
    if _iou(np.asarray(orig, float), np.asarray(refined, float)) < IOU_MIN:
        return False, "iou"
    if ar > GROW_MAX * ao:
        return False, "grow"
    if ar < SHRINK_MIN * ao:
        return False, "shrink"
    return True, ""


def refine_track(track: Track, progress: Callable[[float, str], None] | None = None,
                 max_frames: int | None = None) -> RefineResult:
    """Tighten every box of ``track`` against its own frame.

    ``max_frames`` subsamples evenly when set — useful for a quick preview on a
    300-frame track before committing to the full pass.
    """
    sam = get_tracker()
    res = RefineResult(mode="per_frame")
    stats = {"boxes": 0, "refined": 0, "kept_empty": 0, "kept_iou": 0,
             "kept_grow": 0, "kept_shrink": 0, "no_image": 0}

    fids = track.frame_ids
    if max_frames and len(fids) > max_frames:
        idx = np.linspace(0, len(fids) - 1, max_frames).round().astype(int)
        fids = [fids[i] for i in dict.fromkeys(idx.tolist())]

    for n, fid in enumerate(fids):
        if progress is not None:
            progress(n / max(len(fids), 1), f"SAM 3 frame {n + 1}/{len(fids)}")
        frame = _read_frame(track.key.seq_id, fid)
        orig = track.box_at(fid)
        if frame is None or orig is None:
            stats["no_image"] += 1
            continue
        H, W = frame.shape[:2]
        box = [float(np.clip(orig[0], 0, W - 1)), float(np.clip(orig[1], 0, H - 1)),
               float(np.clip(orig[2], 1, W)), float(np.clip(orig[3], 1, H))]
        if box[2] - box[0] < 1 or box[3] - box[1] < 1:
            stats["kept_empty"] += 1
            continue

        stats["boxes"] += 1
        sam.init_video([frame])
        sam.add_prompts(0, np.asarray([box], np.float32),
                        labels=np.zeros(1, np.int64), obj_ids=[0])
        outs = sam.propagate()
        sam.reset_state()

        refined = None
        if outs and len(outs[0]["boxes"]):
            refined = [float(v) for v in outs[0]["boxes"].numpy()[0]]

        ok, reason = _accept(box, refined)
        res.boxes[fid] = refined if ok else box
        res.accepted[fid] = ok
        res.reasons[fid] = reason
        stats["refined" if ok else "kept_" + reason] += 1

    res.stats = stats
    return res


def propagate_track(track: Track, prompt_frame: int | None = None,
                    progress: Callable[[float, str], None] | None = None,
                    max_frames: int = 200) -> RefineResult:
    """Prompt one frame, let SAM 3 track the object across the sequence.

    Fills frames the track does not cover. ``max_frames`` bounds the clip that
    gets decoded and written to disk — full sequences at 1600x1200 are large
    and SAM 3 holds the whole clip in its inference state.
    """
    sam = get_tracker()
    seq = sequence_by_id(track.key.seq_id)
    res = RefineResult(mode="propagate")

    anchor = prompt_frame if prompt_frame is not None else track.frame_ids[len(track) // 2]
    anchor_box = track.box_at(anchor)
    if anchor_box is None:
        res.stats = {"boxes": 0}
        return res

    # A contiguous window centred on the anchor, clipped to the sequence.
    base = seq.frame_index_base
    lo = max(base, anchor - max_frames // 2)
    hi = min(base + seq.n_frames - 1, lo + max_frames - 1)
    window = list(range(lo, hi + 1))

    if progress is not None:
        progress(0.0, f"decoding {len(window)} frames")
    frames, kept = [], []
    for fid in window:
        img = _read_frame(track.key.seq_id, fid)
        if img is None:
            continue
        frames.append(img)
        kept.append(fid)
    if not frames or anchor not in kept:
        res.stats = {"boxes": 0}
        return res

    if progress is not None:
        progress(0.3, "SAM 3 propagating")
    sam.init_video(frames)
    sam.add_prompts(kept.index(anchor), np.asarray([anchor_box], np.float32),
                    labels=np.zeros(1, np.int64), obj_ids=[0])
    outs = sam.propagate()
    sam.reset_state()

    stats = {"boxes": 0, "refined": 0, "kept_empty": 0, "kept_iou": 0,
             "kept_grow": 0, "kept_shrink": 0, "new": 0}
    for i, fid in enumerate(kept):
        o = outs[i] if i < len(outs) else None
        if o is None or not len(o["boxes"]):
            continue
        new = [float(v) for v in o["boxes"].numpy()[0]]
        orig = track.box_at(fid)
        if orig is None:
            # A frame the track never covered — nothing to guard against, this
            # is the hole being filled.
            res.boxes[fid] = new
            res.accepted[fid] = True
            res.reasons[fid] = "new"
            stats["new"] += 1
            continue
        stats["boxes"] += 1
        ok, reason = _accept(orig, new)
        res.boxes[fid] = new if ok else [float(v) for v in orig]
        res.accepted[fid] = ok
        res.reasons[fid] = reason
        stats["refined" if ok else "kept_" + reason] += 1

    res.stats = stats
    return res
