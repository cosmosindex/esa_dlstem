"""The work list: every space-tracker MOT sequence, once.

The unit is a video. 491 sequences across 5 datasets go past the reviewer in a
fixed order, each carrying the one thing that determines what is asked of them:

``check``
    A completed ground truth exists — SAT-MTB after ``merge_det_to_mot.py``, and
    AIR-MOT, which is the only dataset that already annotates static objects.
    The question is whether the boxes are right.
``annotate``
    The ground truth is movers-only and there is no second source to recover
    from. Only VISO's 9 non-car sequences: static objects have to be drawn, with
    SAM 3 doing the drawing.
``view_only``
    Pure car sequences. Static cars are 4.9-6.5 px and not separable from road
    texture in one frame, so nothing is added and no geometry is edited — the
    sequence is watched and signed off so that "every video was looked at" is
    true rather than nearly true.

Sequences that mix car with plane/ship are ``check``: their plane/ship becomes
all-object while car stays movers-only, which is why results are reported per
category and never as one number.
"""

from __future__ import annotations

from dataclasses import dataclass

from .gtsource import merged_available
from .paths import manifest

CHECK, ANNOTATE, VIEW_ONLY = "check", "annotate", "view_only"

MODE_QUESTION = {
    CHECK: "**Are the boxes right?** Every object that should be here is here — "
           "step the frames, scan the grid, fix what is wrong.",
    ANNOTATE: "**What is missing?** This dataset annotates only moving objects "
              "and has no second source. Draw the static ones with SAM 3.",
    VIEW_ONLY: "**Watch and sign off.** Car only: static cars are below what any "
               "single-frame annotation can resolve, so nothing here is edited.",
}

#: Datasets whose ground truth already covers static objects.
ALL_OBJECT_DATASETS = ("airmot",)


@dataclass(frozen=True)
class Item:
    """One sequence's place in the queue."""

    seq_id: str
    dataset: str
    category: str
    mode: str
    n_frames: int

    @property
    def label(self) -> str:
        return f"{self.seq_id}  ({self.category}, {self.n_frames}f, {self.mode})"


def mode_for(seq) -> str:
    """Which of the three jobs this sequence asks for."""
    cats = set(seq.categories_in_seq or [seq.category])
    if cats <= {"car"}:
        return VIEW_ONLY
    if seq.dataset in ALL_OBJECT_DATASETS or merged_available(seq.id):
        return CHECK
    return ANNOTATE


def build_queue(datasets: tuple[str, ...] | None = None,
                modes: tuple[str, ...] | None = None) -> list[Item]:
    """Every sequence, ordered by dataset then id.

    Deliberately not sorted by "interestingness": the reviewer's place in a
    fixed order is itself state, and a queue that reordered itself as decisions
    accumulated would make "where did I get to" unanswerable.
    """
    items = []
    for seq in manifest().sequences:
        if datasets and seq.dataset not in datasets:
            continue
        mode = mode_for(seq)
        if modes and mode not in modes:
            continue
        items.append(Item(seq.id, seq.dataset, seq.category, mode, seq.n_frames))
    return sorted(items, key=lambda i: (i.dataset, i.seq_id))


def next_undone(items: list[Item], decisions, after: int = -1) -> int:
    """Index of the first sequence past ``after`` with no status yet.

    Wraps once, so finishing the tail sends the reviewer back to whatever was
    skipped rather than to a dead end.
    """
    n = len(items)
    for step in range(1, n + 1):
        i = (after + step) % n
        if decisions.status_of(items[i].seq_id) is None:
            return i
    return max(after, 0)


def progress(items: list[Item], decisions) -> dict:
    """Counts for the header — per mode, so a car sweep cannot flatter the rest."""
    out: dict[str, dict[str, int]] = {}
    for it in items:
        row = out.setdefault(it.mode, {"total": 0, "accepted": 0, "flagged": 0,
                                       "edited": 0})
        row["total"] += 1
        status = decisions.status_of(it.seq_id)
        if status in ("accepted", "flagged"):
            row[status] += 1
        if decisions.is_touched(it.seq_id):
            row["edited"] += 1
    return out
