"""The work list: every released MOT sequence, once.

The unit is a video. All 403 sequences of Space-Tracker-MOT go past the
reviewer in a fixed order, each carrying the one thing that determines what is
asked of them.

Scope needs no rules here any more. The queue is built from the released
package, and the release is already what the two rules produced: AIR-MOT is
absent for want of a redistribution licence, and a sequence is in the package
only if at least one of its tracks is small. :data:`UNLICENSED_DATASETS` and
:func:`is_small` survive so a queue can still be built over a source tree that
has not been filtered, and both are no-ops against the release.

The mode is where the ground truth came from, not a permission: annotation is
open everywhere, and both modes end in "add what is missing with SAM 3". What it
changes is what to expect — whether there is completed geometry to scrutinise
first, or a blank movers-only track list to fill.

``check``
    A second source already completed this ground truth: SAT-MTB non-car, whose
    static tracks were restored from its own per-frame detection XML. So the
    boxes come from a pipeline and are worth scrutinising, and the merge
    recovered only what detection could see: what it missed still has to be
    drawn. The release records this per sequence, as
    ``review.merged_from_detection_xml``.
``annotate``
    The ground truth is movers-only and no second source recovered anything:
    VISO's non-car sequences, and every all-car sequence. The merge recovered no
    car track anywhere — detection XML does not resolve cars — so an all-car
    sequence is never "check".
``view_only``
    Off by default, ``car_view_only=True`` to restore. All-car sequences used to
    be watched and signed off but never edited, on the grounds that a parked car
    is 4.9-6.5 px and not separable from road texture in one frame. That is still
    true of parked cars and false of moving ones, which the movers-only ground
    truth also misses plenty of — so the sequences are open for annotation and
    the judgement is made per object rather than per dataset.

Sequences that mix car with plane/ship are ``check``: their plane/ship becomes
all-object while car stays movers-only, which is why results are reported per
category and never as one number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

from .gtsource import merged_available
from .paths import MOT_MANIFEST, SIZE_SPLIT, manifest

CHECK, ANNOTATE, VIEW_ONLY = "check", "annotate", "view_only"

MODE_QUESTION = {
    CHECK: "**Are the boxes right, and what is still missing?** A second source "
           "restored the static tracks here, so the geometry is the first thing "
           "to check — step the frames, scan the grid, fix what is wrong. It "
           "recovered only what detection could see: add the rest with SAM 3.",
    ANNOTATE: "**What is missing?** Movers-only ground truth with no second "
              "source to recover from — draw what it left out with SAM 3. On an "
              "all-car sequence a parked car is 4.9-6.5 px and may not be "
              "resolvable at all: annotate what you can actually see, and leave "
              "the rest.",
    VIEW_ONLY: "**Watch and sign off.** Car only: static cars are below what any "
               "single-frame annotation can resolve, so nothing here is edited.",
}

#: Datasets whose ground truth already covers static objects.
ALL_OBJECT_DATASETS = ("airmot",)

#: Held out of the queue. AIR-MOT is not licensed for redistribution, so nothing
#: annotated on it can ship with the release and reviewing it buys nothing. Its
#: decisions stay in ``review.json`` untouched — this hides the sequences, it
#: does not discard the 69 that were already signed off. Name it in
#: ``datasets`` to get it back.
#: Datasets that may not be redistributed. None of them reaches the release --
#: this is what keeps a queue honest when it is built over an unfiltered source
#: tree instead.
UNLICENSED_DATASETS = ("airmot",)

#: Sequence-level size bucket, read from ``space_tracker/data/size_split.json``:
#: a sequence is small when the median ``sqrt(w * h)`` over its GT boxes is
#: <= 32 px (COCO's small-object threshold). The release applies a *track*-level
#: criterion instead -- it keeps a sequence when any one track is small, and then
#: keeps every track in it -- so this file holds back 36 released sequences whose
#: own median is large. That is why ``small_only`` is off by default: filtering
#: the release by it would hide sequences the benchmark contains.
SMALL = "small"


@lru_cache(maxsize=1)
def _size_buckets() -> dict[str, str]:
    with open(SIZE_SPLIT) as f:
        return {k: v["bucket"] for k, v in json.load(f)["sequences"].items()}


def is_small(seq_id: str) -> bool:
    """Whether ``seq_id`` falls in the small-object half.

    A sequence missing from the split counts as small: that is a stale-file
    problem, and dropping it silently would leave the queue quietly incomplete
    with nothing to notice it by.
    """
    return _size_buckets().get(seq_id, SMALL) == SMALL


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


@lru_cache(maxsize=1)
def _merged_from_det_xml() -> frozenset[str]:
    """Sequences whose static tracks the release recovered from detection XML.

    Recorded by the build, so the mode no longer depends on a scratch directory
    of merge output being present on this machine.
    """
    doc = json.loads(MOT_MANIFEST.read_text())
    return frozenset(
        r.get("source_sequence_id", r["id"]) for r in doc["sequences"]
        if (r.get("review") or {}).get("merged_from_detection_xml"))


def mode_for(seq, *, car_view_only: bool = False) -> str:
    """Which of the three jobs this sequence asks for.

    ``car_view_only`` restores the old behaviour where an all-car sequence was
    watched and signed off but never edited. It is off by default: a static car
    is 4.9-6.5 px and still not annotatable from one frame, but *moving* cars in
    these sequences are, and the movers-only ground truth misses plenty of them
    — so the sequences are opened for annotation and the reviewer decides, per
    object, what is resolvable.
    """
    cats = set(seq.categories_in_seq or [seq.category])
    if cats <= {"car"}:
        # Not routed through `merged_available` below: the merge wrote a file for
        # every SAT-MTB sequence but recovered no car track in any of them —
        # detection XML does not resolve cars — so that test would answer CHECK,
        # "the ground truth is complete", over one that is still movers-only.
        return VIEW_ONLY if car_view_only else ANNOTATE
    if (seq.dataset in ALL_OBJECT_DATASETS
            or seq.id in _merged_from_det_xml()
            or merged_available(seq.id)):
        return CHECK
    return ANNOTATE


def build_queue(datasets: tuple[str, ...] | None = None,
                modes: tuple[str, ...] | None = None,
                *,
                small_only: bool = False,
                exclude_datasets: tuple[str, ...] = UNLICENSED_DATASETS,
                car_view_only: bool = False,
                sequences: tuple[str, ...] | None = None,
                ) -> list[Item]:
    """Every in-scope sequence, ordered by dataset then id.

    Deliberately not sorted by "interestingness": the reviewer's place in a
    fixed order is itself state, and a queue that reordered itself as decisions
    accumulated would make "where did I get to" unanswerable.

    ``small_only`` keeps the small-object half (:func:`is_small`). Off by
    default: the release is already scoped, and this sequence-level median would
    drop 36 sequences it contains.
    ``exclude_datasets`` is overridden for any dataset named in ``datasets``, so
    asking for a held-out dataset by name returns it rather than an empty queue.

    ``sequences`` names sequence ids outright and is the whole queue when given:
    every other filter is bypassed, on the same principle that naming a held-out
    dataset returns it. The size filter is a *sequence*-level median, and a
    sequence whose median is large can still carry small objects — 37 of the 55
    held out that way do — so there has to be a way to ask for those by name
    without reopening the 55.
    """
    if sequences:
        want_ids = set(sequences)
        by_id = {seq.id: seq for seq in manifest().sequences}
        missing = sorted(want_ids - by_id.keys())
        if missing:
            raise SystemExit("no such sequence: " + ", ".join(missing))
        return sorted(
            (Item(seq.id, seq.dataset, seq.category,
                  mode_for(seq, car_view_only=car_view_only), seq.n_frames)
             for sid, seq in by_id.items() if sid in want_ids),
            key=lambda i: (i.dataset, i.seq_id))

    wanted = set(datasets) if datasets else None
    dropped = set(exclude_datasets) - (wanted or set())
    items = []
    for seq in manifest().sequences:
        if wanted is not None and seq.dataset not in wanted:
            continue
        if seq.dataset in dropped:
            continue
        if small_only and not is_small(seq.id):
            continue
        mode = mode_for(seq, car_view_only=car_view_only)
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
