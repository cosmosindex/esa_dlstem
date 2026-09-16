"""Fold a hand-drawn track into the annotated track it duplicates.

A reviewer draws an object because the grid showed nothing on it. On a merged
SAT-MTB sequence that can be true of the frames they are looking at and false of
the sequence: ``satmtb/airplane/16`` carries ``airplane:4`` over frames 61-293
and nothing over 1-60, so an aircraft that is plainly unannotated at frame 1 is
the *same* aircraft that is annotated from frame 61. Drawing it produces two
track ids for one object, which is precisely the error MOT metrics punish
hardest — an identity switch written into the ground truth itself.

Merging is possible because the drawn layer is keyed by ``category:track_id``
and :func:`~.gtsource.frame_objects` parses that key back into an id. Re-keying
a drawn track to an existing track's id makes the two halves one identity, and
no frame may carry that id twice — so the frames the target already covers
cannot simply be added. What happens to them depends on whether the two boxes
also agree on the object's *extent*:

*they agree*
    The drawn box is a redundant copy and is dropped. Provenance survives — the
    frames that were genuinely added stay marked as drawn.

*they disagree*
    Two boxes can share a centre and still differ in area by a factor of thirty:
    ``satmtb/car/18`` annotates a ship with a 10x8 box on a hull that is 80x36,
    and a reviewer who draws the hull is *correcting* that geometry, not
    duplicating it. Dropping the drawn box there discards the only box anyone
    looked at the imagery to produce — and if the target already covers every
    frame, discards the whole track, silently. So beyond
    :data:`MERGE_AREA_RATIO` the drawn geometry wins: it lands in the reviewed
    layer over a shipped frame, or overwrites a drawn one, and shows as
    corrected-by-hand rather than as annotation nobody has checked.

The identity still merges either way. What changes is only which of the two
boxes survives on the frames both cover.

The test is deliberately strict, but it cannot be IoU. Two aircraft on
neighbouring stands overlap by a few pixels for a few frames; the same aircraft
seen twice agrees almost exactly, over many frames — and IoU says so, at 50 px.
At 5 px it does not: on a 5.5 px car a **one pixel** offset in each axis takes
IoU to 0.50, so an IoU threshold that means "a different object" for aircraft
means "drawn by a different hand" for cars. Measured on
``rscardata/train/011``, 20 drawn/dataset pairs that are plainly one car each
scored IoU 0.30-0.57 and every one of them was rejected.

So agreement is the distance between the two centres **in object widths**, which
is the same test at every scale. On that sequence it separates cleanly: the
pairs that are one car sit at 0.14-0.98 widths, and the next 745 pairs start at
6.8. A drawn track that matches nothing is left exactly as it is.

Every live identity is a candidate to merge *into*, not only the ones the
dataset ships. Two drawn tracks land on one object as easily as a drawn track
lands on an annotated one — an exemplar sweep run twice, or one run that split
an object at an occlusion — and testing only against the dataset misses exactly
those: on ``rscardata/train/011`` two drawn tracks sat 0.08 widths apart over 33
frames and a dataset-only test had nothing to compare them with.

Only a purely drawn track is ever dissolved. Where two identities the dataset
itself ships turn out to be one object, that is a claim about the dataset's own
annotation rather than about this reviewer's work, so it is reported by
:func:`shipped_conflicts` and left alone.

Merging changes what matches next: folding a track into another extends it, and
the extension can then reach a third. So :func:`merge_duplicates` re-runs until
a pass finds nothing, rather than trusting one pass over stale geometry.

One object under two ids arrives in two shapes, and agreement over shared
frames only recognises the first:

*drawn twice*
    Two tracks running side by side over the same frames. They overlap in time,
    so the frames they share are the evidence.

*drawn in two halves*
    A propagation dropped the object partway and the reviewer re-prompted from
    the next frame. The two halves share **no** frames at all — that is what a
    break means — so there is nothing for agreement to measure, and the whole
    test is inapplicable rather than merely strict. On ``sdmcar/train/18-1``
    ``ship:9008`` runs to frame 94 and ``ship:9016`` picks up at 95, 1.4 px
    away, and no shared-frame threshold however loose could ever join them.

The second is handled by :func:`find_continuations`, which tests the *seam*
instead: how far apart the two halves are where one ends and the other begins,
after allowing for the object's own motion across the gap.
"""

from __future__ import annotations

import logging

import numpy as np

from .gtsource import load_frames

log = logging.getLogger(__name__)

#: Median centre distance, as a fraction of the box width, below which two
#: tracks are one object — plus :data:`MERGE_SLACK_PX`, because at 5 px the
#: annotation grid itself is a meaningful fraction of the object.
MERGE_CENTRE_WIDTHS = 0.6
MERGE_SLACK_PX = 1.0
#: Reported alongside, never decisive. See the module docstring for why an IoU
#: threshold cannot be used on objects this small.
MERGE_IOU = 0.6
#: Shared frames required before the median means anything.
MERGE_MIN_SHARED = 5

#: Ratio between two boxes' areas beyond which they are no longer two copies of
#: one annotation but two claims about how big the object is. Deliberately loose
#: — a reviewer redrawing a box they agree with lands within a factor of two,
#: and the case this exists for is off by thirty. Anything under it is treated
#: as a redundant copy and dropped, exactly as before.
MERGE_AREA_RATIO = 3.0

#: Frames a track may be missing before a later one stops being its
#: continuation. A reviewer re-prompts within a frame or two of the break; a
#: longer hole is a gap in the annotation, not a seam, and joining across it
#: would assert boxes for frames nobody looked at.
CONTINUE_MAX_GAP = 5
#: Seam distance in object widths at a one-frame gap, and how much to allow per
#: further frame — the object's velocity has to be extrapolated across the hole,
#: and that estimate decays.
CONTINUE_WIDTHS = 1.0
CONTINUE_WIDTHS_PER_FRAME = 0.25
#: Frames at each end used to estimate velocity. Enough to average out the
#: jitter of a 5 px box, short enough to still be the local direction.
CONTINUE_VELOCITY_FRAMES = 8


def _iou_pairwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, 0], b[:, 0]); y1 = np.maximum(a[:, 1], b[:, 1])
    x2 = np.minimum(a[:, 2], b[:, 2]); y2 = np.minimum(a[:, 3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa + ab - inter, 1e-9)


def _agreement(a: dict, b: dict) -> tuple[float, float, int]:
    """How well two tracks agree over the frames they share.

    Returns ``(centre distance in widths, mean IoU, shared frames)``. The first
    decides; the second is carried so the reviewer sees the number they would
    have judged by. ``inf`` when the tracks barely overlap in time.
    """
    shared = sorted(set(a) & set(b))
    if len(shared) < MERGE_MIN_SHARED:
        return float("inf"), 0.0, len(shared)
    A = np.asarray([a[f] for f in shared], np.float32)
    B = np.asarray([b[f] for f in shared], np.float32)
    ca = np.stack([(A[:, 0] + A[:, 2]) / 2, (A[:, 1] + A[:, 3]) / 2], 1)
    cb = np.stack([(B[:, 0] + B[:, 2]) / 2, (B[:, 1] + B[:, 3]) / 2], 1)
    wa = np.sqrt(np.clip(A[:, 2] - A[:, 0], 0, None) * np.clip(A[:, 3] - A[:, 1], 0, None))
    wb = np.sqrt(np.clip(B[:, 2] - B[:, 0], 0, None) * np.clip(B[:, 3] - B[:, 1], 0, None))
    width = np.maximum(np.maximum(wa, wb), 1e-6)
    dist = np.linalg.norm(ca - cb, axis=1) / width
    return (float(np.median(dist)),
            float(np.mean(_iou_pairwise(A, B))), len(shared))


def _tolerance(a: dict, b: dict, shared) -> float:
    """Match threshold in widths, loosened by :data:`MERGE_SLACK_PX` of grid."""
    sizes = [max(_sqrt_area(a[f]), _sqrt_area(b[f])) for f in shared]
    w = float(np.median(sizes)) if sizes else 0.0
    return MERGE_CENTRE_WIDTHS + (MERGE_SLACK_PX / w if w > 0 else 0.0)


def _sqrt_area(box) -> float:
    x1, y1, x2, y2 = (float(v) for v in box)
    return float(np.sqrt(max(x2 - x1, 0.0) * max(y2 - y1, 0.0)))


def _extent_disagrees(a, b) -> bool:
    """Whether two boxes on one object disagree about how much of it is object.

    Area, not IoU: at 5 px a one-pixel offset halves IoU, which is why nothing
    here decides identity by it. Area is not offset-sensitive, so it still means
    something at that scale — and "how big" is precisely the question.
    """
    sa, sb = max(_sqrt_area(a), 1e-6), max(_sqrt_area(b), 1e-6)
    return (max(sa, sb) / min(sa, sb)) ** 2 > MERGE_AREA_RATIO


def source_tracks(seq_id: str, decisions=None) -> dict[str, dict[int, list[float]]]:
    """Every track the dataset ships, with per-frame fixes applied.

    The drawn layer is excluded on purpose: these are the frames a merge must
    never move or overwrite, and the ones that decide whether an identity may be
    dissolved at all.
    """
    fixes = decisions.boxes(seq_id) if decisions is not None else {}
    out: dict[str, dict[int, list[float]]] = {}
    for fid, objs in load_frames(seq_id).items():
        for o in objs:
            box = fixes.get(o.key, {}).get(fid)
            out.setdefault(o.key, {})[fid] = \
                [float(v) for v in (box if box is not None else o.box)]
    return out


def live_tracks(seq_id: str, decisions) -> dict[str, dict[int, list[float]]]:
    """Every identity as it currently reads, both layers, minus the deleted.

    This is what the reviewer sees and what the export writes, so it is what
    duplicate detection has to run on. A previous merge leaves drawn frames
    under a dataset key, and those frames are as much a part of that identity as
    the shipped ones — comparing against the shipped half alone would miss a
    third track that duplicates only the extension.
    """
    deleted = decisions.deleted(seq_id)
    out = {k: dict(v) for k, v in source_tracks(seq_id, decisions).items()
           if k not in deleted}
    for key, frames in decisions.drawn(seq_id).items():
        if key in deleted or not frames:
            continue
        into = out.setdefault(key, {})
        for f, box in frames.items():
            into.setdefault(int(f), [float(v) for v in box])
    return out


def _rank(key: str, src: dict, tracks: dict) -> tuple:
    """Which of two identities is the authority on an object. Higher wins.

    An identity the dataset ships always outranks a drawn one — it is the one
    the rest of the world already refers to, and it is the only one whose frames
    this tool must not move.

    Within the shipped tier the count is of *shipped* frames, not of everything
    the identity now carries. Counting both layers would let a drawn extension
    vote for the track it was drawn onto: on ``rscardata/train/011`` ``car:27``
    is 105 annotated frames plus a 137-frame extension that runs onto
    ``car:22``'s car, and by total length it would outrank the 189 annotated
    frames it is contaminating. Authority comes from what the dataset says, not
    from what this tool drew on top of it.

    Between two drawn tracks the longer survives, so a merge adds the few frames
    to the many rather than the other way round, and the id breaks the tie so a
    re-run reaches the same answer.
    """
    return (key in src,
            len(src[key]) if key in src else len(tracks.get(key, ())),
            -int(key.split(":")[1]))


def shipped_conflicts(seq_id: str, decisions) -> list[dict]:
    """Pairs of *dataset* identities that read as one object. Changes nothing.

    Not merged automatically, and not by this tool at all: dissolving one would
    rewrite an id the dataset ships and every other consumer of it refers to.
    Reported so a reviewer can look at the imagery and decide.
    """
    live = live_tracks(seq_id, decisions)
    shipped = set(source_tracks(seq_id, decisions)) - decisions.deleted(seq_id)
    keys = sorted(shipped, key=lambda k: (k.split(":")[0], int(k.split(":")[1])))
    out = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if a.split(":", 1)[0] != b.split(":", 1)[0]:
                continue
            shared = set(live[a]) & set(live[b])
            d, iou, n = _agreement(live[a], live[b])
            if d > _tolerance(live[a], live[b], shared):
                continue
            out.append({"a": a, "b": b, "dist": d, "iou": iou, "shared": n,
                        "frames": (min(shared), max(shared)) if shared else None})
    return sorted(out, key=lambda p: p["dist"])


def _centre(box) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], float)


def _velocity(track: dict, at_end: bool,
              n: int = CONTINUE_VELOCITY_FRAMES) -> np.ndarray:
    """Per-frame velocity at one end of a track, as a vector.

    Averaged over the last (or first) ``n`` frames and divided by the frames
    actually spanned, so a hole inside the window does not read as a stall.
    """
    fids = sorted(track)
    if len(fids) < 2:
        return np.zeros(2)
    seg = fids[-n:] if at_end else fids[:n]
    span = seg[-1] - seg[0]
    if span <= 0:
        return np.zeros(2)
    return (_centre(track[seg[-1]]) - _centre(track[seg[0]])) / span


def _seam(a: dict, b: dict) -> tuple[float, int, float]:
    """How well ``b`` continues ``a``: ``(distance in widths, gap, px)``.

    ``a`` is extrapolated forward across the gap at its own closing velocity and
    compared with where ``b`` actually starts. Extrapolating matters even at a
    one-frame gap once the object is quick: predicting nothing would charge a
    car its whole per-frame travel as error, and on the small boxes here that is
    a meaningful fraction of a width.

    ``inf`` when ``b`` does not begin after ``a`` ends, or the hole is too long.
    """
    fa, fb = sorted(a), sorted(b)
    if not fa or not fb:
        return float("inf"), 0, 0.0
    gap = fb[0] - fa[-1]
    if gap < 1 or gap > CONTINUE_MAX_GAP:
        return float("inf"), gap, 0.0
    predicted = _centre(a[fa[-1]]) + _velocity(a, at_end=True) * gap
    px = float(np.linalg.norm(_centre(b[fb[0]]) - predicted))
    w = max(_sqrt_area(a[fa[-1]]), _sqrt_area(b[fb[0]]), 1e-6)
    return px / w, gap, px


def find_continuations(seq_id: str, decisions) -> list[dict]:
    """Drawn tracks that resume an earlier track rather than duplicate it.

    Returns plans in the same shape as :func:`find_merges`, so
    :func:`apply_merges` carries them out unchanged — the mechanics of folding
    one key into another are identical, only the evidence differs.

    Each track may continue at most one other, and is chosen by the closest
    seam, so a re-prompt in dense traffic joins the object it actually left off
    rather than whichever neighbour happens to qualify.
    """
    live = live_tracks(seq_id, decisions)
    src = source_tracks(seq_id, decisions)
    drawn = decisions.drawn(seq_id)
    deleted = decisions.deleted(seq_id)
    plans = []
    for key in sorted(drawn, key=lambda k: (k.split(":")[0], int(k.split(":")[1]))):
        if key in deleted or key in src or not drawn[key]:
            continue
        category = key.split(":", 1)[0]
        mine = live[key]
        best, best_d, best_gap, best_px = None, float("inf"), 0, 0.0
        for other, otk in live.items():
            if other == key or other.split(":", 1)[0] != category:
                continue
            if _rank(other, src, live) <= _rank(key, src, live):
                continue
            d, gap, px = _seam(otk, mine)
            tol = CONTINUE_WIDTHS + CONTINUE_WIDTHS_PER_FRAME * (gap - 1)
            if d > tol or d >= best_d:
                continue
            best, best_d, best_gap, best_px = other, d, gap, px
        if best is None:
            continue
        extra = sorted(set(mine) - set(live[best]))
        span = list(live[best]) + extra
        plans.append({
            "drawn": key, "into": best, "dist": best_d,
            "iou": 0.0, "shared": 0, "adds": extra,
            "into_shipped": best in src, "gap": best_gap, "seam_px": best_px,
            "span_before": (min(live[best]), max(live[best])),
            "span_after": (min(span), max(span)),
        })
    return sorted(plans, key=lambda p: p["dist"])


def find_merges(seq_id: str, decisions) -> list[dict]:
    """Which drawn tracks duplicate which live identity. Changes nothing.

    Returns one record per drawn track that matched, carrying the numbers the
    decision rests on so a reviewer can disagree with it before it is applied.
    A track is only ever dissolved into one that outranks it — see :func:`_rank`
    — so the pass cannot fold two tracks into each other.
    """
    live = live_tracks(seq_id, decisions)
    src = source_tracks(seq_id, decisions)
    drawn = decisions.drawn(seq_id)
    deleted = decisions.deleted(seq_id)
    plans = []
    for key in sorted(drawn, key=lambda k: (k.split(":")[0], int(k.split(":")[1]))):
        # Only a purely drawn identity may be dissolved. Once a key is one the
        # dataset ships, its frames are not this tool's to move.
        if key in deleted or key in src or not drawn[key]:
            continue
        category = key.split(":", 1)[0]
        mine = live[key]
        best, best_d, best_iou, best_shared = None, float("inf"), 0.0, 0
        for skey, stk in live.items():
            if skey == key or skey.split(":", 1)[0] != category:
                continue
            if _rank(skey, src, live) <= _rank(key, src, live):
                continue
            d, iou, shared = _agreement(mine, stk)
            if d >= best_d:
                continue
            if d > _tolerance(mine, stk, set(mine) & set(stk)):
                continue
            best, best_d, best_iou, best_shared = skey, d, iou, shared
        if best is None:
            continue
        extra = sorted(set(mine) - set(live[best]))
        span = list(live[best]) + extra
        plans.append({
            "drawn": key, "into": best, "dist": best_d,
            "iou": best_iou, "shared": best_shared,
            "adds": extra,
            "into_shipped": best in src,
            "span_before": (min(live[best]), max(live[best])),
            "span_after": (min(span), max(span)),
        })
    # Closest first, so that when two drawn tracks fold into one target the
    # better-matching geometry is the one that lands — see `apply_merges`.
    return sorted(plans, key=lambda p: p["dist"])


def apply_merges(seq_id: str, decisions, plans) -> list[str]:
    """Carry out ``plans``. Returns a line per merge, for the reviewer to read."""
    src = source_tracks(seq_id, decisions)
    live = live_tracks(seq_id, decisions)
    drawn = decisions.drawn(seq_id)
    # A target dissolved earlier in this same pass no longer exists to merge
    # into. Rather than chase the chain over geometry that has already moved,
    # the plan is dropped and the next pass — which sees the merged result —
    # decides on it. `merge_duplicates` runs those passes to convergence.
    dissolved = set()
    notes = []
    for p in plans:
        key, into = p["drawn"], p["into"]
        if key in dissolved or into in dissolved:
            continue
        # Only the frames the target does not already cover, in either layer.
        # Keeping the overlap would put the same id on one frame twice, which
        # every consumer of the export would read as two objects.
        merged = dict(decisions.drawn(seq_id).get(into) or {})
        held = set(src.get(into, ())) | set(merged)
        extra = {f: b for f, b in drawn[key].items() if f not in held}

        # The frames both cover. A copy of a box already there is redundant and
        # goes; a box that disagrees about the object's extent is the reviewer
        # correcting geometry nobody had looked at, and it is the box that
        # survives. Without this the overlap is dropped wholesale — and where
        # the target already spans every frame that silently deletes the entire
        # track the reviewer just drew.
        target = live.get(into, {})
        shipped_frames = set(src.get(into, ()))
        corrected = 0
        for f, box in drawn[key].items():
            if f not in held:
                continue
            current = target.get(f)
            if current is None or not _extent_disagrees(box, current):
                continue
            if f in shipped_frames:
                # Over a shipped frame it belongs in the reviewed layer, which is
                # where a human overruling the batch geometry is meant to live —
                # it renders as corrected-by-hand and reverts frame by frame.
                decisions.patch_box(seq_id, into, f, box)
            else:
                merged[f] = box
            corrected += 1

        # Nothing to add and nothing to correct: the whole net effect of this
        # merge is to delete what the reviewer just drew. Doing that outright
        # leaves them looking at a canvas where their track simply is not, with
        # no way to tell a merge from a crash and nothing to restore — so the
        # track is hidden instead of erased. It stays out of the export exactly
        # as an erased one would, and it is one click from coming back.
        if not extra and not corrected:
            decisions.set_deleted(seq_id, key, True)
            dissolved.add(key)
            note = (f"{key} is already annotated as {into} on every frame it "
                    f"covers, at the same geometry — hidden rather than kept, "
                    f"since two ids on one object is the error this exists to "
                    f"prevent. Restore it from `Delete / restore this track` if "
                    f"that call is wrong.")
            log.info("merge: %s | %s", seq_id, note)
            notes.append(note)
            continue

        merged.update(extra)
        decisions.set_drawn(seq_id, key, None)
        dissolved.add(key)
        if merged:
            decisions.set_drawn(seq_id, into, merged)
        lo, hi = p["span_after"]
        if p.get("gap"):
            why = (f"resumes it after a {p['gap']}-frame break, "
                   f"{p['seam_px']:.1f} px = {p['dist']:.2f} widths off where it "
                   f"was heading")
        else:
            why = (f"centres {p['dist']:.2f} widths apart, IoU {p['iou']:.2f}, "
                   f"over {p['shared']} frames")
        note = (f"{key} → {into} ({why}): "
                f"{len(extra)} frames added, {into} now spans {lo}..{hi}")
        if corrected:
            note += (f", and on {corrected} frames the drawn box replaced "
                     f"{into}'s own — they disagreed about the object's extent "
                     f"by more than {MERGE_AREA_RATIO:.0f}x in area")
        log.info("merge: %s | %s", seq_id, note)
        notes.append(note)
    if notes:
        decisions.save()
    return notes


#: Consecutive close frames before a coincidence counts as contamination. Two
#: cars genuinely passing each other are close for a frame or two and rarely
#: below the tolerance at all; a propagation that has changed object stays on it
#: for tens of frames. Measured on ``rscardata/train/011``: real passes ran 1-3
#: frames at 0.6-0.9 widths, contamination 22-82 frames at 0.08-0.31.
OVERLAP_MIN_RUN = 5


def _close_runs(a: dict, b: dict) -> list[list[int]]:
    """Maximal runs of consecutive frames on which two tracks coincide.

    Consecutive in frame number, not merely in the shared list: a gap in one
    track is a gap in the evidence, and stitching across it would report one
    long run where the imagery only supports two short ones.
    """
    close = []
    for f in sorted(set(a) & set(b)):
        ba, bb = a[f], b[f]
        w = max(_sqrt_area(ba), _sqrt_area(bb), 1e-6)
        d = float(np.hypot((ba[0] + ba[2] - bb[0] - bb[2]) / 2,
                           (ba[1] + ba[3] - bb[1] - bb[3]) / 2)) / w
        if d <= MERGE_CENTRE_WIDTHS + MERGE_SLACK_PX / w:
            close.append(f)

    runs, cur = [], []
    for f in close:
        if cur and f != cur[-1] + 1:
            if len(cur) >= OVERLAP_MIN_RUN:
                runs.append(cur)
            cur = []
        cur.append(f)
    if len(cur) >= OVERLAP_MIN_RUN:
        runs.append(cur)
    return runs


def find_overlaps(seq_id: str, decisions) -> list[dict]:
    """Drawn frames sitting on an object another identity already carries.

    Distinct from a duplicate, and not fixable by merging. A duplicate agrees
    with its twin over the whole life of both — ``car:18`` and ``car:30`` on
    ``rscardata/train/011`` coincide on 82 of the 83 frames they share, so they
    are one object under two ids. Contamination agrees over a *stretch*: on the
    same sequence ``car:9011`` runs 30 px from ``car:13`` for its first hundred
    frames and then lands on it from frame 116 to the end. Merging those two
    would claim the first hundred frames were the same car as well.

    So the coinciding frames are removed from the lower-ranked identity instead,
    and only where they run long enough not to be two objects passing. What is
    left of the track is the part that still tracks its own object.
    """
    live = live_tracks(seq_id, decisions)
    src = source_tracks(seq_id, decisions)
    drawn = decisions.drawn(seq_id)
    deleted = decisions.deleted(seq_id)
    out = []
    for key in sorted(drawn, key=lambda k: (k.split(":")[0], int(k.split(":")[1]))):
        if key in deleted or not drawn[key]:
            continue
        category = key.split(":", 1)[0]
        # Only the drawn frames are ever dropped, so a key the dataset ships is
        # still a candidate — its drawn extension can be contaminated while its
        # shipped frames are not.
        mine = {int(f): b for f, b in drawn[key].items()}
        for other, otk in live.items():
            if other == key or other.split(":", 1)[0] != category:
                continue
            if _rank(other, src, live) <= _rank(key, src, live):
                continue
            for run in _close_runs(mine, otk):
                out.append({"key": key, "against": other, "frames": run,
                            "dist": _agreement({f: mine[f] for f in run},
                                               {f: otk[f] for f in run})[0]})
    return sorted(out, key=lambda r: (-len(r["frames"]), r["key"]))


def strike_overlaps(seq_id: str, decisions) -> list[str]:
    """Apply :func:`find_overlaps`. Returns a line per stretch removed.

    A track left with nothing is dropped entirely; one left with a handful of
    frames is kept, because a short honest track is still ground truth whereas a
    long contaminated one is not.
    """
    plans = find_overlaps(seq_id, decisions)
    if not plans:
        return []
    drop: dict[str, set[int]] = {}
    notes = []
    for r in plans:
        drop.setdefault(r["key"], set()).update(r["frames"])
        notes.append(f"{r['key']}: {len(r['frames'])} frames "
                     f"{r['frames'][0]}..{r['frames'][-1]} removed — sitting "
                     f"{r['dist']:.2f} widths from {r['against']}, which "
                     f"already carries that object")
    for key, frames in drop.items():
        kept = {f: b for f, b in decisions.drawn(seq_id)[key].items()
                if int(f) not in frames}
        decisions.set_drawn(seq_id, key, kept or None)
        if not kept:
            notes.append(f"{key}: nothing left of it, dropped")
    decisions.save()
    return notes


#: Passes before :func:`merge_duplicates` gives up. Each pass strictly reduces
#: the number of drawn identities, so this only ever bounds a bug.
MERGE_MAX_PASSES = 10


def merge_duplicates(seq_id: str, decisions) -> list[str]:
    """Find and apply, until a pass finds nothing. A line per merge.

    One pass is not enough: a merge extends the track it lands in, and the
    extension can then reach a third track that shared too few frames to be
    judged before. Each pass is decided on the geometry the previous one
    produced, never on stale boxes.
    """
    notes = []
    for _pass in range(MERGE_MAX_PASSES):
        # Duplicates first: they are decided on many shared frames, so they are
        # the better-evidenced of the two, and folding them shortens the list of
        # ends a continuation could attach to.
        done = apply_merges(seq_id, decisions, find_merges(seq_id, decisions))
        done += apply_merges(seq_id, decisions,
                             find_continuations(seq_id, decisions))
        if not done:
            break
        notes.extend(done)
    return notes
