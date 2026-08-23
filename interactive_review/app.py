"""Video-level ground-truth review for every space-tracker MOT sequence.

One video at a time, in a fixed order, 367 of them. For each: look at what the
ground truth says, fix what is wrong, watch the whole thing played back, sign it
off, move on. Nothing here decides *whether* an object belongs in the ground
truth — ``tools/merge_det_to_mot.py`` already did that, unattended, for the 842
static tracks SAT-MTB's MOT files drop.

Three surfaces, because one frame at browser scale cannot answer every question:

* **grid** — one zoomed tile per object on this frame. The inspection surface: a
  wrong box is visible here and invisible in the whole frame, where a 10 px
  aircraft renders as 6 px.
* **frame** — the whole frame. Navigation and "is something missing".
* **playback** — the sequence as a video, which is the only view that shows
  flicker, drift, and duplicate tracks.

Run::

    CUDA_VISIBLE_DEVICES=0 python -m interactive_review.app
    CUDA_VISIBLE_DEVICES=0 python -m interactive_review.app --datasets satmtb
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import gradio as gr
import numpy as np

from .core.annotate import Draft, propagate_draft, save_draft, segment_frame
from .core.gtsource import (MERGED_ROOT, area_ratio, frame_objects,
                            load_frames, raw_geometry, sequence_summary,
                            set_label_overrides, track_frames)
from .core.paths import CATEGORIES, DEFAULT_OVERRIDES, sequence_by_id
from .core.render import _placeholder, annotate_view
from .core.vdecisions import SequenceDecisions
from .core.video import VIDEO_CACHE
from .core import sizecheck
from .core.vqueue import (ANNOTATE, CHECK, MODE_QUESTION, UNLICENSED_DATASETS,
                          VIEW_ONLY, build_queue, next_undone, progress)
from .core.vrender import LEGEND, fix_view, frame_view, grid_view, size_outliers, tile_at

TILE = 150
COLUMNS = 8
#: The proposal wall's own tile size and width. Separate from the grid's,
#: because a proposal crop only has to be judgeable, not clickable to the pixel.
PROP_TILE = 130
PROP_COLUMNS = 8
PROP_MAX_TILES = 96

#: Starting vocabulary, derived from the notes written during the AIR-MOT pass
#: rather than invented — these are the properties a reviewer actually reached
#: for. The dropdown accepts new values, so it grows with use.
TAG_PRESETS = [
    "static",          # every object in the sequence is stationary
    "single_object",   # one target — usable as an SOT sequence
    "fast_motion",
    "occlusion",
    "sudden_entry",    # an object appears abruptly mid-sequence
    "maneuver",        # the target turns
    "tiny",
    "weather",         # haze, smog, cloud
    "low_contrast",
    "crowded",
]

GUIDE_MD = """
**The loop.** Scan the grid → fix what is wrong → `P` play it back → `A` accept → next video.

| key | |
|---|---|
| `←` `→` | previous / next frame (hold `shift` for ±10) |
| `[` `]` | previous / next video |
| `P` | render and play the whole sequence |
| `A` | accept this video and open the next undone one |
| `F` | flag it — something here needs a decision, come back to it |
| `S` | snap the selected box with SAM 3 |
| `U` | undo the correction on the selected box, this frame |

**Fixing a box.** Click its tile in the grid, then click two opposite corners in
the zoom panel and save. One frame is corrected at a time: fixing frame 41 of a
283-frame track leaves the other 282 exactly as they were.

**Annotating many of the same thing.** Draw one by hand, save it, then
`🔍 Find all like this`: SAM 3 uses that track as a visual exemplar, sweeps the
sequence for everything matching it, and shows the proposals in pink. Adjust the
score, discard them, or save them all as tracks in one go. Nothing is written
until you press save, and every track it makes is an ordinary drawn track you
can delete one at a time.

**Judging what the sweep found.** The `proposals` tab is one tile per match,
strongest first, each labelled with the frame it came from. Click a tile to
strike it out — it dims and leaves the canvas, and clicking again puts it back.
`Save all proposals as tracks` only propagates the ones still standing.

**When a track freezes.** SAM 3 can latch onto the background under a small
object: the box stops moving while the object drives out of it. The propagation
now detects that — a track that was moving over the ground and stops dead is cut
there rather than continued — so the tail is missing rather than wrong. To
repair it: select the track, step to the frame it stopped at, `New object here`,
click the real object, `Segment`, `Propagate`, then `Continue selected track
from here`. Frames before that one are left untouched.

**Static vehicles are not `car`.** SAT-MTB annotates cars only when they move,
and the sweep cannot tell a parked one from a driving one. `Delete static
tracks` measures each drawn track against the background right around it and
deletes the ones that never move over the ground. Deletion is reversible, and
a sequence where everything moves is left untouched.

**Duplicates are merged, not left.** A drawn track that agrees with an existing
one over the frames they share is the same object, and saving it would put two
ids on one aircraft. It is folded into the existing track automatically, keeping
whatever frames the existing one was missing — which is the reason it looked
unannotated in the first place.

**Accept is fragile on purpose.** Any edit clears both the sign-off and the
"watched" flag, so an acceptance always refers to the annotation that was
actually played back.
"""


class Session:
    """Everything the UI needs that is not a Gradio component."""

    def __init__(self, args):
        self.decisions = SequenceDecisions(args.overrides, reviewer=args.reviewer)
        self.items = build_queue(sequences=tuple(args.sequences) if args.sequences else None,
                                 datasets=tuple(args.datasets) if args.datasets else None,
                                 modes=tuple(args.modes) if args.modes else None,
                                 small_only=not args.all_sizes,
                                 exclude_datasets=() if args.include_unlicensed
                                                  else UNLICENSED_DATASETS,
                                 car_view_only=args.car_view_only)
        if not self.items:
            raise SystemExit("queue is empty — check --datasets / --modes")
        #: Whether saving a track may also dissolve it. See `_merge_note`.
        self.auto_merge = bool(getattr(args, "auto_merge", True))
        self.index = next_undone(self.items, self.decisions)
        self.frame_id = self.first_frame()
        self.selected: str | None = None
        self.grid_objs: list = []
        self.corners: list[tuple[float, float]] = []
        self.drawn: list[float] | None = None
        self.show_raw = False
        self.draft: Draft | None = None
        self.draft_mask = None
        self.click_mode = "positive"
        #: Exemplar-sweep proposals: [(frame_id, xyxy, score)]. Never saved until
        #: the reviewer adopts them, and dropped whenever the sequence changes —
        #: a proposal is about one video's pixels and means nothing on the next.
        self.candidates: list[tuple[int, list[float], float]] = []
        #: Category of the track the proposals were copied from. Taken from the
        #: exemplar rather than the dropdown: a sweep for cars run with the
        #: dropdown left on "airplane" would file every car as an aircraft, and
        #: the reviewer would have no reason to look.
        self.candidate_category = "car"
        #: The exemplar the proposals were scored against, as (frame_id, box, score).
        self.candidate_exemplar: tuple[int, list[float], float] | None = None
        #: Set by the sweep so `refresh` opens the proposals tab once.
        self.show_proposals = False
        #: Why the last sweep came back empty. Rendered *into* the proposals tab
        #: rather than only into the header, because the header sits at the top
        #: of the page and the button that failed is most of a screen below it.
        self.sweep_note = ""
        #: Where the last sweep's detections went, for the info panel.
        self.sweep_stats: dict = {}
        #: Indices into `candidates` the reviewer struck out on the wall. Kept
        #: as a rejection set rather than by deleting from the list, so the tile
        #: under the cursor never changes identity between two clicks.
        self.candidate_rejected: set[int] = set()
        self.require_watch = not args.no_watch_required
        #: Transient one-line feedback, shown in the header until the next action.
        self.message = ""
        #: Set by :meth:`goto`. The playback tab keeps showing the *previous*
        #: sequence's video otherwise, which is indistinguishable from the
        #: Accept button having done nothing.
        self.switched = False

    def kept_candidates(self) -> list[tuple[int, list[float], float]]:
        """Proposals still standing — the ones a save would actually adopt."""
        return [c for i, c in enumerate(self.candidates)
                if i not in self.candidate_rejected]

    def refuse(self, why: str) -> bool:
        """Record why an action was declined. Returns True, to `return` through."""
        self.message = why
        return True

    def is_view_only(self) -> bool:
        return self.item.mode == VIEW_ONLY

    # -- current position ---------------------------------------------------

    @property
    def item(self):
        return self.items[self.index]

    @property
    def seq_id(self) -> str:
        return self.item.seq_id

    def first_frame(self) -> int:
        seq = sequence_by_id(self.seq_id)
        return seq.frame_index_base

    def last_frame(self) -> int:
        seq = sequence_by_id(self.seq_id)
        return seq.frame_index_base + seq.n_frames - 1

    def goto(self, index: int) -> None:
        self.index = index % len(self.items)
        self.frame_id = self.first_frame()
        self.selected = None
        self.draft = None
        self.draft_mask = None
        self.candidates = []
        self.candidate_exemplar = None
        self.candidate_rejected = set()
        self.switched = True
        self.clear_draft()

    def clear_draft(self) -> None:
        self.corners = []
        self.drawn = None

    def draft_zoom(self, half: int = 70) -> list[float] | None:
        """Window to render the annotation canvas in, centred on the last click.

        Whole-frame is unusable for placing a prompt on a 10 px object; once
        there is a click to centre on, the canvas follows it.
        """
        d = self.draft
        if d is None:
            return None
        pts = list(d.points) + list(d.box_corners)
        if d.box is not None:
            pts.append(((d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2))
        if not pts:
            return None
        cx, cy = pts[-1]
        return [cx - half, cy - half, cx + half, cy + half]

    def exemplar_track(self) -> tuple[str, dict[int, list[float]]] | None:
        """The hand-drawn track the exemplar sweep should copy.

        The selected one when a drawn track is selected, otherwise the most
        recent one on this sequence — a reviewer who has just saved a car and
        reaches for "find the rest" means that car, and making them re-select it
        first is a click that carries no information.

        A draft in progress wins over anything saved. The object the reviewer
        just segmented is on screen in front of them, and requiring them to save
        it first before it can be copied is a rule with no reason behind it —
        the sweep needs pixels and a box, not a decision.

        Deleted tracks are skipped. A track the reviewer threw away is the one
        statement they have made about it — that it is not a valid annotation —
        and copying it is the opposite of what they asked for. Sorting is by id
        rather than by string, so track 9 does not come after track 10.
        """
        d = self.draft
        if d is not None and d.boxes:
            return f"{d.category}:draft", dict(d.boxes)
        if d is not None and d.seed_box is not None:
            # One frame is a thin exemplar, but it is the one the reviewer is
            # pointing at, and it beats refusing.
            return f"{d.category}:draft", {d.anchor_frame: list(d.seed_box)}
        drawn = self.decisions.drawn(self.seq_id)
        deleted = self.decisions.deleted(self.seq_id)
        live = [k for k in drawn if k not in deleted]
        if not live:
            return None
        if self.selected in live:
            return self.selected, drawn[self.selected]
        key = max(live, key=lambda k: int(k.split(":", 1)[1]))
        return key, drawn[key]

    def candidates_on(self, frame_id: int) -> list[tuple[list[float], float]]:
        """Kept proposals on one frame. Struck-out ones leave the canvas at once
        — the canvas is where the reviewer checks their own decision."""
        return [(b, sc) for fid, b, sc in self.kept_candidates() if fid == frame_id]

    def selected_box(self) -> list[float] | None:
        # frame_objects, not visible_objects: this is a lookup, not a draw. A
        # deleted track still has to be findable, or the button that restores it
        # would have nothing to point at.
        for o in frame_objects(self.seq_id, self.frame_id, self.decisions):
            if o.key == self.selected:
                return [float(v) for v in o.box]
        return None


SESSION: Session | None = None


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def header_md() -> str:
    s = SESSION
    it = s.item
    stats = progress(s.items, s.decisions)
    done = sum(v["accepted"] for v in stats.values())
    parts = " &nbsp;·&nbsp; ".join(
        f"{m}: {v['accepted']}/{v['total']}" + (f" ({v['flagged']}⚑)" if v["flagged"] else "")
        for m, v in sorted(stats.items()))
    status = s.decisions.status_of(s.seq_id)
    badge = {"accepted": "✅ accepted", "flagged": "⚑ flagged",
             "excluded": "✖ excluded from the release"}.get(
                 status, "— not signed off")
    tags = s.decisions.tags(s.seq_id)
    badge += ("  &nbsp;" + " ".join(f"`{t}`" for t in tags)) if tags else ""
    watched = "👁 watched" if s.decisions.get(s.seq_id).get("watched") else "not watched"
    return (f"### {it.seq_id} &nbsp;<sub>{it.dataset} · {it.category} · "
            f"{it.n_frames} frames · **{it.mode}**</sub>\n"
            f"{badge} &nbsp;·&nbsp; {watched} &nbsp;·&nbsp; "
            f"video {s.index + 1}/{len(s.items)} &nbsp;·&nbsp; "
            f"**{done}/{len(s.items)} accepted** &nbsp;·&nbsp; {parts}"
            + (f"\n\n> ⚠ {s.message}" if s.message else ""))


def info_md() -> str:
    s = SESSION
    summary = sequence_summary(s.seq_id)
    edits = s.decisions.edit_counts(s.seq_id)
    prov = " ".join(f"`{k}` {v}" for k, v in sorted(summary["by_provenance"].items()))
    lines = [
        MODE_QUESTION[s.item.mode],
        "",
        f"{summary['tracks']} tracks · {summary['boxes']} boxes · median "
        f"{summary['median_size']:.0f} px · {prov}",
    ]
    if summary["tracks_recovered"]:
        lines.append(f"**{summary['tracks_recovered']} tracks were recovered by the "
                     f"merge** — they have never been looked at by anyone.")
    if any(edits.values()):
        lines.append("edits: " + ", ".join(f"{k.replace('_', ' ')} {v}"
                                           for k, v in edits.items() if v))
    if s.selected:
        out = size_outliers(s.seq_id, s.selected, s.decisions)
        n = len(track_frames(s.seq_id, s.selected))
        ratio = area_ratio(s.seq_id, s.selected, s.decisions)
        lines.append(f"\n**selected `{s.selected}`** — {n} frames"
                     + (f", {ratio:.2f}× the area of the original annotation"
                        if ratio is not None else "")
                     + (f", ⚠ {len(out)} frames with outlying box area: "
                        + ", ".join(f"{f}({r:.1f}×)" for f, r in out[:8])
                        if out else ", box area is stable"))
    if s.candidates:
        per: dict[int, int] = {}
        for fid, _b, sc in s.kept_candidates():
            per[fid] = per.get(fid, 0) + 1
        scores = [sc for _f, _b, sc in s.candidates]
        here = len(s.candidates_on(s.frame_id))
        st = s.sweep_stats
        if s.candidate_rejected:
            lines.append(f"\n**{len(s.kept_candidates())} kept**, "
                         f"{len(s.candidate_rejected)} struck out — only the "
                         f"kept ones are saved")
        lines.append(
            f"\n**{len(s.candidates)} proposals** across {len(per)} frames, "
            f"score {min(scores):.2f}–{max(scores):.2f} &nbsp;·&nbsp; "
            f"{here} on frame {s.frame_id} &nbsp;·&nbsp; "
            f"they would be saved as `{s.candidate_category}`")
        shown = min(len(s.candidates), 96)
        if shown < len(s.candidates):
            # The wall is capped at 96 tiles. Said out loud, because a wall that
            # quietly showed 96 of 300 reads as "that is all there was".
            lines.append(f"the wall shows the **{shown} strongest**; all "
                         f"{len(s.candidates)} would be saved")
        if st:
            lines.append(f"detector matched {st.get('raw', 0)} over "
                         f"{st.get('frames', 0)} frames — "
                         f"{st.get('already_annotated', 0)} already annotated, "
                         f"{st.get('wrong_size', 0)} the wrong size, "
                         f"{st.get('new', 0)} new")
        lines.append("per frame: "
                     + ", ".join(f"{f}({n})" for f, n in sorted(per.items())))
    if s.draft is not None:
        lines.append(f"\n**drafting a new `{s.draft.category}`** from frame "
                     f"{s.draft.anchor_frame} — {s.draft.summary()}"
                     + (f"; segmented at score {s.draft.seed_score:.2f}"
                        if s.draft.seed_box else "")
                     + (f"; propagated over {len(s.draft.boxes)} frames"
                        if s.draft.boxes else ""))
    return "\n".join(lines)


def views() -> tuple:
    s = SESSION
    grid, objs = grid_view(s.seq_id, s.frame_id, s.decisions, tile=TILE,
                           columns=COLUMNS, selected=s.selected, show_raw=s.show_raw)
    s.grid_objs = objs
    frame, _ = frame_view(s.seq_id, s.frame_id, s.decisions, selected=s.selected,
                          show_raw=s.show_raw)
    box = s.selected_box()
    fix = (fix_view(s.seq_id, s.frame_id, box, s.corners, s.drawn,
                    raw_box=(raw_geometry(s.seq_id).get(s.selected, {}).get(s.frame_id)
                             if s.show_raw else None))[0]
           if s.selected else None)
    # `decisions` is what puts the existing objects on the canvas. Without it
    # the canvas shows bare imagery, and a reviewer drafts a second track on an
    # object that is already annotated — or on one they drew themselves a
    # minute ago. `on_canvas_click` passes it, so the click mapping and the
    # picture have to agree on the argument too.
    from .core.exemplar import proposals_view

    if s.candidates:
        proposals = proposals_view(s.seq_id, s.candidates, s.candidate_exemplar,
                                   tile=PROP_TILE, columns=PROP_COLUMNS,
                                   max_tiles=PROP_MAX_TILES,
                                   rejected=s.candidate_rejected)
    elif s.sweep_note:
        proposals = _placeholder(s.sweep_note, w=1000, h=160)
    else:
        proposals = None
    canvas = (annotate_view(s.seq_id, s.frame_id, s.draft, s.draft_mask,
                            zoom_box=s.draft_zoom(), decisions=s.decisions,
                            candidates=s.candidates_on(s.frame_id))[0]
              if (s.draft or s.candidates) else None)
    return grid, frame, fix, canvas, proposals


STATUS_MARK = {"accepted": "✅", "flagged": "⚑", "excluded": "✖", None: "·"}


def picker_label(item) -> str:
    mark = STATUS_MARK[SESSION.decisions.status_of(item.seq_id)]
    edited = "✎" if SESSION.decisions.is_touched(item.seq_id) else ""
    return f"{mark}{edited} {item.seq_id}  ({item.category}, {item.n_frames}f, {item.mode})"


def refresh() -> tuple:
    s = SESSION
    grid, frame, fix, canvas, proposals = views()
    header = header_md()
    # Shown once, then cleared: a warning that outlived the action it refers to
    # would read as a warning about the next one.
    s.message = ""
    # Only a change of sequence resets the tab — doing it on every refresh would
    # throw the reviewer out of the annotate tab on every click inside it.
    if s.switched:
        s.switched = False
        video_u, tabs_u = gr.update(value=None), gr.Tabs(selected="grid")
    elif s.show_proposals:
        # A sweep that leaves the reviewer on whatever tab they were on reads as
        # a sweep that did nothing: the proposals live on a view they are not
        # looking at.
        s.show_proposals = False
        video_u, tabs_u = gr.skip(), gr.Tabs(selected="proposals")
    else:
        video_u = tabs_u = gr.skip()
    # Choices grow with whatever has actually been used, so a custom tag typed
    # once is a click away on every later sequence.
    vocabulary = sorted(set(TAG_PRESETS) | set(s.decisions.tag_counts()))
    return (header, info_md(), grid, frame, fix, canvas, proposals,
            gr.update(minimum=s.first_frame(), maximum=s.last_frame(),
                      value=s.frame_id, step=1),
            gr.update(choices=[picker_label(i) for i in s.items],
                      value=picker_label(s.item)),
            video_u, tabs_u,
            gr.update(choices=vocabulary, value=s.decisions.tags(s.seq_id)))


# ---------------------------------------------------------------------------
# callbacks
# ---------------------------------------------------------------------------

def on_frame(frame_id):
    s = SESSION
    if frame_id is not None:
        s.frame_id = int(frame_id)
    # A draft box belongs to one frame; carrying it to the next would silently
    # corrupt a different box.
    s.clear_draft()
    return refresh()


def step_frame(delta: int):
    s = SESSION
    s.frame_id = int(np.clip(s.frame_id + delta, s.first_frame(), s.last_frame()))
    s.clear_draft()
    return refresh()


def step_video(delta: int):
    SESSION.goto(SESSION.index + delta)
    return refresh()


def jump_to(label: str):
    s = SESSION
    for i, it in enumerate(s.items):
        # Matched on the sequence id inside the label, because the label also
        # carries a status mark that changes as the review proceeds.
        if f" {it.seq_id}  " in label:
            s.goto(i)
            break
    return refresh()


def next_flagged():
    """Flagged sequences are parked, not finished — this is how they come back."""
    s = SESSION
    n = len(s.items)
    for step in range(1, n + 1):
        i = (s.index + step) % n
        if s.decisions.status_of(s.items[i].seq_id) == "flagged":
            s.goto(i)
            return refresh()
    s.message = "nothing is flagged"
    return refresh()


def on_grid_click(evt: gr.SelectData):
    s = SESSION
    x, y = evt.index[0], evt.index[1]
    idx = tile_at(x, y, len(s.grid_objs), TILE, COLUMNS)
    s.selected = s.grid_objs[idx].key if idx is not None else None
    s.clear_draft()
    return refresh()


def on_fix_click(evt: gr.SelectData):
    """Two clicks make a replacement box; a third starts over."""
    s = SESSION
    if not s.selected:
        return refresh()
    _, scale, (ox, oy) = fix_view(s.seq_id, s.frame_id, s.selected_box(),
                                  s.corners, s.drawn)
    px = evt.index[0] / scale + ox
    py = evt.index[1] / scale + oy
    if len(s.corners) >= 2:
        s.corners = []
        s.drawn = None
    s.corners.append((px, py))
    if len(s.corners) == 2:
        (x1, y1), (x2, y2) = s.corners
        s.drawn = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    return refresh()


def save_fix():
    s = SESSION
    if s.is_view_only() and s.refuse(
            "this sequence is car-only and view-only — nothing here is edited"):
        return refresh()
    if s.selected and s.drawn:
        s.decisions.patch_box(s.seq_id, s.selected, s.frame_id, s.drawn)
        s.decisions.save()
        s.clear_draft()
    return refresh()


def undo_fix():
    s = SESSION
    if s.selected:
        s.decisions.patch_box(s.seq_id, s.selected, s.frame_id, None)
        s.decisions.save()
        s.clear_draft()
    return refresh()


def snap_sam3():
    """Offer SAM 3 on the selected box — never apply it automatically.

    The reviewer is looking at this box because something is wrong with it, and
    the same model produced it in the first place; a silent re-run would just
    reproduce the failure.
    """
    s = SESSION
    box = s.selected_box()
    if box is None:
        return refresh()
    from .core.sam3refine import _accept, get_tracker, sam3_autocast
    from .core.render import _read_frame

    frame = _read_frame(s.seq_id, s.frame_id)
    if frame is None:
        return refresh()
    sam = get_tracker()
    H, W = frame.shape[:2]
    prompt = [float(np.clip(box[0], 0, W - 1)), float(np.clip(box[1], 0, H - 1)),
              float(np.clip(box[2], 1, W)), float(np.clip(box[3], 1, H))]
    with sam3_autocast():
        sam.init_video([frame])
        sam.add_prompts(0, np.asarray([prompt], np.float32),
                        labels=np.zeros(1, np.int64), obj_ids=[0])
        outs = sam.propagate()
        sam.reset_state()
    new = ([float(v) for v in outs[0]["boxes"].numpy()[0]]
           if outs and len(outs[0]["boxes"]) else None)
    ok, _ = _accept(prompt, new)
    s.drawn = new if ok else None
    s.corners = []
    return refresh()


def toggle_raw(value: bool):
    SESSION.show_raw = bool(value)
    return refresh()


# -- annotation (mode `annotate`, and anywhere an object is simply missing) ---

def start_draft(category: str):
    s = SESSION
    s.draft = Draft(seq_id=s.seq_id, category=category, anchor_frame=s.frame_id)
    s.draft_mask = None
    return refresh()


def cancel_draft():
    s = SESSION
    s.draft = None
    s.draft_mask = None
    return refresh()


def set_click_mode(mode: str):
    SESSION.click_mode = mode
    return refresh()


def on_canvas_click(evt: gr.SelectData):
    s = SESSION
    if s.draft is None:
        return refresh()
    _, scale, (ox, oy) = annotate_view(s.seq_id, s.frame_id, s.draft, s.draft_mask,
                                       zoom_box=s.draft_zoom(), decisions=s.decisions,
                                       candidates=s.candidates_on(s.frame_id))
    px, py = evt.index[0] / scale + ox, evt.index[1] / scale + oy
    if s.click_mode == "box corner":
        s.draft.add_corner(px, py)
    else:
        s.draft.add_point(px, py, positive=s.click_mode == "positive")
    return refresh()


def undo_prompt():
    s = SESSION
    if s.draft is not None:
        s.draft.undo()
    return refresh()


def segment():
    """Segment the anchor frame from the draft's prompts, and say what happened.

    Three different things used to render as the same nothing: no draft open, a
    draft whose prompt is incomplete, and SAM 3 returning an empty mask. From
    the reviewer's side those are indistinguishable — the canvas simply does not
    change — and the three fixes are unrelated, so each says which it was.

    The middle one is the common case and the least visible: `box corner` mode
    needs *two* opposite corners, and one click leaves a prompt that no model
    ever sees.
    """
    s = SESSION
    if s.draft is None:
        s.refuse("no object being drafted — press `New object here` first")
        return refresh()
    if not s.draft.has_prompt():
        if s.draft.box_corners:
            s.refuse("one corner placed — click the **opposite** corner to "
                     "close the box, then segment")
        else:
            s.refuse("nothing to segment from: click the object, or place two "
                     "opposite corners around it")
        return refresh()

    box, score, mask = segment_frame(s.draft)
    s.draft.seed_box, s.draft.seed_score = box, score
    s.draft_mask = mask
    if box is None:
        s.refuse(f"SAM 3 ran on frame {s.draft.anchor_frame} and came back with "
                 "an empty mask — nudge the box to sit tight on the object, or "
                 "add a negative click on what it is being pulled towards")
    return refresh()


def propagate(backward: bool = False, progress_=gr.Progress()):
    """Carry the drafted object through the video, forward from the anchor.

    Forward only unless asked otherwise: the anchor is the frame the reviewer
    chose because the object is visible there, so everything before it is
    frames they have not looked at, and a reverse pass fills them with boxes on
    whatever the tracker finds. Tick `also track backwards` when the object was
    genuinely there earlier and simply unannotated.
    """
    s = SESSION
    if s.draft is None:
        s.refuse("no object being drafted — press `New object here` first")
        return refresh()
    if s.draft.seed_box is None:
        s.refuse("nothing to propagate yet — `Segment` has to return a box on "
                 "this frame before it can be carried through the video")
        return refresh()
    frozen: dict = {}
    s.draft.boxes = propagate_draft(s.draft,
                                    progress=lambda f, m: progress_(f, desc=m),
                                    frozen=frozen, backward=bool(backward))
    if not s.draft.boxes:
        s.refuse("propagation held the object on no frame at all — re-segment "
                 "from a frame where it is clearer")
    elif 0 in frozen:
        # Not a failure, and not something the reviewer can see: the track just
        # stops, and a short track looks like a hard object rather than a
        # recoverable anchor choice. On rscardata/train/069 anchoring `car:6` at
        # frame 1 freezes at 48 and covers nothing new, while anchoring the same
        # object at frame 100 covers all 332 frames.
        s.message = (f"propagation stopped at frame {frozen[0]} — the object "
                     f"stopped moving over the ground there, so the rest was "
                     f"dropped rather than written as a frozen box. If it is "
                     f"still driving later, start the draft from a frame where "
                     f"it is moving and propagate again.")
    return refresh()


def sweep_similar(thr: float, progress_=gr.Progress()):
    """Find everything in this video that looks like the drawn track."""
    s = SESSION
    if s.is_view_only() and s.refuse(
            "car-only sequence — a static car at 5 px is not annotatable here"):
        return refresh()
    found = s.exemplar_track()
    if found is None:
        s.sweep_note = ("nothing to copy: segment an object on the canvas, or "
                        "select a track you drew earlier, then sweep again")
        s.show_proposals = True
        s.refuse(s.sweep_note)
        return refresh()
    key, track = found
    s.candidate_category = key.split(":", 1)[0]
    s.sweep_note = ""
    s.candidate_rejected = set()
    from .core.exemplar import find_similar

    # An exemplar taken from an unsaved draft is invisible to `decisions`, so
    # without this the sweep hands the reviewer their own object back as its
    # top-scoring find.
    st: dict = {}
    s.candidates = find_similar(s.seq_id, track, decisions=s.decisions,
                                thr=float(thr), stats=st,
                                exclude=track if key.endswith(":draft") else None,
                                progress=lambda f, m: progress_(f, desc=m))
    mid = sorted(track)[len(track) // 2]
    s.candidate_exemplar = (mid, track[mid], 1.0)
    s.show_proposals = True
    s.sweep_stats = st
    if not s.candidates:
        s.candidate_exemplar = None
        # Say which of the three ways it came back empty, because the fix is
        # different for each: raise/lower the score, or accept that everything
        # matching is already in the ground truth.
        if not st.get("raw"):
            why = (f"the detector matched nothing above {thr:.2f} on any of "
                   f"{st.get('frames', 0)} frames — lower the score, or the "
                   f"exemplar is too small to ground on")
        elif st.get("already_annotated") and not st.get("wrong_size"):
            why = (f"every one of the {st['raw']} matches is already annotated "
                   f"— there is nothing missing here to add")
        else:
            why = (f"{st['raw']} matches, but {st.get('already_annotated', 0)} "
                   f"are already annotated and {st.get('wrong_size', 0)} are "
                   f"the wrong size for the exemplar — none left")
        s.sweep_note = why
        s.refuse(why)
    else:
        # Jump to the frame with the most proposals, and open the wall: the
        # canvas can only show one frame's worth, which on its own understates
        # what the sweep found by roughly the number of frames it searched.
        per: dict[int, int] = {}
        for fid, _b, _sc in s.candidates:
            per[fid] = per.get(fid, 0) + 1
        s.frame_id = max(per, key=lambda f: (per[f], -f))
        s.show_proposals = True
    return refresh()


def on_proposal_click(evt: gr.SelectData):
    """Strike out the proposal under the cursor, or put it back.

    A toggle rather than a delete: the sweep's mistakes and its successes look
    alike at 12 px, so a reviewer will strike one out and want it back a second
    later once they have looked at the frame it came from.
    """
    s = SESSION
    if not s.candidates:
        return refresh()
    lead = 1 if s.candidate_exemplar is not None else 0
    n = min(len(s.candidates), PROP_MAX_TILES) + lead
    tile = tile_at(evt.index[0], evt.index[1], n, PROP_TILE, PROP_COLUMNS)
    if tile is None or tile < lead:
        return refresh()              # the exemplar itself is not a proposal
    idx = tile - lead
    if idx in s.candidate_rejected:
        s.candidate_rejected.discard(idx)
    else:
        s.candidate_rejected.add(idx)
        # Follow the frame the struck-out proposal came from, so the canvas
        # shows what was just rejected in its own context.
        s.frame_id = s.candidates[idx][0]
    return refresh()


def restore_proposals():
    SESSION.candidate_rejected = set()
    return refresh()


def reject_below(thr: float):
    """Strike out everything scoring under ``thr`` in one go."""
    s = SESSION
    s.candidate_rejected = {i for i, (_f, _b, sc) in enumerate(s.candidates)
                            if sc < float(thr)}
    s.message = (f"struck out {len(s.candidate_rejected)} proposals below "
                 f"{float(thr):.2f}; {len(s.kept_candidates())} kept")
    return refresh()


def clear_candidates():
    SESSION.candidates = []
    SESSION.candidate_exemplar = None
    SESSION.candidate_rejected = set()
    SESSION.sweep_note = ""
    return refresh()


def adopt_candidates(progress_=gr.Progress()):
    """Propagate every proposal into a track and save them all."""
    s = SESSION
    if not s.candidates:
        s.refuse("run the sweep first")
        return refresh()
    category = s.candidate_category
    kept = s.kept_candidates()
    if not kept:
        s.refuse("every proposal is struck out — nothing to save")
        return refresh()
    from .core.exemplar import adopt

    existing = list(s.decisions.drawn(s.seq_id).values())
    tracks = adopt(s.seq_id, kept, decisions=s.decisions,
                   existing=existing,
                   progress=lambda f, m: progress_(f, desc=m))
    for fb in tracks.values():
        if len(fb) < 2:
            continue                    # a one-frame "track" is a false positive
        key = s.decisions.new_drawn_key(s.seq_id, category)
        s.decisions.set_drawn(s.seq_id, key, fb)
    s.decisions.save()
    n_saved = sum(1 for fb in tracks.values() if len(fb) >= 2)
    merged = _merge_note()
    s.message = (f"saved {n_saved} new {category} track"
                 f"{'s' * (n_saved != 1)} from {len(kept)} kept proposals"
                 + (f" ({len(s.candidate_rejected)} struck out)"
                    if s.candidate_rejected else "")
                 + (f" &nbsp;·&nbsp; {merged}" if merged else ""))
    s.candidates = []
    s.candidate_exemplar = None
    s.candidate_rejected = set()
    s.sweep_note = ""
    return refresh()


def append_to_track():
    """Continue the selected track from the draft's anchor frame onward.

    The repair for a track that was tracking correctly and then latched onto
    the background: step to the frame where it went wrong, point at the real
    object, propagate, and graft that onto the existing track. Frames before
    the anchor are left alone — they were right, which is why the reviewer is
    only replacing the tail — so this never has to be trusted with the part of
    the track that already works.
    """
    s = SESSION
    if s.draft is None or not s.draft.boxes:
        s.refuse("segment the object and press Propagate first")
        return refresh()
    drawn = s.decisions.drawn(s.seq_id)
    if s.selected not in drawn:
        s.refuse("select the hand-drawn track to continue — click it in the grid")
        return refresh()

    anchor = s.draft.anchor_frame
    head = {f: b for f, b in drawn[s.selected].items() if f < anchor}
    tail = {f: b for f, b in s.draft.boxes.items() if f >= anchor}
    if not tail:
        s.refuse("the propagation covered nothing from this frame onward")
        return refresh()
    merged = {**head, **tail}
    s.decisions.set_drawn(s.seq_id, s.selected, merged)
    s.decisions.save()
    was = len(drawn[s.selected])
    s.message = (f"`{s.selected}`: kept {len(head)} frames before {anchor}, "
                 f"replaced the rest with {len(tail)} — {was} frames → "
                 f"{len(merged)}")
    s.draft = None
    s.draft_mask = None
    return refresh()


def save_track():
    """Store the drafted track, then say what became of it.

    A draft can end up contributing nothing in two entirely different ways, and
    both used to clear the canvas and report the same silence. It has no boxes
    at all — `Propagate` was never pressed, or held the object nowhere — in
    which case the draft is *kept*, because discarding an unsaved prompt on the
    reviewer's behalf loses work they cannot get back. Or it was saved and then
    folded into a track that already covered every one of its frames, which is
    the merge behaving correctly and still leaves the drawn layer empty.
    """
    s = SESSION
    if s.is_view_only() and s.refuse(
            "car-only sequence — a static car at 5 px is not annotatable here"):
        return refresh()
    if s.draft is None:
        s.refuse("no object being drafted — press `New object here` first")
        return refresh()
    if not s.draft.boxes:
        s.refuse("this draft has no boxes to save — press `Segment`, then "
                 "`Propagate`, before saving. The draft is still open.")
        return refresh()

    n = len(s.draft.boxes)
    key = save_draft(s.draft, s.decisions)
    s.selected = key or s.selected
    s.draft = None
    s.draft_mask = None

    note = _merge_note()
    # A merge that found the track wholly redundant now hides it rather than
    # erasing it, so it is still in `drawn` — `kept` alone would read that as
    # "the track is fine" and report a save that the reviewer cannot see.
    hidden = bool(key) and key in s.decisions.deleted(s.seq_id)
    kept = len(s.decisions.drawn(s.seq_id).get(key) or {}) if key else 0
    if key and (hidden or not kept):
        s.message = (f"saved {n} frames as {key}, and every one of them was "
                     f"already annotated — {note or 'the track was absorbed'}. "
                     f"Nothing new was added: to fill a gap, anchor the draft "
                     f"inside it.")
    else:
        s.message = note or f"saved {n} frames as {key}"
    return refresh()


def _merge_note() -> str:
    """Report what a merge *would* fold into what. Changes nothing by default.

    Run automatically rather than offered: a duplicate identity is not a
    judgement call the reviewer is in a position to make from the canvas — the
    grid shows one frame, and the track it duplicates may only start 60 frames
    later. Left alone it ships as an identity switch inside the ground truth.

    Safe to run unasked only because a merge can no longer destroy anything: one
    whose entire effect would be to delete the track just drawn hides it instead
    (see :func:`~.merge.apply_merges`), and every merge is logged.
    ``--no-auto-merge`` turns it into a report that the `Merge duplicate tracks`
    button carries out.
    """
    from .core.merge import find_continuations, find_merges, merge_duplicates

    s = SESSION
    if s.auto_merge:
        return " · ".join(f"merged {n}"
                          for n in merge_duplicates(s.seq_id, s.decisions))

    plans = find_merges(s.seq_id, s.decisions) + \
        find_continuations(s.seq_id, s.decisions)
    if not plans:
        return ""
    bits = ", ".join(f"{p['drawn']} looks like {p['into']} "
                     f"({p['dist']:.2f} widths apart)" for p in plans[:3])
    more = f" (+{len(plans) - 3} more)" if len(plans) > 3 else ""
    return (f"nothing was merged — {bits}{more}. Check they really are one "
            f"object, then press `Merge duplicate tracks`.")


def trim_parked(progress_=gr.Progress()):
    """Drop each drawn track's frames from before its object started moving.

    A car that is parked for the first two hundred frames and then drives off is
    one object under two conventions: the car sets annotate movers, so the same
    box is ground truth after it pulls away and a category error before. Neither
    deleting the track nor keeping it whole is right, so the parked head goes and
    the rest stays.

    Measured over the ground, not the image — a parked car travels across the
    frame with the camera, and in image coordinates that is indistinguishable
    from driving.
    """
    s = SESSION
    from .core.motion import trim_parked_starts

    notes = trim_parked_starts(s.seq_id, s.decisions,
                               progress=lambda f, m: progress_(f, desc=m))
    if not notes:
        s.refuse("no drawn track here is parked at its start and moving later — "
                 "one that never moves at all is `Delete static tracks`")
        return refresh()
    s.selected = None
    s.message = ("; ".join(notes) + " &nbsp;·&nbsp; the dataset's own frames are "
                 "untouched, but the dropped drawn frames cannot be restored")
    return refresh()


def strike_switched():
    """Drop the stretches where a drawn track left its object for another one.

    Not the same failure as a duplicate, and merging would make it worse. A
    duplicate agrees with its twin for as long as both exist, so folding them
    together loses nothing. A track that switched agrees only after the switch:
    on `rscardata/train/011` the extension of `car:18` tracks its own car to
    frame 121, and from 122 — an overtake — it is on `car:30`'s car for the rest
    of the sequence. Merging the two ids would claim the first hundred frames
    were that car as well; keeping them puts two boxes on one car for eighty.

    So only the frames after the switch go, and what is left of the track is the
    part that was still following its own object.
    """
    s = SESSION
    from .core.merge import strike_overlaps

    notes = strike_overlaps(s.seq_id, s.decisions)
    if not notes:
        s.refuse("no drawn track sits on an object another id already carries "
                 "for long enough to be a switch rather than a pass")
        return refresh()
    s.selected = None
    s.message = ("; ".join(notes) + " &nbsp;·&nbsp; the dataset's own frames are "
                 "untouched, but the dropped drawn frames cannot be restored")
    return refresh()


def strike_oversize_boxes():
    """Drop the boxes SAM 3 leaked onto the surroundings, by size alone."""
    s = SESSION
    r = sizecheck.strike_oversize(s.seq_id, s.decisions)
    if not (r["tracks_deleted"] or r["frames_dropped"]):
        s.refuse(f"nothing oversized — every drawn box is within "
                 f"{sizecheck.OVERSIZE_FRAME:.0f}x its own track, and no track "
                 f"exceeds {sizecheck.OVERSIZE_TRACK:.0f}x the {r['ref_px']:.1f} px "
                 f"objects this dataset annotates")
        return refresh()
    bits = []
    if r["tracks_deleted"]:
        bits.append(f"deleted {r['tracks_deleted']} oversized track(s)")
    if r["frames_dropped"]:
        bits.append(f"dropped {r['frames_dropped']} leaked frames")
    if r["tracks_emptied"]:
        bits.append(f"{r['tracks_emptied']} track(s) left too short to keep")
    s.selected = None
    s.message = (f"against the {r['ref_px']:.1f} px this dataset annotates: "
                 + ", ".join(bits) + " &nbsp;·&nbsp; the deletions are "
                 "reversible, the dropped frames are not")
    return refresh()


def strike_static(progress_=gr.Progress()):
    """Delete the drawn tracks that are not a moving object.

    Two ways to fail, opposite in both coordinate frames: *parked* — annotated
    under the wrong convention, since SAT-MTB and the car sets label movers only
    and an exemplar sweep cannot tell a parked car from a driving one; *pinned*
    — the propagation lost the object, usually where it drove out of frame, and
    the box stayed put in the image for the rest of the sequence. Nothing is
    removed when neither applies, and the sequence says so.
    """
    s = SESSION
    from .core.motion import static_tracks

    keys, rows = static_tracks(s.seq_id, s.decisions,
                               progress=lambda f, m: progress_(f, desc=m))
    if not rows:
        s.refuse("no drawn tracks on this sequence to measure")
        return refresh()
    if not keys:
        moved = min(net for _k, net, _m, _i, _n, _r in rows)
        s.message = (f"all {len(rows)} drawn tracks move, over the ground and "
                     f"in the image (the least of them by {moved:.0f} px) — "
                     f"nothing removed")
        return refresh()

    # A merge re-keys a drawn track onto an identity the dataset already ships,
    # so after one has run the two layers can share a key — and `deleted` is
    # keyed by identity, not by layer. Marking the whole key deleted would take
    # the dataset's own frames with it: on rscardata/train/011 that is 75 frames
    # of a real car thrown away because the propagation that extended it stuck
    # at the border afterwards. So where the dataset supplies the identity, only
    # the drawn frames go.
    shipped = {o.key for objs in load_frames(s.seq_id).values() for o in objs}
    erased = [k for k in keys if k in shipped]
    if erased:
        s.decisions.purge_drawn(s.seq_id, erased)
    for key in keys:
        if key not in shipped:
            s.decisions.set_deleted(s.seq_id, key, True)
    s.decisions.save()

    detail = ", ".join(
        ((f"{k} pinned in the image, {img:.0f} px over {n} frames"
          if reason == "pinned" else f"{k} parked, {net:.0f} px over the ground")
         + (" (drawn frames erased, dataset track kept)" if k in shipped else ""))
        for k, net, _m, img, n, reason in rows if reason)
    note = ("select one and press `Delete / restore this track` to put it back"
            if len(erased) < len(keys) else "")
    if erased:
        note = (f"the {len(erased)} extension(s) of a dataset track cannot be "
                f"restored — draw again if that was wrong"
                + (" &nbsp;·&nbsp; " + note if note else ""))
    s.message = (f"deleted {len(keys)} of {len(rows)} drawn tracks: "
                 f"{detail} &nbsp;·&nbsp; {note}")
    return refresh()


def merge_tracks():
    """Merge duplicates across the whole sequence, on demand."""
    s = SESSION
    notes = _merge_note()
    s.message = notes or "no drawn track duplicates an existing one here"
    s.selected = None
    return refresh()


def purge_drawn_tracks():
    """Erase every hand-drawn track on this sequence that is marked deleted."""
    s = SESSION
    n = s.decisions.purge_drawn(s.seq_id)
    if not n["tracks"]:
        s.refuse("nothing to erase — this only removes tracks *you drew* that "
                 "are already deleted. Delete one first; the dataset's own boxes "
                 "are never erased")
        return refresh()
    s.decisions.save()
    s.selected = None
    s.message = (f"erased {n['tracks']} drawn track(s), {n['boxes']} boxes — out "
                 f"of the file for good, not recoverable")
    return refresh()


def relabel_track(category: str):
    """Correct the category of the selected track.

    A wrong category is not a wrong box, and nothing else in this tool can
    express it: SAT-MTB labels three ships in ``car/18`` ``airplane``, and every
    per-class result those 450 boxes touched has been scored against the wrong
    class. Deleting the track would lose a real object; correcting the geometry
    would leave the class untouched.

    The track's key carries its category, so relabelling re-keys it — and a key
    the sequence already uses would silently fuse two identities into one, which
    is why that case is refused rather than resolved.
    """
    s = SESSION
    if not s.selected:
        s.refuse("select an object first — click its tile in the grid")
        return refresh()
    if category not in CATEGORIES:
        s.refuse(f"{category!r} is not one of {', '.join(CATEGORIES)}")
        return refresh()
    if s.selected.split(":", 1)[0] == category:
        s.refuse(f"{s.selected} is already a {category}")
        return refresh()

    was = s.selected
    tid = was.split(":", 1)[1]
    wants = f"{category}:{tid}"
    # Both layers, and the label layer applied first: `load_frames` is keyed on
    # the sequence alone, so asking it before the registry is in step would test
    # the collision against the categories this sequence had one edit ago.
    set_label_overrides(s.seq_id, s.decisions.labels(s.seq_id))
    live = {o.key for objs in load_frames(s.seq_id).values() for o in objs}
    live |= set(s.decisions.drawn(s.seq_id))
    if wants in live:
        s.refuse(f"{s.seq_id} already has a {wants} — relabelling {was} would "
                 f"merge two identities into one. Give one of them a different "
                 f"track id first.")
        return refresh()

    s.selected = s.decisions.set_label(s.seq_id, was, category)
    s.decisions.save()
    # Again, and after the write: the collision check above synced the registry
    # to the state *before* this relabel, so `track_frames` below would count
    # the frames of a key that no longer exists and report zero.
    set_label_overrides(s.seq_id, s.decisions.labels(s.seq_id))
    s.message = (f"{was} is now {s.selected} — {len(track_frames(s.seq_id, s.selected))} "
                 f"boxes re-classed. The geometry is untouched; the dataset's own "
                 f"file is untouched. Relabel it back to "
                 f"`{was.split(':', 1)[0]}` to undo.")
    return refresh()


def delete_track():
    s = SESSION
    if s.selected:
        deleted = s.selected in s.decisions.deleted(s.seq_id)
        s.decisions.set_deleted(s.seq_id, s.selected, not deleted)
        s.decisions.save()
    return refresh()


def play(progress_=gr.Progress()):
    from .core.video import render_sequence_video

    s = SESSION
    path = render_sequence_video(s.seq_id, decisions=s.decisions,
                                 progress=lambda f, m: progress_(f, desc=m))
    s.decisions.set_watched(s.seq_id)
    s.decisions.save()
    # Switch to the tab the video is on. Rendering it and leaving the reviewer
    # looking at the grid is indistinguishable from the button doing nothing.
    return str(path), header_md(), gr.Tabs(selected="playback")


def accept(note: str):
    s = SESSION
    # Watching the whole sequence is the sign-off, not a nicety: a box that
    # drifts over 200 frames is invisible in every still view in this tool.
    if (s.require_watch and not s.decisions.get(s.seq_id).get("watched")
            and s.refuse("play the sequence first (P) — accepting is a "
                         "statement that you watched it")):
        return refresh() + ("",)
    s.decisions.set_status(s.seq_id, "accepted", note)
    s.decisions.save()
    s.goto(next_undone(s.items, s.decisions, s.index))
    return refresh() + ("",)


def flag(note: str):
    s = SESSION
    s.decisions.set_status(s.seq_id, "flagged", note)
    s.decisions.save()
    # Cleared, like Accept does: a note left in the box outlives the sequence it
    # was written about, and the next Accept would silently inherit it.
    return refresh() + ("",)


def exclude(note: str):
    """Drop this sequence from the release. Requires a reason."""
    s = SESSION
    if not note.strip() and s.refuse(
            "write why in the note box first — an excluded sequence with no "
            "recorded reason cannot be defended later"):
        return refresh() + (note,)
    s.decisions.set_status(s.seq_id, "excluded", note)
    s.decisions.save()
    s.goto(next_undone(s.items, s.decisions, s.index))
    return refresh() + ("",)


def set_tags(values):
    """Tags are descriptive, so they never disturb a sign-off."""
    s = SESSION
    s.decisions.set_tags(s.seq_id, list(values or []))
    s.decisions.save()
    return refresh()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

HOTKEY_JS = """
() => {
  const click = id => { const b = document.getElementById(id); if (b) b.click(); };
  document.addEventListener('keydown', e => {
    const t = e.target.tagName;
    if (t === 'INPUT' || t === 'TEXTAREA') return;
    const k = e.key.toLowerCase();
    const map = {
      'arrowleft':  () => click(e.shiftKey ? 'k_back10'  : 'k_back'),
      'arrowright': () => click(e.shiftKey ? 'k_fwd10'   : 'k_fwd'),
      '[': () => click('k_prevvid'), ']': () => click('k_nextvid'),
      'p': () => click('k_play'), 'a': () => click('k_accept'),
      'f': () => click('k_flag'), 's': () => click('k_snap'),
      'u': () => click('k_undo'), 'x': () => click('k_exclude'),
    };
    if (map[k]) { e.preventDefault(); map[k](); }
  });
}
"""


def build_ui(args) -> gr.Blocks:
    # Rendered once here, because a Gradio component with no initial value comes
    # up empty and stays empty until the first callback: the reviewer opens the
    # tool onto three blank panels and has to press something to find out that
    # nothing is wrong.
    grid0, frame0, fix0, canvas0, proposals0 = views()

    with gr.Blocks(title="space-tracker MOT review", fill_width=True) as demo:
        header = gr.Markdown(header_md())

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    b_prev_vid = gr.Button("◀ video", elem_id="k_prevvid", scale=0)
                    picker = gr.Dropdown([picker_label(i) for i in SESSION.items],
                                         value=picker_label(SESSION.item),
                                         container=False, filterable=True, scale=4)
                    b_next_vid = gr.Button("video ▶", elem_id="k_nextvid", scale=0)
                    b_flagged = gr.Button("⚑ next flagged", scale=0)
                with gr.Tabs() as tabs:
                  with gr.Tab("grid — click a tile to select", id="grid"):
                    grid = gr.Image(grid0, label=None, show_label=False,
                                    interactive=False, height=760)
                  with gr.Tab("whole frame", id="frame"):
                    frame = gr.Image(frame0, label=None, show_label=False,
                                     interactive=False, height=760)
                  with gr.Tab("annotate — add an object that is not there", id="annotate"):
                      canvas = gr.Image(canvas0, label=None, show_label=False,
                                        interactive=False, height=700)
                      with gr.Row():
                          cat = gr.Dropdown(CATEGORIES, value="airplane",
                                            label="category", scale=1)
                          click_mode = gr.Radio(["positive", "negative", "box corner"],
                                                value="positive", label="click places",
                                                scale=2)
                      with gr.Row():
                          b_draft = gr.Button("New object here", variant="primary")
                          b_segment = gr.Button("Segment")
                          b_prop = gr.Button("Propagate")
                          back_prop = gr.Checkbox(
                              False, label="also track backwards",
                              info="off: the track starts at the frame you "
                                   "segmented", scale=0)
                          b_save_track = gr.Button("Save as track", variant="primary")
                          b_append = gr.Button("Continue selected track from here")
                          b_undo_prompt = gr.Button("Undo prompt")
                          b_cancel = gr.Button("Cancel")
                      with gr.Row():
                          b_sweep = gr.Button("🔍 Find all like this",
                                              variant="secondary")
                          b_adopt = gr.Button("Save all proposals as tracks",
                                              variant="primary")
                          b_clear_cand = gr.Button("Discard proposals")
                          b_merge = gr.Button("Merge duplicate tracks")
                          b_static = gr.Button("Delete static tracks")
                          b_oversize = gr.Button("Drop oversized boxes")
                          b_switched = gr.Button("Drop stretches that switched object")
                          b_parked = gr.Button("Trim the parked start")
                          sweep_thr = gr.Slider(0.05, 0.9, value=0.35, step=0.05,
                                                label="exemplar score", scale=2)
                  with gr.Tab("proposals — click a tile to strike it out",
                              id="proposals"):
                      proposals = gr.Image(proposals0, label=None, show_label=False,
                                           interactive=False, height=700)
                      with gr.Row():
                          b_restore = gr.Button("Restore all struck-out")
                          b_reject_below = gr.Button("Strike out below the score")
                  with gr.Tab("playback", id="playback"):
                    video = gr.Video(label=None, show_label=False, height=680,
                                     autoplay=True)
                with gr.Row():
                    b_back = gr.Button("◀", elem_id="k_back", scale=0)
                    slider = gr.Slider(SESSION.first_frame(), SESSION.last_frame(),
                                       value=SESSION.frame_id, step=1, label="frame",
                                       scale=6)
                    b_fwd = gr.Button("▶", elem_id="k_fwd", scale=0)
                gr.Markdown(LEGEND)

            with gr.Column(scale=2):
                info = gr.Markdown(info_md())
                fix = gr.Image(fix0, label="selected object — click two opposite "
                               "corners", interactive=False, height=420)
                with gr.Row():
                    b_snap = gr.Button("Snap with SAM 3", elem_id="k_snap")
                    b_save = gr.Button("Save this frame", variant="primary")
                    b_undo = gr.Button("Undo", elem_id="k_undo")
                with gr.Row():
                    b_delete = gr.Button("Delete / restore this track")
                    b_purge = gr.Button("Erase deleted tracks I drew")
                    show_raw = gr.Checkbox(False, label="show geometry before SAM 3",
                                           container=False)
                with gr.Row():
                    relabel_to = gr.Dropdown(CATEGORIES, value=None, container=False,
                                             scale=1,
                                             label="category this object really is")
                    b_relabel = gr.Button("Relabel this track", scale=1)
                tags = gr.Dropdown(TAG_PRESETS, value=[], multiselect=True,
                                   allow_custom_value=True, label="tags — what "
                                   "this video *is*, independent of review status")
                note = gr.Textbox(label="note", placeholder="why is it flagged / "
                                  "excluded?", lines=1)
                with gr.Row():
                    b_play = gr.Button("▶ Play whole sequence", elem_id="k_play")
                    b_accept = gr.Button("✅ Accept & next", variant="primary",
                                         elem_id="k_accept")
                    b_flag = gr.Button("⚑ Flag", elem_id="k_flag")
                    b_exclude = gr.Button("✖ Exclude", elem_id="k_exclude")
                with gr.Accordion("how this works", open=False):
                    gr.Markdown(GUIDE_MD)

        # Hidden twins for the ±10 hotkeys — Gradio has no hotkey API, so every
        # key has to reach a real button.
        with gr.Row(visible=False):
            b_back10 = gr.Button(elem_id="k_back10")
            b_fwd10 = gr.Button(elem_id="k_fwd10")

        out = [header, info, grid, frame, fix, canvas, proposals, slider,
               picker, video, tabs, tags]
        b_back.click(lambda: step_frame(-1), outputs=out)
        b_fwd.click(lambda: step_frame(1), outputs=out)
        b_back10.click(lambda: step_frame(-10), outputs=out)
        b_fwd10.click(lambda: step_frame(10), outputs=out)
        b_prev_vid.click(lambda: step_video(-1), outputs=out)
        b_next_vid.click(lambda: step_video(1), outputs=out)
        b_flagged.click(next_flagged, outputs=out)
        # `.input`, not `.change`: the nav buttons write the picker's value back,
        # and `.change` would fire on that write and re-enter jump_to.
        picker.input(jump_to, inputs=picker, outputs=out)
        slider.release(on_frame, inputs=slider, outputs=out)
        grid.select(on_grid_click, outputs=out)
        fix.select(on_fix_click, outputs=out)
        canvas.select(on_canvas_click, outputs=out)
        b_draft.click(start_draft, inputs=cat, outputs=out)
        b_segment.click(segment, outputs=out)
        b_prop.click(propagate, inputs=back_prop, outputs=out)
        b_save_track.click(save_track, outputs=out)
        b_append.click(append_to_track, outputs=out)
        b_undo_prompt.click(undo_prompt, outputs=out)
        b_cancel.click(cancel_draft, outputs=out)
        b_sweep.click(sweep_similar, inputs=sweep_thr, outputs=out)
        b_adopt.click(adopt_candidates, outputs=out)
        b_clear_cand.click(clear_candidates, outputs=out)
        proposals.select(on_proposal_click, outputs=out)
        b_restore.click(restore_proposals, outputs=out)
        b_reject_below.click(reject_below, inputs=sweep_thr, outputs=out)
        b_merge.click(merge_tracks, outputs=out)
        b_static.click(strike_static, outputs=out)
        b_oversize.click(strike_oversize_boxes, outputs=out)
        b_switched.click(strike_switched, outputs=out)
        b_parked.click(trim_parked, outputs=out)
        click_mode.input(set_click_mode, inputs=click_mode, outputs=out)
        show_raw.input(toggle_raw, inputs=show_raw, outputs=out)
        b_snap.click(snap_sam3, outputs=out)
        b_save.click(save_fix, outputs=out)
        b_undo.click(undo_fix, outputs=out)
        b_delete.click(delete_track, outputs=out)
        b_relabel.click(relabel_track, inputs=[relabel_to], outputs=out)
        b_purge.click(purge_drawn_tracks, outputs=out)
        b_play.click(play, outputs=[video, header, tabs])
        b_accept.click(accept, inputs=note, outputs=out + [note])
        b_flag.click(flag, inputs=note, outputs=out + [note])
        b_exclude.click(exclude, inputs=note, outputs=out + [note])
        tags.input(set_tags, inputs=tags, outputs=out)
    return demo


def main() -> None:
    global SESSION
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--overrides", type=Path, default=DEFAULT_OVERRIDES)
    ap.add_argument("--reviewer", default="")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="satmtb viso sdmcar rscardata (airmot is held out by "
                         "default and comes back only if you name it)")
    ap.add_argument("--all-sizes", action="store_true",
                    help="queue the large-object half too; off by default "
                         "because the results this feeds are about small objects")
    ap.add_argument("--car-view-only", action="store_true",
                    help="make all-car sequences watch-and-sign-off again, "
                         "refusing every edit, as they were before")
    ap.add_argument("--include-unlicensed", action="store_true",
                    help="queue AIR-MOT as well — its annotations are kept but "
                         "cannot be redistributed, so it is out of scope")
    ap.add_argument("--modes", nargs="*", default=None,
                    choices=[CHECK, ANNOTATE, VIEW_ONLY])
    ap.add_argument("--sequences", nargs="*", default=None, metavar="ID",
                    help="review exactly these sequence ids and nothing else, "
                         "bypassing the licence and size filters. For sequences "
                         "held out because their *median* object is large but "
                         "which still carry small ones")
    ap.add_argument("--no-auto-merge", dest="auto_merge", action="store_false",
                    help="report duplicates instead of folding them in on save; "
                         "the `Merge duplicate tracks` button then applies them")
    ap.add_argument("--no-watch-required", action="store_true",
                    help="allow accepting a sequence that has not been played "
                         "back; off by default, because the sign-off is the "
                         "watching")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    # Merges happen without being asked for and can hide a track the reviewer
    # just drew. Without this the only account of that is a line in the browser
    # that the next click replaces.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")

    SESSION = Session(args)
    modes = Counter(i.mode for i in SESSION.items)
    print(f"{len(SESSION.items)} sequences in the queue "
          f"({', '.join(f'{m} {n}' for m, n in sorted(modes.items()))}), "
          f"resuming at {SESSION.seq_id} ({SESSION.index + 1})")
    # Not a cosmetic warning: without the merged tree `merged_available()` is
    # False everywhere, so every sequence the merge completed is queued as
    # `annotate` — "draw what is missing" — instead of `check`. The queue looks
    # normal and asks the wrong question of 154 SAT-MTB sequences.
    if not MERGED_ROOT.is_dir():
        print(f"[warn] merged ground truth not found at {MERGED_ROOT}\n"
              f"       set SPACE_TRACKER_MERGED (and SPACE_TRACKER_FILL) or "
              f"SAT-MTB shows its raw, movers-only ground truth and lands in "
              f"'annotate' rather than 'check'")
    # Playback is written outside Gradio's own temp tree, and Gradio refuses to
    # serve a file it has not been told about — the video would render fine and
    # then fail to load in the browser.
    # `show_error` puts the exception in the browser's error box. Off by
    # default, which leaves a reviewer looking at the word "Error" with the
    # traceback in a terminal they may not still have — the failure is visible
    # and undiagnosable at the same time. Nothing here is exposed to anyone the
    # reviewer would need protecting from.
    build_ui(args).launch(server_name="0.0.0.0", server_port=args.port,
                          share=args.share, js=HOTKEY_JS,
                          show_error=True,
                          allowed_paths=[str(VIDEO_CACHE)])


if __name__ == "__main__":
    main()
