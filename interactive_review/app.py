"""Video-level ground-truth review for every space-tracker MOT sequence.

One video at a time, in a fixed order, 491 of them. For each: look at what the
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
from pathlib import Path

import gradio as gr
import numpy as np

from .core.annotate import Draft, propagate_draft, save_draft, segment_frame
from .core.gtsource import (MERGED_ROOT, area_ratio, frame_objects,
                            raw_geometry, sequence_summary, track_frames)
from .core.paths import CATEGORIES, DEFAULT_OVERRIDES, sequence_by_id
from .core.render import annotate_view
from .core.vdecisions import SequenceDecisions
from .core.video import VIDEO_CACHE
from .core.vqueue import (ANNOTATE, CHECK, MODE_QUESTION, VIEW_ONLY, build_queue,
                          next_undone, progress)
from .core.vrender import LEGEND, fix_view, frame_view, grid_view, size_outliers, tile_at

TILE = 150
COLUMNS = 8

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

**Accept is fragile on purpose.** Any edit clears both the sign-off and the
"watched" flag, so an acceptance always refers to the annotation that was
actually played back.
"""


class Session:
    """Everything the UI needs that is not a Gradio component."""

    def __init__(self, args):
        self.decisions = SequenceDecisions(args.overrides, reviewer=args.reviewer)
        self.items = build_queue(datasets=tuple(args.datasets) if args.datasets else None,
                                 modes=tuple(args.modes) if args.modes else None)
        if not self.items:
            raise SystemExit("queue is empty — check --datasets / --modes")
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
        self.require_watch = not args.no_watch_required
        #: Transient one-line feedback, shown in the header until the next action.
        self.message = ""
        #: Set by :meth:`goto`. The playback tab keeps showing the *previous*
        #: sequence's video otherwise, which is indistinguishable from the
        #: Accept button having done nothing.
        self.switched = False

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

    def selected_box(self) -> list[float] | None:
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
    canvas = annotate_view(s.seq_id, s.frame_id, s.draft, s.draft_mask,
                           zoom_box=s.draft_zoom())[0] if s.draft else None
    return grid, frame, fix, canvas


STATUS_MARK = {"accepted": "✅", "flagged": "⚑", "excluded": "✖", None: "·"}


def picker_label(item) -> str:
    mark = STATUS_MARK[SESSION.decisions.status_of(item.seq_id)]
    edited = "✎" if SESSION.decisions.is_touched(item.seq_id) else ""
    return f"{mark}{edited} {item.seq_id}  ({item.category}, {item.n_frames}f, {item.mode})"


def refresh() -> tuple:
    s = SESSION
    grid, frame, fix, canvas = views()
    header = header_md()
    # Shown once, then cleared: a warning that outlived the action it refers to
    # would read as a warning about the next one.
    s.message = ""
    # Only a change of sequence resets the tab — doing it on every refresh would
    # throw the reviewer out of the annotate tab on every click inside it.
    if s.switched:
        s.switched = False
        video_u, tabs_u = gr.update(value=None), gr.Tabs(selected="grid")
    else:
        video_u = tabs_u = gr.skip()
    # Choices grow with whatever has actually been used, so a custom tag typed
    # once is a click away on every later sequence.
    vocabulary = sorted(set(TAG_PRESETS) | set(s.decisions.tag_counts()))
    return (header, info_md(), grid, frame, fix, canvas,
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
    from .core.sam3refine import _accept, get_tracker
    from .core.render import _read_frame

    frame = _read_frame(s.seq_id, s.frame_id)
    if frame is None:
        return refresh()
    sam = get_tracker()
    H, W = frame.shape[:2]
    prompt = [float(np.clip(box[0], 0, W - 1)), float(np.clip(box[1], 0, H - 1)),
              float(np.clip(box[2], 1, W)), float(np.clip(box[3], 1, H))]
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
                                       zoom_box=s.draft_zoom(), decisions=s.decisions)
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
    s = SESSION
    if s.draft is None:
        return refresh()
    box, score, mask = segment_frame(s.draft)
    s.draft.seed_box, s.draft.seed_score = box, score
    s.draft_mask = mask
    return refresh()


def propagate(progress_=gr.Progress()):
    s = SESSION
    if s.draft is None or s.draft.seed_box is None:
        return refresh()
    s.draft.boxes = propagate_draft(s.draft,
                                    progress=lambda f, m: progress_(f, desc=m))
    return refresh()


def save_track():
    s = SESSION
    if s.is_view_only() and s.refuse(
            "car-only sequence — a static car at 5 px is not annotatable here"):
        return refresh()
    if s.draft is None:
        return refresh()
    key = save_draft(s.draft, s.decisions)
    s.selected = key or s.selected
    s.draft = None
    s.draft_mask = None
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
                    grid = gr.Image(label=None, show_label=False, interactive=False,
                                    height=760)
                  with gr.Tab("whole frame", id="frame"):
                    frame = gr.Image(label=None, show_label=False, interactive=False,
                                     height=760)
                  with gr.Tab("annotate — add an object that is not there", id="annotate"):
                      canvas = gr.Image(label=None, show_label=False,
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
                          b_save_track = gr.Button("Save as track", variant="primary")
                          b_undo_prompt = gr.Button("Undo prompt")
                          b_cancel = gr.Button("Cancel")
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
                fix = gr.Image(label="selected object — click two opposite corners",
                               interactive=False, height=420)
                with gr.Row():
                    b_snap = gr.Button("Snap with SAM 3", elem_id="k_snap")
                    b_save = gr.Button("Save this frame", variant="primary")
                    b_undo = gr.Button("Undo", elem_id="k_undo")
                with gr.Row():
                    b_delete = gr.Button("Delete / restore this track")
                    show_raw = gr.Checkbox(False, label="show geometry before SAM 3",
                                           container=False)
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

        out = [header, info, grid, frame, fix, canvas, slider, picker, video,
               tabs, tags]
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
        b_prop.click(propagate, outputs=out)
        b_save_track.click(save_track, outputs=out)
        b_undo_prompt.click(undo_prompt, outputs=out)
        b_cancel.click(cancel_draft, outputs=out)
        click_mode.input(set_click_mode, inputs=click_mode, outputs=out)
        show_raw.input(toggle_raw, inputs=show_raw, outputs=out)
        b_snap.click(snap_sam3, outputs=out)
        b_save.click(save_fix, outputs=out)
        b_undo.click(undo_fix, outputs=out)
        b_delete.click(delete_track, outputs=out)
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
                    help="airmot satmtb viso sdmcar rscardata")
    ap.add_argument("--modes", nargs="*", default=None,
                    choices=[CHECK, ANNOTATE, VIEW_ONLY])
    ap.add_argument("--no-watch-required", action="store_true",
                    help="allow accepting a sequence that has not been played "
                         "back; off by default, because the sign-off is the "
                         "watching")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    SESSION = Session(args)
    print(f"{len(SESSION.items)} sequences in the queue, "
          f"resuming at {SESSION.seq_id} ({SESSION.index + 1})")
    if not MERGED_ROOT.is_dir():
        print(f"[warn] {MERGED_ROOT} not found — SAT-MTB will show its raw, "
              f"movers-only ground truth")
    # Playback is written outside Gradio's own temp tree, and Gradio refuses to
    # serve a file it has not been told about — the video would render fine and
    # then fail to load in the browser.
    build_ui(args).launch(server_name="0.0.0.0", server_port=args.port,
                          share=args.share, js=HOTKEY_JS,
                          allowed_paths=[str(VIDEO_CACHE)])


if __name__ == "__main__":
    main()
