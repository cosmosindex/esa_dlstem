# Video-level ground-truth review

One video at a time, all 491 space-tracker MOT sequences, in a fixed order.

```
CUDA_VISIBLE_DEVICES=0 python -m interactive_review.app
CUDA_VISIBLE_DEVICES=0 python -m interactive_review.app --datasets satmtb --modes check
```

Whether an object *belongs* in the ground truth is not asked here —
`tools/merge_det_to_mot.py` already restored SAT-MTB's 842 missing static tracks,
unattended. What is asked is whether the boxes are right, and the unit of sign-off
is a whole video.

## What each video asks for

| mode | sequences | question |
|---|---|---|
| `check` | 223 (SAT-MTB 154 merged, AIR-MOT 69 already all-object) | are the boxes right? |
| `annotate` | 9 (VISO non-car) | movers-only GT, no second source — draw the static objects with SAM 3 |
| `view_only` | 259 (all-car) | watch and sign off; static cars are 4.9–6.5 px and out of scope |

## The three surfaces

| view | what it is for |
|---|---|
| **grid** | one zoomed tile per object on this frame, all at the same tile size. The inspection surface — a wrong box is obvious here and invisible in the whole frame. |
| **whole frame** | navigation, and "is something missing". |
| **playback** | the sequence as video: flicker, drift, duplicate tracks. |

Colour is provenance, not category: 🟩 shipped with the dataset · 🟧 recovered from
detection XML · 🟠 hole-filled · 🟦 corrected by hand · 🟪 drawn by hand.

## The loop

Scan the grid → fix what is wrong → `P` play it back → `A` accept → next video.

`←` `→` frame (`shift` for ±10) · `[` `]` video · `S` snap the selected box with
SAM 3 · `U` undo this frame's correction · `F` flag.

Click a tile to select an object, then click two opposite corners in the zoom panel
and save. Corrections are per frame: fixing frame 41 of a 283-frame track leaves the
other 282 exactly as they were.

**Accepting requires having watched it.** `A` is refused until the sequence has
been played back, because a box that drifts over 200 frames is invisible in every
still view in this tool. (`--no-watch-required` turns the check off.) Acceptance is
also deliberately fragile: any edit clears both the sign-off and the "watched"
flag, so it always refers to the annotation that was actually played.

`⚑ Flag` parks a sequence instead of finishing it — `⚑ next flagged` walks back
through them. The video picker marks every sequence: `✅` accepted, `⚑` flagged,
`·` untouched, `✎` edited.

`view_only` sequences refuse geometry edits and new tracks outright, rather than
relying on the reviewer to remember that car is out of scope.

## Files

```
app.py                 Gradio UI, queue navigation, hotkeys
core/gtsource.py       the completed ground truth (raw / merged / reviewed), with provenance
core/vqueue.py         the 491-sequence work list and its three modes
core/vdecisions.py     per-sequence corrections and sign-off, one JSON
core/vrender.py        grid / frame / fix views
core/video.py          whole-sequence playback
core/sam3refine.py     SAM 3 box refinement and propagation
core/annotate.py       click-to-segment annotation (used by `annotate` mode)
```

Decisions land in `docs/annotation_review/review.json`. Raw dataset files are never
written.
