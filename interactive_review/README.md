# Video-level ground-truth review

One video at a time, in a fixed order: the 367 space-tracker MOT sequences
that are in scope — licensed for redistribution, and in the small-object half.

```
CUDA_VISIBLE_DEVICES=0 python -m interactive_review.app
CUDA_VISIBLE_DEVICES=0 python -m interactive_review.app --datasets satmtb --modes check
```

Whether an object *belongs* in the ground truth is not asked here —
`tools/merge_det_to_mot.py` already restored SAT-MTB's 842 missing static tracks,
unattended. What is asked is whether the boxes are right, and the unit of sign-off
is a whole video.

## What is in scope

Two rules cut the 491 MOT sequences down to 367. Both hide work; neither deletes
any — every decision already recorded stays in `review.json` and is still
exported.

| rule | drops | why |
|---|---|---|
| licensed for redistribution | AIR-MOT, 69 | no licence was granted, so nothing annotated on it can ship |
| small-object half | SAT-MTB 50, VISO 5 | the results this feeds are about small objects; a sequence is small when its median `sqrt(w·h)` is ≤ 32 px, read from `docs/size_split/size_split.json` so the queue and the size-split experiments cannot drift apart |

`--all-sizes` puts the large half back, `--include-unlicensed` puts AIR-MOT back,
and naming a held-out dataset in `--datasets` overrides its exclusion.

## What each video asks for

The mode is **where the ground truth came from, not a permission**. Annotation is
open in both: the difference is whether there is completed geometry to scrutinise
first, or a movers-only track list to fill.

| mode | sequences | what to expect |
|---|---|---|
| `check` | 104 (SAT-MTB non-car) | `merge_det_to_mot.py` restored 842 static tracks from detection XML — scrutinise that geometry, then add what detection could not see |
| `annotate` | 263 (259 all-car + 4 VISO non-car) | movers-only GT, nothing recovered — draw what is missing with SAM 3 |

All-car sequences are annotatable. They used to be `view_only` — watched, signed
off, never edited — because a parked car is 4.9–6.5 px and not separable from road
texture in one frame. That still holds for parked cars and not for moving ones,
which the movers-only ground truth misses too, so the call is now made per object.
`--car-view-only` puts the old refusal back.

## The three surfaces

| view | what it is for |
|---|---|
| **grid** | one zoomed tile per object on this frame, all at the same tile size. The inspection surface — a wrong box is obvious here and invisible in the whole frame. |
| **whole frame** | navigation, and "is something missing". |
| **playback** | the sequence as video: flicker, drift, duplicate tracks. |

Colour is provenance, not category: 🟩 shipped with the dataset · 🟧 recovered from
detection XML · 🟠 hole-filled · 🟦 corrected by hand · 🟪 drawn by hand.

The grid is single-frame and single-object by construction, so two things it will
never show: an object that is *stuck* (its tile looks the same as a tracked one),
and two identities on one car (the crop removes exactly the relationship). Both
are playback's job — which is why accepting requires having played the video.

## The loop

Scan the grid → fix what is wrong → `P` play it back → `A` accept → next video.

`←` `→` frame (`shift` for ±10) · `[` `]` video · `S` snap the selected box with
SAM 3 · `U` undo this frame's correction · `F` flag · `X` exclude.

Click a tile to select an object, then click two opposite corners in the zoom panel
and save. Corrections are per frame: fixing frame 41 of a 283-frame track leaves the
other 282 exactly as they were.

Three verdicts, kept apart because one Flag button used to carry all three:

| | key | meaning | export |
|---|---|---|---|
| ✅ accept | `A` | ships | in |
| ⚑ flag | `F` | parked, a decision is owed | in, marked `flagged`; `--accepted-only` drops it |
| ✖ exclude | `X` | dropped from the release, and a note is required | **out**, audited in `excluded.json` |

Tags (`static`, `single_object`, `fast_motion`, `occlusion`, `tiny`, …) say what the
video *is*, so they deliberately touch neither `status` nor `watched` — "everything
in it is static" is a property of the imagery, not a claim that anyone reviewed it.

**Accepting requires having watched it.** `A` is refused until the sequence has
been played back, because a box that drifts over 200 frames is invisible in every
still view in this tool. (`--no-watch-required` turns the check off.) Acceptance is
also deliberately fragile: any edit clears both the sign-off and the "watched"
flag, so it always refers to the annotation that was actually played.

`⚑ next flagged` walks back through the parked ones. The video picker marks every
sequence: `✅` accepted, `⚑` flagged, `·` untouched, `✎` edited.

`view_only` sequences refuse geometry edits and new tracks outright, rather than
relying on the reviewer to remember that car is out of scope.

## Drawing an object that has no annotation

`New object here` → click two opposite corners → `Segment` → `Propagate` →
`Save as track`. The canvas zooms 8× onto the last click and draws the **completed**
ground truth underneath, so nobody re-annotates an object the merge already restored.

- **Prompt with a box, not with clicks.** On this imagery a positive click returns
  the whole *road* more often than the car; two corners recover the same cars to
  within a pixel.
- **Propagation runs forward only**, from the frame the object was actually seen on.
  `also track backwards` opts in. A reverse pass fills earlier frames with boxes on
  whatever the tracker latches onto, and those frames look exactly like real ones.
- **A track starts where you segmented it**, not at frame 0.
- **Propagation stops when the object stops moving**, and says which frame it cut at.
  That is correct for a tracker stuck on background and wrong for a genuinely slow
  car — re-anchor on a frame where it is driving.
- `🔍 Find all like this` sweeps the sequence for objects resembling the draft and
  offers them as proposals to accept, score-filter, or discard in bulk.
- `Continue selected track from here` extends an existing identity instead of
  opening a new one.

Every refusal now says which one it is: no draft, one corner placed, empty mask,
nothing to propagate, frozen cut, nothing to save.

## Track cleanup

Five buttons, plus checks that run on every save. Each of them removes only frames
that were **drawn**; a dataset's own frames are never moved, because a merge can
leave a drawn track sharing an id with a dataset track and the deletion list is
keyed by identity rather than by layer.

| button | catches |
|---|---|
| `Merge duplicate tracks` | one object under two ids — and a track that resumes another after a propagation break |
| `Delete static tracks` | objects that never move, in either coordinate frame |
| `Drop oversized boxes` | frames where the SAM 3 mask leaked onto the surroundings |
| `Drop stretches that switched object` | a track that left its object for another one mid-sequence |
| `Trim the parked start` | a car parked for the first N frames, then driving — on the car sets only movers are in scope |
| `Erase deleted tracks I drew` | discards the frames behind tracks already marked deleted |

Merging is deliberately conservative, for reasons that are all measurements on
5-pixel objects:

- **Identity is judged by centre distance in object widths, not by IoU.** A
  one-pixel offset in each axis takes IoU on a 5.5 px box to 0.50. On one sequence
  the true same-car pairs sat at 0.14–0.98 widths and the next 745 pairs began at
  6.8 — while IoU put those same true pairs at 0.30–0.57, indistinguishable from
  the rest.
- **A duplicate is not a switched object.** A duplicate agrees over the *whole*
  range the two tracks share; a switch agrees only over a stretch. Merging a switch
  would assert the earlier frames were the same car too, so the test is the
  *fraction* of shared frames that coincide, never how many.
- **A resumed track shares no frames at all**, so agreement cannot be measured.
  Instead the *seam* is: extrapolate the first track's closing velocity across the
  gap and see whether the second starts where that lands.
- **Only shipped frames confer authority.** Ranking by total frame count let a
  drawn extension outvote the dataset track it was contaminating, so the
  contamination survived every pass.
- **Not-moving has two opposite forms**, and each test clears the other: an object
  parked on the ground travels with the camera, and an object pinned in the image
  appears — ground-relative — to move by the camera's own travel.

Nothing here is applied silently: each button reports what it changed, and pairs of
*dataset* identities that read as one object are listed for a human rather than
merged.

## Files

```
app.py                 Gradio UI, queue navigation, hotkeys
core/gtsource.py       the completed ground truth (raw / merged / reviewed), with provenance
core/vqueue.py         the 367-sequence work list, its scope rules and three modes
core/vdecisions.py     per-sequence corrections and sign-off, one JSON
core/vrender.py        grid / frame / fix views
core/video.py          whole-sequence playback
core/sam3refine.py     SAM 3 box refinement and propagation
core/annotate.py       click-to-segment annotation (used by `annotate` mode)
core/exemplar.py       sweep a sequence for objects resembling the draft
core/merge.py          duplicates, continuations, contaminated stretches
core/motion.py         parked / pinned detection, motion onset
core/sizecheck.py      boxes that departed from a track's own size
core/tracks.py         track-level reads shared by the above
export.py              11-column CSV per sequence + the reviewed manifest
```

Decisions land in `docs/annotation_review/review.json`. Raw dataset files are never
written.

## Running it over days

The app holds `review.json` in memory and rewrites it on every action, so **never
edit that file while the server is up** — the next click puts the in-memory copy
back. Stop the process first, or work on a copy.

Launch it detached (`setsid nohup … & disown`) and it survives the session that
started it; a lost UI is then almost always the SSH tunnel, not the server. Check
with a local request to the port before restarting anything. `pkill -f
interactive_review.app` also matches the shell running it, so kill and relaunch
have to be separate commands.
