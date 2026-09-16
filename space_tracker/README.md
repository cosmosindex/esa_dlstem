# Space-Tracker — dataset manual

A detection and tracking benchmark for **small objects in spaceborne video**, in
two parts — single-object (SOT) and multi-object (MOT) — consolidated from seven
source datasets, re-annotated where the sources were incomplete, and rewritten
into one annotation format per task.

|     | sequences | frames  | tracks | boxes     | classes |
|-----|----------:|--------:|-------:|----------:|---------|
| SOT |       395 | 198,481 |    395 |   185,376 | car, car-large, airplane, ship, train |
| MOT |       403 |  93,207 | 29,367 | 3,570,015 | car, airplane, ship, train |

MOT class folders: car 237, mixed 69, ship 61, airplane 35, train 1.
MOT boxes by class: car 3,251,921 · airplane 200,803 · ship 115,523 · train 1,768.

Every SOT target and 97.7 % of MOT tracks have a median `sqrt(area)` of at most
32 px; 74.9 % of all boxes are under `8×8` px.

This directory is the loading code. It reads the released package and nothing
else — no source dataset is needed, and no field is recomputed at read time, so
what you get is exactly what the package ships.

---

## 1. Download

The benchmark is one directory holding both parts. It is distributed as two
archives, which unpack into the same root:

| archive | unpacked | contents |
|---|---:|---|
| `space_tracker_sot.tar` | ≈ 58 GB | `sot/` — 395 sequences, frames + ground truth + COCO-VID annotations |
| `space_tracker_mot.tar` | ≈ 30 GB | `mot/` — 403 sequences, frames + ground truth + COCO-VID annotations |

Either part stands alone; take only the one you need.

> **Where to get them.** Both archives will be published as a **Google Drive**
> folder, with an `md5sums.txt` beside them. The link is withheld while the
> paper is under review and will be filled in here on acceptance; the project
> page, <https://anonymous.4open.science/r/59C2/>, carries the code and the
> annotation tool in the meantime.

Unpack, then point the loader at the directory that holds `sot/` and `mot/`:

```bash
export SPACE_TRACKER_ROOT=/path/to/space_tracker
python -m space_tracker --verify
```

`--verify` stats every ground-truth file and the first and last frame of every
sequence, and exits non-zero if anything the manifests promise is not on disk.
Without it you get the summary alone:

```
Space-Tracker at /path/to/space_tracker
SOT: 395 sequences (train 274, val 50, test 71)
MOT: 403 sequences (train 283, val 40, test 80)
  sot: 198,481 frames; class folders: car 312, car-large 37, ship 36, airplane 9, train 1
  mot: 93,207 frames; class folders: car 237, mixed 69, ship 61, airplane 35, train 1
```

The loading code has **no required dependencies** beyond the standard library.
Decoding a frame needs Pillow or OpenCV; `load_coco()` needs `pycocotools`.
Copy this directory next to your own code, or add the repository root to
`PYTHONPATH`.

---

## 2. What is new here

This is not a re-hosting of seven datasets.

1. **Annotations the sources do not have.** SAT-MTB labels only *moving*
   objects, so its static aircraft and ships were absent; they were recovered
   from its own detection XML and, where nothing existed to recover, drawn by
   hand. SDM-Car and RsCarData annotate cars and nothing else — the aircraft and
   ships visible in the same scenes were never labelled, and now are. Across the
   403 released sequences the audit **adds 1,256 tracks and 256,969 boxes**,
   removes 167 spurious tracks, corrects 9 class labels and re-fits 94,579 boxes
   to a SAM 3 mask.
2. **A class folder that reflects the pixels, not the filename.** Because of (1),
   a source sequence filed under `car/` may well contain ships. Every sequence's
   class is derived from the boxes it actually carries, and **24 sequences moved
   out of the folder their source named them for**. Names were reassigned for the
   same reason: `satmtb/car/01` is an airport apron with 27 aircraft in it, and is
   released as `mixed_satmtb_0052`.
3. **One format per task**, replacing six mutually incompatible MOT formats and
   three SOT ones, plus a COCO-VID JSON written from the same intermediate.
4. **A `motion_state` on every track**, so one ground truth serves two evaluation
   protocols (§6).

---

## 3. Layout

```
space_tracker/                      <- $SPACE_TRACKER_ROOT
  splits.json                       train / val / test, one entry per sequence
  sot/
    <class>/<sequence>/
      img/000001.jpg                frames, 1-indexed, original pixels
      groundtruth.txt               15-column CSV, one row per frame
      seqinfo.ini
    annotations/
      space_tracker_sot.json        COCO-VID, every sequence
      per_class/<class>.json
    space_tracker_sot.json          manifest: sequences, attributes, provenance
  mot/
    <class>/<sequence>/
      img1/000001.png               frames, 1-indexed
      gt/gt.txt                     MOTChallenge CSV
      gt/motion_state.txt           one row per track
      seqinfo.ini
    annotations/
      space_tracker_mot.json        COCO-VID, every sequence
      per_class/<class>.json
    space_tracker_mot.json          manifest
```

`<class>` for MOT is `car`, `airplane`, `ship`, `train`, or `mixed` for a
sequence holding more than one. A sequence name states its class and its source:
`mixed_sdmcar_0030`, `car_rscardata_0002`. The source's own id is kept in the
manifest as `source_sequence_id`.

---

## 4. Quick start

```python
from space_tracker import SpaceTracker

st = SpaceTracker("/path/to/space_tracker")     # or omit, and set $SPACE_TRACKER_ROOT
print(st.summary())
```

### Single-object tracking

One target, one row per frame, `visible=False` where the target has left the
scene.

```python
for seq in st.sot.filter(split="test", attribute="OCC"):
    box = seq.init_box                       # first visible box: what you initialise with
    for f in seq.frames(visible_only=True):
        f.frame_id                           # int, 1-indexed
        f.image_path                         # pathlib.Path
        f.box                                # (x, y, w, h), absolute pixels, top-left origin
        f.box_xyxy                           # the same box as (x1, y1, x2, y2)
        f.polygon                            # oriented corners where the source has them, else None
        img = seq.load_image(f.frame_id)     # HxWx3 uint8 RGB
```

### Multi-object tracking

Every object in the field of view is annotated, large ones included.

```python
seq = st.mot["mixed_sdmcar_0030"]

for frame_id, objects in seq.frames():       # every frame, including empty ones
    for o in objects:
        o.track_id, o.category, o.box        # (x, y, w, h)

seq.objects(95)                              # just one frame
seq.tracks                                   # {track_id: MOTTrack}

small  = [t for t in seq.tracks.values() if t.is_small]
moving = [t for t in seq.tracks.values() if t.motion_state == "moving"]
```

### Selecting sequences

`filter()` takes a string or a list of strings for each argument, and `and`s
them together. Anything it does not cover goes through `predicate`.

```python
st.sot.filter(split="test")                       # train / val / test
st.sot.filter(category="ship")                    # the class folder
st.sot.filter(dataset="ootb")                     # the source dataset
st.sot.filter(attribute=["OCC", "SOB"])           # any of the 18 taxonomy attributes
st.sot.filter(attribute="STO")                    # or an occlusion sub-type
st.sot.filter(unified_attribute="OCC")            # restricted to the 5 pooled rows (§7)
st.sot.filter(max_size=8)                         # median sqrt(area) at most 8 px
st.sot.filter(predicate=lambda s: s.n_frames > 500)

st.mot.filter(split="test", contains="ship")      # what the boxes hold, not the folder name
st.mot.filter(category="mixed")                   # what the folder is called
st.mot.filter(has_small_tracks=True)

st.sot["car_ootb_0001"]        # by name
st.sot[0]                      # by position
st.sequence("mixed_sdmcar_0030")   # by name, either part
len(st.mot), list(st.mot.ids)
```

### COCO and TrackEval

The same boxes are described twice, both written from the same pass, neither
derived at read time.

```python
coco = st.mot.load_coco()                 # pycocotools.COCO over every sequence
coco = st.mot.load_coco("car")            # or one class
raw  = st.mot.load_coco(raw=True)         # parsed JSON, including the video extension
```

In every JSON a box is `[x, y, width, height]` in absolute pixels with a
top-left origin, with `area` and `iscrowd` — the COCO detection contract
unchanged, so `pycocotools` loads these files as they are. What COCO has no
field for follows the video extension that TAO, BDD100K and YouTube-VIS share:
`videos` at the top level, `video_id` + `frame_id` on each image, `track_id` on
each annotation, and a `tracks` table carrying `is_small`,
`median_sqrt_area_px` and `motion_state`.

The per-sequence files are the benchmark convention. **TrackEval reads
`mot/<class>/` as a MOTChallenge benchmark directory directly** — no conversion.
SOT ships `<seq>/groundtruth.txt` + `img/` the way OTB and LaSOT do.

---

## 5. Annotation format

### `sot/<class>/<seq>/groundtruth.txt` — 15 columns

`frame_id, x, y, w, h, visible, state, px1, py1, px2, py2, px3, py3, px4, py4`

One row per frame. All-`-1` geometry with `visible = 0` means the target is
absent, which unifies SatSOT's literal `none` with SV248S's separate state file.
`state` keeps SV248S's native flag (0 visible, 1 invisible, 2 occluded) and is
`-1` elsewhere, because "occluded but present" is a distinction `visible`
deliberately collapses. `px1..py4` are oriented-box corners, `-1` where the
source annotates no orientation; rows that carry a polygon carry its
axis-aligned box alongside, and the polygon is also the COCO `segmentation`.

### `mot/<class>/<seq>/gt/gt.txt` — MOTChallenge, 9 columns

`frame, id, x, y, w, h, conf, class, visibility`

`conf` and `visibility` are `1` on every row: the sources annotate neither, and
the columns are kept so that MOTChallenge parsers read the file unmodified.
Class ids are `1 = car, 2 = airplane, 3 = ship, 4 = train`.

### `mot/<class>/<seq>/gt/motion_state.txt` — one row per track

`track_id, motion_state, moving_fraction, median_speed_mps, centre_span_over_size`

`motion_state` is `moving`, `intermittent`, `static` or `unknown`. The same
values are in the COCO-VID `tracks` table. See §6.

### `seqinfo.ini`

MOTChallenge's own keys, plus the two constants it has no field for:

```ini
[Sequence]
name = mixed_satmtb_0001
imDir = img1
frameRate = 10
seqLength = 93
imWidth = 1080
imHeight = 1080
imExt = .png
gsd = 0.92
platform = Jilin-1 03
category = mixed
sourceDataset = satmtb
```

`gsd` is the ground sample distance in metres and `frameRate` the acquisition
rate; together they convert a displacement in pixels per frame into one in
metres per second, which is how `motion_state` was assigned. Both are published
per sequence, each with a provenance tag in the manifest — `paper` (stated by
the source publication), `derived` (measured from the distributed files, as
SDM-Car's rate was read out of its AVI containers), `inherited` (taken from a
matched parent scene) or `unverified`. **Nothing is guessed.** All 403 MOT
sequences carry both. On the SOT side 246 do; the remaining 149 come from OOTB
and SatSOT, which mix imaging platforms without saying which filmed what, so
their `frameRate` stays at MOTChallenge's `-1` and `gsd` is omitted.

```python
seq.gsd_m, seq.fps, seq.platform    # None where the source publishes none
seq.acquisition                     # the same, with each value's provenance
```

### Conventions

* **Everything is 1-indexed** — frame ids, image basenames, track ids, category
  ids — in both parts.

---

## 6. Motion state and the two MOT protocols

Exhaustive annotation creates a problem on the metric side: a stationary object
is a trivial association problem, so scoring static tracks raises MOTA and IDF1
for reasons unrelated to a tracker's ability to associate, and numbers computed
on the enriched annotation would not be comparable with results published on the
original moving-only labels.

Every track therefore carries a `motion_state`, and one ground truth serves two
protocols: **full**, over all tracks, and **moving-only**, which drops the static
ones and reproduces the earlier convention. No second ground truth is stored;
moving-only is a filter.

```python
full   = list(seq.tracks.values())
moving = [t for t in seq.tracks.values() if t.motion_state != "static"]
```

The flag is computed from ground speed in metres per second rather than pixels
per frame, because the sources differ in ground sample distance and frame rate
and a pixel threshold would denote a different physical speed in each. A track
is **static** when its box centre never leaves the object's own footprint *and*
its median speed stays below 1 m/s; **moving** when at least 95 % of ten-frame
windows exceed that speed; **intermittent** in between — a vehicle that stops, a
vessel that berths; **unknown** for a single-box track, which carries no
displacement to measure. Both conditions are required because either alone
misfires: speed alone labels four fifths of parked aircraft *intermittent*,
since a stationary box wobbles; displacement alone reads a moving train as
static, a long thin object travelling its own length barely leaving its
footprint.

Static tracks hold 4.4 % of all boxes and occur in 116 of the 403 sequences.
`tools/add_motion_state.py` recomputes the flags from the release.

---

## 7. The SOT attribute taxonomy

18 attributes, consolidated from the 33 native labels of the three SOT sources.
**Every one of the 18 is annotated on the released sequences and every one is
filterable**, alongside the five sub-types that drill into the `OCC` row:

```python
st.sot.attributes            # the 18 names
st.sot.pooled_attributes     # the 5 that more than one source annotates
st.sot.occlusion_subtypes    # POC, FOC, STO, LTO, CO
st.sot.attribute_taxonomy    # each one's group, full name, definition, source labels, pooled flag
```

The taxonomy has three groups. The distinction between them is about **how a row
is scored, never about what is attached to a sequence**:

| group | attributes | how it is reported |
|---|---|---|
| **pooled** (5) | `SOB` 236 · `ROT` 176 · `OCC` 135 · `IV` 119 · `BC` 86 | more than one source annotates it, so the row is scored over every source that does — an attribute-level comparison across datasets, which no single source supports |
| **single-source** (13) | `BCH` 91 · `IBG` 76 · `SM` 58 · `LT` 37 · `MB` 29 · `AM` 16 · `ND` 15 · `LQ` 13 · `BJT` 13 · `IM` 12 · `OON` 11 · `DEF` 4 · `ARC` 1 | exactly one source annotates it among the released sequences, so it is reported on that source alone — evaluated, not discarded |
| **occlusion sub-types** (5) | `POC` 38 · `FOC` 20 · `STO` 79 · `LTO` 9 · `CO` 34 | a drill-down into the pooled `OCC` row, **not** extra rows of the 18 |

Counts are released sequences carrying the attribute. `DEF` and `ARC` are each
defined by two source datasets, but the 32 px size criterion (§8) leaves DEF in
OOTB alone and ARC in SatSOT alone, which is why neither can be pooled here.

Each sequence carries three views of its labels, and the first two are complete:

| field | what it is |
|---|---|
| `seq.native_attrs` | the source dataset's own labels, unchanged |
| `seq.taxonomy_attrs` | **every** taxonomy attribute it carries, plus any occlusion sub-type — the full list |
| `seq.pooled_attrs` | the subset of those that are pooled (alias: `seq.unified_attrs`) |

`filter(attribute=...)` reaches all 23 rows; `filter(unified_attribute=...)` is
the restricted form and rejects a non-pooled name rather than silently returning
nothing.

Merging by name is not merging by meaning, so each entry in
`attribute_taxonomy` reproduces the source labels it was built from and any
numeric criterion the source publishes — only SV248S publishes any. Where an
attribute reduces to a quantity measurable identically on every source, the
merge was verified by measuring it: rotation does, and OOTB's labels behave like
a 15° threshold where SV248S's behave like 35°.

---

## 8. Scope: what "small" means

An object is small when the **median of `sqrt(area)` over its own track** is at
most 32 px. A median, not a single frame, because one frame's box is too noisy
to judge on. Size is measured on the axis-aligned box — the same `area` the COCO
files carry — so this selection is reproducible from the package alone.
Measuring OOTB on its oriented polygon instead would judge it on a smaller
footprint than the two sources that annotate no orientation.

* **SOT** keeps a sequence when its single target is small. 68 sequences were
  dropped for being too large: 35 trains, 31 aircraft, 2 ships. Filter further
  with `st.sot.filter(max_size=...)`.
* **MOT** keeps a sequence when *any* track in it is small, and then keeps every
  track in that sequence annotated, large ones included. An unlabelled object
  sitting in an annotated frame would cost every method a false positive it does
  not deserve. Use `track.is_small` to cut a strictly-small subset.

---

## 9. Splits

Scene-disjoint, stratified, 70 / 10 / 20 within each part:

|     | scene groups | train | val | test |
|-----|-------------:|------:|----:|-----:|
| SOT |           99 |   274 |  50 |   71 |
| MOT |          254 |   283 |  40 |   80 |

The unit of the split is the **parent satellite scene**, not the sequence: many
sequences are crops of the same overhead capture, and splitting on sequences
would put the same pixels on both sides. Within that constraint the partition is
optimised so that source dataset, object category, object scale, speed and — on
the SOT part — each attribute land in the target proportion at once. The
procedure is deterministic and reproduces the same split on every run.

`splits.json` ships with this code, so `seq.split` is populated before you have
downloaded anything; a `splits.json` placed at the package root takes precedence,
and so does one passed explicitly:

```python
st = SpaceTracker(root, splits="my_splits.json")
```

---

## 10. Cleaning applied at the release boundary

None of this originates in the re-annotation; all of it is in the sources as
shipped, and all of it is recorded in the manifests.

| | |
|---|---|
| 2,342 boxes clipped to the image | the sources keep writing a track after its object has left the frame |
| 2,063 boxes dropped | they lay wholly outside the image; a box over pixels that do not exist is not ground truth |
| 10 tracks dropped | nothing left after the above |
| 24 sequences regrouped | their class folder now matches their boxes |

### A source label error found and corrected

SAT-MTB tags every detection object with both a coarse class (`<name>`) and a
fine one (`<subname>`). Across all 32,953 of its detection XML files the two
agree everywhere except twice:

| sequence | objectID | coarse `<name>` | fine `<subname>` | boxes |
|---|---|---|---|---:|
| `ship_satmtb_0193` (source `ship/51`) | 0011 | `airplane` | **speed_boat** | 326 |
| `ship_satmtb_0200` (source `ship/58`) | 0008 | `train` | **yacht** | 326 |

A speed boat and a yacht are ships. Both objects had been carried into the
ground truth under the wrong coarse class, and both were independently flagged
by a size check — at 9.8 px and 8.9 px they sat below every other aircraft and
train track in the benchmark. Both were reviewed on the video and corrected to
`ship`. Geometry is untouched; only the class changed. The correction is a rule,
not a patch: where a detection object's coarse class contradicts the class its
own fine label belongs to, the fine label wins.

---

## 11. Sources and licensing

SOT: OOTB, SatSOT, SV248S. MOT: SAT-MTB, VISO (non-car), SDM-Car, RsCarData.
VISO's car subset is excluded because RsCarData is that subset re-annotated, and
loading both would double-count it. AIR-MOT is **not** included: no
redistribution licence was obtained, so any result computed on an earlier
internal export that contained it is not comparable to this release.

The benchmark is released under **CC BY-NC-SA 4.0** — the ShareAlike clause on
VISO's imagery is what fixes it, being the strictest term among the seven. The
terms are in [`../LICENSE`](../LICENSE); see [`../DATASETS.md`](../DATASETS.md)
for each source's own licence, where it came from, and what redistributing it
obliges.

---

## 12. API reference

```
SpaceTracker(root=None, splits=None)
  .root .sot .mot .splits
  .summary()                       one line per part
  .sequence(name)                  look up in either part
  ["sot"] / ["mot"]

Half (st.sot, st.mot)
  len() iter() [name] [index] in .get(name)
  .sequences .ids .categories .counts()
  .filter(split=, category=, dataset=, ids=, predicate=, **part-specific)
  .coco_path .per_class_coco_path(cls) .load_coco(cls=None, raw=False)
  st.sot only:  .attributes .pooled_attributes .occlusion_subtypes
                .attribute_taxonomy
                filter(attribute=, unified_attribute=, max_size=, min_size=)
  st.mot only:  filter(contains=, has_small_tracks=)

Sequence (both parts)
  .id .name .dataset .source_sequence_id .category .split
  .platform .gsd_m .fps .acquisition
  .n_frames .width .height .frame_ids
  .dir .gt_path .image_path(frame_id) .load_image(frame_id) .seqinfo .record

SOTSequence
  .n_visible_frames .median_sqrt_area_px
  .native_attrs .taxonomy_attrs .pooled_attrs (= .unified_attrs) .has_attribute(a)
  .groundtruth                     list[SOTFrame], every row
  .frames(visible_only=False)      iterator
  .init_box                        first visible box

MOTSequence
  .categories_in_sequence .n_tracks .n_small_tracks .n_boxes .review
  .objects(frame_id=None)          list[MOTObject], or the whole {frame_id: [...]}
  .objects_by_frame                the whole dict
  .frames()                        (frame_id, objects) for every frame
  .tracks                          {track_id: MOTTrack}

SOTFrame   frame_id image_path visible box box_xyxy state polygon
MOTObject  frame_id track_id box box_xyxy category_id category conf visibility
MOTTrack   track_id category_id category n_boxes is_small median_sqrt_area_px
           motion_state moving_fraction median_speed_mps centre_span_over_size
```

---

## 13. Also in this directory

`release.py` is what the above documents, and it is all a reader of the
benchmark needs. `data/sequence_constants.json` holds the per-sequence
acquisition constants of §5, so the release carries them without depending on
anything upstream.

The remaining modules — `manifest.py`, `data.py`, `manifest_mot.py`,
`data_mot.py`, `benchmark.py`, `benchmark_mot.py`, `metrics.py` — are the
**internal layer**: they index the *source* datasets in their original, mutually
incompatible on-disk formats, and they exist because the annotation tool
(`../annotation_tool/`, which is released as a tool in its own right) and the build scripts
(`../tools/build_space_tracker_release.py` and friends) work upstream of the
release. They are imported lazily, so they cost nothing here.

The two indexes they read, `space_tracker.json` and `space_tracker_mot.json`,
are **not published**: their derived fields describe the source datasets rather
than this benchmark, and their sequence counts (463 SOT, 491 MOT) are not the
benchmark's. Rebuild them from the sources with
`../tools/build_space_tracker_manifest.py` and
`../tools/build_space_tracker_mot_manifest.py` if you need to re-run the build
pipeline. Nothing in the released package depends on them.

---

## 14. Citation

BibTeX will be added with the camera-ready release.
