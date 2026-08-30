# Space-Tracker

A tracking benchmark for **small objects in satellite video**, in two halves —
single-object (SOT) and multi-object (MOT) — assembled from seven source
datasets, re-annotated where the sources were incomplete, and rewritten into one
annotation format per task.

|     | sequences | frames  | tracks | boxes     | classes |
|-----|----------:|--------:|-------:|----------:|---------|
| SOT |       395 | 198,481 |    395 |   185,376 | car, car-large, airplane, ship, train |
| MOT |       403 |  93,207 | 29,367 | 3,570,015 | car, airplane, ship, train |

Every target is small: **97.7 %** of MOT tracks and **all** SOT targets have a
median `sqrt(area)` of at most 32 px, and **74.9 %** of all boxes in the
benchmark are under `8×8` px.

This directory is the loading code. It reads the released package and nothing
else — no source dataset is needed, and no field is recomputed at read time, so
what you get is exactly what the package ships.

---

## 1. Download

The benchmark is one directory holding two halves. It is distributed as two
archives, which unpack into the same root:

| archive | unpacked | contents |
|---|---:|---|
| `space_tracker_sot.tar` | ≈ 58 GB | `sot/` — 395 sequences, frames + ground truth + COCO-VID annotations |
| `space_tracker_mot.tar` | ≈ 30 GB | `mot/` — 403 sequences, frames + ground truth + COCO-VID annotations |

Either half stands alone; take only the one you need.

> **Where to get them.** The download link is on the project page,
> <https://anonymous.4open.science/r/59C2/> (anonymised for review).

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

## 2. Layout

```
space_tracker/                      <- $SPACE_TRACKER_ROOT
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

## 3. Quick start

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

small = [t for t in seq.tracks.values() if t.is_small]
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
st.sot.filter(unified_attribute="OCC")            # the 6 attributes at least two sources annotate
st.sot.filter(max_size=8)                         # median sqrt(area) at most 8 px
st.sot.filter(predicate=lambda s: s.n_frames > 500)

st.mot.filter(split="test", contains="ship")      # what the boxes hold, not the folder name
st.mot.filter(category="mixed")                   # what the folder is called
st.mot.filter(has_small_tracks=True)

st.sot["car_ootb_0001"]        # by name
st.sot[0]                      # by position
st.sequence("mixed_sdmcar_0030")   # by name, either half
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

## 4. Annotation format

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

`motion_state` is `moving`, `intermittent`, `static` or `unknown`. It supports
the two evaluation protocols the benchmark reports: over all tracks, and over
moving tracks alone. The same values are in the COCO-VID `tracks` table.

### Conventions

* **Everything is 1-indexed** — frame ids, image basenames, track ids, category
  ids — in both halves.
* `seqinfo.ini` reports `frameRate = -1`. None of the seven sources publishes a
  capture rate, and writing `30` would be a number readers might trust.

---

## 5. Splits

Scene-disjoint, stratified, 70 / 10 / 20 within each half:

|     | train | val | test |
|-----|------:|----:|-----:|
| SOT |   274 |  50 |   71 |
| MOT |   283 |  40 |   80 |

The unit of the split is the **parent satellite scene**, not the sequence: many
sequences are crops of the same overhead capture, and splitting on sequences
would put the same pixels on both sides. `splits.json` ships with this code, so
`seq.split` is populated before you have downloaded anything; a `splits.json`
placed at the package root takes precedence, and so does one passed explicitly:

```python
st = SpaceTracker(root, splits="my_splits.json")
```

---

## 6. The SOT attribute taxonomy

18 attributes, consolidated from the 33 native labels of the three SOT sources.
Six are **shared** — annotated by at least two sources and collapsed into one
row (`BC`, `IV`, `ROT`, `OCC`, `SOB`, `DEF`); two are aspect-ratio attributes
(`ARC`, `OON`); ten are **dataset-unique** (`LQ`, `BJT`, `BCH`, `ND`, `IBG`,
`SM`, `LT`, `MB`, `IM`, `AM`).

```python
st.sot.attributes            # the 18 names
st.sot.attribute_taxonomy    # each one's group, full name, definition, source labels
st.sot.occlusion_subtypes    # POC, FOC, STO, LTO, CO
```

Each sequence carries three views of its labels:

| field | what it is |
|---|---|
| `seq.native_attrs` | the source dataset's own labels, unchanged |
| `seq.unified_attrs` | the six shared attributes |
| `seq.taxonomy_attrs` | all 18, plus any occlusion sub-type that applies |

The five sub-types of `OCC` are a drill-down into that one row. They are **not**
extra rows of the taxonomy and should not be counted as such.

---

## 7. Scope: what "small" means

An object is small when the **median of `sqrt(area)` over its own track** is at
most 32 px. A median, not a single frame, because one frame's box is too noisy
to judge on. Size is measured on the axis-aligned box — the same `area` the COCO
files carry — so this selection is reproducible from the package alone.

* **SOT** keeps a sequence when its single target is small. 68 sequences were
  dropped for being too large: 35 trains, 31 aircraft, 2 ships. Filter further
  with `st.sot.filter(max_size=...)`.
* **MOT** keeps a sequence when *any* track in it is small, and then keeps every
  track in that sequence annotated, large ones included. An unlabelled object
  sitting in an annotated frame would cost every method a false positive it does
  not deserve. Use `track.is_small` to cut a strictly-small subset.

---

## 8. Sources and licensing

SOT: OOTB, SatSOT, SV248S. MOT: SAT-MTB, VISO (non-car), SDM-Car, RsCarData.
VISO's car subset is excluded because RsCarData is that subset re-annotated, and
loading both would double-count it. AIR-MOT is **not** included: no
redistribution licence was obtained, so any result computed on an earlier
internal export that contained it is not comparable to this release.

Each source dataset carries its own licence; consult them before redistributing.
See [`../DATASETS.md`](../DATASETS.md) for where each source comes from.

---

## 9. API reference

```
SpaceTracker(root=None, splits=None)
  .root .sot .mot .splits
  .summary()                       one line per half
  .sequence(name)                  look up in either half
  ["sot"] / ["mot"]

Half (st.sot, st.mot)
  len() iter() [name] [index] in .get(name)
  .sequences .ids .categories .counts()
  .filter(split=, category=, dataset=, ids=, predicate=, **half-specific)
  .coco_path .per_class_coco_path(cls) .load_coco(cls=None, raw=False)
  st.sot only:  .attributes .occlusion_subtypes .attribute_taxonomy
                filter(attribute=, unified_attribute=, max_size=, min_size=)
  st.mot only:  filter(contains=, has_small_tracks=)

Sequence (both halves)
  .id .name .dataset .source_sequence_id .category .split
  .n_frames .width .height .frame_ids
  .dir .gt_path .image_path(frame_id) .load_image(frame_id) .seqinfo .record

SOTSequence
  .n_visible_frames .median_sqrt_area_px
  .native_attrs .unified_attrs .taxonomy_attrs .has_attribute(a)
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

## 10. Also in this directory

`release.py` is what the above documents, and it is all a reader of the
benchmark needs.

The remaining modules — `manifest.py`, `data.py`, `manifest_mot.py`,
`data_mot.py`, `benchmark.py`, `benchmark_mot.py`, `metrics.py`, and the two
`space_tracker*.json` indexes beside them — are the **internal layer**: they
index the *source* datasets in their original, mutually incompatible on-disk
formats, and they exist because the annotation tool (`../interactive_review/`)
and the build scripts (`../tools/build_space_tracker_release.py` and friends)
work upstream of the release. They are imported lazily, so they cost nothing
here, but their derived fields describe the source datasets rather than this
benchmark, and their sequence counts (463 SOT, 491 MOT) are **not** the
benchmark's. Do not quote them.

---

## 11. Citation

BibTeX will be added with the camera-ready release.
