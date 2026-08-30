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
benchmark are under `8×8` px. Splits are scene-disjoint and stratified,
70 / 10 / 20 within each half.

Unlike the earlier manifest-only release, the benchmark now ships as a **single
self-contained package**: frames, ground truth in one format per task, and
COCO-VID annotations. Nothing has to be assembled from the seven sources.

**→ [`space_tracker/README.md`](space_tracker/README.md) — download, layout,
annotation format, and the loading API.**

## Getting started

```bash
export SPACE_TRACKER_ROOT=/path/to/space_tracker    # where you unpacked the download
python -m space_tracker --verify                    # check the package is complete
```

```python
from space_tracker import SpaceTracker

st = SpaceTracker()                       # reads $SPACE_TRACKER_ROOT

# SOT — one target, one row per frame, selectable by attribute and by size.
for seq in st.sot.filter(split="test", attribute="OCC"):
    box = seq.init_box                    # what a single-object tracker is initialised with
    for f in seq.frames(visible_only=True):
        f.image_path, f.box               # pathlib.Path, (x, y, w, h) in absolute pixels

# MOT — every object in the field of view is annotated.
for seq in st.mot.filter(split="test", contains="ship"):
    for frame_id, objects in seq.frames():
        for o in objects:
            o.track_id, o.category, o.box
    small = [t for t in seq.tracks.values() if t.is_small]
```

The loading code needs nothing beyond the standard library. Reading frames needs
Pillow or OpenCV; `st.mot.load_coco()` needs `pycocotools`. **TrackEval** reads
`mot/<class>/` as a MOTChallenge benchmark directory with no conversion.

## What is in this repository

| | |
|---|---|
| [`space_tracker/`](space_tracker/README.md) | the benchmark: download instructions, annotation format, loading API |
| [`interactive_review/`](interactive_review/) | the annotation tool — SAM 3 in the loop, per-sequence audit trail — used to build the MOT half and offered for extending it |
| [`evaluation/`](evaluation/), [`tools/`](tools/) | the evaluated trackers and the scripts that produce the paper's tables |
| [`DATASETS.md`](DATASETS.md) | where each of the seven source datasets comes from |

## Attribute showcase

Ground truth plus all seven SOT trackers overlaid frame by frame, on three
released sequences that between them cover background clutter, illumination
variation, occlusion, similar distractor objects, low texture and motion blur.
Each is named by its released id, with the source sequence it came from.

**`car_ootb_0033`** (`ootb/car_39`) — Background Clutter · Illumination Variation · Occlusion · Similar Object · Less Texture · Isotropic Motion — median target scale 8.6 px

![car_ootb_0033 — BC · IV · OCC · SOB · LT · IM](docs/figures/attributes_videos/car_39_combined_trackers.gif)

**`car_ootb_0040`** (`ootb/car_45`) — Background Clutter · Illumination Variation · Motion Blur — 11.2 px

![car_ootb_0040 — BC · IV · MB](docs/figures/attributes_videos/car_45_combined_trackers.gif)

**`ship_ootb_0069`** (`ootb/ship_4`) — Background Clutter · Illumination Variation · Less Texture · Motion Blur — 17.0 px

![ship_ootb_0069 — BC · IV · LT · MB](docs/figures/attributes_videos/ship_4_combined_trackers.gif)

## The SOT attribute taxonomy

18 attributes, consolidated from the 33 native labels of the three SOT sources:
six **shared** rows annotated by at least two sources (`BC`, `IV`, `ROT`, `OCC`,
`SOB`, `DEF` — e.g. `OCC` merges SatSOT's `POC`/`FOC`, SV248S's `STO`/`LTO`/`CO`
and OOTB's `PO`/`FO`), two aspect-ratio attributes (`ARC`, `OON`), and ten
dataset-unique ones. Each sequence carries its native labels, its six unified
attributes and its full taxonomy list. The mapping is in
[`space_tracker/README.md`](space_tracker/README.md#6-the-sot-attribute-taxonomy)
and, machine-readable, in `st.sot.attribute_taxonomy`.

## Reproducing the paper numbers

- **SOT** — `tools/reaggregate_sot_per_sequence.py` recomputes the headline SR/NPR/PR/P@5 numbers from existing `per_image_metrics.json` files. `tools/sot_unified_attribute_table.py` produces the unified-attribute breakdown CSVs.
- **MOT** — `compute_hota.py` computes HOTA / MOTA / IDF1 over predictions persisted to disk; the `MOT_<date>/<tracker>/` experiment layout is documented inside that script.

## Citation

If you use Space-Tracker, please cite the paper (BibTeX entry forthcoming with the camera-ready release).
