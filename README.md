# Space-Tracker

A detection and tracking benchmark for **small objects in spaceborne video**, in
two parts — single-object (SOT) and multi-object (MOT) — consolidated from seven
source datasets, re-annotated where the sources were incomplete, and released as
one self-contained package with the frames included.

|     | sequences | frames  | tracks | boxes     | classes |
|-----|----------:|--------:|-------:|----------:|---------|
| SOT |       395 | 198,481 |    395 |   185,376 | car, car-large, airplane, ship, train |
| MOT |       403 |  93,207 | 29,367 | 3,570,015 | car, airplane, ship, train |

Targets are small by construction: every SOT target and 97.7 % of MOT tracks
have a median `sqrt(area)` of at most 32 px, and **74.9 % of all boxes are under
`8×8` px**. Splits are scene-disjoint and stratified, 70 / 10 / 20 within each
part.

## Highlights

- **Exhaustive annotation where the sources are partial.** Satellite video
  benchmarks label moving objects only. We keep that convention for *car*, where
  its physical justification holds, and drop it for *airplane*, *ship* and
  *train*, which are annotated frame by frame irrespective of motion. Sequences
  labelled for one class only are completed for the whole vocabulary.
- **An audit of every MOT sequence.** All 403 were opened and reviewed: **+1,256
  tracks and +256,969 boxes added**, 167 spurious tracks removed, 9 class labels
  corrected, and 94,579 boxes re-fitted to a SAM 3 mask.
- **Three tasks, one protocol.** Because MOT is annotated exhaustively, the same
  data supports single-frame **detection** on the per-frame labels and **MOT** on
  the identity-linked tracks, alongside the SOT part.
- **Two MOT protocols from one ground truth.** Every track carries a
  `motion_state`, so results can be reported *full* or *moving-only* — the latter
  reproduces the earlier convention and stays comparable with published numbers.
  No second ground truth is stored.
- **An 18-attribute SOT taxonomy** aligning the 33 native labels of three source
  benchmarks, five of which more than one source annotates and can therefore be
  scored across datasets rather than within one.
- **The annotation tool is released too.** Point it at a video nothing has ever
  labelled and it returns tracking ground truth, from a handful of prompted
  frames rather than a box on every frame: SAM 3 propagates one prompt into a
  track, and one annotated object becomes a visual exemplar that sweeps the
  sequence for every other instance of it. Nothing it proposes is written until
  a human adopts it. See [`annotation_tool/`](annotation_tool/README.md).

## Where the benchmark sits

| Dataset | Task | #Seq | #Frames | #Labels | <8×8 px | GSD (m) | FPS | #Attr |
|---|---|---:|---:|---:|---:|---|---|---:|
| SatSOT | SOT | 105 | 27,664 | 27,061 | 26.2 % | 0.92–1.1 | 10 / 25 | 11 |
| SV248S | SOT | 248 | 156,621 | 144,119 | 80.3 % | 0.92 | 25 | 10 |
| OOTB | SOT | 110 | 29,890 | 29,890 | 13.7 % | 0.92–1.1 | mixed | 12 |
| VISO | Det, SOT, MOT | 47 | 16,204 | 844,345 | — | 0.92 | 10 | 8 |
| SAT-MTB | Det, MOT, Seg | 249 | 48,390 | 1,118,584 | — | 0.92 | 10 | — |
| RsCarData | Det, MOT | 77 | 29,689 | 848,786 | — | 0.92 | 10 | — |
| SDM-Car | Det | 99 | 16,423 | 1,522,577 | — | 0.75 | 6 / 8 | — |
| **Space-Tracker** | **Det, SOT, MOT** | **798** | **291,688** | **3,755,391** | **74.9 %** | 0.75–1.1 | 6–25 | **18** |

Counts are on the portion used here. RsCarData re-annotates imagery VISO also
contains, so those two rows overlap. AIR-MOT is excluded: no redistribution
licence was obtained for it.

## Getting started

The benchmark ships as one directory holding both parts, frames included —
nothing has to be assembled from the seven sources.

```bash
export SPACE_TRACKER_ROOT=/path/to/space_tracker   # where you unpacked the download
python -m space_tracker --verify                   # check the package is complete
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
    seq.gsd_m, seq.fps                    # metres per pixel, frames per second
    for frame_id, objects in seq.frames():
        for o in objects:
            o.track_id, o.category, o.box
    moving = [t for t in seq.tracks.values() if t.motion_state == "moving"]
```

The loading code needs nothing beyond the standard library. Reading frames needs
Pillow or OpenCV; `st.mot.load_coco()` needs `pycocotools`. **TrackEval** reads
`mot/<class>/` as a MOTChallenge benchmark directory with no conversion.

**→ [`space_tracker/README.md`](space_tracker/README.md) — download, layout,
annotation format, attribute taxonomy and the full loading API.**

To rerun the benchmark rather than only read the release, point the roots at
your own copies; `${DATA_ROOT}` in a config expands at load time, so no config
needs editing per machine.

```bash
export DATA_ROOT=/path/to/data        # data/<dataset>, release/space_tracker, experiments/Detection
export EXPERIMENT_ROOT=/path/to/runs  # where training and evaluation write
export CHECKPOINT_ROOT=/path/to/ckpts # pretrained tracker / detector weights
```

## Benchmark results

### Single-object tracking — 395 sequences

Metrics are averaged per sequence. **Tiny** restricts them to the 238 sequences
below 8 px, where PR saturates and is replaced by P@5 (fraction of frames with
centre error < 5 px). Predictions and OOTB's oriented ground truth are scored as
horizontal boxes, so every method faces one localisation task. `†` marks a
third-party re-implementation or checkpoint.

| Method | SR ↑ | NPR ↑ | PR ↑ | Abst. ↓ | | Tiny SR ↑ | Tiny NPR ↑ | P@5 ↑ |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| *Siamese correlation* ||||| ||||
| SiamRPN++ † | 0.211 | 0.236 | 0.530 | 0.93 | | 0.113 | 0.119 | 0.272 |
| SiamFC † | 0.381 | 0.420 | 0.694 | **0.76** | | 0.328 | 0.354 | **0.602** |
| *Transformer* ||||| ||||
| OSTrack-384 | 0.244 | 0.276 | 0.508 | **0.76** | | 0.118 | 0.126 | 0.283 |
| ODTrack | 0.237 | 0.254 | 0.564 | **0.76** | | 0.126 | 0.119 | 0.332 |
| LoRAT-g378 | **0.390** | 0.453 | 0.589 | 17.86 | | **0.337** | **0.387** | 0.469 |
| *Small-object specific* ||||| ||||
| SmallTrack † | 0.264 | 0.326 | 0.597 | 0.79 | | 0.135 | 0.184 | 0.369 |
| *Video foundation models* ||||| ||||
| SAM 2 | 0.291 | 0.348 | 0.503 | 26.82 | | 0.190 | 0.213 | 0.320 |
| SAMURAI | 0.301 | 0.356 | 0.539 | 9.65 | | 0.201 | 0.224 | 0.345 |
| SAM 3 | 0.383 | **0.466** | **0.704** | 10.83 | | 0.287 | 0.345 | 0.550 |

### Multi-object tracking — non-car test split (airplane, ship), 22 sequences

Every method consumes the same Faster R-CNN detections (mAP₅₀ 0.607) at one
operating point, so the spread is association behaviour. Scoring is
class-agnostic, since the query-based methods emit no category. IoU ≥ 0.5.

| Method | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDsw ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| *Tracking-by-detection* |||||||
| SORT | 0.583 | 0.427 | 0.807 | 0.818 | 0.224 | 0.644 | 156 |
| ByteTrack | 0.584 | 0.439 | 0.788 | 0.818 | 0.270 | 0.647 | 234 |
| OC-SORT | 0.579 | 0.429 | 0.791 | 0.817 | 0.229 | 0.643 | 229 |
| BoT-SORT | 0.577 | 0.430 | 0.785 | 0.818 | 0.228 | 0.635 | 396 |
| BoT-SORT-ReID | 0.583 | 0.428 | 0.806 | 0.817 | 0.222 | 0.645 | 224 |
| *Learned association* |||||||
| TrackTrack | 0.587 | 0.436 | 0.799 | 0.817 | 0.261 | 0.657 | 100 |
| MASA | 0.571 | 0.426 | 0.776 | 0.817 | 0.220 | 0.628 | 274 |
| *Joint detection and tracking* |||||||
| FairMOT | 0.612 | 0.492 | 0.787 | 0.790 | 0.542 | 0.738 | 58 |
| TGraM | 0.562 | 0.438 | 0.736 | 0.795 | 0.469 | 0.676 | 325 |
| *Query-based end-to-end* |||||||
| MOTRv2 | **0.715** | **0.582** | **0.894** | **0.857** | **0.598** | **0.787** | **1** |
| MOTIP | 0.670 | 0.536 | 0.856 | 0.849 | 0.499 | 0.734 | 360 |

### Multi-object tracking — car test split, 48 sequences

Every row consumes the same HiEUM clip detections (mAP₅₀ 0.212). The table has
no JDT or query-based rows because those families cannot be trained on this
part at all: car ground truth annotates moving objects only, so a single-frame
detection stage is shown a parked and a moving vehicle — the same pixels — as
background and as foreground. For the same reason **MOTA is negative
throughout**: correctly finding a stationary car counts as a false positive.
HOTA and its DetA / AssA decomposition are the informative columns.

| Method | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDsw ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| SORT | **0.268** | **0.196** | 0.373 | 0.682 | −0.075 | 0.278 | 3,622 |
| ByteTrack | 0.236 | 0.134 | **0.420** | 0.692 | 0.046 | 0.251 | 300 |
| OC-SORT | 0.221 | 0.193 | 0.257 | 0.681 | −0.062 | 0.218 | 7,874 |
| BoT-SORT | 0.252 | 0.194 | 0.333 | 0.682 | −0.089 | 0.255 | 30,538 |
| BoT-SORT-ReID | 0.267 | 0.178 | 0.405 | 0.687 | 0.015 | **0.293** | 3,416 |
| TrackTrack | 0.198 | 0.103 | 0.385 | **0.705** | **0.059** | 0.216 | **233** |
| MASA | 0.253 | 0.194 | 0.336 | 0.680 | −0.086 | 0.264 | 3,309 |

## What is in this repository

| | |
|---|---|
| [`space_tracker/`](space_tracker/README.md) | the benchmark: download, annotation format, loading API |
| [`annotation_tool/`](annotation_tool/README.md) | **an open annotation tool for tracking** — one prompt becomes a track, one track becomes an exemplar that finds every other instance of it. Built this benchmark's MOT ground truth; released for annotating any video whose cost is the sheer number of similar objects |
| [`evaluation/`](evaluation/) | one runner per benchmarked method. Each names its upstream repository, the commit it was pinned to and every local modification, in its own docstring — the trackers themselves are cloned beside this one, not vendored here |
| [`configs/`](configs/) | the configuration each run was launched with. `<method>_space_tracker*.yaml` evaluates on the released package; the SOT files named after a source dataset are how the published numbers were produced (see below) |
| [`tools/`](tools/) | release build, dataset analysis, and the scripts that produce the paper's tables and figures |
| [`DATASETS.md`](DATASETS.md) | the seven source datasets, their origin and their licences |

## Reproducing the paper numbers

Every table and figure is generated by a script; none is edited by hand.

- **SOT** — `tools/make_wacv_sot_table.py` (headline SR / NPR / PR / P@5),
  `tools/make_wacv_attr_tables.py` (per-attribute breakdown),
  `tools/make_wacv_sot_composition_table.py`, `tools/make_wacv_obb_table.py`
  (oriented-box supplement), `tools/plot_wacv_size_curve.py`.
- **MOT** — `compute_hota.py` computes HOTA / MOTA / IDF1 over predictions
  persisted to disk; the `MOT_<date>/<tracker>/` experiment layout is documented
  inside it. `tools/make_wacv_mot_table.py`, `tools/make_wacv_mot_car_table.py`,
  `tools/make_wacv_car_oracle_table.py`, `tools/make_wacv_det_table.py` and
  `tools/plot_wacv_mot_size_curve.py` turn those results into the tables.
- **Curation** — `tools/make_wacv_curation_tables.py` recomputes the audit and
  curation-matrix counts from the release and the review log.
- **Evaluating on the release** — `configs/SOT/<tracker>_space_tracker.yaml`
  runs a tracker over all 395 SOT sequences as one dataset, straight out of the
  downloaded package. The published SOT numbers came the other way round: each
  tracker was run on OOTB, SatSOT and SV248S in their original layouts
  (`configs/SOT/<tracker>_{ootb,satsot,sv248s}.yaml`) and
  `tools/make_wacv_sot_table.py` aggregated the result over the released
  sequences. The two paths read identical ground truth —
  `tools/check_sot_release_equivalence.py` checks that box by box, and reports
  395/395 sequences and 185,376 boxes identical.
- **MOTRv2 and MOTIP** are the two methods that cannot be imported alongside
  this project — their package names collide with ours — so they are run from
  inside their own clones rather than through a runner in `evaluation/`.
  `tools/export_motrv2.py`, `tools/make_motrv2_det_db.py` and
  `tools/export_motip_val.py` write what they read, and carry the clone
  instructions and the list of local modifications.
- **Release** — `tools/build_space_tracker_release.py` builds the package from
  the seven sources; `tools/verify_space_tracker_release.py` checks it.

## Licence

Both the benchmark and this code are released under **CC BY-NC-SA 4.0** — see
[`LICENSE`](LICENSE) for the terms and for why that licence and no other:
Space-Tracker redistributes imagery from seven sources, so the strictest of
their terms has to hold for the package as a whole. Redistribution permission
was obtained for every source; per-source terms are in
[`DATASETS.md`](DATASETS.md).

## Citation

BibTeX will be added with the camera-ready release.
