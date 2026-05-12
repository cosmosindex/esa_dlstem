# Space-tracker

Unified satellite-video tracking benchmark. Two parallel surfaces:

| Task | Manifest | Datasets | API |
|---|---|---|---|
| **SOT** | [`space_tracker.json`](space_tracker.json) — 463 seqs | SatSOT, SV248S, OOTB | [`Benchmark`](#sot--quick-start) |
| **MOT** | [`space_tracker_mot.json`](space_tracker_mot.json) — 491 seqs | AIRMOT, SAT-MTB, VISO (non-car), SDM-Car, RsCarData | [`MOTBenchmark`](#mot--quick-start) |

This directory ships:

- **`space_tracker.json`** — SOT manifest cataloguing 463 sequences (relative paths to imagery / GT, native + unified + full-taxonomy attribute labels, sequence-level scale stats, tiny flag). ~260 KB; safe to commit and diff. Pinned by Croissant metadata — see `croissant.json`.
- **`space_tracker_mot.json`** — MOT manifest cataloguing 491 sequences (relative paths, per-dataset native GT format tag, image-source mode, resolution, track count, official split). ~280 KB.
- **`manifest.py`** / **`manifest_mot.py`** — JSON loaders + filtering APIs.
- **`data.py`** / **`data_mot.py`** — per-frame loaders bridging the manifest to on-disk imagery and GT for each source dataset.
- **`metrics.py`** — SOT-side per-sequence SR / NPR / PR / P@5 (PR over CLE ∈ [0, 50] px; NPR over normalised CLE ∈ [0, 0.5]). MOT-side scoring is not bundled — feed predictions into TrackEval / py-motmetrics.
- **`benchmark.py`** / **`benchmark_mot.py`** — high-level `Benchmark` and `MOTBenchmark` classes. Most users only touch these.

## Conventions

Neither manifest ships raw imagery or GT — readers are expected to download each source dataset themselves under its original license, the same way LaSOT and GOT-10k handle it. See [`../DATASETS.md`](../DATASETS.md) for direct download links.

### SOT attribute taxonomy (two layers)

1. **Unified attributes** (6 shared rows): `BC`, `IV`, `ROT`, `OCC`, `SOB`, `DEF`. These consolidate per-dataset native labels (e.g. OCC = SatSOT \{POC, FOC\} ∪ SV248S \{STO, LTO, CO\} ∪ OOTB \{PO, FO\}). Mapping lives in `manifest['unified_attributes']`; per-sequence labels are in `seq.unified_attrs`.
2. **Full paper taxonomy** (23 rows total, mirrors `split_attributes_table.tex`): the 6 unified rows plus aspect-ratio (`ARC`, `OON`), 10 dataset-unique-other (`LQ`, `BJT`, `BCH`, `ND`, `IBG`, `SM`, `LT`, `MB`, `IM`, `AM`), and 5 occlusion sub-types (`POC`, `FOC`, `STO`, `LTO`, `CO`). Mapping lives in `manifest['attribute_taxonomy']`; per-sequence labels are in `seq.taxonomy_attrs`.

The two layers are consistent: any sequence with an occlusion sub-type also carries `OCC` in its unified set, and the unified set is always a subset of the taxonomy set. Use `unified_attrs` for cross-dataset headline numbers; use `taxonomy_attrs` to drill down (e.g. OCC → POC vs. FOC vs. STO/LTO/CO).

### MOT class taxonomy

Four unified categories: `car`, `airplane`, `ship`, `train`. VISO's `plane` directory is folded into `airplane`. VISO's `car` subset is **excluded** from `space_tracker_mot.json` because it is re-annotated and shipped as RsCarData under the HiEUM (TPAMI'24) protocol — loading both would double-count. SAT-MTB sequences that mix multiple coarse classes are tagged `category="mixed"` with the actual class set in `categories_in_seq`.

### Aggregation

- **SOT**: per-sequence — each sequence contributes one curve per metric, and the headline number is the equal-weight mean across sequences. This matches OTB / LaSOT / GOT-10k / TrackingNet / OOTB. Cross-dataset unified-attribute scores additionally arithmetic-mean across the datasets that annotate the attribute.
- **MOT**: per-sequence HOTA / MOTA / IDF1 / DetA / AssA at IoU≥0.5. Scoring is delegated to TrackEval / py-motmetrics — `MOTBenchmark` only emits predictions in the right shape.

## SOT — Quick start

```python
from space_tracker import Benchmark

bench = Benchmark.load(
    manifest="space_tracker/space_tracker.json",
    dataset_roots={
        "ootb":   "/data/OOTB",
        "satsot": "/data/SatSOT",
        "sv248s": "/data/SV248S",
    },
)

# 1a. Iterate over a slice (e.g. all tiny sequences with the OCC attribute).
for seq in bench.filter(unified_attrs=["OCC"], tiny=True):
    for frame in bench.frames(seq):
        if not frame.visible:
            continue
        # frame.image_path  — pathlib.Path
        # frame.gt_box_xyxy — np.ndarray (4,) xyxy
        # frame.gt_obb_8pt  — np.ndarray (8,) corners (OOTB only)
        ...

# 1b. Drill into a unified row using its sub-types (filter via taxonomy_attrs).
for sub in bench.manifest.occlusion_subtypes():     # ["POC","FOC","STO","LTO","CO"]
    seqs = bench.filter(taxonomy_attrs=[sub])
    print(sub, "→", len(seqs), "sequences")

# 1c. Filter by a dataset-unique paper-taxonomy attribute (e.g. SV248S IBG).
for seq in bench.filter(taxonomy_attrs=["IBG"]):
    ...

# 2. Run a tracker and score it.
def my_tracker(frames_iter, init_box):
    """One prediction per visible-GT frame; yield None for misses."""
    for frame in frames_iter:
        yield init_box   # naive baseline: report frame-0 GT every frame

result = bench.evaluate(my_tracker, unified_attrs=["OCC", "SOB"])
print(result.summary())
```

`result.overall`, `result.per_dataset`, `result.per_unified_attribute`, and `result.per_sequence` are dicts of `{n_sequences, n_frames, SR, NPR, PR, P@5}`.

## MOT — Quick start

```python
from space_tracker import MOTBenchmark

bench = MOTBenchmark.load(
    manifest="space_tracker/space_tracker_mot.json",
    dataset_roots={
        "airmot":    "/data/AIR-MOT-100",
        "satmtb":    "/data/SAT-MTB",
        "viso":      "/data/VISO",
        "sdmcar":    "/data/SDM-Car",
        "rscardata": "/data/RsCarData",
    },
)

# 1a. Filter sequences. `categories` matches against the set of classes
# actually present in the sequence (categories_in_seq), so a SAT-MTB
# sequence whose primary class is airplane but which also contains cars
# WILL be selected when you ask for ["car"].
for seq in bench.filter(categories=["car"], splits=["test"]):
    for frame in bench.frames(seq):
        # frame.frame_id   — int (native frame id; 0-indexed for SDM-Car, 1-indexed elsewhere)
        # frame.image      — np.ndarray HxWxC uint8 RGB
        # frame.image_path — pathlib.Path (frames mode) or None (SDM-Car video mode)
        # frame.objects    — list[MOTObject(track_id, category, bbox_xyxy)]
        ...

# 1b. Slice by dataset / split independently.
satmtb_val = bench.filter(datasets=["satmtb"], splits=["val"])
all_test   = bench.filter(splits=["test"])

# 2. Collect predictions from your tracker. The contract: yield exactly
# seq.n_frames lists, one per frame in capture order. Each list contains
# (track_id, category, bbox_xyxy) tuples.
import numpy as np

def my_tracker(frames_iter, image_size):
    width, height = image_size
    for frame in frames_iter:
        # Toy "echo the GT" baseline:
        yield [(o.track_id, o.category, o.bbox_xyxy) for o in frame.objects]

preds = bench.run(my_tracker, categories=["car"], splits=["test"])
# preds[seq_id] = list[ list[ (track_id, category, np.ndarray) ] ] — one inner list per frame.
```

Scoring is intentionally out-of-scope for `MOTBenchmark`. Convert `preds` to MOT-Challenge txt or COCO-MOT JSON, then run [TrackEval](https://github.com/JonathonLuiten/TrackEval) (HOTA family) or `py-motmetrics` (CLEAR / IDF1). The manifest's `evaluation` block names the metrics + IoU threshold the paper reports.

### MOT image-source modes

| Dataset | `image_format` | Where pixels come from |
|---|---|---|
| AIRMOT, SAT-MTB, VISO, RsCarData | `frames` | One JPEG/PNG per frame; iter globs the image directory. |
| SDM-Car | `video` | One `.avi` per sequence; iter walks the video via OpenCV `VideoCapture`. Random access is slow — pre-extract frames offline if you do many runs. |

### MOT ground-truth formats

Six on-disk formats are dispatched by `seq.gt_format`. The loader handles them all transparently — listed here for reference:

| `gt_format` | Dataset | Schema |
|---|---|---|
| `mot_csv_9col`             | AIRMOT       | `frame, track, x, y, w, h, conf, cls, vis` (xywh) |
| `mot_csv_11col`            | SAT-MTB      | `frame, track, x, y, w, h, conf, cls_id, r1, r2, r3` (xywh) |
| `viso_dual`                | VISO         | car/train: comma+xywh; plane/ship: space+xyxy (auto-detected) |
| `mot_csv_10col_0idx`       | SDM-Car      | `frame, track, x, y, w, h, -1, -1, -1, -1` (xywh, **0-indexed frames**) |
| `coco_mot_json`            | RsCarData (train/val) | one COCO-MOT JSON per split; bbox is xywh |
| `pascal_voc_xml_per_frame` | RsCarData (test) | HiEUM re-curated XML, one file per frame (xyxy) — used preferentially over the COCO JSON for the test split |

## Reproducing the paper numbers

- **SOT.** `tools/reaggregate_sot_per_sequence.py` re-aggregates existing `per_image_metrics.json` files using the per-sequence protocol — useful if you have already run the seven trackers reported in the paper and want to recompute the headline numbers without re-running inference. `tools/sot_unified_attribute_table.py` produces the unified-attribute breakdown CSVs.
- **MOT.** `tools/compute_hota.py` computes HOTA / MOTA / IDF1 over predictions persisted to disk; the `MOT_<date>/<tracker>/` experiment layout is documented in the repo's main `README.md`.

## Citation

If you use Space-tracker, please cite the paper (see top-level repo for the BibTeX entry).
