# Space-tracker

Unified satellite-video tracking benchmark. Two parallel surfaces:

| Task | Manifest | Datasets | API |
|---|---|---|---|
| **SOT** | [`space_tracker/space_tracker.json`](space_tracker/space_tracker.json) — 463 seqs | SatSOT, SV248S, OOTB | `Benchmark` |
| **MOT** | [`space_tracker/space_tracker_mot.json`](space_tracker/space_tracker_mot.json) — 491 seqs | AIRMOT, SAT-MTB, VISO (non-car), SDM-Car, RsCarData | `MOTBenchmark` |

Both manifests are thin JSON pointers to per-sequence images + native ground-truth files. They do **not** redistribute raw imagery or GT — readers are expected to download each source dataset themselves under its original license, the same way LaSOT and GOT-10k handle it. See [`DATASETS.md`](DATASETS.md) for direct download links to every dataset used in the paper.

Detailed conventions, taxonomies, and quick-start examples are in [`space_tracker/README.md`](space_tracker/README.md).

## SOT — quick start

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

# Sub-select sequences by unified attribute / tiny flag / native taxonomy attr.
for seq in bench.filter(unified_attrs=["OCC"], tiny=True):
    for frame in bench.frames(seq):
        if not frame.visible:
            continue
        # frame.image_path  — pathlib.Path
        # frame.gt_box_xyxy — np.ndarray (4,) xyxy
        # frame.gt_obb_8pt  — np.ndarray (8,) corners (OOTB only)
        ...

def my_tracker(frames_iter, init_box):
    """One prediction per visible-GT frame; yield None for misses."""
    for frame in frames_iter:
        yield init_box

result = bench.evaluate(my_tracker, unified_attrs=["OCC", "SOB"])
print(result.summary())   # SR / NPR / PR / P@5 per dataset, per attribute, overall
```

The 6-row unified attribute taxonomy (`BC`, `IV`, `ROT`, `OCC`, `SOB`, `DEF`) consolidates per-dataset native labels — e.g. `OCC = SatSOT{POC, FOC} ∪ SV248S{STO, LTO, CO} ∪ OOTB{PO, FO}`. A 23-row full paper taxonomy (`taxonomy_attrs`) lets you drill down into the sub-types. See [`space_tracker/README.md`](space_tracker/README.md#sot-attribute-taxonomy-two-layers) for the full mapping.

## MOT — quick start

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

# Filter sequences by dataset / unified category / split.
for seq in bench.filter(categories=["car"], splits=["test"]):
    for frame in bench.frames(seq):
        # frame.frame_id    — int (native frame id; 0-indexed for SDM-Car, 1-indexed elsewhere)
        # frame.image       — np.ndarray HxWxC uint8 RGB
        # frame.objects     — list[MOTObject(track_id, category, bbox_xyxy)]
        ...

def my_tracker(frames_iter, image_size):
    """Yield one list[(track_id, category, bbox_xyxy)] per frame, in order."""
    for frame in frames_iter:
        yield [(o.track_id, o.category, o.bbox_xyxy) for o in frame.objects]

preds = bench.run(my_tracker, categories=["car"], splits=["test"])
# preds[seq_id] = list of per-frame prediction lists.
```

`MOTBenchmark.run` only collects predictions — scoring (HOTA, MOTA, IDF1, DetA, AssA at IoU ≥ 0.5) is delegated to [TrackEval](https://github.com/JonathonLuiten/TrackEval) or `py-motmetrics`. The four unified categories are `car`, `airplane`, `ship`, `train`; VISO's `car` subset is excluded from the MOT manifest because it is re-annotated and shipped as RsCarData under the HiEUM protocol.

## Reproducing the paper numbers

- **SOT** — `tools/reaggregate_sot_per_sequence.py` recomputes the headline SR/NPR/PR/P@5 numbers from existing `per_image_metrics.json` files. `tools/sot_unified_attribute_table.py` produces the unified-attribute breakdown CSVs.
- **MOT** — `tools/compute_hota.py` computes HOTA / MOTA / IDF1 over predictions persisted to disk; the `MOT_<date>/<tracker>/` experiment layout is documented inside that script.

## Citation

If you use Space-tracker, please cite the paper (BibTeX entry forthcoming with the camera-ready release).
