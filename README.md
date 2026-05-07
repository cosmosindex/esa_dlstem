# Space-tracker

A unified single-object-tracking (SOT) benchmark across **SatSOT**, **SV248S**, and **OOTB**, with a shared attribute taxonomy and the per-sequence evaluation protocol used in the paper.

This directory ships:

- **`space_tracker.json`** — manifest cataloguing all 463 sequences (relative paths to imagery / GT, native + unified attribute labels, sequence-level scale stats, tiny flag). ~210 KB; safe to commit and diff.
- **`manifest.py`** — JSON loader + filtering API (`Manifest.filter(...)`).
- **`data.py`** — per-frame loader bridging the manifest to on-disk imagery and GT for each of the three source datasets.
- **`metrics.py`** — per-sequence SR / NPR / PR / P@5 (PR over CLE ∈ [0, 50] px; NPR over normalised CLE ∈ [0, 0.5]).
- **`benchmark.py`** — the high-level `Benchmark` class. Most users only touch this.

## Conventions

The manifest does **not** ship raw imagery or GT — readers are expected to download SatSOT, SV248S, and OOTB themselves under their original licenses, the same way LaSOT and GOT-10k handle it.

Attribute taxonomy: 6 unified attributes (BC, IV, ROT, OCC, SOB, DEF) consolidate the per-dataset native labels (e.g. OCC = SatSOT \{POC, FOC\} ∪ SV248S \{STO, LTO, CO\} ∪ OOTB \{PO, FO\}). The full mapping is in `manifest['unified_attributes']`.

Aggregation: per-sequence — each sequence contributes one curve per metric, and the headline number is the equal-weight mean across sequences. This matches OTB / LaSOT / GOT-10k / TrackingNet / OOTB. Cross-dataset unified-attribute scores additionally arithmetic-mean across the datasets that annotate the attribute.

## Quick start

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

# 1. Iterate over a slice (e.g. all tiny sequences with the OCC attribute).
for seq in bench.filter(unified_attrs=["OCC"], tiny=True):
    for frame in bench.frames(seq):
        if not frame.visible:
            continue
        # frame.image_path  — pathlib.Path
        # frame.gt_box_xyxy — np.ndarray (4,) xyxy
        # frame.gt_obb_8pt  — np.ndarray (8,) corners (OOTB only)
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

## Reproducing the paper numbers

`tools/reaggregate_sot_per_sequence.py` re-aggregates existing
`per_image_metrics.json` files using the same per-sequence protocol — useful if you have already run the seven trackers reported in the paper and want to recompute the headline numbers without re-running inference. `tools/sot_unified_attribute_table.py` produces the unified-attribute breakdown CSVs.

## Citation

If you use Space-tracker, please cite the paper (see top-level repo for the BibTeX entry).
