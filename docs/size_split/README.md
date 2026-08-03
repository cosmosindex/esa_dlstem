# Small / large object split

Every Space-tracker dataset (3 SOT + 5 MOT, 954 sequences) is partitioned into
a **small-object** and a **large-object** half so trackers can be scored on the
two size regimes separately.

## Criterion

A sequence is **small** when the median `sqrt(w * h)` over all its GT boxes is
`<= 32` px — COCO's small-object threshold (area `<= 32 * 32`), applied at the
sequence level.

The median, rather than "contains at least one small box", is what keeps
sequences whose annotations merely jitter across the line out of the small
bucket. Under the any-small rule `airmot/47` would be small on the strength of
6 boxes out of 2696 (median 38.0 px), and `satmtb/airplane/18` on 1 box out of
321 (median 39.9 px). 11 MOT and 4 SOT sequences differ between the two rules;
`per_sequence_size.csv` carries `frac_small_area` if you want to re-cut it.

Splitting at the sequence level (not the box level) is required for MOT: pulling
individual boxes out of a sequence would truncate tracks mid-flight and make
association metrics meaningless.

## Counts

| task | dataset | small | large |
|---|---|---:|---:|
| SOT | OOTB | 74 | 36 |
| SOT | SatSOT | 75 | 30 |
| SOT | SV248S | 246 | 2 |
| MOT | AIR-MOT-100 | 45 | 24 |
| MOT | SAT-MTB | 187 | 50 |
| MOT | VISO | 4 | 5 |
| MOT | SDM-Car | 99 | 0 |
| MOT | RsCarData | 77 | 0 |
| | **total** | **807** | **147** |

## Caveats

- **Size and class are confounded on MOT.** The small bucket holds all 259 car
  sequences plus every `mixed` one; 35 of the 44 sequences that any rule sends
  to the large bucket are airplane. A small-vs-large gap on MOT therefore
  cannot be read as a pure size effect.
- **SDM-Car and RsCarData have no large half** — both are car-only satellite
  video where no sequence comes close to 32 px. SV248S is nearly as lopsided
  (2 large sequences).
- **SOT sequences are size-homogeneous**; only 7 of 463 contain boxes on both
  sides of the threshold, all of them hugging 32 px (`ootb/ship_10` median
  32.2, `sv248s/04/000000` median 31.6). The split is essentially exact there.

## Artefacts

- `size_split.json` — the source of truth: `seq_id -> {bucket,
  median_sqrt_area_px, n_boxes, frac_small_area, ...}`.
- `per_sequence_size.csv` — full per-sequence size stats (percentiles, both the
  area-based and the side-based small criterion, per-class breakdown for MOT).
- A symlink tree under `<trafic>/_size_split/`, where each bucket is a drop-in
  dataset root:

  ```
  _size_split/OOTB/small/car_1 -> <trafic>/OOTB/car_1
  ```

  Links carry absolute targets and total ~8 MB, so the tree can live anywhere
  and deleting it never touches the source data.

## Usage

```python
# as a dataset root
ds = OOTBDataset(root="<trafic>/_size_split/OOTB/small",
                 split="no_split", mode="detection")

# or filter the Space-tracker manifest directly
import json
split = json.load(open("docs/size_split/size_split.json"))["sequences"]
small = [s for s in bench.manifest.sequences if split[s.id]["bucket"] == "small"]
```

All 16 buckets have been verified to load through their own dataset class with
the expected sequence count, and to yield real pixels + GT via the
`space_tracker` loaders.

## Regenerating

```bash
python tools/analyze_size_split.py --out-dir docs/size_split          # size stats
python tools/build_size_split.py \
    --out docs/size_split/size_split.json \
    --stats-csv docs/size_split/per_sequence_size.csv \
    --link-root <trafic>/_size_split                                  # manifest + tree
```
