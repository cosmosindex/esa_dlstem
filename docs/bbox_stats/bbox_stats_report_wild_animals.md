# Wild Animal Datasets — Bounding Box Size Statistics

**Size threshold:** small = area < 32×32 = 1024 px²

> Splits below are the **actual experiment splits** produced by `datasets/birdsai_mot.py`
> (`_SPLIT_SEED = 42`): TrainReal → `train`; TestReal → `val` (30%) + `test` (70%),
> assigned **per source video** (random shuffle, no class/size stratification).
> Species are folded to the experiment vocabulary: `-1→unknown, 0→human, 1→elephant,
> 2→lion, 3→giraffe`; every other species (`dog`, crocodile, hippo, zebra, rhino) → `unknown`.

## 1. Per-Split, Per-Category — fine granularity (5-class, used by detection experiments)

| dataset | split | category | total_boxes | small (<32²) | small_% | large (≥32²) | large_% | avg_w | avg_h | avg_area | min_area | max_area |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BIRDSAI | train | elephant | 42973 | 5772 | 13.4% | 37201 | 86.6% | 47.1 | 52.2 | 2855 | 154 | 14641 |
| BIRDSAI | train | giraffe | 9979 | 8937 | 89.6% | 1042 | 10.4% | 18.7 | 24.8 | 520 | 30 | 2684 |
| BIRDSAI | train | human | 12531 | 10002 | 79.8% | 2529 | 20.2% | 15.7 | 25.5 | 478 | 25 | 2080 |
| BIRDSAI | train | lion | 1024 | 1024 | 100.0% | 0 | 0.0% | 12.5 | 13.2 | 171 | 63 | 357 |
| BIRDSAI | train | unknown | 20692 | 20681 | 99.9% | 11 | 0.1% | 12.5 | 12.2 | 161 | 25 | 4160 |
| BIRDSAI | val | human | 5310 | 5052 | 95.1% | 258 | 4.9% | 13.6 | 18.5 | 300 | 36 | 1326 |
| BIRDSAI | val | lion | 351 | 351 | 100.0% | 0 | 0.0% | 11.4 | 11.1 | 127 | 90 | 208 |
| BIRDSAI | val | unknown | 1456 | 1456 | 100.0% | 0 | 0.0% | 11.1 | 11.9 | 133 | 49 | 285 |
| BIRDSAI | test | elephant | 48055 | 6676 | 13.9% | 41379 | 86.1% | 49.2 | 65.8 | 3502 | 130 | 10355 |
| BIRDSAI | test | giraffe | 3231 | 3040 | 94.1% | 191 | 5.9% | 13.7 | 15.3 | 264 | 56 | 2952 |
| BIRDSAI | test | human | 16804 | 16138 | 96.0% | 666 | 4.0% | 19.6 | 31.3 | 665 | 165 | 9216 |
| BIRDSAI | test | unknown | 3815 | 3815 | 100.0% | 0 | 0.0% | 16.9 | 21.0 | 400 | 56 | 812 |

## 2. Per-Split, Per-Category — coarse granularity (2-class: animal / human)

| dataset | split | category | total_boxes | small (<32²) | small_% | large (≥32²) | large_% | avg_w | avg_h | avg_area | min_area | max_area |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BIRDSAI | train | animal | 74668 | 36414 | 48.8% | 38254 | 51.2% | 33.3 | 37.0 | 1760 | 25 | 14641 |
| BIRDSAI | train | human | 12531 | 10002 | 79.8% | 2529 | 20.2% | 15.7 | 25.5 | 478 | 25 | 2080 |
| BIRDSAI | val | animal | 1807 | 1807 | 100.0% | 0 | 0.0% | 11.1 | 11.8 | 132 | 49 | 285 |
| BIRDSAI | val | human | 5310 | 5052 | 95.1% | 258 | 4.9% | 13.6 | 18.5 | 300 | 36 | 1326 |
| BIRDSAI | test | animal | 55101 | 13531 | 24.6% | 41570 | 75.4% | 44.9 | 59.7 | 3098 | 56 | 10355 |
| BIRDSAI | test | human | 16804 | 16138 | 96.0% | 666 | 4.0% | 19.6 | 31.3 | 665 | 165 | 9216 |

## 3. Per-Split Totals

| dataset | split | videos | total_boxes | small (<32²) | small_% | large (≥32²) | large_% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BIRDSAI | train | 32 | 87199 | 46416 | 53.2% | 40783 | 46.8% |
| BIRDSAI | val | 5 | 7117 | 6859 | 96.4% | 258 | 3.6% |
| BIRDSAI | test | 11 | 71905 | 29669 | 41.3% | 42236 | 58.7% |
| BIRDSAI | **all** | 48 | 166221 | 82944 | 49.9% | 83277 | 50.1% |

## 4. val / test Distribution Mismatch (⚠️)

The 30/70 val/test split is a **plain random shuffle of source videos** — it does **not**
stratify by class or object size. The resulting val and test sets are nearly disjoint in
both respects:

**Class composition (fine, % of split's boxes):**

| category | val | test |
| --- | --- | --- |
| human | 74.6% | 23.4% |
| unknown | 20.5% | 5.3% |
| lion | 4.9% | **0% (absent)** |
| elephant | **0% (absent)** | 66.8% |
| giraffe | **0% (absent)** | 4.5% |

- val has **no elephant and no giraffe**; test has **no lion**.
- test's dominant class (elephant, 66.8%) has **zero samples in val** → val cannot monitor
  the detector's performance on the test set's majority class.

**Object size (median box side = √area):**

| metric | val | test |
| --- | --- | --- |
| small (<32px side) | 96.4% | 41.3% |
| medium (32–96px) | 3.6% | 58.2% |
| median box side | 12.4 px | 49.0 px |

- val is almost entirely tiny objects (human-dominated); test is over half medium-sized
  (elephant-dominated). Selecting models / thresholds on val poorly represents test.

## Notes

- **BIRDSAI**: MOT CSV format (frame, object_id, x, y, w, h, class, species, occlusion, noise).
  Coarse category = `class` column (0=animal, 1=human); fine category = `species` column
  (-1=unknown, 0=human, 1=elephant, 2=lion, 3=giraffe, 4=dog, 5=crocodile, 6=hippo,
  7=zebra, 8=rhino), with `dog`/crocodile/hippo/zebra/rhino folded into `unknown`.
- Boxes counted are annotated frames that have a matching image on disk (the same frames
  the loader serves). Per-video assignment prevents frame-level leakage across splits.
- Regenerate: `python /tmp/birdsai_split_table.py` (replicates `BIRDSAIMOTDataset` splits,
  seed=42).
