# LMOD — Bounding Box Size Statistics

**Dataset:** LMOD — multi-class moving-object detection on Jilin-1 satellite videos.
**Class taxonomy:** 4 categories — `car`, `plane`, `ship`, `train`.
**Annotation format:** Pascal-VOC XML, axis-aligned `(xmin, ymin, xmax, ymax)`.
**Split:** 80 / 10 / 10 by frame within each sequence (temporal order); every sequence contributes to all three splits.
**Size threshold:** small = area < 32×32 = 1024 px².

Analysis script: [`tools/analyze_lmod_bbox.py`](../../tools/analyze_lmod_bbox.py).

## Per coarse category (all splits)

| dataset | split | category | total_boxes | small (<32²) | small_% | large (≥32²) | large_% | avg_w | avg_h | avg_area | min_area | max_area |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LMOD | all | car | 459536 | 459536 | 100.0% | 0 | 0.0% | 4.7 | 4.4 | 23 | 1 | 625 |
| LMOD | all | plane | 9390 | 4202 | 44.7% | 5188 | 55.3% | 36.4 | 32.5 | 1251 | 16 | 3481 |
| LMOD | all | ship | 10536 | 10536 | 100.0% | 0 | 0.0% | 11.6 | 8.7 | 114 | 20 | 595 |
| LMOD | all | train | 693 | 144 | 20.8% | 549 | 79.2% | 25.3 | 62.7 | 1699 | 35 | 4131 |

## Per coarse category, per split

| dataset | split | category | total_boxes | small (<32²) | small_% | large (≥32²) | large_% | avg_w | avg_h | avg_area | min_area | max_area |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LMOD | train | car | 375557 | 375557 | 100.0% | 0 | 0.0% | 4.8 | 4.5 | 24 | 1 | 625 |
| LMOD | train | plane | 7643 | 3449 | 45.1% | 4194 | 54.9% | 36.3 | 32.4 | 1245 | 342 | 3481 |
| LMOD | train | ship | 8424 | 8424 | 100.0% | 0 | 0.0% | 11.9 | 8.9 | 119 | 24 | 595 |
| LMOD | train | train | 486 | 144 | 29.6% | 342 | 70.4% | 23.9 | 57.0 | 1498 | 35 | 4131 |
| LMOD | val | car | 42695 | 42695 | 100.0% | 0 | 0.0% | 4.5 | 4.1 | 20 | 1 | 294 |
| LMOD | val | plane | 862 | 382 | 44.3% | 480 | 55.7% | 36.5 | 32.9 | 1273 | 16 | 3364 |
| LMOD | val | ship | 1056 | 1056 | 100.0% | 0 | 0.0% | 10.5 | 7.8 | 94 | 24 | 437 |
| LMOD | val | train | 103 | 0 | 0.0% | 103 | 100.0% | 24.2 | 77.9 | 1885 | 1872 | 2204 |
| LMOD | test | car | 41284 | 41284 | 100.0% | 0 | 0.0% | 4.4 | 4.0 | 19 | 1 | 221 |
| LMOD | test | plane | 885 | 371 | 41.9% | 514 | 58.1% | 36.5 | 33.0 | 1273 | 361 | 3420 |
| LMOD | test | ship | 1056 | 1056 | 100.0% | 0 | 0.0% | 10.5 | 7.8 | 96 | 20 | 437 |
| LMOD | test | train | 104 | 0 | 0.0% | 104 | 100.0% | 33.0 | 74.4 | 2454 | 2204 | 2808 |

## Sequence / frame counts

| split | category | sequences | frames |
| --- | --- | --- | --- |
| train | car | 8 | 3249 |
| train | plane | 4 | 1656 |
| train | ship | 4 | 1024 |
| train | train | 1 | 825 |
| train | **subtotal** | **17** | **6754** |
| val | car | 8 | 406 |
| val | plane | 4 | 207 |
| val | ship | 4 | 128 |
| val | train | 1 | 103 |
| val | **subtotal** | **17** | **844** |
| test | car | 8 | 407 |
| test | plane | 4 | 207 |
| test | ship | 4 | 128 |
| test | train | 1 | 104 |
| test | **subtotal** | **17** | **846** |
| **all** | **total** | **51** | **8444** |

> Sequence / frame counts are aggregated across categories: a sequence containing both `car` and `plane` is counted under each category (so the subtotal can exceed the actual number of distinct sequences). Frame counts are the full split-frame count of each sequence.

## Dataset total

| dataset | total_boxes | small (<32²) | small_% | large (≥32²) | large_% | avg_area |
| --- | --- | --- | --- | --- | --- | --- |
| LMOD | 480155 | 474418 | 98.8% | 5737 | 1.2% | 51 |

## Notes

- LMOD is **multi-object** per frame: `total_boxes` = sum over all annotated instances in all frames, not number of frames.
- Annotations are HBB only; no OBB.
- Splits are *intra-sequence temporal* (first 80 % train, next 10 % val, last 10 % test) — every sequence appears in all three splits.
- Categories are taken verbatim from the XML `<name>` field; no remapping.
