# SatSOT — Bounding Box Size Statistics

**Dataset:** SatSOT — single-object tracking dataset from Jilin-1 satellite video.
**Annotation format:** per-frame axis-aligned `[x, y, w, h]` in `<seq>/groundtruth.txt`. Frames where the target is absent or fully occluded are written as the literal `none` and stored as NaN by the dataset class.
**Counting rule:** absent-frame rows (`none` / NaN) are excluded; all other rows are counted.
**Size threshold:** small = area < 32×32 = 1024 px².
**Categories:** `car`, `plane`, `ship`, `train` (inferred from sequence-name prefix).
**Split:** class-stratified 80/10/10 (seed = 42), with at least 1 sequence per split per category.

Analysis script: [`tools/analyze_satsot_bbox.py`](../tools/analyze_satsot_bbox.py).

## 1. Per-Category (overall, all splits)

| dataset | split | category | total_boxes | small (<32²) | small_% | large (≥32²) | large_% | avg_w | avg_h | avg_area | min_area | max_area |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SatSOT | all | car | 19389 | 19389 | 100.0% | 0 | 0.0% | 10.8 | 9.6 | 112 | 9 | 480 |
| SatSOT | all | plane | 2713 | 1105 | 40.7% | 1608 | 59.3% | 39.3 | 40.4 | 1810 | 306 | 4692 |
| SatSOT | all | ship | 1549 | 1549 | 100.0% | 0 | 0.0% | 27.2 | 17.4 | 439 | 70 | 645 |
| SatSOT | all | train | 3410 | 135 | 4.0% | 3275 | 96.0% | 171.9 | 91.2 | 36446 | 574 | 796576 |

## 2. Per-Split, Per-Category

| dataset | split | category | total_boxes | small (<32²) | small_% | large (≥32²) | large_% | avg_w | avg_h | avg_area | min_area | max_area |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SatSOT | train | car | 16431 | 16431 | 100.0% | 0 | 0.0% | 10.5 | 9.3 | 105 | 9 | 432 |
| SatSOT | train | plane | 2233 | 625 | 28.0% | 1608 | 72.0% | 42.8 | 44.1 | 2081 | 504 | 4692 |
| SatSOT | train | ship | 967 | 967 | 100.0% | 0 | 0.0% | 30.1 | 14.7 | 443 | 70 | 645 |
| SatSOT | train | train | 2810 | 135 | 4.8% | 2675 | 95.2% | 171.3 | 82.4 | 41515 | 574 | 796576 |
| SatSOT | val | car | 1413 | 1413 | 100.0% | 0 | 0.0% | 12.3 | 12.4 | 161 | 21 | 400 |
| SatSOT | val | plane | 300 | 300 | 100.0% | 0 | 0.0% | 18.3 | 19.4 | 355 | 306 | 380 |
| SatSOT | val | ship | 300 | 300 | 100.0% | 0 | 0.0% | 32.0 | 16.0 | 512 | 512 | 512 |
| SatSOT | val | train | 360 | 0 | 0.0% | 360 | 100.0% | 210.6 | 193.2 | 17957 | 4884 | 71442 |
| SatSOT | test | car | 1545 | 1545 | 100.0% | 0 | 0.0% | 12.8 | 10.4 | 135 | 20 | 480 |
| SatSOT | test | plane | 180 | 180 | 100.0% | 0 | 0.0% | 30.5 | 28.9 | 882 | 810 | 1023 |
| SatSOT | test | ship | 282 | 282 | 100.0% | 0 | 0.0% | 12.2 | 28.3 | 346 | 208 | 574 |
| SatSOT | test | train | 240 | 0 | 0.0% | 240 | 100.0% | 120.8 | 41.9 | 4824 | 1702 | 10620 |

## 3. Dataset Total

| dataset | total_boxes | small (<32²) | small_% | large (≥32²) | large_% | avg_area |
| --- | --- | --- | --- | --- | --- | --- |
| SatSOT | 27061 | 22178 | 82.0% | 4883 | 18.0% | 4879 |

## 4. Frame Presence Breakdown

Boxes counted in the tables above come from frames where the target is present. Absent rows (`none` in `groundtruth.txt`) are excluded.

| category | present (counted) | absent (excluded) | total_frames |
| --- | --- | --- | --- |
| car | 19389 | 559 | 19948 |
| plane | 2713 | 0 | 2713 |
| ship | 1549 | 44 | 1593 |
| train | 3410 | 0 | 3410 |

## 5. Extremely Small Objects (max(w, h) ≤ 2 px)

| category | total | extreme (≤2px) | extreme_% | (w × h) breakdown |
| --- | --- | --- | --- | --- |
| car | 19389 | 0 | 0.0% | — |
| plane | 2713 | 0 | 0.0% | — |
| ship | 1549 | 0 | 0.0% | — |
| train | 3410 | 0 | 0.0% | — |

## Notes

- SatSOT is a **single-object tracking** dataset: each sequence has exactly one target, so `total_boxes ≈ #frames` (minus absent frames).
- Sequence categories are inferred from the directory-name prefix (`car_03` → `car`, `plane_07` → `plane`, …).
- Widths/heights are reported as-is from `groundtruth.txt` (integer or sub-pixel depending on sequence); area is rounded to integers for display.
