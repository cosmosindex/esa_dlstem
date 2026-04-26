# SV248S — Bounding Box Size Statistics

**Dataset:** SV248S SOT — 248 single-target sequences across 6 parent videos.
**Annotation format:** per-frame axis-aligned `[x, y, w, h]` in `.rect`, with `.state` flag per frame (0 = visible, 1 = invisible, 2 = occluded).
**Counting rule:** bboxes on frames flagged `invisible` (state = 1) are skipped — the target has disappeared and the rect is stale. Visible (state = 0) and occluded (state = 2) frames are counted.
**Size threshold:** small = area < 32×32 = 1024 px².
**Split:** class-stratified 80/10/10 (seed = 42), with tiny classes (plane, ship) pre-assigned round-robin so every split covers every category.

Analysis script: [`tools/analyze_sv248s_bbox.py`](../tools/analyze_sv248s_bbox.py).

## 1. Per-Category (overall, all splits)

| dataset | split | category | total_boxes | small (<32²) | small_% | large (≥32²) | large_% | avg_w | avg_h | avg_area | min_area | max_area |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SV248S | all | car | 115519 | 115519 | 100.0% | 0 | 0.0% | 6.0 | 5.3 | 33 | 9 | 196 |
| SV248S | all | car-large | 22998 | 22998 | 100.0% | 0 | 0.0% | 12.5 | 8.8 | 111 | 22 | 523 |
| SV248S | all | plane | 4239 | 2747 | 64.8% | 1492 | 35.2% | 35.2 | 31.6 | 1190 | 526 | 3242 |
| SV248S | all | ship | 1363 | 1363 | 100.0% | 0 | 0.0% | 11.9 | 9.9 | 128 | 33 | 244 |

## 2. Per-Split, Per-Category

| dataset | split | category | total_boxes | small (<32²) | small_% | large (≥32²) | large_% | avg_w | avg_h | avg_area | min_area | max_area |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SV248S | train | car | 92146 | 92146 | 100.0% | 0 | 0.0% | 5.9 | 5.3 | 32 | 9 | 124 |
| SV248S | train | car-large | 17345 | 17345 | 100.0% | 0 | 0.0% | 12.0 | 8.7 | 101 | 22 | 504 |
| SV248S | train | plane | 1247 | 748 | 60.0% | 499 | 40.0% | 42.8 | 35.9 | 1722 | 736 | 3242 |
| SV248S | train | ship | 490 | 490 | 100.0% | 0 | 0.0% | 10.0 | 10.1 | 101 | 83 | 129 |
| SV248S | val | car | 10979 | 10979 | 100.0% | 0 | 0.0% | 6.8 | 6.1 | 44 | 13 | 196 |
| SV248S | val | car-large | 2747 | 2747 | 100.0% | 0 | 0.0% | 10.8 | 7.1 | 75 | 36 | 157 |
| SV248S | val | plane | 1496 | 1496 | 100.0% | 0 | 0.0% | 24.4 | 25.7 | 624 | 526 | 736 |
| SV248S | val | ship | 383 | 383 | 100.0% | 0 | 0.0% | 6.5 | 6.5 | 42 | 33 | 46 |
| SV248S | test | car | 12394 | 12394 | 100.0% | 0 | 0.0% | 5.8 | 5.2 | 30 | 12 | 79 |
| SV248S | test | car-large | 2906 | 2906 | 100.0% | 0 | 0.0% | 17.1 | 10.9 | 202 | 50 | 523 |
| SV248S | test | plane | 1496 | 503 | 33.6% | 993 | 66.4% | 39.8 | 33.8 | 1311 | 901 | 1693 |
| SV248S | test | ship | 490 | 490 | 100.0% | 0 | 0.0% | 18.0 | 12.4 | 223 | 203 | 244 |

## 3. Dataset Total

| dataset | total_boxes | small (<32²) | small_% | large (≥32²) | large_% | avg_area |
| --- | --- | --- | --- | --- | --- | --- |
| SV248S | 144119 | 142627 | 99.0% | 1492 | 1.0% | 80 |

## 4. Frame State Breakdown

Boxes counted in the tables above come from frames with `state ∈ {0, 2}`. Invisible frames (state = 1) are excluded.

| category | visible (state=0) | occluded (state=2) | invisible (state=1, excluded) | counted_total |
| --- | --- | --- | --- | --- |
| car | 112235 | 3284 | 10899 | 115519 |
| car-large | 22632 | 366 | 1496 | 22998 |
| plane | 4148 | 91 | 0 | 4239 |
| ship | 1363 | 0 | 107 | 1363 |

## 5. Extremely Small Objects (max(w, h) ≤ 2 px)

| category | total | extreme (≤2px) | extreme_% | (w × h) breakdown |
| --- | --- | --- | --- | --- |
| car | 115519 | 0 | 0.0% | — |
| car-large | 22998 | 0 | 0.0% | — |
| plane | 4239 | 0 | 0.0% | — |
| ship | 1363 | 0 | 0.0% | — |

## Notes

- SV248S is a **single-object tracking** dataset: each sequence has exactly one target, so `total_boxes ≈ #frames` (minus invisible frames).
- `car-large` is the SV248S-specific class for buses/trucks and is separate from `car`.
- Widths and heights are reported **as-is** from `.rect` (which carries sub-pixel precision), but averages/min/max of `area` are rounded to integers for display.
