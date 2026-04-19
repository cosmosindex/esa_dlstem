# OOTB — Bounding Box Size Statistics

**Dataset:** OOTB (Object Tracking Benchmark on satellite video, ISPRS 2024) — single-object tracking with **oriented bounding boxes** (OBB).
**Annotation format:** per-frame `[x1, y1, x2, y2, x3, y3, x4, y4]` (4 OBB corners) in `<seq>/groundtruth.txt`. Categories: `car`, `plane`, `ship`, `train`.
**Counting rule:** every frame has a valid OBB (no `none` / NaN markers); all rows are counted.
**Size threshold:** small = area < 32×32 = 1024 px².
**Split:** hybrid 80/10/10 (seed = 42), iterative-stratification on class ⊕ 12 attribute flags.

Two flavours of `(w, h)` are reported:
- **OBB sides** (default, sections 1–3): `w = shorter side`, `h = longer side` of the OBB.
- **AABB** (section 4): width/height of the axis-aligned bounding box that encloses the OBB — this is what an HBB detector actually sees.

Analysis script: [`tools/analyze_ootb_bbox.py`](../tools/analyze_ootb_bbox.py).

## 1. Per-Category (overall, all splits, OBB sides)

| dataset | split | category | total_boxes | small (<32²) | small_% | large (≥32²) | large_% | avg_w | avg_h | avg_area | min_area | max_area |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OOTB | all | car | 12168 | 12168 | 100.0% | 0 | 0.0% | 7.3 | 14.9 | 116 | 30 | 546 |
| OOTB | all | plane | 8004 | 1755 | 21.9% | 6249 | 78.1% | 38.2 | 44.0 | 1827 | 528 | 7820 |
| OOTB | all | ship | 8023 | 8023 | 100.0% | 0 | 0.0% | 10.1 | 21.3 | 248 | 33 | 774 |
| OOTB | all | train | 1695 | 250 | 14.7% | 1445 | 85.3% | 13.8 | 125.0 | 1747 | 638 | 3920 |

## 2. Per-Split, Per-Category (OBB sides)

| dataset | split | category | total_boxes | small (<32²) | small_% | large (≥32²) | large_% | avg_w | avg_h | avg_area | min_area | max_area |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OOTB | train | car | 10076 | 10076 | 100.0% | 0 | 0.0% | 7.2 | 14.9 | 113 | 41 | 406 |
| OOTB | train | plane | 6304 | 505 | 8.0% | 5799 | 92.0% | 39.8 | 46.5 | 1965 | 945 | 7820 |
| OOTB | train | ship | 5676 | 5676 | 100.0% | 0 | 0.0% | 10.5 | 21.5 | 261 | 33 | 774 |
| OOTB | train | train | 1455 | 250 | 17.2% | 1205 | 82.8% | 13.7 | 115.0 | 1574 | 638 | 2977 |
| OOTB | val | car | 1200 | 1200 | 100.0% | 0 | 0.0% | 8.5 | 18.7 | 183 | 42 | 546 |
| OOTB | val | plane | 450 | 200 | 44.4% | 250 | 55.6% | 44.2 | 48.4 | 2577 | 528 | 4216 |
| OOTB | val | ship | 1027 | 1027 | 100.0% | 0 | 0.0% | 9.6 | 24.9 | 270 | 78 | 531 |
| OOTB | val | train | 120 | 0 | 0.0% | 120 | 100.0% | 17.1 | 228.6 | 3920 | 3920 | 3920 |
| OOTB | test | car | 892 | 892 | 100.0% | 0 | 0.0% | 6.3 | 9.6 | 63 | 30 | 98 |
| OOTB | test | plane | 1250 | 1050 | 84.0% | 200 | 16.0% | 28.2 | 29.8 | 860 | 720 | 1560 |
| OOTB | test | ship | 1320 | 1320 | 100.0% | 0 | 0.0% | 8.8 | 17.7 | 177 | 60 | 438 |
| OOTB | test | train | 120 | 0 | 0.0% | 120 | 100.0% | 11.7 | 143.4 | 1678 | 1678 | 1678 |

## 3. Dataset Total (OBB sides)

| dataset | total_boxes | small (<32²) | small_% | large (≥32²) | large_% | avg_area |
| --- | --- | --- | --- | --- | --- | --- |
| OOTB | 29890 | 22196 | 74.3% | 7694 | 25.7% | 702 |

## 4. Per-Category — AABB (axis-aligned bbox of the OBB)

AABB area is always ≥ OBB area; the gap grows with rotation and aspect ratio (most pronounced for `train`, which is long & often diagonal).

| dataset | split | category | total_boxes | small (<32²) | small_% | large (≥32²) | large_% | avg_w | avg_h | avg_area | min_area | max_area |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OOTB | all | car | 12168 | 12124 | 99.6% | 44 | 0.4% | 13.8 | 13.7 | 203 | 50 | 1154 |
| OOTB | all | plane | 8004 | 750 | 9.4% | 7254 | 90.6% | 51.6 | 50.7 | 2880 | 729 | 13661 |
| OOTB | all | ship | 8023 | 7527 | 93.8% | 496 | 6.2% | 20.9 | 17.6 | 418 | 33 | 1860 |
| OOTB | all | train | 1695 | 0 | 0.0% | 1695 | 100.0% | 109.9 | 53.8 | 5622 | 1094 | 25145 |

## 5. Sequence / Frame Counts per Category

| category | sequences | frames |
| --- | --- | --- |
| car | 45 | 12168 |
| plane | 25 | 8004 |
| ship | 30 | 8023 |
| train | 10 | 1695 |
| **total** | **110** | **29890** |

## 6. Extremely Small Objects (max(w, h) ≤ 2 px, OBB sides)

| category | total | extreme (≤2px) | extreme_% | (w × h) breakdown |
| --- | --- | --- | --- | --- |
| car | 12168 | 0 | 0.0% | — |
| plane | 8004 | 0 | 0.0% | — |
| ship | 8023 | 0 | 0.0% | — |
| train | 1695 | 0 | 0.0% | — |

## Notes

- OOTB is a **single-object tracking** dataset: each sequence has exactly one target, so `total_boxes = #frames`.
- Sequence categories are inferred from the directory-name prefix (`car_3` → `car`, `plane_07` → `plane`, …).
- OBB-side stats match the OOTB row in `docs/bbox_stats_report_trafic.md`; AABB stats (section 4) are larger because the rotation-bounding box absorbs the diagonal.
