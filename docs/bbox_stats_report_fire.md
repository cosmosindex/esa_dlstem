# Fire Datasets — Bounding Box Size Statistics

**Size threshold:** small = area < 32×32 = 1024 px²

## 1. Per-Dataset, Per-Split, Per-Category

| dataset | split | category | total_boxes | small (<32²) | small_% | large (≥32²) | large_% | avg_w | avg_h | avg_area | min_area | max_area |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RGBT-3M | test | fire | 3401 | 2447 | 71.9% | 954 | 28.1% | 24.4 | 29.4 | 929 | 18 | 15498 |
| RGBT-3M | test | person | 1740 | 1026 | 59.0% | 714 | 41.0% | 25.9 | 34.2 | 1127 | 24 | 11808 |
| RGBT-3M | test | smoke | 4086 | 601 | 14.7% | 3485 | 85.3% | 185.7 | 147.6 | 42957 | 56 | 306081 |
| RGBT-3M | train | fire | 7914 | 5615 | 71.0% | 2299 | 29.0% | 24.7 | 29.9 | 961 | 0 | 41013 |
| RGBT-3M | train | person | 4148 | 2499 | 60.2% | 1649 | 39.8% | 25.3 | 33.4 | 1086 | 15 | 12879 |
| RGBT-3M | train | smoke | 9488 | 1381 | 14.6% | 8107 | 85.4% | 186.6 | 147.3 | 42892 | 56 | 306081 |

## 2. Per-Dataset Totals

| dataset | total_boxes | small (<32²) | small_% | large (≥32²) | large_% | avg_area |
| --- | --- | --- | --- | --- | --- | --- |
| RGBT-3M | 30777 | 13569 | 44.1% | 17208 | 55.9% | 19486 |

## Notes

- **RGBT-3M**: YOLO format — normalized w/h converted to pixels using fixed image size 640×480.
  Classes: smoke (0), fire (1), person (2). Train/test split via subfolder.
