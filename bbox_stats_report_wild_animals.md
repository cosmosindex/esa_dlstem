# Wild Animal Datasets — Bounding Box Size Statistics

**Size threshold:** small = area < 32×32 = 1024 px²

## 1. Per-Dataset, Per-Split, Per-Category

| dataset | split | category | total_boxes | small (<32²) | small_% | large (≥32²) | large_% | avg_w | avg_h | avg_area | min_area | max_area |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BIRDSAI | test | dog | 2715 | 2715 | 100.0% | 0 | 0.0% | 19.8 | 25.0 | 517 | 165 | 812 |
| BIRDSAI | test | elephant | 48055 | 6676 | 13.9% | 41379 | 86.1% | 49.2 | 65.8 | 3502 | 130 | 10355 |
| BIRDSAI | test | giraffe | 3231 | 3040 | 94.1% | 191 | 5.9% | 13.7 | 15.3 | 264 | 56 | 2952 |
| BIRDSAI | test | human | 22114 | 21190 | 95.8% | 924 | 4.2% | 18.1 | 28.2 | 577 | 36 | 9216 |
| BIRDSAI | test | lion | 351 | 351 | 100.0% | 0 | 0.0% | 11.4 | 11.1 | 127 | 90 | 208 |
| BIRDSAI | test | unknown | 2556 | 2556 | 100.0% | 0 | 0.0% | 10.5 | 11.7 | 124 | 49 | 285 |
| BIRDSAI | train | elephant | 42973 | 5772 | 13.4% | 37201 | 86.6% | 47.1 | 52.2 | 2855 | 154 | 14641 |
| BIRDSAI | train | giraffe | 9979 | 8937 | 89.6% | 1042 | 10.4% | 18.7 | 24.8 | 520 | 30 | 2684 |
| BIRDSAI | train | human | 12531 | 10002 | 79.8% | 2529 | 20.2% | 15.7 | 25.5 | 478 | 25 | 2080 |
| BIRDSAI | train | lion | 1024 | 1024 | 100.0% | 0 | 0.0% | 12.5 | 13.2 | 171 | 63 | 357 |
| BIRDSAI | train | unknown | 20692 | 20681 | 99.9% | 11 | 0.1% | 12.5 | 12.2 | 161 | 25 | 4160 |
| WUR_MOTS | test | cattle | 2470 | 8 | 0.3% | 2462 | 99.7% | 94.2 | 71.1 | 7002 | 572 | 18172 |
| WUR_MOTS | train | cattle | 18177 | 2006 | 11.0% | 16171 | 89.0% | 62.3 | 63.6 | 4385 | 48 | 23320 |

## 2. Per-Dataset Totals

| dataset | total_boxes | small (<32²) | small_% | large (≥32²) | large_% | avg_area |
| --- | --- | --- | --- | --- | --- | --- |
| BIRDSAI | 166221 | 82944 | 49.9% | 83277 | 50.1% | 1932 |
| WUR_MOTS | 20647 | 2014 | 9.8% | 18633 | 90.2% | 4698 |

## Notes

- **BIRDSAI**: MOT CSV format (frame, object_id, x, y, w, h, class, species, occlusion, noise).
  Category derived from `species` column: -1=unknown, 0=human, 1=elephant, 2=lion,
  3=giraffe, 4=dog, 5=crocodile, 6=hippo, 7=zebra, 8=rhino.
  Splits: TrainReal → train, TestReal → test.
- **WUR_MOTS**: Instance segmentation masks (uint16 PNG, MOTS format).
  Pixel value = class_id × 1000 + instance_id; background = 0.
  Only class_id=1 (cattle) present. Bbox computed as tight bounding box of each instance mask.
  Image size: 1360×1000 px. Splits: train / testing → test.
