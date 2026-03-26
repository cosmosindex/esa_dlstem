# Traffic Datasets — Bounding Box Size Statistics

**Size threshold:** small = area < 32×32 = 1024 px²

## 1. Per-Dataset, Per-Split, Per-Category

| dataset | split | category | total_boxes | small (<32²) | small_% | large (≥32²) | large_% | avg_w | avg_h | avg_area | min_area | max_area |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OOTB | no_split | car | 12168 | 12168 | 100.0% | 0 | 0.0% | 7.3 | 14.9 | 116 | 30 | 546 |
| OOTB | no_split | plane | 8004 | 1755 | 21.9% | 6249 | 78.1% | 38.2 | 44.0 | 1827 | 528 | 7820 |
| OOTB | no_split | ship | 8023 | 8023 | 100.0% | 0 | 0.0% | 10.1 | 21.3 | 248 | 33 | 774 |
| OOTB | no_split | train | 1695 | 250 | 14.7% | 1445 | 85.3% | 13.8 | 125.0 | 1747 | 638 | 3920 |
| SAT-MTB | no_split | car | 704552 | 703979 | 99.9% | 573 | 0.1% | 5.1 | 5.0 | 28 | 2 | 2728 |
| SAT-MTB | no_split | train | 13541 | 12598 | 93.0% | 943 | 7.0% | 20.4 | 14.2 | 2687 | 3 | 100443 |
| SAT-MTB | test | airplane | 15618 | 3499 | 22.4% | 12119 | 77.6% | 48.9 | 47.9 | 2778 | 64 | 13037 |
| SAT-MTB | test | ship | 22454 | 20542 | 91.5% | 1912 | 8.5% | 19.7 | 18.8 | 633 | 18 | 33152 |
| SAT-MTB | test | train | 1786 | 0 | 0.0% | 1786 | 100.0% | 304.5 | 202.9 | 65222 | 7666 | 116812 |
| SAT-MTB | train | airplane | 301669 | 290837 | 96.4% | 10832 | 3.6% | 7.9 | 7.2 | 148 | 12 | 10718 |
| SAT-MTB | train | ship | 56183 | 50492 | 89.9% | 5691 | 10.1% | 16.8 | 14.5 | 474 | 10 | 10906 |
| SAT-MTB | train | train | 2781 | 613 | 22.0% | 2168 | 78.0% | 225.3 | 119.1 | 38866 | 25 | 256635 |
| SatSOT | no_split | car | 19389 | 19389 | 100.0% | 0 | 0.0% | 10.8 | 9.6 | 112 | 9 | 480 |
| SatSOT | no_split | plane | 2713 | 1105 | 40.7% | 1608 | 59.3% | 39.3 | 40.4 | 1810 | 306 | 4692 |
| SatSOT | no_split | ship | 1549 | 1549 | 100.0% | 0 | 0.0% | 27.2 | 17.4 | 439 | 70 | 645 |
| SatSOT | no_split | train | 3410 | 135 | 4.0% | 3275 | 96.0% | 171.9 | 91.2 | 36446 | 574 | 796576 |
| VISO | test | car | 170371 | 170371 | 100.0% | 0 | 0.0% | 5.4 | 6.3 | 35 | 3 | 324 |
| VISO | test | plane | 608 | 0 | 0.0% | 608 | 100.0% | 51.7 | 49.4 | 2687 | 1360 | 4556 |
| VISO | test | ship | 427 | 427 | 100.0% | 0 | 0.0% | 20.1 | 17.1 | 349 | 119 | 986 |
| VISO | train | car | 525502 | 525502 | 100.0% | 0 | 0.0% | 7.2 | 6.5 | 52 | 1 | 594 |
| VISO | train | plane | 1288 | 744 | 57.8% | 544 | 42.2% | 39.7 | 38.4 | 1688 | 400 | 3876 |
| VISO | train | ship | 1059 | 1026 | 96.9% | 33 | 3.1% | 26.5 | 16.0 | 457 | 84 | 1820 |
| VISO | val | car | 130519 | 130519 | 100.0% | 0 | 0.0% | 5.5 | 4.8 | 28 | 4 | 392 |
| VISO | val | plane | 373 | 4 | 1.1% | 369 | 98.9% | 38.4 | 41.9 | 1612 | 208 | 1980 |

## 2. Per-Dataset Totals

| dataset | total_boxes | small (<32²) | small_% | large (≥32²) | large_% | avg_area |
| --- | --- | --- | --- | --- | --- | --- |
| OOTB | 29890 | 22196 | 74.3% | 7694 | 25.7% | 702 |
| SAT-MTB | 1118584 | 1082560 | 96.8% | 36024 | 3.2% | 366 |
| SatSOT | 27061 | 22178 | 82.0% | 4883 | 18.0% | 4879 |
| VISO | 830147 | 828593 | 99.8% | 1554 | 0.2% | 51 |

## Notes

- **OOTB**: Oriented Bounding Boxes — w/h computed as shorter/longer side of OBB; no official train/val/test split.
- **SatSOT**: Axis-aligned x,y,w,h; no official train/val/test split.
- **SAT-MTB**: MOT format; `car` category has no split assignment in the xlsx → marked as `no_split`.
- **VISO**: COCO format used (same data as VOC/MOT/SOT formats); has train/val/test splits.

---

## LMOD — Detailed Statistics

**LMOD** (Large-scale and Multiclass Moving Object Detection Dataset for Satellite Videos)  
8 sequences (Seq1–Seq8), 4 062 frames, annotation format: Pascal VOC XML (per-frame detection, no tracking IDs).  
Categories: car, plane, ship, train.

### Per-Category Summary (same format as above)

| dataset | split | category | total_boxes | small (<32²) | small_% | large (≥32²) | large_% | avg_w | avg_h | avg_area | min_area | max_area |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LMOD | no_split | car | 459723 | 459723 | 100.0% | 0 | 0.0% | 4.7 | 4.4 | 23 | 0 | 625 |
| LMOD | no_split | plane | 9390 | 4202 | 44.7% | 5188 | 55.3% | 36.4 | 32.5 | 1251 | 16 | 3481 |
| LMOD | no_split | ship | 10536 | 10536 | 100.0% | 0 | 0.0% | 11.6 | 8.7 | 114 | 20 | 595 |
| LMOD | no_split | train | 693 | 144 | 20.8% | 549 | 79.2% | 25.3 | 62.7 | 1699 | 35 | 4131 |

### Dataset Total

| dataset | total_boxes | small (<32²) | small_% | large (≥32²) | large_% | avg_area |
| --- | --- | --- | --- | --- | --- | --- |
| LMOD | 480342 | 474605 | 98.8% | 5737 | 1.2% | 51 |

### Extremely Small Objects (max(w,h) ≤ 2 px)

| category | total | extreme (≤2px) | extreme_% | (w,h) breakdown |
| --- | --- | --- | --- | --- |
| car | 459723 | 5635 | 1.2% | (0×0): 162, (0×1): 6, (0×2): 3, (1×1): 19, (1×2): 12, (2×1): 79, (2×2): 5354 |
| plane | 9390 | 0 | 0.0% | — |
| ship | 10536 | 0 | 0.0% | — |
| train | 693 | 0 | 0.0% | — |

### Small Object Size Distribution (area < 1024 px², by max side length)

**car** (459,723 small objects)

| max(w,h) px | count | % of small |
| ---: | ---: | ---: |
| 0 | 162 | 0.0% |
| 1 | 25 | 0.0% |
| 2 | 5,448 | 1.2% |
| 3 | 50,786 | 11.0% |
| 4 | 141,097 | 30.7% |
| 5 | 123,393 | 26.8% |
| 6 | 73,894 | 16.1% |
| 7 | 30,831 | 6.7% |
| 8 | 12,424 | 2.7% |
| 9 | 5,734 | 1.2% |
| 10 | 3,448 | 0.8% |
| 11 | 2,939 | 0.6% |
| 12 | 2,861 | 0.6% |
| 13 | 2,338 | 0.5% |
| 14 | 1,034 | 0.2% |
| 15 | 549 | 0.1% |
| 16 | 325 | 0.1% |
| 17 | 640 | 0.1% |
| 18 | 722 | 0.2% |
| 19 | 427 | 0.1% |
| 20 | 47 | 0.0% |
| 21 | 549 | 0.1% |
| 22 | 37 | 0.0% |
| 23 | 4 | 0.0% |
| 24 | 4 | 0.0% |
| 25 | 5 | 0.0% |

**plane** (4,202 small objects)

| max(w,h) px | count | % of small |
| ---: | ---: | ---: |
| 4 | 1 | 0.0% |
| 6 | 1 | 0.0% |
| 18 | 1 | 0.0% |
| 19 | 40 | 1.0% |
| 20 | 352 | 8.4% |
| 21 | 628 | 14.9% |
| 22 | 84 | 2.0% |
| 23 | 7 | 0.2% |
| 24 | 1 | 0.0% |
| 25 | 11 | 0.3% |
| 26 | 50 | 1.2% |
| 27 | 368 | 8.8% |
| 28 | 218 | 5.2% |
| 29 | 211 | 5.0% |
| 30 | 222 | 5.3% |
| 31 | 138 | 3.3% |
| 32 | 104 | 2.5% |
| 33 | 432 | 10.3% |
| 34 | 282 | 6.7% |
| 35 | 286 | 6.8% |
| 36 | 343 | 8.2% |
| 37 | 187 | 4.5% |
| 38 | 114 | 2.7% |
| 39 | 117 | 2.8% |
| 40 | 4 | 0.1% |

**ship** (10,536 small objects)

| max(w,h) px | count | % of small |
| ---: | ---: | ---: |
| 5 | 2 | 0.0% |
| 6 | 652 | 6.2% |
| 7 | 1,388 | 13.2% |
| 8 | 1,985 | 18.8% |
| 9 | 1,168 | 11.1% |
| 10 | 1,154 | 11.0% |
| 11 | 461 | 4.4% |
| 12 | 77 | 0.7% |
| 13 | 629 | 6.0% |
| 14 | 266 | 2.5% |
| 15 | 280 | 2.7% |
| 16 | 218 | 2.1% |
| 17 | 244 | 2.3% |
| 18 | 288 | 2.7% |
| 19 | 603 | 5.7% |
| 20 | 324 | 3.1% |
| 21 | 51 | 0.5% |
| 22 | 125 | 1.2% |
| 23 | 135 | 1.3% |
| 24 | 103 | 1.0% |
| 25 | 70 | 0.7% |
| 26 | 33 | 0.3% |
| 27 | 92 | 0.9% |
| 28 | 4 | 0.0% |
| 29 | 14 | 0.1% |
| 30 | 1 | 0.0% |
| 31 | 1 | 0.0% |
| 32 | 1 | 0.0% |
| 33 | 3 | 0.0% |
| 35 | 101 | 1.0% |
| 36 | 51 | 0.5% |
| 37 | 1 | 0.0% |
| 38 | 11 | 0.1% |

**train** (144 small objects)

| max(w,h) px | count | % of small |
| ---: | ---: | ---: |
| 7 | 2 | 1.4% |
| 8 | 2 | 1.4% |
| 9 | 2 | 1.4% |
| 10 | 12 | 8.3% |
| 11 | 8 | 5.6% |
| 12 | 5 | 3.5% |
| 13 | 14 | 9.7% |
| 14 | 11 | 7.6% |
| 15 | 7 | 4.9% |
| 16 | 9 | 6.2% |
| 17 | 10 | 6.9% |
| 18 | 19 | 13.2% |
| 19 | 10 | 6.9% |
| 31 | 7 | 4.9% |
| 33 | 1 | 0.7% |
| 34 | 1 | 0.7% |
| 37 | 3 | 2.1% |
| 39 | 8 | 5.6% |
| 43 | 3 | 2.1% |
| 45 | 10 | 6.9% |

### Notes (LMOD)

- **LMOD**: Pascal VOC XML per-frame annotations; no tracking IDs; no official train/val/test split.
- Extremely small objects (max side ≤ 2 px) are a subset of small objects.
- Small defined as area < 32×32 = 1024 px², consistent with the rest of this report.
