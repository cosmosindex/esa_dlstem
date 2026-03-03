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
- **Cross-Videos**: Only video metadata, no per-frame bbox annotations → excluded.
