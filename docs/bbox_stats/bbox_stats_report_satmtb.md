# SAT-MTB — Bounding Box Size Statistics

**Dataset:** SAT-MTB (Multi-Task Benchmark for satellite videos) — supports detection (HBB / OBB), MOT, and instance segmentation.
**Class taxonomy:** 4 coarse classes (`airplane`, `car`, `ship`, `train`) and 14 fine classes (WA/NA/RA/FA/CA, SB/YH/CS/FH/NV/OS, LC/SC, TN).
**Split:** official `train` / `test` from `data_split.xlsx`; `val` carved from 30 % of `test` (seed=42, stratified by category).
**Size threshold:** small = area < 32×32 = 1024 px².

**Annotation availability by task (this distribution):**

| task | airplane | car | ship | train | granularity in source files |
| --- | :---: | :---: | :---: | :---: | --- |
| det_hbb | ✓ | ✗ | ✓ | ✓ | coarse only (XML `<name>` ∈ {airplane, ship, train}) |
| det_obb | ✓ | ✗ | ✓ | ✓ | coarse only (XML `<name>` ∈ {airplane, ship, train}) |
| mot     | ✓ | ✓ | ✓ | ✓ | coarse only (CSV class id) |
| seg     | ✓ | ✗ | ✓ | ✓ | **fine** (JSON `name`) + coarse (`supercategory`) |

**Fine-grained labels:** the seg JSONs use names like `wide_bodied_aircraft`, `speed_boat`, etc. (with underscores). Per-fine-class tables below appear under `seg` only.

**Box shapes:** `det_hbb`, `mot`, `seg` use axis-aligned `(xmin, ymin, xmax, ymax)`. `det_obb` reduces the 4 OBB corners to their enclosing AABB (this is what the dataset loader returns), so its `(w, h)` here is **AABB-of-OBB**, not OBB short/long sides.

Analysis script: [`tools/analyze_satmtb_bbox.py`](../../tools/analyze_satmtb_bbox.py).

## Task: `det_hbb`

### det_hbb — per coarse category (all splits)

| dataset | split | category | total_boxes | small (<32²) | small_% | large (≥32²) | large_% | avg_w | avg_h | avg_area | min_area | max_area |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAT-MTB | all | airplane | 193849 | 39211 | 20.2% | 154638 | 79.8% | 51.8 | 51.1 | 3217 | 64 | 13541 |
| SAT-MTB | all | ship | 97162 | 77450 | 79.7% | 19712 | 20.3% | 24.2 | 22.6 | 971 | 10 | 33153 |
| SAT-MTB | all | train | 6795 | 1013 | 14.9% | 5782 | 85.1% | 237.0 | 143.8 | 43144 | 20 | 256631 |

### det_hbb — per coarse category, per split

| dataset | split | category | total_boxes | small (<32²) | small_% | large (≥32²) | large_% | avg_w | avg_h | avg_area | min_area | max_area |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAT-MTB | train | airplane | 104549 | 20488 | 19.6% | 84061 | 80.4% | 53.5 | 52.6 | 3451 | 84 | 13541 |
| SAT-MTB | train | ship | 74939 | 57143 | 76.3% | 17796 | 23.7% | 25.6 | 23.7 | 1069 | 10 | 10906 |
| SAT-MTB | train | train | 4156 | 939 | 22.6% | 3217 | 77.4% | 221.2 | 121.2 | 37823 | 20 | 256631 |
| SAT-MTB | val | airplane | 19329 | 699 | 3.6% | 18630 | 96.4% | 46.4 | 45.9 | 2233 | 334 | 9572 |
| SAT-MTB | val | ship | 4350 | 3369 | 77.4% | 981 | 22.6% | 30.0 | 29.0 | 1825 | 10 | 33153 |
| SAT-MTB | val | train | 849 | 74 | 8.7% | 775 | 91.3% | 172.1 | 130.9 | 22889 | 31 | 49262 |
| SAT-MTB | test | airplane | 69971 | 18024 | 25.8% | 51947 | 74.2% | 50.7 | 50.3 | 3138 | 64 | 13038 |
| SAT-MTB | test | ship | 17873 | 16938 | 94.8% | 935 | 5.2% | 17.2 | 16.4 | 348 | 31 | 2267 |
| SAT-MTB | test | train | 1790 | 0 | 0.0% | 1790 | 100.0% | 304.4 | 202.6 | 65105 | 7666 | 116812 |

### det_hbb — sequence / frame counts

| split | category | sequences | frames |
| --- | --- | --- | --- |
| train | airplane | 37 | 7267 |
| train | ship | 41 | 10006 |
| train | train | 6 | 1659 |
| train | **subtotal** | **84** | **18932** |
| val | airplane | 8 | 1746 |
| val | ship | 9 | 2109 |
| val | train | 1 | 298 |
| val | **subtotal** | **18** | **4153** |
| test | airplane | 17 | 4083 |
| test | ship | 20 | 5186 |
| test | train | 3 | 599 |
| test | **subtotal** | **40** | **9868** |
| **all** | **total** | **142** | **32953** |

### det_hbb — dataset total

| dataset | total_boxes | small (<32²) | small_% | large (≥32²) | large_% | avg_area |
| --- | --- | --- | --- | --- | --- | --- |
| SAT-MTB | 297806 | 117674 | 39.5% | 180132 | 60.5% | 3395 |

## Task: `det_obb`

### det_obb — per coarse category (all splits)

| dataset | split | category | total_boxes | small (<32²) | small_% | large (≥32²) | large_% | avg_w | avg_h | avg_area | min_area | max_area |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAT-MTB | all | airplane | 146124 | 12694 | 8.7% | 133430 | 91.3% | 58.3 | 57.4 | 3905 | 141 | 13541 |
| SAT-MTB | all | ship | 56027 | 39271 | 70.1% | 16756 | 29.9% | 30.8 | 28.3 | 1334 | 10 | 33153 |
| SAT-MTB | all | train | 4256 | 613 | 14.4% | 3643 | 85.6% | 262.3 | 156.2 | 51616 | 25 | 256631 |

### det_obb — per coarse category, per split

| dataset | split | category | total_boxes | small (<32²) | small_% | large (≥32²) | large_% | avg_w | avg_h | avg_area | min_area | max_area |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAT-MTB | train | airplane | 89718 | 11299 | 12.6% | 78419 | 87.4% | 57.5 | 56.6 | 3871 | 141 | 13541 |
| SAT-MTB | train | ship | 43706 | 28627 | 65.5% | 15079 | 34.5% | 32.1 | 29.8 | 1434 | 10 | 9465 |
| SAT-MTB | train | train | 2902 | 588 | 20.3% | 2314 | 79.7% | 215.4 | 106.6 | 29247 | 53 | 114810 |
| SAT-MTB | val | airplane | 14310 | 409 | 2.9% | 13901 | 97.1% | 73.0 | 73.0 | 6033 | 342 | 13038 |
| SAT-MTB | val | ship | 3430 | 3142 | 91.6% | 288 | 8.4% | 26.0 | 25.9 | 1732 | 36 | 33153 |
| SAT-MTB | val | train | 804 | 25 | 3.1% | 779 | 96.9% | 343.1 | 266.5 | 98647 | 25 | 256631 |
| SAT-MTB | test | airplane | 42096 | 986 | 2.3% | 41110 | 97.7% | 55.0 | 54.0 | 3253 | 399 | 12663 |
| SAT-MTB | test | ship | 8891 | 7502 | 84.4% | 1389 | 15.6% | 26.1 | 22.0 | 688 | 18 | 4177 |
| SAT-MTB | test | train | 550 | 0 | 0.0% | 550 | 100.0% | 391.6 | 257.2 | 100889 | 86773 | 116812 |

### det_obb — sequence / frame counts

| split | category | sequences | frames |
| --- | --- | --- | --- |
| train | airplane | 30 | 5716 |
| train | ship | 26 | 5722 |
| train | train | 5 | 1135 |
| train | **subtotal** | **61** | **12573** |
| val | airplane | 6 | 1317 |
| val | ship | 7 | 1489 |
| val | train | 1 | 298 |
| val | **subtotal** | **14** | **3104** |
| test | airplane | 15 | 3139 |
| test | ship | 15 | 3687 |
| test | train | 1 | 275 |
| test | **subtotal** | **31** | **7101** |
| **all** | **total** | **106** | **22778** |

### det_obb — dataset total

| dataset | total_boxes | small (<32²) | small_% | large (≥32²) | large_% | avg_area |
| --- | --- | --- | --- | --- | --- | --- |
| SAT-MTB | 206407 | 52578 | 25.5% | 153829 | 74.5% | 4191 |

## Task: `mot`

### mot — per coarse category (all splits)

| dataset | split | category | total_boxes | small (<32²) | small_% | large (≥32²) | large_% | avg_w | avg_h | avg_area | min_area | max_area |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAT-MTB | all | airplane | 30244 | 6769 | 22.4% | 23475 | 77.6% | 47.4 | 46.5 | 2626 | 16 | 13037 |
| SAT-MTB | all | car | 1015310 | 1015177 | 100.0% | 133 | 0.0% | 5.4 | 5.1 | 29 | 2 | 8158 |
| SAT-MTB | all | ship | 67447 | 59795 | 88.7% | 7652 | 11.3% | 19.6 | 17.5 | 601 | 6 | 33152 |
| SAT-MTB | all | train | 5583 | 819 | 14.7% | 4764 | 85.3% | 245.2 | 147.8 | 46584 | 6 | 256635 |

### mot — per coarse category, per split

| dataset | split | category | total_boxes | small (<32²) | small_% | large (≥32²) | large_% | avg_w | avg_h | avg_area | min_area | max_area |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAT-MTB | train | airplane | 13370 | 2164 | 16.2% | 11206 | 83.8% | 48.5 | 47.9 | 2656 | 87 | 10718 |
| SAT-MTB | train | car | 665830 | 665830 | 100.0% | 0 | 0.0% | 5.6 | 5.1 | 31 | 2 | 375 |
| SAT-MTB | train | ship | 45645 | 39690 | 87.0% | 5955 | 13.0% | 20.1 | 17.4 | 681 | 6 | 33152 |
| SAT-MTB | train | train | 3732 | 697 | 18.7% | 3035 | 81.3% | 214.9 | 109.3 | 30300 | 6 | 114809 |
| SAT-MTB | val | airplane | 4474 | 1006 | 22.5% | 3468 | 77.5% | 40.3 | 38.3 | 1741 | 32 | 4329 |
| SAT-MTB | val | car | 91228 | 91095 | 99.9% | 133 | 0.1% | 5.2 | 5.4 | 35 | 3 | 8158 |
| SAT-MTB | val | ship | 5261 | 5261 | 100.0% | 0 | 0.0% | 10.7 | 11.5 | 152 | 31 | 665 |
| SAT-MTB | val | train | 750 | 0 | 0.0% | 750 | 100.0% | 372.4 | 232.4 | 88292 | 27069 | 116812 |
| SAT-MTB | test | airplane | 12400 | 3599 | 29.0% | 8801 | 71.0% | 48.8 | 47.9 | 2912 | 16 | 13037 |
| SAT-MTB | test | car | 258252 | 258252 | 100.0% | 0 | 0.0% | 4.8 | 4.7 | 24 | 5 | 305 |
| SAT-MTB | test | ship | 16541 | 14844 | 89.7% | 1697 | 10.3% | 21.2 | 19.6 | 522 | 18 | 4177 |
| SAT-MTB | test | train | 1101 | 122 | 11.1% | 979 | 88.9% | 261.5 | 220.4 | 73370 | 12 | 256635 |

### mot — sequence / frame counts

| split | category | sequences | frames |
| --- | --- | --- | --- |
| train | airplane | 37 | 7267 |
| train | car | 62 | 9975 |
| train | ship | 42 | 10145 |
| train | train | 8 | 1515 |
| train | **subtotal** | **149** | **28902** |
| val | airplane | 8 | 1746 |
| val | car | 9 | 1565 |
| val | ship | 8 | 1752 |
| val | train | 2 | 475 |
| val | **subtotal** | **27** | **5538** |
| test | airplane | 17 | 4083 |
| test | car | 21 | 3865 |
| test | ship | 20 | 5404 |
| test | train | 3 | 598 |
| test | **subtotal** | **61** | **13950** |
| **all** | **total** | **237** | **48390** |

### mot — dataset total

| dataset | total_boxes | small (<32²) | small_% | large (≥32²) | large_% | avg_area |
| --- | --- | --- | --- | --- | --- | --- |
| SAT-MTB | 1118584 | 1082560 | 96.8% | 36024 | 3.2% | 366 |

## Task: `seg`

### seg — per coarse category (all splits)

| dataset | split | category | total_boxes | small (<32²) | small_% | large (≥32²) | large_% | avg_w | avg_h | avg_area | min_area | max_area |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAT-MTB | all | airplane | 101036 | 29570 | 29.3% | 71466 | 70.7% | 44.4 | 42.6 | 2214 | 54 | 6678 |
| SAT-MTB | all | ship | 101549 | 85049 | 83.8% | 16500 | 16.2% | 23.3 | 20.8 | 886 | 7 | 27118 |
| SAT-MTB | all | train | 5965 | 1347 | 22.6% | 4618 | 77.4% | 206.2 | 105.2 | 30689 | 9 | 126827 |

### seg — per coarse category, per split

| dataset | split | category | total_boxes | small (<32²) | small_% | large (≥32²) | large_% | avg_w | avg_h | avg_area | min_area | max_area |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAT-MTB | train | airplane | 96592 | 27947 | 28.9% | 68645 | 71.1% | 44.8 | 43.0 | 2259 | 54 | 6678 |
| SAT-MTB | train | ship | 79890 | 64809 | 81.1% | 15081 | 18.9% | 24.8 | 21.8 | 974 | 7 | 17067 |
| SAT-MTB | train | train | 3960 | 1243 | 31.4% | 2717 | 68.6% | 167.3 | 64.0 | 14020 | 24 | 107332 |
| SAT-MTB | val | airplane | 346 | 346 | 100.0% | 0 | 0.0% | 30.2 | 29.9 | 902 | 830 | 1005 |
| SAT-MTB | val | ship | 4237 | 3057 | 72.2% | 1180 | 27.8% | 31.0 | 30.1 | 1895 | 10 | 27118 |
| SAT-MTB | test | airplane | 4098 | 1277 | 31.2% | 2821 | 68.8% | 36.9 | 31.9 | 1271 | 193 | 3526 |
| SAT-MTB | test | ship | 17422 | 17183 | 98.6% | 239 | 1.4% | 14.7 | 13.7 | 234 | 20 | 2220 |
| SAT-MTB | test | train | 2005 | 104 | 5.2% | 1901 | 94.8% | 282.9 | 186.6 | 63612 | 9 | 126827 |

### seg — per fine class (all splits)

| dataset | split | coarse | fine | total_boxes | small (<32²) | small_% | large (≥32²) | large_% | avg_w | avg_h | avg_area | min_area | max_area |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAT-MTB | all | airplane | corporate_aircraft | 5943 | 5943 | 100.0% | 0 | 0.0% | 17.8 | 18.0 | 338 | 55 | 990 |
| SAT-MTB | all | airplane | four_engine_aircraft | 6073 | 0 | 0.0% | 6073 | 100.0% | 78.1 | 74.8 | 5874 | 1339 | 6678 |
| SAT-MTB | all | airplane | narrow_bodied_aircraft | 48279 | 16505 | 34.2% | 31774 | 65.8% | 35.4 | 34.4 | 1271 | 54 | 3180 |
| SAT-MTB | all | airplane | rear_engine_aircraft | 11946 | 6498 | 54.4% | 5448 | 45.6% | 35.0 | 30.1 | 1109 | 362 | 2141 |
| SAT-MTB | all | airplane | wide_bodied_aircraft | 28795 | 624 | 2.2% | 28171 | 97.8% | 61.7 | 59.6 | 3869 | 80 | 6673 |
| SAT-MTB | all | ship | cruise | 6244 | 4952 | 79.3% | 1292 | 20.7% | 42.5 | 22.0 | 1384 | 55 | 15612 |
| SAT-MTB | all | ship | freighter | 15501 | 4371 | 28.2% | 11130 | 71.8% | 52.1 | 49.4 | 3038 | 43 | 17067 |
| SAT-MTB | all | ship | naval_vessels | 4145 | 1202 | 29.0% | 2943 | 71.0% | 54.5 | 71.1 | 4355 | 36 | 27118 |
| SAT-MTB | all | ship | other_ship | 3367 | 2921 | 86.8% | 446 | 13.2% | 25.4 | 20.9 | 547 | 40 | 1747 |
| SAT-MTB | all | ship | speed_boat | 40368 | 40319 | 99.9% | 49 | 0.1% | 9.3 | 8.7 | 96 | 7 | 6264 |
| SAT-MTB | all | ship | yacht | 31924 | 31284 | 98.0% | 640 | 2.0% | 19.0 | 15.4 | 326 | 9 | 6351 |
| SAT-MTB | all | train | train | 5965 | 1347 | 22.6% | 4618 | 77.4% | 206.2 | 105.2 | 30689 | 9 | 126827 |

### seg — per fine class, per split

| dataset | split | coarse | fine | total_boxes | small (<32²) | small_% | large (≥32²) | large_% | avg_w | avg_h | avg_area | min_area | max_area |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAT-MTB | train | airplane | corporate_aircraft | 5335 | 5335 | 100.0% | 0 | 0.0% | 17.8 | 18.0 | 339 | 55 | 990 |
| SAT-MTB | train | airplane | four_engine_aircraft | 6073 | 0 | 0.0% | 6073 | 100.0% | 78.1 | 74.8 | 5874 | 1339 | 6678 |
| SAT-MTB | train | airplane | narrow_bodied_aircraft | 46276 | 16159 | 34.9% | 30117 | 65.1% | 35.3 | 34.4 | 1265 | 54 | 3180 |
| SAT-MTB | train | airplane | rear_engine_aircraft | 10656 | 6068 | 56.9% | 4588 | 43.1% | 34.4 | 30.2 | 1101 | 362 | 2141 |
| SAT-MTB | train | airplane | wide_bodied_aircraft | 28252 | 385 | 1.4% | 27867 | 98.6% | 62.1 | 60.0 | 3908 | 80 | 6673 |
| SAT-MTB | train | ship | cruise | 5525 | 4233 | 76.6% | 1292 | 23.4% | 45.6 | 20.4 | 1481 | 55 | 15612 |
| SAT-MTB | train | ship | freighter | 14325 | 3745 | 26.1% | 10580 | 73.9% | 53.1 | 50.8 | 3175 | 43 | 17067 |
| SAT-MTB | train | ship | naval_vessels | 3728 | 1202 | 32.2% | 2526 | 67.8% | 48.8 | 66.4 | 3346 | 36 | 7821 |
| SAT-MTB | train | ship | other_ship | 2798 | 2491 | 89.0% | 307 | 11.0% | 25.8 | 21.2 | 537 | 44 | 1183 |
| SAT-MTB | train | ship | speed_boat | 32160 | 32112 | 99.9% | 48 | 0.1% | 9.0 | 8.4 | 95 | 7 | 6264 |
| SAT-MTB | train | ship | yacht | 21354 | 21026 | 98.5% | 328 | 1.5% | 19.7 | 15.2 | 334 | 9 | 6351 |
| SAT-MTB | train | train | train | 3960 | 1243 | 31.4% | 2717 | 68.6% | 167.3 | 64.0 | 14020 | 24 | 107332 |
| SAT-MTB | val | airplane | narrow_bodied_aircraft | 346 | 346 | 100.0% | 0 | 0.0% | 30.2 | 29.9 | 902 | 830 | 1005 |
| SAT-MTB | val | ship | cruise | 287 | 287 | 100.0% | 0 | 0.0% | 25.9 | 37.0 | 959 | 943 | 978 |
| SAT-MTB | val | ship | freighter | 312 | 0 | 0.0% | 312 | 100.0% | 54.3 | 47.0 | 2547 | 2442 | 2584 |
| SAT-MTB | val | ship | naval_vessels | 417 | 0 | 0.0% | 417 | 100.0% | 105.8 | 112.4 | 13375 | 6436 | 27118 |
| SAT-MTB | val | ship | other_ship | 426 | 287 | 67.4% | 139 | 32.6% | 28.4 | 24.2 | 777 | 320 | 1747 |
| SAT-MTB | val | ship | speed_boat | 214 | 214 | 100.0% | 0 | 0.0% | 10.8 | 8.7 | 108 | 10 | 154 |
| SAT-MTB | val | ship | yacht | 2581 | 2269 | 87.9% | 312 | 12.1% | 18.8 | 16.7 | 398 | 19 | 1534 |
| SAT-MTB | test | airplane | corporate_aircraft | 608 | 608 | 100.0% | 0 | 0.0% | 17.6 | 18.2 | 327 | 193 | 430 |
| SAT-MTB | test | airplane | narrow_bodied_aircraft | 1657 | 0 | 0.0% | 1657 | 100.0% | 40.7 | 37.2 | 1518 | 1242 | 1872 |
| SAT-MTB | test | airplane | rear_engine_aircraft | 1290 | 430 | 33.3% | 860 | 66.7% | 39.6 | 29.0 | 1170 | 712 | 1751 |
| SAT-MTB | test | airplane | wide_bodied_aircraft | 543 | 239 | 44.0% | 304 | 56.0% | 40.1 | 37.8 | 1815 | 352 | 3526 |
| SAT-MTB | test | ship | cruise | 432 | 432 | 100.0% | 0 | 0.0% | 13.2 | 31.8 | 420 | 403 | 448 |
| SAT-MTB | test | ship | freighter | 864 | 626 | 72.5% | 238 | 27.5% | 35.3 | 27.1 | 932 | 221 | 2220 |
| SAT-MTB | test | ship | other_ship | 143 | 143 | 100.0% | 0 | 0.0% | 8.5 | 5.8 | 50 | 40 | 54 |
| SAT-MTB | test | ship | speed_boat | 7994 | 7993 | 100.0% | 1 | 0.0% | 10.1 | 9.6 | 103 | 20 | 2124 |
| SAT-MTB | test | ship | yacht | 7989 | 7989 | 100.0% | 0 | 0.0% | 17.2 | 15.5 | 282 | 28 | 729 |
| SAT-MTB | test | train | train | 2005 | 104 | 5.2% | 1901 | 94.8% | 282.9 | 186.6 | 63612 | 9 | 126827 |

### seg — sequence / frame counts

| split | category | sequences | frames |
| --- | --- | --- | --- |
| train | airplane | 37 | 7267 |
| train | ship | 41 | 10006 |
| train | train | 6 | 1659 |
| train | **subtotal** | **84** | **18932** |
| val | airplane | 8 | 1746 |
| val | ship | 9 | 2109 |
| val | train | 1 | 298 |
| val | **subtotal** | **18** | **4153** |
| test | airplane | 17 | 4083 |
| test | ship | 20 | 5186 |
| test | train | 3 | 599 |
| test | **subtotal** | **40** | **9868** |
| **all** | **total** | **142** | **32953** |

### seg — dataset total

| dataset | total_boxes | small (<32²) | small_% | large (≥32²) | large_% | avg_area |
| --- | --- | --- | --- | --- | --- | --- |
| SAT-MTB | 208550 | 115966 | 55.6% | 92584 | 44.4% | 2382 |

## Notes

- SAT-MTB is **multi-object**: `total_boxes` = sum over all instances in all frames, not number of frames.
- For `det_obb`, the loader (`SATMTBDataset._parse_det_obb`) projects the 4 OBB corners to an axis-aligned bbox; that is what trains/evals an HBB detector. If you need true OBB short/long sides, parse `<robndbox>` in `det/OBB/<frame>.xml` directly.
- `car` sequences only ship MOT-format annotations, so `car` rows appear only under `mot`.
- Per-fine-class breakdown only appears under `seg`. HBB/OBB XMLs in this distribution carry only the coarse name; MOT CSVs carry only the coarse class id.
