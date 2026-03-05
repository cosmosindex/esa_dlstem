# Video Length Summary

> **Metric**: number of frames per video
> Organised by **use case → dataset → split → category**

---

## 1. Fire

### RGBT-3M

Multi-class detection dataset (smoke / fire / person mixed per video).
Category is reported as **all** (videos are not organised by object type).

| Split | Category | # Videos | Total Frames | Min | Max | Mean | Median |
|-------|----------|----------:|-------------:|----:|----:|-----:|-------:|
| train | all | 8 | 7 854 | 270 | 1 927 | 981.8 | 886.0 |
| test  | all | 8 | 3 366 | 117 |   825 | 420.8 | 379.5 |

---

## 2. Traffic

### OOTB

Single-object tracking dataset. No official train/val/test split.

| Split | Category | # Videos | Total Frames | Min | Max | Mean | Median |
|-------|----------|----------:|-------------:|----:|----:|-----:|-------:|
| no_split | car   | 45 | 24 336 | 278 | 1 500 | 540.8 | 400.0 |
| no_split | plane | 25 | 16 008 | 180 | 1 500 | 640.3 | 520.0 |
| no_split | ship  | 30 | 16 046 | 300 | 1 000 | 534.9 | 600.0 |
| no_split | train | 10 |  3 390 | 240 |   810 | 339.0 | 240.0 |

### SatSOT

Single-object tracking dataset. No official train/val/test split.

| Split | Category | # Videos | Total Frames | Min | Max | Mean | Median |
|-------|----------|----------:|-------------:|----:|----:|-----:|-------:|
| no_split | car   | 65 | 39 896 | 260 | 1 500 | 613.8 | 600.0 |
| no_split | plane |  9 |  5 426 | 360 | 1 000 | 602.9 | 600.0 |
| no_split | ship  |  5 |  3 186 | 600 |   654 | 637.2 | 640.0 |
| no_split | train | 26 |  6 820 | 240 |   500 | 262.3 | 240.0 |

### SAT-MTB

Multi-object tracking dataset. Train/test split defined via `data_split.xlsx`
(airplane / ship / train have official splits; **car** has no split assignment).

| Split | Category | # Videos | Total Frames | Min | Max | Mean | Median |
|-------|----------|----------:|-------------:|----:|----:|-----:|-------:|
| train    | airplane | 37 |  7 267 |  45 | 326 | 196.4 | 145.0 |
| train    | ship     | 41 | 10 006 |  56 | 432 | 244.0 | 262.0 |
| train    | train    |  6 |  1 659 | 197 | 317 | 276.5 | 286.0 |
| test     | airplane | 25 |  5 829 |  67 | 432 | 233.2 | 243.0 |
| test     | ship     | 29 |  7 295 |  85 | 432 | 251.6 | 287.0 |
| test     | train    |  4 |    897 |  49 | 298 | 224.2 | 275.0 |
| no_split | car      | 92 | 15 405 |  30 | 300 | 167.4 | 200.0 |
| no_split | train    |  6 |    880 | 100 | 200 | 146.7 | 140.0 |

### VISO-SOT

Per-video frame counts from the SOT subfolder.
No train/val/test split is available at the SOT video level.

| Split | Category | # Videos | Total Frames | Min | Max | Mean | Median |
|-------|----------|----------:|-------------:|----:|----:|-----:|-------:|
| no_split | car   | 38 | 13 421 | 170 | 750 | 353.2 | 326.0 |
| no_split | plane |  6 |  1 913 | 180 | 500 | 318.8 | 312.5 |
| no_split | ship  |  2 |    620 | 300 | 320 | 310.0 | 310.0 |
| no_split | train |  1 |    250 | 250 | 250 | 250.0 | 250.0 |

### VISO-COCO

Split-level **aggregate** frame counts from the COCO subfolder (train2017 / val2017 / test2017).
COCO image filenames are sequentially numbered independently per split, so individual
videos cannot be separated at this level — each row represents the entire split's frame total.

| Split | Category | Total Frames |
|-------|----------|-------------:|
| train | car      |  9 417 |
| train | plane    |  1 005 |
| train | ship     |    570 |
| train | train    |    198 |
| val   | car      |  1 287 |
| val   | plane    |    300 |
| test  | car      |  2 717 |
| test  | plane    |    608 |
| test  | ship     |    351 |
| test  | train    |     49 |

---

## 3. Wild Animal

### BIRDSAI

Aerial wildlife surveillance dataset. Category = dominant species per video
(determined by bounding-box annotation count).
Species: elephant / giraffe / human / lion / unknown (unannotated species).

| Split | Category | # Videos | Total Frames | Min | Max | Mean | Median |
|-------|----------|----------:|-------------:|----:|----:|-----:|-------:|
| train | elephant | 5  | 12 832 |   284 | 10 896 | 2 566.4 | 618.0  |
| train | giraffe  | 2  |  3 956 | 1 950 |  2 006 | 1 978.0 | 1 978.0 |
| train | human    | 5  | 28 828 | 1 752 | 15 914 | 5 765.6 | 3 504.0 |
| train | lion     | 1  |    470 |   470 |    470 |   470.0 | 470.0  |
| train | unknown  | 19 | 35 236 |   898 |  2 008 | 1 854.5 | 1 912.0 |
| test  | elephant | 6  | 18 170 |   536 |  6 418 | 3 028.3 | 2 140.0 |
| test  | giraffe  | 2  |  1 028 |   416 |    612 |   514.0 | 514.0  |
| test  | human    | 5  | 21 958 | 1 394 |  6 064 | 4 391.6 | 5 168.0 |
| test  | lion     | 1  |    368 |   368 |    368 |   368.0 | 368.0  |
| test  | unknown  | 2  |  1 142 |   526 |    616 |   571.0 | 571.0  |

---

## Notes

- **RGBT-3M**: videos contain mixed object categories (smoke / fire / person); category is reported as `all`.
- **OOTB / SatSOT / VISO-SOT**: no official train/val/test split → reported as `no_split`.
- **SAT-MTB**: `car` category is absent from `data_split.xlsx` → reported as `no_split`; the 6 `train` sequences with `no_split` are camera-named sequences not assigned in the xlsx.
- **VISO-COCO**: COCO splits (train/val/test) cannot be mapped back to individual SOT videos because image filenames are independently renumbered per split. `val` split only contains car and plane; `test` split contains all four categories.
- **BIRDSAI**: category assigned as the dominant species (highest annotation count) per video; a single video may contain multiple species.
