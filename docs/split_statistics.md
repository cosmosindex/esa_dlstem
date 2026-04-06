# Train / Val / Test Split Statistics

> Per-dataset breakdown of video counts by split and category.
> Splits without an official definition are created via **stratified random split** (seeded for reproducibility).

---

## OOTB

Split strategy: **80 / 10 / 10** stratified by category, `seed=42`.

| Category | Train | Val | Test | Total |
|----------|------:|----:|-----:|------:|
| car      |    36 |   4 |    5 |    45 |
| plane    |    20 |   2 |    3 |    25 |
| ship     |    24 |   3 |    3 |    30 |
| train    |     8 |   1 |    1 |    10 |
| **Total**|**88** |**10**|**12**|**110**|

---

## BIRDSAI

Split strategy: **Official** TrainReal/TestReal from dataset.
TrainReal → train. Val split from TestReal sequences (30/70 stratified by source video, `seed=42`).
Two categories: `animal` (class=0) and `human` (class=1). Category determined by looking up track_id in MOT CSV.

### tracking_split="perfect" (no occluded frames)

| Category   | Train | Val | Test | Total |
|------------|------:|----:|-----:|------:|
| animal     |   322 |  13 |   83 |   418 |
| human      |    61 |  12 |   55 |   128 |
| **Total**  |**383**|**25**|**138**|**546**|

| Split     | Sequences | Frames | Source Videos |
|-----------|----------:|-------:|--------------:|
| train     |       383 | 70,261 |            24 |
| val       |        25 |  5,575 |             3 |
| test      |       138 | 28,170 |             7 |
| **Total** |   **546** |**104,006**|        **34** |

### tracking_split="full" (includes occluded frames, skipped during loading)

| Split     | Sequences | Valid Frames | Source Videos |
|-----------|----------:|-------------:|--------------:|
| train     |       268 |       82,004 |            32 |
| val       |        23 |        6,109 |             5 |
| test      |        73 |       64,171 |            11 |
| **Total** |   **364** | **152,284**  |        **48** |

---

## BIRDSAI_MOT

Split strategy: TrainReal → **train**. TestReal → **val (30%) + test (70%)** by video, `seed=42`.
Uses raw MOT CSV annotations (multiple objects per frame).
Two categories: `animal` (class=0) and `human` (class=1).

| Category | Train | Val | Test | Total |
|----------|------:|----:|-----:|------:|
| animal   |    27 |   2 |   10 |    39 |
| human    |     6 |   3 |    2 |    11 |
| **Total**|**32** |**5**|**11**|**48** |

> Note: 1 train video and 1 test video contain both animal and human annotations (mixed).

| Split     | Videos | Frames  | Boxes   |
|-----------|-------:|--------:|--------:|
| train     |     32 |  21,209 |  87,199 |
| val       |      5 |   3,258 |   6,856 |
| test      |     11 |  12,236 |  71,905 |
| **Total** | **48** |**36,703**|**165,960**|

---

## LMOD

Split strategy: **80 / 10 / 10** by frame within each sequence (temporal order, no shuffle).
8 sequences (Seq1–Seq8), Pascal VOC XML annotations (per-frame detection, no tracking IDs).
Label typos fixed: `cat` → `car`, `w` → `car` (Seq2).

| Category | Train Boxes | Val Boxes | Test Boxes | Total Boxes |
|----------|----------:|--------:|---------:|----------:|
| car      |   375,733 |  42,695 |   41,295 |   459,723 |
| plane    |     7,643 |     862 |      885 |     9,390 |
| ship     |     8,424 |   1,056 |    1,056 |    10,536 |
| train    |       486 |     103 |      104 |       693 |
| **Total**|**392,286**|**44,716**|**43,340**|**480,342**|

| Split     | Videos | Frames | Boxes   |
|-----------|-------:|-------:|--------:|
| train     |      8 |  3,249 | 392,286 |
| val       |      8 |    406 |  44,716 |
| test      |      8 |    407 |  43,340 |
| **Total** | **24** |**4,062**|**480,342**|

---

## IRSatVideo-LEO

Split strategy: **Official** train/test from dataset.
Val carved from **30% of official test** (stratified by geographic region, `seed=42`).
Single category: `target` (satellite objects). Track IDs from XML object names (`target0`, `target1`, …).
Binary segmentation masks available per frame.

| Region             | Train | Val | Test | Total |
|--------------------|------:|----:|-----:|------:|
| AfricaWest         |     1 |   0 |    0 |     1 |
| EastAfrica         |    17 |   1 |    1 |    19 |
| EastAustralia      |    12 |   1 |    3 |    16 |
| EastEurope         |    15 |   1 |    1 |    17 |
| EastNorthAisa      |    13 |   2 |    6 |    21 |
| NorthAfrica        |    10 |   0 |    1 |    11 |
| NorthAmericaEast   |    13 |   1 |    2 |    16 |
| NorthAmericaNorth  |    11 |   0 |    0 |    11 |
| NorthAmericaWest   |     7 |   1 |    2 |    10 |
| NorthAustralia     |    14 |   1 |    2 |    17 |
| NorthEurope        |     9 |   0 |    1 |    10 |
| NorthNorthAisa     |     8 |   1 |    2 |    11 |
| WestAfrica         |     8 |   1 |    2 |    11 |
| WestAustralia      |     5 |   1 |    1 |     7 |
| WestEurope         |     7 |   2 |    3 |    12 |
| WestNorthAisa      |    10 |   0 |    0 |    10 |
| **Total**          |**160**|**13**|**27**|**200**|

| Split     | Videos | Frames  |
|-----------|-------:|--------:|
| train     |    160 |  73,699 |
| val       |     13 |   5,267 |
| test      |     27 |  12,055 |
| **Total** |**200** |**91,021**|

---

## SatSOT

Split strategy: **80 / 10 / 10** stratified by category, `seed=42`. No official split.
SOT dataset — single object per sequence, bbox format xywh → xyxy.
Frames with `none` GT (target absent/occluded) return empty annotations.

| Category | Train | Val | Test | Total |
|----------|------:|----:|-----:|------:|
| car      |    52 |   6 |    7 |    65 |
| plane    |     7 |   1 |    1 |     9 |
| ship     |     3 |   1 |    1 |     5 |
| train    |    21 |   3 |    2 |    26 |
| **Total**|**83** |**11**|**11**|**105**|

| Split     | Videos | Frames |
|-----------|-------:|-------:|
| train     |     83 | 22,718 |
| val       |     11 |  2,478 |
| test      |     11 |  2,468 |
| **Total** |**105** |**27,664**|

---

## AIR-MOT

Split strategy: **80 / 10 / 10** stratified by dominant class per sequence, `seed=42`. No official split.
MOT dataset from Jilin-1 satellite video. 100 sequences total, **31 have empty annotations** and are excluded.
2 classes: `airplane` (class 1 in raw annotations) and `car` (class 2).
Some sequences have black padding bars (bottom and/or right); annotations are within valid content.

| Category  | Train | Val | Test | Total |
|-----------|------:|----:|-----:|------:|
| airplane  |    24 |   3 |    3 |    30 |
| car       |    31 |   4 |    4 |    39 |
| **Total** |**55** |**7**|**7** |**69** |

| Split     | Videos | Frames |
|-----------|-------:|-------:|
| train     |     55 | 15,559 |
| val       |      7 |  2,282 |
| test      |      7 |  2,099 |
| **Total** | **69** |**19,940**|

---

## SAT-MTB

Split strategy: **Official** train/test from `data_split.xlsx`.
Val carved from **30 % of official test** (stratified by category, `seed=42`).
Multi-task dataset — annotation availability varies by category and task.

### task=det_hbb (Detection — Horizontal Bounding Boxes)

3 categories (no car). 142 videos total.

| Category  | Train | Val | Test | Total |
|-----------|------:|----:|-----:|------:|
| airplane  |    37 |   8 |   17 |    62 |
| ship      |    41 |   9 |   20 |    70 |
| train     |     6 |   1 |    3 |    10 |
| **Total** |**84** |**18**|**40**|**142**|

| Split     | Videos | Frames |
|-----------|-------:|-------:|
| train     |     84 | 18,932 |
| val       |     18 |  4,153 |
| test      |     40 |  9,868 |
| **Total** |**142** |**32,953**|

### task=det_obb (Detection — Oriented Bounding Boxes)

3 categories (no car). Fewer sequences have OBB annotations. 106 videos total.

| Category  | Train | Val | Test | Total |
|-----------|------:|----:|-----:|------:|
| airplane  |    30 |   6 |   15 |    51 |
| ship      |    26 |   7 |   15 |    48 |
| train     |     5 |   1 |    1 |     7 |
| **Total** |**61** |**14**|**31**|**106**|

| Split     | Videos | Frames |
|-----------|-------:|-------:|
| train     |     61 | 12,573 |
| val       |     14 |  3,104 |
| test      |     31 |  7,101 |
| **Total** |**106** |**22,778**|

### task=mot (Multi-Object Tracking)

4 categories (includes car). 237 videos total.

| Category  | Train | Val | Test | Total |
|-----------|------:|----:|-----:|------:|
| airplane  |    37 |   8 |   17 |    62 |
| car       |    62 |   9 |   21 |    92 |
| ship      |    42 |   8 |   20 |    70 |
| train     |     8 |   2 |    3 |    13 |
| **Total** |**149**|**27**|**61**|**237**|

| Split     | Videos | Frames |
|-----------|-------:|-------:|
| train     |    149 | 28,902 |
| val       |     27 |  5,538 |
| test      |     61 | 13,950 |
| **Total** |**237** |**48,390**|

### task=seg (Instance Segmentation)

3 categories (no car). Same sequences as det_hbb. 142 videos total.

| Category  | Train | Val | Test | Total |
|-----------|------:|----:|-----:|------:|
| airplane  |    37 |   8 |   17 |    62 |
| ship      |    41 |   9 |   20 |    70 |
| train     |     6 |   1 |    3 |    10 |
| **Total** |**84** |**18**|**40**|**142**|

---

## VISO

Split strategy: **Official** COCO/VOC frame-level split mapped to sequence-level
by majority vote. Ship has no val (only 2 sequences); the *train* category has no
val/test (single sequence). Annotation format varies by category: car/train use
comma-delimited xywh, plane/ship use space-delimited xyxy.

| Category | Train | Val | Test | Total |
|----------|------:|----:|-----:|------:|
| car      |    24 |   4 |   10 |    38 |
| plane    |     4 |   1 |    1 |     6 |
| ship     |     1 |   0 |    1 |     2 |
| train    |     1 |   0 |    0 |     1 |
| **Total**|**30** |**5**|**12**|**47** |

| Split     | Videos | Frames  |
|-----------|-------:|--------:|
| train     |     30 |  10,902 |
| val       |      5 |   1,741 |
| test      |     12 |   3,561 |
| **Total** | **47** |**16,204**|

---

## SV248S

Split strategy: **80 / 10 / 10** stratified by category, `seed=42`. No official split.
SOT dataset — single object per sequence, bbox format xywh → xyxy.
Frames with state=1 (invisible) return empty annotations; state=2 (occluded) keep bbox.
Polygon annotations available for mask-level segmentation.

| Category  | Train | Val | Test | Total |
|-----------|------:|----:|-----:|------:|
| car       |   162 |  20 |   20 |   202 |
| car-large |    30 |   4 |    3 |    37 |
| plane     |     4 |   1 |    1 |     6 |
| ship      |     1 |   1 |    1 |     3 |
| **Total** |**197**|**26**|**25**|**248**|

| Split     | Videos | Frames  |
|-----------|-------:|--------:|
| train     |    197 | 124,403 |
| val       |     26 |  16,111 |
| test      |     25 |  16,107 |
| **Total** |**248** |**156,621**|

---

## SDM-Car

Split strategy: **Official split** from dataset (train / validation / test directories).

| Category  | Train | Val | Test | Total |
|-----------|------:|----:|-----:|------:|
| car       |    64 |  15 |   20 |    99 |

| Split     | Videos | Frames |
|-----------|-------:|-------:|
| train     |     64 | 10,483 |
| val       |     15 |  2,409 |
| test      |     20 |  3,531 |
| **Total** | **99** |**16,423**|
