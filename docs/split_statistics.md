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
