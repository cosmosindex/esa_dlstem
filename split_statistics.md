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
Single category: `animal`.

### tracking_split="perfect" (no occluded frames)

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

| Split     | Videos | Frames  | Boxes   |
|-----------|-------:|--------:|--------:|
| train     |     32 |  21,209 |  87,199 |
| val       |      5 |   3,258 |   6,856 |
| test      |     11 |  12,236 |  71,905 |
| **Total** | **48** |**36,703**|**165,960**|
