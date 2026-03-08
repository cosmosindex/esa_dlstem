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
