# SatSOT — Split balance across sequence attributes

> Investigation of whether the previous class-only stratified split yields
> balanced coverage of the 11 sequence attributes defined in the SatSOT paper
> (ARC, BC, BJT, DEF, FOC, IV, LQ, POC, ROT, SOB, TO).

Analysis script: [`tools/analyze_satsot_split.py`](../tools/analyze_satsot_split.py).
Run with `micromamba run -n esa_dlstem python tools/analyze_satsot_split.py`.

---

## 1. Dataset-wide attribute prevalence

Attributes are stored as a per-sequence list of strings in `SatSOT.json`.
The canonical 11-attribute ordering used for stratification is:

```
ARC, BC, BJT, DEF, FOC, IV, LQ, POC, ROT, SOB, TO
```

Across all 105 sequences:

| Attr | # positive | % of dataset |
|------|-----------:|-------------:|
| ARC  | 26 | 24.8 % |
| BC   | 45 | 42.9 % |
| BJT  | 14 | 13.3 % |
| DEF  |  6 |  5.7 % |
| FOC  | 12 | 11.4 % |
| IV   |  3 |  2.9 % |
| LQ   | 13 | 12.4 % |
| POC  | 34 | 32.4 % |
| ROT  | 56 | 53.3 % |
| SOB  | 27 | 25.7 % |
| TO   | 21 | 20.0 % |

**IV** (3 positives) and **DEF** (6 positives) are very rare — at 80/10/10
target ratios neither can be split into each bucket by a naive stratifier.

Categories are also skewed: car 65 / train 26 / plane 9 / ship 5 — plane
and ship are tiny and need round-robin pre-assignment to survive the split.

---

## 2. Previous split — class-stratified 80/10/10, `seed=42`

The previous split balanced **class** well but not attributes:

| | train | val | test |
|--|------:|----:|-----:|
| car   | 52 | 6 | 7 |
| plane |  7 | 1 | 1 |
| ship  |  3 | 1 | 1 |
| train | 21 | 3 | 2 |
| **TOTAL** | 83 | 11 | 11 |

| Attr | total | train | val | test | train % | val % | test % |
|------|------:|------:|----:|-----:|--------:|------:|------:|
| ARC  | 26 | 21 | 3 | 2 | 80.8 | 11.5 |  7.7 |
| BC   | 45 | 37 | 4 | 4 | 82.2 |  8.9 |  8.9 |
| BJT  | 14 | 10 | 1 | 3 | 71.4 |  7.1 | 21.4 |
| DEF  |  6 |  4 | 1 | 1 | 66.7 | 16.7 | 16.7 |
| FOC  | 12 |  7 | 2 | 3 | 58.3 | 16.7 | 25.0 |
| IV   |  3 |  3 | **0** | **0** | 100.0 |  **0.0** |  **0.0** |
| LQ   | 13 | 12 | 1 | **0** |  92.3 |  7.7 |  **0.0** |
| POC  | 34 | 25 | 4 | 5 | 73.5 | 11.8 | 14.7 |
| ROT  | 56 | 50 | 4 | 2 | 89.3 |  7.1 |  3.6 |
| SOB  | 27 | 22 | 1 | 4 | 81.5 |  3.7 | 14.8 |
| TO   | 21 | 18 | 1 | 2 | 85.7 |  4.8 |  9.5 |

**Mean RMS deviation from target 80/10/10 across 11 attrs: 0.0734**

Three attributes fail the per-split coverage requirement:

- **IV** has 0 sequences in both val and test.
- **LQ** has 0 sequences in test.

A per-attribute precision/success breakdown on val/test is therefore not
reportable for IV (anywhere) or LQ (test) under the old split.

---

## 3. Options explored

### Option A — iterative stratification on class ⊕ 11 attributes

Sechidis et al. 2011 multi-label iterative stratification, using the 4-way
class one-hot concatenated with the 11 attributes as labels.

| | train | val | test |
|--|------:|----:|-----:|
| car   | 52 | 7 | 6 |
| plane |  7 | 1 | 1 |
| ship  |  4 | 1 | **0** |
| train | 21 | 2 | 3 |
| **TOTAL** | 84 | 11 | 10 |

Attribute mean RMS deviation: **0.0370** (2× better than previous). But:
- **ship** has 0 sequences in test — pure iterative strat dumps the 5 ships
  across train/val only, since ship's per-split target (0.5 per bucket) is
  below 1 and the greedy rule picks train.
- **IV** still has 0 in val and test for the same reason (3 positives, all
  three go to train because its per-split target is 2.4 / 0.3 / 0.3 and the
  deficit rule always picks train for the first 3 placements).

### Option B — hybrid (**adopted, `datasets/satsot.py` as of 2026-04-20**)

Three-step splitter designed so **test covers every category and every attribute**:

1. **Tiny-class round-robin** (n ≤ `_SMALL_CAT_THRESH = 10`). plane (9)
   and ship (5) get pre-assigned test → val → train, seeded by
   `(42, crc32(category))`. Forces ≥ 1 sequence per split for every class.
2. **Rare-attribute round-robin** (positive count ≤ `_RARE_ATTR_THRESH = 9`).
   For each rare attribute, find the splits that no pre-assigned positive
   currently covers, then force one unassigned positive into each missing
   split. IV (3 positives) and DEF (6 positives) need this on SatSOT —
   without it, iterative strat's deficit rule dumps all their positives into
   train. Seeded by `(42, crc32(attr_name))`.
3. **Iterative stratification** (Sechidis et al. 2011) on all remaining
   sequences, using the class one-hot ⊕ 11 attribute flags as labels, with
   the step-1/step-2 pre-assignments passed in as hard constraints.

Result (produced by `SatSOTDataset._hybrid_split`, `seed=42`):

| | train | val | test |
|--|------:|----:|-----:|
| car   | 52 | 6 | 7 |
| plane |  3 | 3 | 3 |
| ship  |  1 | 2 | 2 |
| train | 21 | 3 | 2 |
| **TOTAL** | 77 | 14 | 14 |

| Attr | total | train | val | test | train % | val % | test % |
|------|------:|------:|----:|-----:|--------:|------:|------:|
| ARC  | 26 | 21 | 3 | 2 | 80.8 | 11.5 |  7.7 |
| BC   | 45 | 36 | 4 | 5 | 80.0 |  8.9 | 11.1 |
| BJT  | 14 | 11 | 1 | 2 | 78.6 |  7.1 | 14.3 |
| DEF  |  6 |  4 | 1 | 1 | 66.7 | 16.7 | 16.7 |
| FOC  | 12 | 10 | 1 | 1 | 83.3 |  8.3 |  8.3 |
| IV   |  3 |  1 | 1 | 1 | 33.3 | 33.3 | 33.3 |
| LQ   | 13 | 10 | 1 | 2 | 76.9 |  7.7 | 15.4 |
| POC  | 34 | 25 | 4 | 5 | 73.5 | 11.8 | 14.7 |
| ROT  | 56 | 45 | 6 | 5 | 80.4 | 10.7 |  8.9 |
| SOB  | 27 | 21 | 3 | 3 | 77.8 | 11.1 | 11.1 |
| TO   | 21 | 17 | 2 | 2 | 81.0 |  9.5 |  9.5 |

**Mean RMS deviation: 0.0564.** Every attribute has ≥ 1 positive sequence
in every split, and every class has ≥ 1 too. IV's 33/33/33 row inflates the
global RMS — it's the price of making IV reportable at all, since its only
3 positives can't simultaneously be rare and 80/10/10-balanced.

Reproduce with `python tools/analyze_satsot_split.py`.

---

## 4. Side-by-side summary

| Metric | Previous (class only) | Option A (class + attrs) | **Adopted (hybrid)** |
|---|---:|---:|---:|
| Attr mean RMS dev            | 0.0734 | 0.0370 | 0.0564 |
| Attrs with an empty split    | 2 (IV val+test, LQ test) | 1 (IV val+test) | **0** |
| Classes with an empty split  | 0 | 1 (ship test) | **0** |
| Test covers every category?  | ✅ | ❌ (no ship) | ✅ |
| Test covers every attribute? | ❌ (IV, LQ missing) | ❌ (IV missing) | ✅ |

---

## 5. Tradeoffs

1. **Split composition changes.** Switching splitters re-shuffles which
   sequences land in each split. Any result that was evaluated on the old
   SatSOT val or test set is no longer directly comparable. Re-run the
   SOT evaluation matrix (SiamRPN, OSTrack, ODTrack, LoRAT, SAMURAI, SAM 2,
   SAM 3) on the new splits before comparing to historical numbers.
2. **Train shrinks 83 → 77, val/test grow 11 → 14 each.** The overall
   test fraction moves from 10.5 % to 13.3 %. This is the cost of forcing
   plane to 3/3/3 and ship to 1/2/2 instead of 7/1/1 and 3/1/1 — 4 extra
   planes and 2 extra ships move out of train so val/test actually have
   small-class coverage.
3. **IV gets 33/33/33 instead of 100/0/0.** The per-split attribute balance
   is intentionally sacrificed for the small/rare attributes so that they
   remain reportable on val/test. With only 3 IV-positive sequences there
   is no "balanced" alternative — either you report IV or you don't.
4. **Seeded reproducibility still holds** — the hybrid is deterministic at
   `seed=42`. Category and attribute hashing use `zlib.crc32` so the split
   is stable across Python sessions.

---

## 6. Status (2026-04-20)

Implemented in `datasets/satsot.py` as `SatSOTDataset._hybrid_split`:

- `_build_index` now encodes each sequence's `attr` list from `SatSOT.json`
  into an 11-element binary vector in `self._attr_cache` (canonical
  `ATTR_NAMES` order). Sequences missing from the metadata file get an
  empty row and only contribute to class balance.
- `_stratified_split` replaced by `_hybrid_split` + `_iterative_stratify`.
- `_SMALL_CAT_THRESH = 10` (plane=9, ship=5 both need round-robin).
- `_RARE_ATTR_THRESH = 9` (IV=3, DEF=6 both need round-robin).
- `sequence_attributes()` still returns `{video_id: [attr_name, ...]}` —
  the callback-visible format is unchanged; only the internal storage
  switched from string-list to binary vector.
- Pending: re-run the SatSOT SOT eval matrix. Old results are no longer
  directly comparable because both val-set and test-set composition changed.
