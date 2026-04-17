# OOTB — Split balance across sequence attributes

> Investigation of whether the previous class-only stratified split yields
> balanced coverage of the 12 sequence attributes defined in the OOTB paper
> (Chen et al., ISPRS 2024: DEF, IPR, PO, FO, IV, MB, BC, OON, SA, LT, IM, AM).

Analysis script: [`tools/analyze_ootb_split.py`](../tools/analyze_ootb_split.py).
Run with `micromamba run -n esa_dlstem python tools/analyze_ootb_split.py`.

---

## 1. Dataset-wide attribute prevalence

Every `anno/<seq>.txt` file contains 16 binary flags: 4 category one-hot
(car, ship, train, plane) followed by 12 attributes in this order:

```
DEF, IPR, PO, FO, IV, MB, BC, OON, SA, LT, IM, AM
```

Across all 110 sequences:

| Attr | # positive | % of dataset |
|------|-----------:|-------------:|
| DEF  | 16  | 14.5 % |
| IPR  | 42  | 38.2 % |
| PO   | 17  | 15.5 % |
| FO   |  8  |  7.3 % |
| IV   | 62  | 56.4 % |
| MB   | 45  | 40.9 % |
| BC   | 81  | 73.6 % |
| OON  | 20  | 18.2 % |
| SA   | 46  | 41.8 % |
| LT   | 46  | 41.8 % |
| IM   | 12  | 10.9 % |
| AM   | 17  | 15.5 % |

FO (8 seqs) and IM (12 seqs) are the rare ones — with a 10 % test target
they sit right at the edge of being assigned to zero test sequences by a
class-only split.

---

## 2. Previous split — class-stratified 80/10/10, `seed=42`

The previous split balances **class** well but not attributes:

| Attr | total | train | val | test | train % | val % | test % |
|------|------:|------:|----:|-----:|--------:|------:|------:|
| DEF  | 16  | 13 |  2 |  1 | 81.2 | 12.5 |  6.2 |
| IPR  | 42  | 32 |  5 |  5 | 76.2 | 11.9 | 11.9 |
| PO   | 17  | 14 |  **0** |  3 | 82.4 |  **0.0** | 17.6 |
| FO   |  8  |  4 |  1 |  3 | 50.0 | 12.5 | 37.5 |
| IV   | 62  | 47 |  6 |  9 | 75.8 |  9.7 | 14.5 |
| MB   | 45  | 38 |  4 |  3 | 84.4 |  8.9 |  6.7 |
| BC   | 81  | 67 |  7 |  7 | 82.7 |  8.6 |  8.6 |
| OON  | 20  | 15 |  3 |  2 | 75.0 | 15.0 | 10.0 |
| SA   | 46  | 38 |  4 |  4 | 82.6 |  8.7 |  8.7 |
| LT   | 46  | 38 |  2 |  6 | 82.6 |  4.3 | 13.0 |
| IM   | 12  | 10 |  **0** |  2 | 83.3 |  **0.0** | 16.7 |
| AM   | 17  | 14 |  2 |  1 | 82.4 | 11.8 |  5.9 |

**Mean RMS deviation from target 80/10/10 across 12 attrs: 0.0543**

Two attributes (**PO** and **IM**) have zero sequences in val, so a
per-attribute val breakdown is not reportable under the old split. FO and
DEF skew heavily (FO test = 37.5 %, DEF test = 6.2 %).

---

## 3. Options explored

### Option A — iterative stratification on class ⊕ 12 attributes

Sechidis et al. 2011 multi-label iterative stratification, using the 4-way
class one-hot concatenated with the 12 attributes as labels.

| | train | val | test |
|--|------:|----:|-----:|
| car        | 36 | 5 | 4 |
| plane      | 20 | 2 | 3 |
| ship       | 22 | 4 | 4 |
| train      |  8 | 1 | 1 |

Attribute mean RMS deviation: **0.0250** (2.2× better than the previous split).
Every attribute has ≥ 1 sequence in every split, and every class keeps ≥ 1
too — OOTB's smallest class (`train`, 10 seqs) is large enough that
iterative stratification already lands it at 8/1/1 without any special
handling.

### Option B — hybrid (round-robin tiny classes + iterative strat)

The same hybrid mechanism adopted for SV248S, parameterised by a
small-class threshold. With `_SMALL_CAT_THRESH = 10` (SV248S default) the
OOTB `train` class is pulled into round-robin pre-assignment and lands at
3/3/4, which is far worse than 8/1/1. Because no OOTB class needs the
safety net, `_SMALL_CAT_THRESH` is set to **5** in the OOTB splitter — no
class has ≤ 5 sequences, so the pre-assignment step is a no-op and the
result matches Option A.

Implementation: `OOTBDataset._hybrid_split` (kept with the same name so
the splitter structure mirrors SV248S).

Result (produced by `OOTBDataset._hybrid_split`, `seed=42`):

| | train | val | test |
|--|------:|----:|-----:|
| car        | 36 | 5 | 4 |
| plane      | 20 | 2 | 3 |
| ship       | 22 | 4 | 4 |
| train      |  8 | 1 | 1 |
| **TOTAL**  | 86 | 12 | 12 |

| Attr | total | train | val | test | train % | val % | test % |
|------|------:|------:|----:|-----:|--------:|------:|------:|
| DEF  | 16  | 13 |  2 |  1 | 81.2 | 12.5 |  6.2 |
| IPR  | 42  | 34 |  4 |  4 | 81.0 |  9.5 |  9.5 |
| PO   | 17  | 13 |  2 |  2 | 76.5 | 11.8 | 11.8 |
| FO   |  8  |  6 |  1 |  1 | 75.0 | 12.5 | 12.5 |
| IV   | 62  | 51 |  5 |  6 | 82.3 |  8.1 |  9.7 |
| MB   | 45  | 36 |  5 |  4 | 80.0 | 11.1 |  8.9 |
| BC   | 81  | 60 | 10 | 11 | 74.1 | 12.3 | 13.6 |
| OON  | 20  | 16 |  3 |  1 | 80.0 | 15.0 |  5.0 |
| SA   | 46  | 37 |  4 |  5 | 80.4 |  8.7 | 10.9 |
| LT   | 46  | 35 |  4 |  7 | 76.1 |  8.7 | 15.2 |
| IM   | 12  | 10 |  1 |  1 | 83.3 |  8.3 |  8.3 |
| AM   | 17  | 13 |  2 |  2 | 76.5 | 11.8 | 11.8 |

**Mean RMS deviation: 0.0250.** Every attribute has ≥ 1 positive sequence
in every split (including every attribute in test), and every class does too.

Reproduce with `python tools/report_ootb_split.py`.

---

## 4. Side-by-side summary

| Metric | Previous (class only) | **Adopted (class + 12 attrs)** |
|---|---:|---:|
| Attr mean RMS dev            | 0.0543 | **0.0250** |
| Attrs with an empty split    | 2 (PO val, IM val) | **0** |
| Classes with an empty split  | 0 | **0** |
| Worst attr deviation         | FO 50.0 / 12.5 / 37.5 | DEF 81.2 / 12.5 / 6.2 |

---

## 5. Tradeoffs

1. **Split composition changes.** Switching splitters re-shuffles which
   sequences land in each split. Any checkpoint / result that was
   evaluated on the old OOTB val or test set is no longer directly
   comparable. Re-run the SOT evaluation matrix (SAM2, SAM3, SAMURAI,
   OSTrack, ODTrack, LoRAT, SiamRPN) on the new splits before comparing
   to historical numbers.
2. **Small class coverage.** `train` (10 seqs) still lands at 8/1/1 under
   the new split, matching the old behaviour.
3. **Seeded reproducibility still holds** — the hybrid is deterministic
   at `seed=42`.

---

## 6. Status (2026-04-17)

Implemented in `datasets/ootb.py` as `OOTBDataset._hybrid_split`:
- `_build_index` now loads each `anno/<seq>.txt` into `self._attr_cache`
  (keeping only the 12-attribute suffix; the category one-hot is already
  recovered from the directory name).
- `_stratified_split` replaced by `_hybrid_split` + `_iterative_stratify`.
- `_SMALL_CAT_THRESH = 5` — no OOTB class is small enough to trigger
  round-robin pre-assignment; the step is kept for parity with the
  SV248S splitter.
- Category hashing uses `zlib.crc32` so the split is stable across
  Python sessions (built-in `hash` is salted per-process).
- `docs/split_statistics.md` updated.
- Pending: re-run the OOTB SOT eval matrix. Old results are no longer
  directly comparable because both test-set and val-set composition
  changed.
