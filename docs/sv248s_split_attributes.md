# SV248S — Split balance across sequence attributes

> Investigation of whether the current class-only stratified split yields
> balanced coverage of the 10 sequence attributes defined in the SV248S paper
> (Table 5: STO, LTO, DS, IV, BCH, SM, ND, CO, BCL, IPR).

Analysis script: [`tools/analyze_sv248s_split.py`](../tools/analyze_sv248s_split.py).
Run with `micromamba run -n esa_dlstem python tools/analyze_sv248s_split.py`.

---

## 1. Dataset-wide attribute prevalence

Every `.attr` file contains 10 binary flags, in this order:

```
STO, LTO, DS, IV, BCH, SM, ND, CO, BCL, IPR
```

Across all 248 sequences:

| Attr | # positive | % of dataset |
|------|-----------:|-------------:|
| STO  | 79  | 31.9 % |
| LTO  |  9  |  3.6 % |
| DS   | 179 | 72.2 % |
| IV   | 79  | 31.9 % |
| BCH  | 91  | 36.7 % |
| SM   | 58  | 23.4 % |
| ND   | 15  |  6.0 % |
| CO   | 34  | 13.7 % |
| BCL  | 76  | 30.6 % |
| IPR  | 109 | 44.0 % |

LTO (9 seqs) and ND (15 seqs) are the rare ones — these drive most of the
imbalance below.

---

## 2. Current split — class-stratified 80/10/10, `seed=42`

Current split balances **class** well but not attributes:

| Attr | total | train | val | test | train % | val % | test % |
|------|------:|------:|----:|-----:|--------:|------:|------:|
| STO  | 79  | 64 |  5 | 10 | 81.0 |  6.3 | 12.7 |
| LTO  |  9  |  8 |  0 |  1 | 88.9 |  **0.0** | 11.1 |
| DS   | 179 | 141 | 19 | 19 | 78.8 | 10.6 | 10.6 |
| IV   | 79  | 66 |  9 |  4 | 83.5 | 11.4 |  5.1 |
| BCH  | 91  | 71 |  8 | 12 | 78.0 |  8.8 | 13.2 |
| SM   | 58  | 50 |  4 |  4 | 86.2 |  6.9 |  6.9 |
| ND   | 15  | 14 |  1 |  0 | 93.3 |  6.7 |  **0.0** |
| CO   | 34  | 27 |  3 |  4 | 79.4 |  8.8 | 11.8 |
| BCL  | 76  | 60 |  9 |  7 | 78.9 | 11.8 |  9.2 |
| IPR  | 109 | 83 | 14 | 12 | 76.1 | 12.8 | 11.0 |

**Mean RMS deviation from target 80/10/10 across 10 attrs: 0.0368**

Two problems stand out:

- **LTO** (long-term occlusion) has 0 sequences in val.
- **ND** (natural disturbance) has 0 sequences in test.

So if we report a per-attribute breakdown on val/test (common in SOT
benchmarks), LTO val and ND test are untestable today.

---

## 3. Options explored

### Option A — iterative stratification on class ⊕ attributes

Sechidis et al. 2011 multi-label iterative stratification, using the 4-way
class one-hot concatenated with the 10 attributes as labels.

| | train | val | test |
|--|------:|----:|-----:|
| car        | 157 | 22 | 23 |
| car-large  |  29 |  4 |  4 |
| plane      |  5  |  1 |  **0** |
| ship       |  3  |  **0** |  **0** |

Attribute mean RMS deviation: **0.0111** (3× better than current).
But rare classes get sacrificed — **no planes in test, no ships in val/test.**
With targets like 0.6 ship per split and greedy deficit rule, all 3 ships
land in train.

### Option B — hybrid (**adopted, `datasets/sv248s.py` as of 2026-04-16**)

1. **Pre-assign tiny classes** (plane, ship — both ≤ 10 seqs) round-robin in
   order test → val → train, seeded by `(42, crc32(category))`. This forces
   ≥ 1 sequence per split for every class.
2. Run iterative stratification (Sechidis et al. 2011) on **all** samples
   using the class one-hot ⊕ the 10 binary attributes as labels, with the
   pre-assignment passed in as a hard constraint.

Result (produced by `SV248SDataset._hybrid_split`, `seed=42`):

| | train | val | test |
|--|------:|----:|-----:|
| car        | 161 | 19 | 22 |
| car-large  |  29 |  4 |  4 |
| plane      |   2 |  2 |  2 |
| ship       |   1 |  1 |  1 |
| **TOTAL**  | 193 | 26 | 29 |

| Attr | total | train | val | test | train % | val % | test % |
|------|------:|------:|----:|-----:|--------:|------:|------:|
| STO  | 79  | 63  |  8 |  8 | 79.7 | 10.1 | 10.1 |
| LTO  |  9  |  7  |  1 |  1 | 77.8 | 11.1 | 11.1 |
| DS   | 179 | 143 | 17 | 19 | 79.9 |  9.5 | 10.6 |
| IV   | 79  | 63  |  8 |  8 | 79.7 | 10.1 | 10.1 |
| BCH  | 91  | 73  |  9 |  9 | 80.2 |  9.9 |  9.9 |
| SM   | 58  | 46  |  6 |  6 | 79.3 | 10.3 | 10.3 |
| ND   | 15  | 12  |  1 |  2 | 80.0 |  6.7 | 13.3 |
| CO   | 34  | 27  |  4 |  3 | 79.4 | 11.8 |  8.8 |
| BCL  | 76  | 61  |  8 |  7 | 80.3 | 10.5 |  9.2 |
| IPR  | 109 | 84  | 15 | 10 | 77.1 | 13.8 |  9.2 |

**Mean RMS deviation: 0.0104 (3.5× better than current).** Every attribute
has ≥ 1 positive sequence in every split, and every class does too.

Reproduce with `python tools/report_sv248s_split.py`.

---

## 4. Side-by-side summary

| Metric | Current (class only) | Option A (class + attrs) | **Option B (hybrid)** |
|---|---:|---:|---:|
| Attr mean RMS dev            | 0.0368 | 0.0111 | **0.0104** |
| Attrs with an empty bucket   | 2 (LTO, ND) | 0 | **0** |
| Classes with an empty bucket | 0 | 2 (plane, ship) | **0** |
| Worst attr deviation         | ND 93.3/6.7/0 | IPR 76.1/13.8/10.1 | IPR 77.1/13.8/9.2 |

---

## 5. Tradeoffs to discuss before implementing

1. **Split composition changes.** Switching splitters re-shuffles which
   sequences land in each split. Any checkpoint / result that was evaluated
   on the old val or test set stops being directly comparable. If we adopt
   Option B, we should re-run the SOT evaluations that already reported
   SV248S numbers (SAM2, ODTrack, LoRAT, OSTrack, SAMURAI, SiamRPN, etc.).
2. **Plane goes from 4/1/1 → 2/2/2.** You lose 2 planes of training data
   in exchange for a more defensible per-split plane evaluation. With only
   6 planes this is a judgement call.
3. **Small val/test shift.** 26→29 val, 25→26 test. Still roughly 10 / 10 %.
4. **Seeded reproducibility still holds** — the hybrid is deterministic at
   `seed=42`.

---

## 6. Status (2026-04-16)

Implemented in `datasets/sv248s.py` as `SV248SDataset._hybrid_split`:
- `_build_index` now loads each `.attr` file into `self._attr_cache`.
- `_stratified_split` replaced by `_hybrid_split` + `_iterative_stratify`.
- Category hashing uses `zlib.crc32` so the split is stable across Python
  sessions (built-in `hash` is salted per-process).
- `docs/split_statistics.md` updated.
- Pending: re-run the SOT eval matrix (SAM2, ODTrack, LoRAT, OSTrack,
  SAMURAI, SiamRPN, SAM3) on the new test set. Old results are no longer
  directly comparable because test-set composition changed.
