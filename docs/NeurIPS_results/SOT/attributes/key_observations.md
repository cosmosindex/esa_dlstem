# Key Observations — Per-Attribute SOT Performance

Per-sequence aggregation, PR over CLE ∈ [0, 50] px. Numbers come from
`/work/anon/experiments/NeurIPS/SOT_whole_dataset_04_22/analysis/unified_attributes/`:

- `unified_attr.csv` — 6 unified attributes (BC, IV, ROT, OCC, SOB, DEF), averaged across the datasets that annotate them.
- `unique_attr.csv` — 18 dataset-unique attributes, evaluated on the single annotating dataset.
- `per_dataset_attr.csv` — intermediate (tracker × dataset × unified attribute).
- `attribute_counts.csv` — sequence and frame counts per attribute.

Trackers compared: SiamRPN++, OSTrack-384, ODTrack, LoRAT-g378, SAM 2, SAMURAI, SAM 3.

---

## 1. Unified attributes (BC, IV, ROT, OCC, SOB, DEF)

Sub-table layout: 6 rows × 3 metrics = **18 cells**.

- **SAM 3 wins all 18 cells.** Best in every (attribute, metric) combination.
- **LoRAT-g378 is the consistent runner-up — 16/18 cells.**
- The two exceptions are both on **OCC**:
  - **OCC SR / NPR**: 2nd is **SAMURAI** (0.289 / 0.344) — narrowly above LoRAT (0.286 / 0.344, by 0.0002 on NPR). The SAM-family memory mechanism slightly outperforms LoRAT once any-occlusion is collapsed across spatial (PO/FO) and temporal (STO/LTO/CO) axes.
  - **OCC PR**: 2nd is **ODTrack** (0.486). Pretrained transformers with strong center localization keep PR high under occlusion even when SR is moderate.
- The unified OCC numbers hide a real qualitative split — see §2 below.

## 2. Dataset-unique attributes (18 attributes)

54 cells (18 rows × 3 metrics).

- **SAM 3: 41/54 best.** Dominates static / appearance / spatial-occlusion / motion attributes, especially:
  - Spatial occlusion (POC, FOC) — every metric.
  - Aspect / quality (OON, LQ) — every metric.
  - Tiny / motion (TO, BJT, IM, AM) — every metric.
  - LT, MB — most metrics.
- **LoRAT-g378: 9/54 best.** Concentrated on time-structured SV248S attributes:
  - **LTO** (long-term occlusion, >50 consecutive OCC frames) — best on SR / NPR / PR.
  - **CO** (continuous occlusion, two or more STO/LTO events) — best on SR; tied on NPR/PR with SAM 3.
  - **BCH** (background change), **ND** (natural disturbance), **IBG** (indistinguishable bg.), **SM** (slow motion) — best on SR / NPR.
  - **ARC** (aspect-ratio change) — best on every metric.
  - **MB** SR.
- **The OCC story splits cleanly along annotation axis:**
  - *Spatial* axis (PO/FO/POC/FOC): **SAM 3 best** — promptable mask + memory recovers well from instantaneous full/partial coverage.
  - *Temporal* axis (LTO/CO): **LoRAT best** — long-context ViT excels when occlusion is sustained over many frames.
  - This is exactly the pattern the unified OCC row papers over: averaging spatial + temporal occlusion gives SAM 3 a slight overall lead, but LTO/CO alone reverse the ranking.

## 3. Cross-tracker patterns

- **SiamRPN++ (CVPR'19)** is rarely competitive in the global ranking, but stays on the podium for **PR-style metrics under conditional partial-information regimes**: STO PR (2nd), LTO PR (2nd), BJT NPR/PR (2nd). Lightweight Siamese correlation is still robust at coarse localization.
- **ODTrack** mostly ranks PR-2nd on POC/FOC/TO — token propagation is good at *where* even when *how-tightly* (SR/NPR) is weak.
- **OSTrack-384** rarely wins or finishes 2nd; consistent mid-pack on most attributes. AM PR is its lone 2nd-place.
- **SAM 2** vs **SAMURAI**: SAMURAI's motion-aware memory consistently nudges SAM 2 in attributes with steady motion (POC, IM, IV, ROT) but not enough to challenge LoRAT or SAM 3 anywhere.
- **SAM 3** improvements over SAM 2 are largest on text-/concept-driven attributes (OON, LQ, LT, BJT, IM, AM) — consistent with SAM 3's added concept conditioning over SAM 2's mask-only segmentation.

## 4. Sample-size caveats

| Attribute | n_seq | Notes |
|---|---:|---|
| LTO | 9 | Very small — LTO numbers should be read as indicative, not statistically tight. |
| FOC | 12 | Small. |
| IM (OOTB) | 12 | Small. |
| LQ | 13 | Small. |
| BJT | 14 | Small. |
| ND | 15 | Small. |
| AM (OOTB) | 17 | Small. |
| OON | 20 | Borderline. |
| TO | 21 | Borderline. |
| DEF | 22 (cross-dataset) | Borderline. |

Larger / safer subsets: SOB (252), ROT (207), IV (144), OCC (148), STO (79), CO (34), BCH (91), IBG (76), SM (58), MB (45), LT (46), BC (126).

## 5. Implications for the paper narrative

1. The headline "SAM 3 dominates" claim is robust at the **whole-dataset level**, the **tiny subset**, and on most unified / unique attributes — but the **temporal-occlusion case (LTO, CO)** is a clean counterexample where pretrained long-context ViT (LoRAT) beats foundation models. Worth highlighting as a non-trivial finding, not glossing over.
2. The unified OCC row hides this. The supplementary unique-attribute table is what surfaces it; consider referencing the supplementary explicitly when discussing OCC in the main text.
3. SiamRPN++'s residual strength on PR-only / coarse-localization metrics under partial occlusion / jitter suggests the discriminative-correlation paradigm is not strictly dominated; useful as a low-cost baseline narrative.
4. SatSOT IV is a 3-sequence subset → unified IV's average is dominated by SV248S (79) and OOTB (62). Mention in caption / discussion if IV-specific claims are made.
