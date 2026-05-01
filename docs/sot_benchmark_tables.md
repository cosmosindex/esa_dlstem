# SOT Benchmark — Evaluation Design & Table Structure

*Updated: PR AUC range revised to [0, 50] px; tiny-object subset added with median √area < 8 px cutoff (COCO-aligned, satellite-scaled).*

---

## 1. Metrics Definition

| Symbol | Definition | Range / Type |
|---|---|---|
| **P@5** | Fraction of frames with centre-location error (CLE) < 5 px | Single-threshold scalar |
| **PR** | Precision curve AUC | CLE ∈ [0, 50] px¹ |
| **NPR** | Normalized precision curve AUC | normalized CLE ∈ [0, 0.5]² |
| **SR** | Success curve AUC | IoU ∈ [0, 1] |

> ¹ **PR AUC is integrated over CLE ∈ [0, 50] px**, following the precision plot range originally defined by OTB [Wu et al., TPAMI 2015] and adopted by subsequent generic-vision SOT benchmarks. We deviate from OOTB's [0, 30] px range to (a) avoid PR saturation on relatively larger targets in OOTB, and (b) keep PR numerically comparable across our three datasets, which span heterogeneous target-size distributions. For fine-grained small-object localization, **P@5 is reported separately as a diagnostic scalar**, and a dedicated **tiny-object subset analysis** (Section 4) further isolates the small-target regime.
>
> ² NPR AUC is integrated over normalized CLE ∈ [0, 0.5], aligning with the post-2018 mainstream SOT convention — TrackingNet [Müller et al., ECCV 2018], LaSOT [Fan et al., CVPR 2019], and GOT-10k [Huang et al., TPAMI 2021] all report normalized precision as the AUC over [0, 0.5]. This deliberately deviates from OOTB's [0, 1] range for two reasons: (a) beyond ~0.5 nearly every competitive tracker saturates, so the high-threshold segment contributes little discriminative signal and compresses the gap between methods; (b) on satellite video where GT diagonals are small, normalized CLE values above 0.5 indicate near-total localization failure, which is already captured by SR at low IoU. The single-threshold NP@0.5 used by OOTB can still be reported as a legacy scalar (`norm_precision_05`) in supplementary for direct comparability with OOTB's published numbers.

> **Selected metrics: SR, NPR, PR, P@5**
> - SR, NPR, PR aligned with TrackingNet / LaSOT / GOT-10k convention (NPR range) and OTB convention (PR range).
> - P@5 retained from OOTB as a small-object diagnostic.

### Ranking Policy

- **Primary metric**: SR (IoU-based, most universally comparable)
- **Secondary metric**: NPR (scale-invariant, preferred for small targets)
- **Reference metric**: PR (included for completeness; no best/2nd-best annotation to reduce visual noise)
- **Diagnostic metric**: P@5 (single-threshold precision at 5 px; not ranked, reported for fine-grained localization diagnostics)
- **Best / 2nd-best annotation scope**: Bold (best) and underline (2nd-best) are applied **per column, within each dataset** — i.e. for each `(dataset, metric)` pair independently. They are applied only to SR and NPR columns; PR and P@5 columns are left unannotated.
- **Cross-dataset aggregate column**: The rightmost block of the main table adds an **Avg** super-column with arithmetic mean of SR, NPR, and PR across the three datasets (equal weight per dataset). Bold / underline are also applied within this Avg super-column (SR and NPR only). P@5 is not aggregated across datasets — it is a diagnostic scalar whose meaning depends on per-dataset pixel scale, so averaging is not well-defined.

---

## 2. SOT Model List (11 models)

| Model | Venue | Architecture | Notes | Eval Tier | Datasets |
|---|---|---|---|---|---|
| SiamRPN++ | CVPR 2019 | Siamese 经典 | 全领域公共锚点，引用极高 | Pretrained | All |
| OSTrack-256 | ECCV 2022 | One-stream Transformer | 范式奠基 | Pretrained | All |
| ODTrack | AAAI 2024 | One-stream Transformer | Token propagation | Pretrained | All |
| LoRAT | ECCV 2024 | Large ViT + LoRA | Scaling ViT 方向代表 | Pretrained | All |
| LoRATv2 | NeurIPS 2025 | Causal temporal + LoRA | 时序建模 SOTA | ⚠ Pending — code not released | — |
| DreamTrack | CVPR 2025 | Temporal future prediction | 卫星匀速运动契合 | ⚠ Pending — code not released | — |
| DF | JSTARS 2022 | SV-specific CF | 唯一领域专用传统方法 | ⚠ Pending — MATLAB/Python incompatibility | — |
| SAM 2 | ICCV 2024 | Foundation Model | Zero-shot 代际对比起点 | Zero-shot | All |
| SAM 3 / SAM 3.1 | Meta Nov 2025 | Foundation Model | Text-prompted, novelty 极高 | Zero-shot | All |
| SAMURAI *(optional)* | arXiv 2024 | SAM 2 + motion-aware | 读完论文后决定是否纳入 | Zero-shot | All |
| STAR† | TGRS 2025 | SV-specific Transformer | Official fine-tuned checkpoint | Fine-tuned† | All |

> **†STAR**: Uses a publicly released checkpoint fine-tuned on satellite video data. Results on SatSOT reflect in-distribution evaluation (different split of the same source domain); results on SV248S and OOTB are zero-shot with respect to the fine-tuning corpus. Reported separately as a *domain-adapted baseline*.
>
> **—** indicates results not yet available at submission time.

---

## 3. Overall Results Table — All Sequences

### 3.1 Row Grouping (by Eval Tier)

```
─── Generic Pretrained Trackers ──────────────────────────────
  SiamRPN++ / OSTrack-256 / ODTrack / LoRAT

─── Foundation Model Zero-shot ───────────────────────────────
  SAM 2 / SAMURAI (if included) / SAM 3 / SAM 3.1

─── Domain-Adapted (†) ───────────────────────────────────────
  STAR†
```

### 3.2 Table Header

```
                    SatSOT                    SV248S                    OOTB                  Avg (across 3)
Model    Venue    SR   NPR   PR   P@5     SR   NPR   PR   P@5     SR   NPR   PR   P@5     SR   NPR   PR
```

- 15 data columns (4 metrics × 3 datasets + 3 aggregate metrics) + Model + Venue = **17 columns total**
- The **Avg** super-column reports the arithmetic mean of SR / NPR / PR across SatSOT, SV248S, and OOTB (equal-weight per dataset). P@5 is not aggregated.
- **Best / 2nd-best annotation is per column**: applied independently within each `(dataset, metric)` cell and within each `Avg` metric cell (SR and NPR only). PR and P@5 stay unannotated everywhere.
- P@5 rendered in a lighter/smaller style (e.g. `\small{}` or gray) to signal its diagnostic role and avoid competing visually with the three ranked metrics.
- If horizontal space is tight, the Avg super-column can be split into a separate narrow table rather than dropped — it is load-bearing for giving readers a single-number summary.
- Format: `table*` (full-width, spanning both columns) + `\footnotesize`
- Style: `booktabs` — `\toprule`, `\midrule` between groups, `\bottomrule`; thin `\cmidrule` between each dataset super-column and the Avg super-column.

### 3.3 Suggested Caption

> Comparison of SOT trackers on SatSOT, SV248S, and OOTB. SR, NPR, and PR denote the AUC of the success, normalized precision, and precision plots respectively (PR over CLE ∈ [0, 50] px following OTB convention; NPR over normalized CLE ∈ [0, 0.5] following TrackingNet / LaSOT). Primary ranking is by SR (↑); NPR (↑) is used as secondary metric. P@5 (fraction of frames with CLE < 5 px) is reported alongside as a diagnostic for fine-grained small-object localization; it is not ranked. **Bold** = best, underline = second best, applied **per column within each dataset** (and within the Avg super-column) to SR and NPR only. The **Avg** super-column reports equal-weight arithmetic mean of SR / NPR / PR across the three datasets and serves as a single-number summary in the spirit of TrackingNet / LaSOT / GOT-10k aggregate reporting; per-dataset columns remain the primary unit of comparison. †STAR uses a domain-specific fine-tuned checkpoint (see Section X). — indicates results unavailable at submission time.

---

## 4. Tiny-Object Subset Analysis

Satellite video tracking is fundamentally a small-object problem, but the overall results table aggregates over all sequences and can mask tracker behaviour in the truly small-target regime where the benchmark's unique value lies. Section 4 isolates this regime.

### 4.1 Subset Definition

A sequence is classified as **tiny** if the **median equivalent side length** (`sqrt(area)`) **of the GT bounding box across all annotated frames is below 8 px**, equivalently **median GT area < 64 px²**:

```
side_t = sqrt(w_t · h_t)                  for frame t
sequence_tiny ⟺ median_t(side_t) < 8 px   (⟺ median area < 64 px²)
```

**Design choices and rationale:**

- **Equivalent side (sqrt(area)) vs. diagonal vs. side alone**: We use `sqrt(area)` because it is the most widely adopted scale measure in the small-object detection / tracking literature (analogous to COCO's area-based small/medium/large stratification, where small = area < 32² = 1024 px²; we adopt 8² = 64 px² for the much smaller targets in satellite video). It is shape-agnostic — for square targets it equals the side length, and for elongated targets (vehicles, ships) it gives a well-defined effective scale that w or h alone cannot. The diagonal alternative (`sqrt(w² + h²)`) yields a similar partition but lacks the precedent in small-object literature.
- **Median vs. mean**: We use the median across frames to be robust to occasional GT size fluctuations (e.g. brief sensor proximity, partial occlusion, annotation jitter). A sequence whose target is consistently sub-8 px (equivalent side) across the bulk of frames is what we want to capture.
- **Sequence-level vs. frame-level**: The split is performed at sequence level rather than frame level, because all four metrics (SR, NPR, PR, P@5) are computed as per-sequence scores then averaged across sequences. Frame-level filtering would break this evaluation unit, fragment temporal continuity, and require redefining the AUC computation. Sequence-level partitioning preserves the standard evaluation protocol — only the set of sequences changes.

### 4.2 Subset Statistics (to be populated)

Before reporting tiny-subset results, we report subset sizes per dataset to allow readers to assess statistical reliability:

| Dataset | Total Sequences | Tiny Sequences (median √area < 8 px) | Tiny Fraction |
|---|---|---|---|
| SatSOT |  105 |  25 | 23.8% |
| SV248S |  248 | 207 | 83.5% |
| OOTB   |  110 |  16 | 14.5% |

> Computed over the *whole* dataset (no_split) by `tools/reaggregate_sot_per_sequence.py` from raw GT (OOTB polygon shoelace area; SatSOT/SV248S `w·h` of `xywh`, with SatSOT `none` and SV248S `state==1` invisible frames excluded). Per-sequence median of `sqrt(area)` across valid annotated frames; sequence is *tiny* iff median < 8 px. Per-sequence median √area values are written to `analysis/per_seq/tiny_subsets.json` for inspection.
>
> **Reliability note**: SV248S's 207-sequence tiny subset is large; OOTB's 16-sequence subset is at the small end of the reliability range and per-tracker numbers there will be noisier. SatSOT (25) is in between.

### 4.3 Tiny Subset Results Table — Metrics

**Reported metrics: SR, NPR, P@5**

Rationale for the metric choice on the tiny subset:

- **SR** (IoU AUC): Maintained for continuity with the overall table. IoU remains meaningful on tiny targets, though absolute values will be lower.
- **NPR** (normalized CLE AUC over [0, 0.5]): Critical on tiny targets — its scale invariance prevents the metric from collapsing to near-zero, which is a known issue with raw CLE on very small targets.
- **P@5** (CLE < 5 px): Replaces PR as the primary precision metric on this subset. For targets with median equivalent side < 8 px, a 5 px error already approaches the target's full extent; the [0, 50] px range used by PR AUC is dominated by signal that is meaningless at this scale (any tracker achieving 30 px on a 6 px target has effectively lost the target). P@5 ≈ "error within roughly one target extent" is the only precision criterion with operational meaning here.

**PR is dropped from the tiny subset table** — its AUC over [0, 50] px is uninformative when the target itself is < 8 px.

### 4.4 Tiny Subset Table Header

```
                    SatSOT (tiny)             SV248S (tiny)             OOTB (tiny)               Avg (across 3)
Model    Venue    SR   NPR   P@5            SR   NPR   P@5            SR   NPR   P@5            SR   NPR
```

- 11 data columns (3 metrics × 3 datasets + 2 aggregate metrics) + Model + Venue = **13 columns total**
- Same row grouping as the overall table.
- Best / 2nd-best annotation: applied to SR and NPR (per dataset and within Avg). P@5 unannotated, consistent with the overall table.
- Avg super-column aggregates SR and NPR only (P@5 not aggregated, same reason as overall table).

### 4.5 Suggested Caption

> Tracker performance on the tiny-object subset of SatSOT, SV248S, and OOTB. A sequence is classified as tiny if the median GT equivalent side length (`sqrt(area)`) across all annotated frames is below 8 px (equivalently, median area < 64 px²; subset sizes reported in Table 4.2). The 8 px cutoff is chosen as the satellite-video analogue of COCO's small-object threshold (sqrt(area) < 32 px) and reflects the much smaller target scale of spaceborne imagery. SR and NPR follow the same definitions as in the overall table; P@5 (fraction of frames with CLE < 5 px) replaces PR AUC as the primary precision metric, since for sub-8-px targets the [0, 50] px PR range is dominated by errors larger than the target itself and ceases to be meaningful. **Bold** = best, underline = second best, applied per column within each dataset and within the Avg super-column, on SR and NPR only. The Avg super-column reports equal-weight arithmetic mean of SR and NPR across the three datasets. †STAR uses a domain-specific fine-tuned checkpoint (see Section X). — indicates results unavailable at submission time.

### 4.6 Narrative Use

The tiny-subset table supports the central narrative point of a satellite-video benchmark: **which methods retain their overall ranking when restricted to the small-object regime, and which methods rely on the relatively larger targets in the overall set to inflate their average performance?** Concrete questions the analysis section should answer:

1. Do any pretrained generic trackers (OSTrack, ODTrack, LoRAT) collapse on the tiny subset relative to their overall ranking?
2. Do foundation models (SAM 2, SAM 3) degrade gracefully or catastrophically as target size shrinks?
3. Does the domain-adapted STAR† maintain its advantage on tiny targets, or is its overall lead driven by larger sequences?
4. Is there a tracker that is *not* SOTA overall but *is* SOTA on tiny targets? — Such a finding would be a strong contribution.

---

## 5. Open Items

1. **Subset statistics**: Run sequence-level median `sqrt(area)` computation across SatSOT, SV248S, OOTB and populate Table 4.2. If any dataset's tiny subset is below ~10 sequences, decide on handling (flag vs. drop vs. supplementary).
2. **Distribution check**: Plot the per-dataset histogram of sequence-level median `sqrt(area)` as an appendix figure, both to justify the 8 px cutoff visually and to give readers a sense of where each dataset sits on the size spectrum.
3. **Legacy NP@0.5**: Decide whether to include the OOTB-style single-threshold NP@0.5 in supplementary for direct comparability with OOTB's published numbers.
4. **SAMURAI inclusion**: Final decision on whether to include in the model list.
