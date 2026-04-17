# SOT Benchmark — Evaluation Design & Table Structure

---

## 1. Metrics Definition

| Symbol | Definition | Type |
|---|---|---|
| **P@5** | CLE < 5px 的帧占比，单点值 | Single-threshold scalar |
| **PR** | Precision curve 在 0–30px 全段的 AUC¹ | Area under curve |
| **NPR** | Normalized precision curve 在 0–0.5 全段的 AUC² | Area under curve |
| **SR** | Success curve 在 0–1 IoU 全段的 AUC | Area under curve |

> ¹ PR AUC is integrated over 0–30 px rather than the generic-vision default of 0–50 px, following OOTB's protocol for satellite video, which argues that GV's 20 px threshold corresponds to SV's 5 px due to lower spatial resolution and smaller objects. The reduced upper bound keeps the AUC concentrated in the pixel-accuracy regime that actually matters for satellite targets.
>
> ² NPR AUC is integrated over normalised CLE ∈ [0, 0.5] rather than the [0, 1] range adopted by OOTB [Chen et al., ISPRS 2024, §5.5.2 — "predefined threshold varied from 0 to 1"]. We deliberately deviate here to align with the post-2018 mainstream SOT convention — TrackingNet [Müller et al., ECCV 2018], LaSOT [Fan et al., CVPR 2019], and GOT-10k [Huang et al., TPAMI 2021] all report normalised precision as the AUC over [0, 0.5]. Two reasons motivate the narrower bound: (a) beyond ~0.5 nearly every competitive tracker saturates, so the high-threshold segment contributes little discriminative signal and compresses the gap between methods; (b) on satellite video where GT diagonals are small, normalised CLE values above 0.5 indicate near-total localisation failure, which is already captured by SR at low IoU. The single-threshold NP@0.5 used by OOTB is still reported as a legacy scalar (`norm_precision_05`) for direct comparability with OOTB's published numbers.

> **Selected metrics for this benchmark: SR, NPR, PR**
> (P@20 excluded as primary metric; may appear in supplementary for compatibility with prior work.)
>
> PR and NPR denote the AUC of the precision and normalized precision plots respectively, following OOTB [Chen et al., ISPRS 2024].
>
> **P@5 is reported alongside PR/NPR/SR as a diagnostic metric specifically targeting small-object localization, following OOTB [Chen et al., 2024] which adopts PR@5px as its primary precision metric.** It is not part of the ranking policy but is included so readers can verify fine-grained localization accuracy on sub-pixel-scale satellite targets.

### Ranking Policy

- **Primary metric**: SR (IoU-based, most universally comparable)
- **Secondary metric**: NPR (scale-invariant, preferred for small targets)
- **Reference metric**: PR (included for completeness; no best/2nd-best annotation to reduce visual noise)
- **Diagnostic metric**: P@5 (single-threshold precision at 5 px; not ranked, reported for fine-grained localization diagnostics following OOTB's primary precision protocol)
- **Best / 2nd-best annotation scope**: Bold (best) and underline (2nd-best) are applied **per column, within each dataset** — i.e. for each `(dataset, metric)` pair independently. They are applied only to SR and NPR columns; PR and P@5 columns are left unannotated to reduce visual noise. A tracker that is SOTA on SatSOT but mediocre on OOTB will therefore be bolded in the SatSOT SR column but not the OOTB SR column, reflecting per-dataset performance rather than a global composite.
- **Cross-dataset aggregate column**: The rightmost block of the main table adds an **Avg** super-column with arithmetic mean of SR, NPR, and PR across the three datasets (equal weight per dataset). Bold / underline are also applied within this Avg super-column (SR and NPR only) to serve as a single-number summary for readers who want a global ranking view, following the TrackingNet / LaSOT / GOT-10k benchmark convention. This is a reference only — per-dataset columns remain the primary unit of comparison because the three datasets differ in resolution, category mix, and difficulty.

---

## 2. SOT Model List (11 models — updated April 14)

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
| STAR† | TGRS 2025 | SV-specific Transformer | Official fine-tuned checkpoint; see note below | Fine-tuned† | All |

> **†STAR**: Uses a publicly released checkpoint fine-tuned on satellite video data. Results on SatSOT reflect in-distribution evaluation (different split of the same source domain); results on SV248S and OOTB are zero-shot with respect to the fine-tuning corpus. Reported separately as a *domain-adapted baseline*.
>
> **—** indicates results not yet available at submission time.

---

## 3. Results Table Structure

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
- The **Avg** super-column reports the arithmetic mean of SR / NPR / PR across SatSOT, SV248S, and OOTB (equal-weight per dataset). P@5 is not aggregated — it is a diagnostic scalar whose meaning depends on the dataset's pixel scale, so averaging across datasets is not well-defined.
- **Best / 2nd-best annotation is per column**: applied independently within each `(dataset, metric)` cell and within each `Avg` metric cell (SR and NPR only). PR and P@5 stay unannotated everywhere.
- P@5 rendered in a lighter/smaller style (e.g. `\small{}` or gray) to signal its diagnostic role and avoid competing visually with the three ranked metrics.
- If horizontal space is tight, the Avg super-column can be split into a separate narrow table rather than dropped — it is load-bearing for giving readers a single-number summary.
- Format: `table*` (full-width, spanning both columns) + `\footnotesize`
- Style: `booktabs` — `\toprule`, `\midrule` between groups, `\bottomrule`; thin `\cmidrule` between each dataset super-column and the Avg super-column.

### 3.3 Suggested Caption

> Comparison of SOT trackers on SatSOT, SV248S, and OOTB. SR, NPR, and PR denote the AUC of the success, normalized precision, and precision plots respectively. Primary ranking is by SR (↑); NPR (↑) is used as secondary metric. P@5 (fraction of frames with centre-location error < 5 px) is reported alongside as a diagnostic for fine-grained small-object localization, following OOTB [Chen et al., 2024] which adopts PR@5px as its primary precision metric; it is not ranked. **Bold** = best, underline = second best, applied **per column within each dataset** (and within the Avg super-column) to SR and NPR only. The **Avg** super-column reports equal-weight arithmetic mean of SR / NPR / PR across the three datasets and serves as a single-number summary in the spirit of TrackingNet / LaSOT / GOT-10k aggregate reporting; per-dataset columns remain the primary unit of comparison. †STAR uses a domain-specific fine-tuned checkpoint (see Section X). — indicates results unavailable at submission time.
