# SOT Benchmark — Evaluation Design & Table Structure

---

## 1. Metrics Definition

| Symbol | Definition | Type |
|---|---|---|
| **P@20** | CLE < 20px 的帧占比，单点值 | Single-threshold scalar |
| **PR** | Precision curve 在 0–50px 全段的 AUC | Area under curve |
| **NPR** | Normalized precision curve 在 0–0.5 全段的 AUC | Area under curve |
| **SR** | Success curve 在 0–1 IoU 全段的 AUC | Area under curve |

> **Selected metrics for this benchmark: SR, NPR, PR**
> (P@20 excluded as primary metric; may appear in supplementary for compatibility with prior work.)
>
> PR and NPR denote the AUC of the precision and normalized precision plots respectively, following OOTB [Chen et al., ISPRS 2024].

### Ranking Policy

- **Primary metric**: SR (IoU-based, most universally comparable)
- **Secondary metric**: NPR (scale-invariant, preferred for small targets)
- **Reference metric**: PR (included for completeness; no best/2nd-best annotation to reduce visual noise)
- Best result: **bold**; 2nd-best result: underlined — applied to SR and NPR columns only.

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
                    SatSOT            SV248S            OOTB
Model    Venue    SR   NPR   PR     SR   NPR   PR     SR   NPR   PR
```

- 9 data columns + Model + Venue = **11 columns total**
- Format: `table*` (full-width, spanning both columns) + `\footnotesize`
- Style: `booktabs` — `\toprule`, `\midrule` between groups, `\bottomrule`

### 3.3 Suggested Caption

> Comparison of SOT trackers on SatSOT, SV248S, and OOTB. SR, NPR, and PR denote the AUC of the success, normalized precision, and precision plots respectively. Primary ranking is by SR (↑); NPR (↑) is used as secondary metric. **Bold** = best, underline = second best, applied to SR and NPR only. †STAR uses a domain-specific fine-tuned checkpoint (see Section X). — indicates results unavailable at submission time.
