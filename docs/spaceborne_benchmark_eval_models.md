# Spaceborne Video Benchmark — Model Evaluation Plan

> **Target venue**: NeurIPS 2026 Datasets & Benchmarks Track
> **Last updated**: April 2026 — includes CVPR 2025 models and SAM 3 / SAM 3.1 (Meta, Nov 2025)

---

## Overview

| Track | # Models | Datasets Used |
|---|---|---|
| Detection | 14 | SAT-MTB (HBB/OBB), VISO, SDM-Car, IRSatVideo-LEO |
| SOT | 14 | SatSOT, SV248S, SAT-MTB, OOTB, IRSatVideo-LEO |
| MOT | 13 | AIR-MOT, SAT-MTB, VISO, SDM-Car, IRSatVideo-LEO, LMOD |
| Segmentation | 12 | SAT-MTB seg, SAT-MTB-SOS, SV248S, IRSatVideo-LEO |
| **Total** | **53** | |

---

## Evaluation Strategy

All models must be evaluated on **our custom splits** (different from original papers — numbers are not comparable). Three tiers of evaluation effort:

| Tier | Models | What to do | Est. effort per model |
|---|---|---|---|
| **Zero-shot inference** | Foundation models (SAM 2, Grounding DINO, YOLO-World, SEEM, MASA…) | Load official pretrained weights, run on our test set directly | ~0.5 day |
| **Pretrained eval** | Classic & recent baselines (ByteTrack, OSTrack, Mask2Former…) | Load COCO/LaSOT/MOT17 pretrained weights, eval on our test set. Tests OOD generalisation — this IS what the benchmark measures. | ~1 day |
| **Fine-tuned upper bound** | 2–3 representative models per track | Fine-tune on our train split, eval on test. Provides upper bound reference. | ~2–3 days |

> ⚠ **Do NOT fine-tune all 44 models.** NeurIPS 2026 D&B deadline is ~June 2026. Prioritise zero-shot FM results (core finding) and pretrained baselines.

---

## Track 1: Object Detection

**Datasets**: SAT-MTB det_hbb (142 seqs), SAT-MTB det_obb (106 seqs), VISO (47 seqs), SDM-Car (99 seqs), IRSatVideo-LEO (200 seqs, TIR)

**Primary metric**: AP50, AP50:95

**Sub-tracks**: Universal HBB · OBB-specific · IR-specific

### Universal HBB Models (all VNIR datasets)

| Model | Venue | Type | Eval tier |
|---|---|---|---|
| Faster R-CNN (ResNet-50/101) | NeurIPS 2015 | Two-stage anchor | Pretrained eval |
| YOLOv8-s / YOLOv8-l | 2023 | One-stage (two sizes) | Pretrained eval |
| FCOS | ICCV 2019 | Anchor-free | Pretrained eval |
| Deformable DETR | ICLR 2021 | Transformer | Pretrained eval |
| DINO-Det (Swin-L) | ICLR 2023 | Transformer, COCO SOTA | Pretrained eval |
| RT-DETR | CVPR 2024 | Real-time transformer | Pretrained eval |
| LSKNet | ICCV 2023 | Remote sensing specific, large kernel | Pretrained eval |
| Florence-2 | arXiv 2023 / CVPR 2024 | Open-set det + grounding, strong zero-shot on small objects | Zero-shot |
| Grounding DINO | ECCV 2024 | Open-vocab, zero-shot | Zero-shot |
| YOLO-World | CVPR 2024 | Real-time open-vocab | Zero-shot |

### OBB-specific Models (SAT-MTB det_obb only)

| Model | Venue | Notes | Eval tier |
|---|---|---|---|
| Oriented R-CNN / RoI Trans. | ICCV 2021 | Standard OBB baseline | Pretrained eval |
| H2RBox-v2 | ICLR 2023 | Weakly supervised OBB; matches your annotation style | Pretrained eval |

### IR-specific Models (IRSatVideo-LEO only)

| Model | Venue | Notes | Eval tier |
|---|---|---|---|
| DNANet / ISNet | TGRS 2022/2023 | Small target detection for TIR | Pretrained eval |
| UIU-Net | TGRS 2023 | U-Net in U-Net for infrared small targets | Pretrained eval |

---

## Track 2: Single Object Tracking (SOT)

**Datasets**: SatSOT (105 seqs), SV248S (248 seqs, 156K frames), SAT-MTB subset (VNIR), OOTB (110 seqs, OBB), IRSatVideo-LEO (200 seqs, TIR)

**Primary metrics**: AUC, Precision (P), Normalised Precision (NP)

**Sub-tracks**: Standard HBB · OBB (OOTB) · TIR (IRSatVideo-LEO)

| Model | Venue | Type | Eval tier |
|---|---|---|---|
| SiamRPN++ | CVPR 2019 | Siamese | Pretrained eval |
| Ocean | ECCV 2020 | Object-aware siamese | Pretrained eval |
| TransT | CVPR 2021 | First strong transformer SOT | Pretrained eval |
| OSTrack-256 | ECCV 2022 | One-stream, fast & strong | Pretrained eval |
| MixFormer-ViT | CVPR 2022 | Mixed attention modules | Pretrained eval |
| ARTrack | CVPR 2023 | Autoregressive tracking | Pretrained eval |
| ROMTrack | ICCV 2023 | Robust to appearance change — key in satellite imagery | Pretrained eval |
| ODTrack | AAAI 2024 | Online token propagation | Pretrained eval |
| SiamBAN-OBB / SiamFC++ | — | For OOTB oriented bbox evaluation | Pretrained eval |
| SAM 2 (prompted SOT) | Meta 2024 | Promptable video seg + track | Zero-shot |
| SAMURAI | arXiv 2024 | SAM 2 adapted for zero-shot visual tracking, motion-aware memory | Zero-shot |
| SAM 3 / SAM 3.1 (text-prompted SOT) | Meta arXiv Nov 2025 | Text-prompt driven tracking — prompt with "car" or "airplane", no bbox init needed; compare directly with SAM 2 to show generational gap | Zero-shot |
| DreamTrack | CVPR 2025 | Temporal future prediction for SOT; SOTA on TrackingNet — relevant for satellite motion regularity | Pretrained eval |
| UNINEXT | CVPR 2023 | Universal instance perception | Zero-shot |

> **Note**: SV248S (248 seqs, 156K frames) is the largest SOT dataset here — use it as the primary table. SAM 2 zero-shot on TIR (IRSatVideo-LEO) is a compelling cross-domain experiment worth highlighting as a standalone finding. DreamTrack vs. OSTrack is a clean 2022→2025 progression story showing how temporal modelling helps in satellite video's regular motion patterns.

---

## Track 3: Multi-Object Tracking (MOT)

**Datasets**: AIR-MOT (69 seqs), SAT-MTB mot (237 seqs), VISO (47 seqs), SDM-Car (99 seqs), IRSatVideo-LEO (200 seqs, TIR), LMOD (8 seqs — see note below)

**Primary metrics**: HOTA, MOTA, IDF1

| Model | Venue | Type | Eval tier |
|---|---|---|---|
| SORT | ICRA 2016 | Kalman + IoU | Pretrained eval |
| DeepSORT | ICIP 2017 | Re-ID baseline | Pretrained eval |
| CenterTrack | ECCV 2020 | Heatmap-based joint det+track | Pretrained eval |
| ByteTrack | ECCV 2022 | Keeps low-confidence detections — matches satellite scenario | Pretrained eval |
| OC-SORT | CVPR 2023 | Handles occlusion, relevant for SAT-MTB | Pretrained eval |
| BoT-SORT | arXiv 2022 | Camera motion compensation — useful for satellite platform motion | Pretrained eval |
| StrongSORT | TMM 2023 | Strong re-ID, good on SDM-Car small vehicles | Pretrained eval |
| Deep OC-SORT | CVPR 2023 | Combines DeepSORT + OC-SORT improvements | Pretrained eval |
| TGraM / MO-TAMA | TGRS 2022 | Domain-specific, trained on AIR-MOT — important upper bound | Fine-tune |
| MASA | CVPR 2024 | SAM-adapter for any-category tracking | Zero-shot |
| UNINEXT | CVPR 2023 | Unified prompt-based MOT | Zero-shot |
| MOTIP | CVPR 2025 | MOT as in-context ID prediction, end-to-end, no heuristic association — SOTA on multiple benchmarks | Pretrained eval |
| TrackTrack | CVPR 2025 | Track-Perspective-Based Association + Track-Aware Initialization; online MOT | Pretrained eval |

> ⚠ **LMOD (8 seqs only)**: Too small for a standalone quantitative track. Fold into MOT as a "large-scale sub-challenge" — report HOTA/IDF1 in a small supplementary table + qualitative visualisations. Do not create a separate section for it.

---

## Track 4: Segmentation

**Datasets**:
- **Instance segmentation**: SAT-MTB seg (142 seqs), SV248S (248 seqs, poly masks), IRSatVideo-LEO (200 seqs, binary masks)
- **Video Object Segmentation (VOS)**: SAT-MTB-SOS (113 seqs)

**Metrics**:
- Instance seg: AP, AP50, AP75
- VOS: J&F score (NOT mAP — keep these in separate tables)

### Instance Segmentation Models

| Model | Venue | Type | Eval tier |
|---|---|---|---|
| Mask R-CNN (ResNet-50) | ICCV 2017 | Proposal-based baseline | Pretrained eval |
| SOLOv2 | NeurIPS 2020 | Instance seg without proposals | Pretrained eval |
| QueryInst | ICCV 2021 | Query-based | Pretrained eval |
| SparseInst | CVPR 2022 | Fast sparse instance representation | Pretrained eval |
| Mask2Former | CVPR 2022 | Unified panoptic/instance seg, current standard | Pretrained eval |
| SAM (ViT-H) zero-shot | Meta 2023 | Promptable seg on static frames | Zero-shot |

### Video Object Segmentation (VOS) Models

| Model | Venue | Type | Eval tier |
|---|---|---|---|
| Cutie | CVPR 2024 | Memory-based VOS with object-level tokens; strong long-term tracking | Pretrained eval |
| DEVA | ICCV 2023 | Decoupled VOS — fits SAT-MTB-SOS directly | Pretrained eval |
| SAM 2 (video mode) | Meta 2024 | Video object seg | Zero-shot |
| EntitySAM | CVPR 2025 | Extends SAM 2 with automatic prompt generation — no initial mask needed, handles new objects mid-sequence | Zero-shot |
| SAM 3 / SAM 3.1 | Meta arXiv Nov 2025 *(ICLR 2026 under review)* | Unified det + seg + track via text/exemplar prompts; doubles SAM 2 accuracy on concept-level PCS; SAM 3.1 adds shared-memory multi-object tracking — most capable zero-shot model for all four tracks | Zero-shot |
| SEEM / SegGPT | 2023 | In-context seg | Zero-shot |

> ⚠ **Keep instance seg and VOS in separate tables.** They use incompatible protocols (AP vs J&F). Mixing them will confuse reviewers.

---

## Key Design Decisions & Findings to Highlight

### Two modality tracks
VNIR (optical) and TIR (IRSatVideo-LEO) are fundamentally different domains. Report them in separate columns or tables. The gap between pretrained-on-optical and zero-shot-on-TIR performance is itself a core benchmark finding.

### OBB is a differentiator
OOTB + SAT-MTB det_obb is nearly unique among spaceborne benchmarks. Emphasise the OBB sub-track as a distinguishing contribution vs. DOTA-series benchmarks.

### Foundation model story
Zero-shot FM performance on satellite video is the most NeurIPS-relevant narrative. Expected findings:
- **SAM 1 → SAM 2 → SAM 3 progression**: running all three on your datasets gives a built-in generational comparison that reviewers will find compelling
- SAM 3 text-prompted tracking on spaceborne video is a completely novel experiment — no published results exist yet
- SAM 3 supports concept-level detection + tracking in one model, directly usable across all four tracks
- EntitySAM (CVPR 2025) extends SAM 2 without needing initial prompts — directly testable on SAT-MTB-SOS
- Grounding DINO / YOLO-World show open-vocab capability but miss small/dim targets
- MASA bridges MOT and FM paradigms cleanly
- MOTIP (CVPR 2025) vs. ByteTrack: end-to-end learned association vs. heuristic — gap on satellite data is a key finding

> ⚠ **SAM 3 paper status**: arXiv Nov 2025, submitted to ICLR 2026 (under double-blind review). Cite as preprint in your paper and note it is not yet formally published.

### Domain gap analysis
Use the pretrained (out-of-domain) vs fine-tuned (in-domain) gap on 2–3 representative models to quantify how challenging the domain shift is — this justifies why your benchmark is needed.

---

## Compute Estimation (rough)

| Task | # Models | Avg GPU-hours per model | Total |
|---|---|---|---|
| Det zero-shot / pretrained | 14 | ~4 h | ~56 h |
| SOT pretrained + zero-shot | 13 | ~6 h | ~78 h |
| MOT pretrained + zero-shot | 13 | ~5 h | ~65 h |
| Seg pretrained + zero-shot | 11 | ~6 h | ~66 h |
| Fine-tune (2–3 per track × 4 tracks) | ~10 | ~24 h | ~240 h |
| **Total estimate** | | | **~505 GPU-hours** |

> This is a rough estimate assuming A100-class GPUs. Parallelise where possible across datasets — most models can run detection on all datasets in one job.
