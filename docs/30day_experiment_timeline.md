# 30-Day Experiment Timeline
# Spaceborne Video Benchmark — NeurIPS 2026 D&B Track

> **Deadline**: May 7, 2026 (Day 30)
> **GPU**: 2× NVIDIA RTX 5000 Ada Generation (32 GB each), CUDA 13.1
> **Strategy**: Zero-shot → Pretrained eval → Fine-tune, per track
> **Track order**: SOT (Days 1–10) → Detection (Days 11–20) → MOT (Days 21–28) → Buffer (Days 28–30)

---

## Legend

| Symbol | Meaning |
|---|---|
| 🟡 Zero-shot | Load official pretrained weights, run directly on test set — no training |
| 🔵 Pretrained eval | Load COCO/LaSOT/MOT17 pretrained weights, eval on our test set (OOD generalisation) |
| 🔴 Fine-tune | Fine-tune on our train split, eval on test — upper bound reference |
| 🟢 Analysis | Result compilation, writing, table generation |

---

## Phase 1 — Single Object Tracking (SOT) · Days 1–10

**Datasets**: SatSOT (105 seqs), SV248S (248 seqs, 156K frames), SAT-MTB subset, OOTB (110 seqs, OBB), IRSatVideo-LEO (200 seqs, TIR)
**Primary metrics**: AUC, Precision (P), Normalised Precision (NP)

| Day | Date | GPU-0 | GPU-1 | Milestone |
|---|---|---|---|---|
| 1 | Apr 8 | 🟡 SAM 2 — zero-shot, all SOT datasets, env setup | 🟡 SAMURAI — zero-shot, all SOT datasets | Zero-shot start |
| 2 | Apr 9 | 🟡 SAM 3 / 3.1 — zero-shot, text-prompted SOT | 🟡 UNINEXT — zero-shot, all SOT datasets | |
| 3 | Apr 10 | 🔵 SiamRPN++ — pretrained, SatSOT + SV248S | 🔵 Ocean — pretrained, SatSOT + SV248S | Pretrained start |
| 4 | Apr 11 | 🔵 TransT — pretrained, all SOT datasets | 🔵 OSTrack-256 — pretrained, all SOT datasets | |
| 5 | Apr 12 | 🔵 MixFormer-ViT — pretrained, all SOT datasets | 🔵 ARTrack — pretrained, all SOT datasets | |
| 6 | Apr 13 | 🔵 ROMTrack — pretrained, all SOT datasets | 🔵 ODTrack — pretrained, all SOT datasets | |
| 7 | Apr 14 | 🔵 DreamTrack — pretrained, all SOT datasets | 🔵 SiamBAN-OBB — pretrained, OOTB only | |
| 8 | Apr 15 | 🔴 OSTrack fine-tune — train on SatSOT + SV248S train splits | 🔴 ARTrack fine-tune — train on SatSOT + SV248S train splits | Fine-tune start |
| 9 | Apr 16 | 🔴 OSTrack fine-tune — continued + eval test split | 🔴 ARTrack fine-tune — continued + eval test split | |
| 10 | Apr 17 | 🟢 SOT result analysis — AUC/P/NP tables, pretrained vs fine-tuned gap, write SOT section draft, flag reruns | ← same | **SOT done** |

---

## Phase 2 — Object Detection · Days 11–20

**Datasets**: SAT-MTB det_hbb (142 seqs), SAT-MTB det_obb (106 seqs), VISO (47 seqs), SDM-Car (99 seqs), IRSatVideo-LEO (200 seqs, TIR)
**Primary metrics**: AP50, AP50:95

| Day | Date | GPU-0 | GPU-1 | Milestone |
|---|---|---|---|---|
| 11 | Apr 18 | 🟡 Grounding DINO — zero-shot, SAT-MTB HBB + VISO | 🟡 YOLO-World — zero-shot, SAT-MTB HBB + VISO | Zero-shot start |
| 12 | Apr 19 | 🟡 Florence-2 — zero-shot, all det datasets incl. SDM-Car | 🔵 Faster R-CNN — pretrained, all VNIR det datasets | |
| 13 | Apr 20 | 🔵 YOLOv8-s/l — pretrained, all VNIR det datasets | 🔵 FCOS — pretrained, all VNIR det datasets | Pretrained start |
| 14 | Apr 21 | 🔵 Deformable DETR — pretrained, all VNIR det datasets | 🔵 DINO-Det (Swin-L) — pretrained, all VNIR det datasets | |
| 15 | Apr 22 | 🔵 RT-DETR — pretrained, all VNIR det datasets | 🔵 LSKNet — pretrained, SAT-MTB + VISO | |
| 16 | Apr 23 | 🔵 Oriented R-CNN — pretrained, SAT-MTB det_obb + OOTB | 🔵 H2RBox-v2 — pretrained, SAT-MTB det_obb + OOTB | |
| 17 | Apr 24 | 🔵 DNANet / ISNet — pretrained, IRSatVideo-LEO only | 🔵 UIU-Net — pretrained, IRSatVideo-LEO only | |
| 18 | Apr 25 | 🟢 Detection SOTA analysis — rank by AP50, select best model for MOT detector | 🔴 DINO-Det fine-tune — on SAT-MTB train split | |
| 19 | Apr 26 | 🔴 YOLOv8 fine-tune — upper bound for fast detector | 🔴 DINO-Det fine-tune — continued + eval test split | Fine-tune start |
| 20 | Apr 27 | 🟢 Detection result analysis + MOT detector prep — compile AP tables, confirm SOTA detector for MOT, generate detection CSV/JSON files for all MOT datasets, write detection section draft | ← same | **Det done · MOT det ready** |

---

## Phase 3 — Multi-Object Tracking (MOT) · Days 21–28

**Datasets**: AIR-MOT (69 seqs), SAT-MTB mot (237 seqs), VISO (47 seqs), SDM-Car (99 seqs), IRSatVideo-LEO (200 seqs, TIR), LMOD (8 seqs — qualitative only)
**Primary metrics**: HOTA, MOTA, IDF1
**Fixed detector**: DINO-Det pretrained (confirmed on Day 20) — all TbD trackers use the same detection input

| Day | Date | GPU-0 | GPU-1 | Milestone |
|---|---|---|---|---|
| 21 | Apr 28 | 🟡 MASA — zero-shot, all MOT datasets | 🟡 SAM 3 (MOT mode) — zero-shot, AIR-MOT + SAT-MTB | Zero-shot start |
| 22 | Apr 29 | 🟡 UNINEXT — zero-shot MOT, all datasets | 🔵 SORT + DeepSORT — pretrained, all MOT datasets (fast, batch together) | Pretrained start |
| 23 | Apr 30 | 🔵 ByteTrack — pretrained, all MOT datasets | 🔵 OC-SORT + BoT-SORT — pretrained, all MOT datasets | |
| 24 | May 1 | 🔵 StrongSORT + Deep OC-SORT — pretrained, all MOT datasets | 🔵 MOTIP — pretrained, SAT-MTB + AIR-MOT | |
| 25 | May 2 | 🔵 TrackTrack + CenterTrack — pretrained, all MOT datasets | 🔴 TGraM / MO-TAMA fine-tune — on AIR-MOT train split | Fine-tune start |
| 26 | May 3 | 🔴 ByteTrack fine-tune — on SAT-MTB mot train split | 🔴 TGraM fine-tune — continued + eval test split | |
| 27 | May 4 | 🟢 MOT result analysis — HOTA/MOTA/IDF1 tables, TbD vs E2E comparison, LMOD qualitative figs, write MOT section draft | ← same | **MOT done** |

---

## Phase 4 — Buffer + Paper Assembly · Days 28–30

| Day | Date | Task |
|---|---|---|
| 28 | May 5 | Rerun any failed or suspicious results — missing metrics, NaN values, model crashes, TIR zero-shot SAM 3 recheck |
| 29 | May 6 | Main result tables + figures — LaTeX tables, per-dataset AP/AUC/HOTA, modality gap figures, FM zero-shot vs pretrained bars |
| 30 | May 7 | Final paper assembly — abstract, intro, benchmark design section, conclusion, references, supplementary appendix |

---

## Key Notes & Risk Points

### 1. SV248S is the biggest bottleneck
156K frames — even on RTX 5000 a single SOT model can take 8–12 hours.
**Action**: On Day 1, first run the full pipeline on SatSOT (27K frames) to verify format and metrics are correct. Only then add SV248S to the queue. Do not run SV248S blind on Day 1.

### 2. Day 20 detector choice determines MOT quality
The MOT evaluation depends entirely on the detection files generated from the chosen detector.
**Action**: Fix DINO-Det (pretrained) as the default MOT detector regardless of fine-tuned results. Write clearly in your paper: *"All TbD trackers are evaluated with pretrained DINO-Det as the shared detector."* This keeps MOT results comparable and reproducible.

### 3. End-to-end MOT models need separate treatment
MOTIP and TGraM / MO-TAMA are end-to-end joint detection + tracking models — they do not use the shared detector.
**Action**: Report them in a separate sub-table labelled "End-to-End (E2E)" in the MOT results section, clearly distinguished from Tracking-by-Detection (TbD) methods. Do not compare E2E AP directly with TbD AP.

### 4. Segmentation track — optional fast path
Segmentation is not in the 30-day plan. If you want minimal seg coverage without extra GPU time:
- SAM 2, SAM 3, EntitySAM are already running on Days 1–2 and Day 21 — grab their segmentation mask outputs at the same time (zero marginal cost).
- Report J&F score on SAT-MTB-SOS only, no fine-tune, one paragraph in the paper.
- Do not start a full seg track — it will break the timeline.

### 5. Daily backup
At the end of every day, `rsync` all result files to `compute01.cosmos-index.com`.
```bash
rsync -avz ./results/ ziwen@compute01.cosmos-index.com:~/benchmark_results/
```
One corrupted local disk should not cost you a week of experiments.

### 6. SAM 3 paper citation status
SAM 3 (arXiv Nov 2025) is currently under double-blind review at ICLR 2026. Cite as:
```
Carion et al., "SAM 3: Segment Anything with Concepts", arXiv:2511.16719, 2025.
```
Check ICLR 2026 acceptance status before submission (~May 2026) and update citation format if accepted.

### 7. TIR vs VNIR — always report separately
IRSatVideo-LEO is a fundamentally different modality. Never mix TIR and VNIR numbers in the same table row.
Use separate columns or sub-tables labelled **VNIR** and **TIR** consistently throughout the paper.

### 8. Compute estimate (revised)
| Track | Models | Est. GPU-hours | Notes |
|---|---|---|---|
| SOT | 14 | ~90 h | SV248S dominates |
| Detection | 14 | ~60 h | IR models faster (fewer seqs) |
| MOT | 13 | ~65 h | Depends on det file I/O speed |
| Fine-tune (all tracks) | ~8 | ~200 h | 2 models per track × 4 tracks |
| **Total** | | **~415 GPU-hours** | ~8.6 days on 2 GPUs running 24/7 — fits in 30 days with buffer |
