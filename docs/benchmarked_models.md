# Benchmarked Models & Selection Rationale

This document consolidates every model we actually run in the benchmark,
grouped by task, together with the reason each was selected. Numbers are
produced on **our own splits** and are therefore not directly comparable to the
originating papers.

Tasks covered: **Detection**, **Single-Object Tracking (SOT)**, and
**Multi-Object Tracking (MOT)**. Under MOT we further separate
**tracking-by-detection (TBD)**, **joint detection & tracking (JDT)**, a
**query-based end-to-end** paradigm, and **open-vocabulary / text-prompted**
trackers.

---

## 1. Detection

**Datasets:** FireRGBT (RGB-T wildfire), BIRDSAI (TIR aerial wildlife),
SAT-MTB det/HBB (satellite airplane/ship/train).
**Primary metric:** mAP@0.5 (per-class AP).

We benchmark three detectors that span the modern design spectrum, all trained
under an identical data/split/augmentation recipe so the comparison is
apples-to-apples:

| Model | Family | Why we chose it |
|---|---|---|
| **Faster R-CNN (ResNet-50 FPN)** | Two-stage, anchor-based | The canonical high-recall two-stage baseline. Multi-scale FPN + data-driven anchors make it the strongest reference on small satellite/thermal objects; it is our "fine-tuned task-specific detector" upper reference. |
| **YOLO11l** | One-stage, anchor-free | The current one-stage real-time standard. Represents the deployment-oriented single-shot design; trained via a manual loop (Lightning's warmup-LR path explodes YOLO loss). |
| **DINOv3 (ViT-B/16, frozen) + dense FCOS head** | Foundation-model features + lightweight head | Tests the "frozen foundation features + simple head" hypothesis: how far do self-supervised web-scale ViT features transfer to remote-sensing/thermal detection with only a 4.9M-param head trained. The in-repo DETR head collapses on frozen features + small data, so we use a dense FCOS head. |

**Key finding across all three datasets:** the frozen DINOv3 detector *matches*
the fine-tuned detectors on large / easy classes (e.g. FireRGBT smoke AP 0.93)
but lags on tiny objects (fire/person, thermal wildlife) — a stride-16 frozen
feature map cannot localise small targets as well as a fine-tuned multi-scale
detector. This is the central "frozen features vs. fine-tuned detector"
narrative of the detection track.

> Detailed comparisons: `docs/use_case_results/fire_detection_compare.md`,
> `docs/use_case_results/birdsai_detection_sam3gt_compare.md`, and the SAT-MTB
> detection comparison.

---

## 2. Single-Object Tracking (SOT)

**Datasets:** OOTB (OBB), SatSOT, SV248S (largest, 153K frames).
**Primary metrics:** Success Rate (AUC), Precision (P), Normalised Precision (NP).

We benchmark **seven** trackers spanning four generations of SOT design — from
Siamese, through one-stream ViT trackers, to promptable video foundation
models. Each is run at the **strongest publicly-released variant** of its origin
paper (see the strength audit below).

| Tracker | Venue | Paradigm | Why we chose it |
|---|---|---|---|
| **SiamRPN++** | CVPR 2019 | Siamese correlation | The classic Siamese-tracker anchor of the field; establishes the pre-transformer baseline. |
| **OSTrack** | ECCV 2022 | One-stream ViT (joint feature+relation) | The efficient, strong one-stream transformer tracker; fast and widely used as the modern reference. |
| **ODTrack** | AAAI 2024 | Online token propagation ViT | Video-level online token propagation; a recent transformer SOTA representative. |
| **LoRAT** | ECCV 2024 | LoRA-adapted large ViT (DINOv2 ViT-g) | The large-backbone SOTA point — largest backbone × largest input resolution, showing the ceiling of scaled ViT trackers. |
| **SAM 2** | Meta 2024 | Promptable video segmentation (bbox→mask track) | The video-segmentation foundation model used as a promptable tracker; anchors the "foundation model as tracker" story. |
| **SAMURAI** | 2024 | SAM 2 + Kalman motion-aware memory | Zero-shot motion-aware adaptation of SAM 2; tests whether an explicit motion model helps on regular satellite motion. |
| **SAM 3** | Meta 2025 | Unified promptable det+seg+track (+ temporal disambiguation) | The newest foundation model; gives a built-in SAM 2 → SAM 3 generational comparison on spaceborne video, with temporal disambiguation enabled. |

**Design intent:** the set forms a clean generational progression
(Siamese → one-stream ViT → online-token ViT → scaled ViT → video foundation
models), letting us quantify how much each architectural jump buys on
small/low-contrast satellite targets.

### SOT variant / checkpoint strength audit

All seven trackers run at their repo-ceiling public configuration. **6/7 are
unambiguously the strongest release; SiamRPN++ carries one footnote.**

| Tracker | Variant used | Ceiling? | Note |
|---|---|---|---|
| SiamRPN++ | `r50_l234_dwxcorr` | ✅ short-term ceiling | ⚠ checkpoint is community-fork-trained with non-default adjust widths `[128,256,512]` (official is `[256,256,256]`) → **not** directly comparable to PySOT-reported SiamRPN++ numbers. Must be footnoted in paper text. |
| OSTrack | `vitb_384_mae_ce_32x4_ep300` | ✅ paper main result (repo has no ViT-L) | ViT-B, 384² input |
| ODTrack | `baseline_large` (ViT-L) | ✅ paper main result | 1.25 GB ViT-L checkpoint |
| LoRAT | `g-378` (ViT-g/14 DINOv2, 378²) | ✅ largest backbone × input | — |
| SAM 2 | `sam2.1-hiera-large` | ✅ largest Hiera backbone | — |
| SAMURAI | `large` (SAM 2.1 Hiera-L + Kalman) | ✅ matches paper main result | — |
| SAM 3 | `facebook/sam3` + temporal disambiguation | ✅ (sole public release) | — |

---

## 3. Multi-Object Tracking (MOT)

**Datasets:** RsCarData, SAT-MTB mot, SDM-Car (car); AIR-MOT, VISO, LMOD,
BIRDSAI (multi-class / other).
**Primary metrics:** HOTA (with its DetA × AssA decomposition), MOTA, IDF1, IDsw.

**Hard rule (project-wide):** in MOT the model must perform detection on its own
every frame — **no ground-truth box ever enters the model** (not as a first-frame
prompt, not as a clip seed). MOT and SOT are different tasks and their pipelines
are never conflated. (Text-prompted open-vocabulary methods are allowed because
they discover instances, not seed from GT.)

The MOT comparison is deliberately organised around **two axes — detection
ability and association ability — each analysed vs. object pixel size** to expose
the small-object cliff. HOTA's DetA × AssA split is the native tool for this. The
association-only comparison (Exp. 2) feeds the **same GT boxes to every method**
as an equalizer, isolating pure association ability across paradigms.

### 3a. Tracking-by-Detection (TBD)

All TBD trackers consume the **same cached HiEUM detections** (Soft-NMS-decayed
car boxes), so any difference between them is pure association ability on an
identical detection front-end. Appearance-aware variants add a FastReID
embedding per detection.

| Tracker | Venue | Association design | Why we chose it |
|---|---|---|---|
| **SORT** | ICRA 2016 | Kalman + IoU (motion only) | The minimal motion-only baseline — the floor of the TBD family. |
| **ByteTrack** | ECCV 2022 | Two-stage IoU incl. low-score boxes | Keeps low-confidence detections — directly matches the low-SNR satellite scenario. |
| **OC-SORT** | CVPR 2023 | Observation-centric re-update | Explicit occlusion/non-linear-motion handling, relevant on SAT-MTB. |
| **BoT-SORT** | 2022 | Kalman + camera-motion compensation | CMC targets satellite/platform ego-motion. |
| **BoT-SORT-ReID** | 2022 | BoT-SORT + FastReID appearance | Adds appearance to test whether ReID helps at tiny scales. |
| **TrackTrack** | CVPR 2025 | Track-perspective association + track-aware init | The newest online TBD SOTA representative; run on the same HiEUM+FastReID cache. |

> **ReID domain-shift caveat:** the FastReID / MOT17 ReID weights are trained on
> ~128×384 pedestrian crops; aerial cars are ~10×10 px and near-square, so
> appearance cues are weak and matching effectively falls back to IoU. This
> asymmetry is reported as a "ReID training data" column, not hidden.
> Setup: `models/trackers/{sort,bytetrack,ocsort,botsort,botsort_reid,tracktrack}.py`.

### 3b. Joint Detection & Tracking (JDT — one-shot)

JDT models do their **own detection every frame** (heatmap/wh/reg decode) with a
center-sampled ReID embedding, trained end-to-end. They cannot consume the
shared HiEUM cache, so each is *trained* on our splits (one union model across
the three car datasets, evaluated per test split). We report DetA so the
detection gap vs. the TBD front-end stays visible.

| Model | Venue | Type | Why we chose it |
|---|---|---|---|
| **HiEUM** | TPAMI 2024 | Spatio-temporal moving-object detector (car-only) | The RS-specialized moving-object detector; also serves as the **shared detection front-end** for all TBD trackers. |
| **FairMOT** | IJCV 2021 | Generic one-shot JDT (joint det + ReID) | The canonical one-shot JDT baseline. Uses an HRNet-18 backbone (DCNv2-free) so it builds on current CUDA; ImageNet-initialised. |
| **TGraM** | TGRS 2022 | RS-specialized JDT (graph spatio-temporal + adversarial multi-task) | The remote-sensing-specialized JDT counterpart, giving a clean generic-JDT vs. RS-JDT comparison row. Trained from scratch (no ImageNet init for its custom MobileNetV3-Small). |

**Why GT-box association is still fair to JDT:** feeding perfect GT boxes to a
JDT whose stride-4 ReID embedding cannot separate sub-4px objects yields a low
AssA — and that low number *is* the finding ("JDT loses on representation"), not
an artifact. Fair means identical input at the measured boundary, not
compensating for a paradigm's weakness.

### 3c. Query-based end-to-end (third paradigm)

| Model | Venue | Type | Why we chose it |
|---|---|---|---|
| **MOTRv2** | CVPR 2023 | Deformable-DETR + track queries, YOLOX/HiEUM proposals | A genuine **third paradigm** beside TBD and JDT — identity is carried by propagated track queries with **no** center-sampled ReID. Chosen over MOTR because it (a) is trainable on a single GPU by delegating detection to external proposals, and (b) exposes a natural **proposal port** for the GT-box association oracle. Its detection axis = its proposal source (report it explicitly, like the ReID-data column). |

### 3d. Open-vocabulary / text-prompted trackers

These discover instances from a text noun-phrase (e.g. "car", "airplane") with
no GT box — compliant with the no-GT-leakage rule and representing the
foundation-model MOT paradigm.

| Model | Type | Why we chose it |
|---|---|---|
| **SAM 3 / SAM 3.1** | Text-prompted unified det+seg+track | Novel text-prompted MOT on spaceborne video (no published prior results). SAM 3.1 is preferred for dense multi-object scenes (~7× multi-object speedup makes full-dataset inference tractable); SAM 3 and 3.1 are equivalent for low-object-count sequences. |
| **SAM 3 + RAFT** | SAM 3 detections + RAFT optical-flow gating | Adds an optical-flow motion filter that removes static tracklets, isolating *moving* objects — the RS moving-object-tracking definition. |

> Setup: `configs/MOT/{sam3,sam3p1,fairmot,tgram,hieum}_*.yaml`,
> `models/{hieum,sam3}.py`, `eval_{fairmot,tgram,tracktrack,hieum}.py`, `MOTRv2/`.

---

## Cross-cutting design principles

- **Own-splits, not paper numbers.** Every model is re-run on our splits; the
  out-of-domain (pretrained) vs. in-domain (fine-tuned) gap is itself a benchmark
  finding.
- **Strongest public variant.** Each baseline runs at its repo-ceiling
  configuration (see the SOT audit; the one exception, SiamRPN++, is footnoted).
- **Decompose, don't force one "fair" number.** Cross-paradigm MOT is compared
  along detection and association axes vs. object size; confounds (ReID training
  data, proposal source) are reported as columns rather than equalized away.
- **No GT leakage in MOT.** Only open-vocabulary / self-detecting methods; GT
  boxes appear solely in the isolated association-only oracle experiment.
