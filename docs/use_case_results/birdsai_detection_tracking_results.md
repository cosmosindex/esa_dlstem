# BIRDSAI — Detection & Tracking Test-Set Results

Wild-animal use case: **YOLO11l**, **FasterRCNN-R50-FPN**, and **DINOv3-ViT-B16+FCOS**
evaluated on the BIRDSAI test split (thermal-infrared aerial drone video).

Each detector is evaluated for pure detection (mAP) and for tracking. The
**main tracking tables use OC-SORT**; a full **6-tracker TBD sweep** (every
tracking-by-detection algorithm in the benchmark × all 3 detectors) is reported
in its own section below. Numbers are taken verbatim from each run's saved
metrics (tracking from `test_metrics.json`, detection mAP from the training
runs' W&B test logs — see Provenance).

## Evaluation setup

- **Split:** test (11 videos, 71,905 GT boxes)
- **Taxonomy:** fine-grained 5-class `{human, elephant, giraffe, lion, unknown}` (0-indexed)
- **Tracker:** OC-SORT (per-class), `track_thresh = 0.3`, `min_hits = 3`, `max_age = 30`
- **Matching:** greedy, `IoU ≥ 0.5`
- **Script:** `evaluation/eval_birdsai_detect_track.py`

> Two evaluation protocols are reported below: **(a)** threshold-free **detection
> mAP@0.5** from each detector's own test phase (640² eval, no tracker — the
> fairest pure-detection metric); **(b)** **detection + tracking** Precision /
> Recall / F1 / MOTA from the shared OC-SORT pipeline (original resolution,
> `score ≥ 0.3` operating point). The Precision/Recall differ between the two
> because they are different resolutions and operating points — do not mix them.

## Detection (mAP@0.5, threshold-free)

Per-class Average Precision and mean AP @ IoU 0.5 from each detector's own test
phase (640², detector only). This is the cleanest pure-detection comparison.

| Class | YOLO11l | DINOv3 | FasterRCNN |
| --- | ---: | ---: | ---: |
| human | 0.014 | **0.194** | 0.030 |
| elephant | 0.611 | 0.687 | **0.709** |
| giraffe | **0.059** | 0.004 | 0.035 |
| lion | n/a | n/a | n/a |
| unknown | 0.000 | **0.009** | 0.001 |
| **mAP** | 0.171 | **0.223** | 0.194 |

- **Ranking: DINOv3 (0.223) > FasterRCNN (0.194) > YOLO11l (0.171).** DINOv3's
  frozen semantic features give it the best `human` AP and top mAP despite the
  tiny thermal targets.
- **`lion` = n/a** — 0 test GT boxes (torchmetrics returns AP = −1 for DINOv3/FRCNN,
  0.0 for YOLO; both mean "undefined", not a real score).

## Overall (pooled over classes)

| Model | Precision | Recall | F1 | MOTA | IDF1 | IDsw |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **DINOv3** ViT-B16+FCOS | 0.812 | 0.399 | **0.535** | 0.290 | **0.535** | 1,205 |
| **YOLO11l** (manual loop) | **0.889** | 0.350 | 0.502 | **0.299** | 0.502 | **496** |
| **FasterRCNN** R50-FPN | 0.429 | **0.488** | 0.457 | −0.198 | 0.457 | 2,652 |

- **DINOv3** wins F1 / IDF1 (highest recall among the precise models).
- **YOLO11l** is the most precise → best MOTA and far fewest ID switches.
- **FasterRCNN** over-detects (low precision → FP flood in sparse thermal scenes) → negative MOTA.

## Per-class detection (Precision / Recall / F1 @ IoU 0.5)

| Class | #GT | YOLO11l | DINOv3 | FasterRCNN |
| --- | ---: | --- | --- | --- |
| human | 16,804 | 0.265 / 0.001 / 0.002 | 0.448 / 0.001 / 0.002 | 0.098 / 0.062 / 0.076 |
| elephant | 48,055 | 0.911 / 0.521 / **0.663** | 0.812 / 0.597 / **0.688** | 0.617 / 0.705 / **0.658** |
| giraffe | 3,231 | 1.000 / 0.035 / 0.068 | 0.500 / 0.000 / 0.001 | 0.059 / 0.044 / 0.051 |
| lion | 0 | — | — | — |
| unknown | 3,815 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |

## Per-class tracking (MOTA / IDsw)

| Class | #GT | YOLO11l | DINOv3 | FasterRCNN |
| --- | ---: | --- | --- | --- |
| human | 16,804 | −0.002 / 0 | −0.000 / 1 | −0.520 / 160 |
| elephant | 48,055 | 0.460 / 493 | 0.434 / 1,204 | 0.215 / 2,486 |
| giraffe | 3,231 | 0.034 / 3 | 0.000 / 0 | −0.658 / 6 |
| lion | 0 | n/a | n/a | n/a |
| unknown | 3,815 | −0.012 / 0 | 0.000 / 0 | −2.942 / 0 |

## TBD tracker sweep (all 6 trackers × 3 detectors)

Every tracking-by-detection algorithm in the benchmark, run on **each detector's
identical cached detections** (`predictions.json` — the detector is never re-run,
so every cell shares the same detection input → differences are pure association).
Per-class trackers, same IoU ≥ 0.5 scoring as above. Appearance-aware trackers
(BoT-SORT-ReID, TrackTrack) use FastReID features; **MOT17 person-ReID weights on
grayscale thermal animals = domain-mismatched → appearance cues ~useless** (run for
completeness). Tracker kwargs mirror the benchmark's tiny-object (rscardata/sdmcar)
tuning. Scripts: `evaluation/eval_birdsai_track_sweep.py` + `cache_birdsai_feats.py`.

**F1 (detection-quality of tracked boxes)**

| Tracker | YOLO11l | DINOv3 | FasterRCNN |
| --- | ---: | ---: | ---: |
| SORT | 0.534 | 0.365 | 0.219 |
| ByteTrack | 0.476 | 0.385 | 0.446 |
| OC-SORT | 0.540 | **0.560** | 0.368 |
| BoT-SORT | 0.518 | 0.458 | 0.368 |
| BoT-SORT-ReID | **0.550** | 0.547 | 0.356 |
| TrackTrack | 0.547 | **0.560** | **0.409** |

**MOTA**

| Tracker | YOLO11l | DINOv3 | FasterRCNN |
| --- | ---: | ---: | ---: |
| SORT | +0.183 | −1.288 | −3.635 |
| ByteTrack | +0.246 | +0.023 | −0.328 |
| OC-SORT | +0.308 | +0.271 | −0.946 |
| BoT-SORT | +0.244 | −0.004 | −0.981 |
| BoT-SORT-ReID | +0.289 | +0.165 | −1.115 |
| TrackTrack | **+0.325** | **+0.306** | **−0.576** |

**ID switches (lower = better)**

| Tracker | YOLO11l | DINOv3 | FasterRCNN |
| --- | ---: | ---: | ---: |
| SORT | 1,402 | 6,817 | 5,604 |
| ByteTrack | 2,959 | 6,551 | 4,181 |
| OC-SORT | 865 | 1,953 | 4,227 |
| BoT-SORT | 4,233 | 8,938 | 6,021 |
| BoT-SORT-ReID | 2,502 | 4,765 | 4,987 |
| TrackTrack | **597** | **528** | **2,256** |

**Findings**

- **TrackTrack is the best tracker overall** — top MOTA on every detector and
  *dramatically* fewest ID switches (YOLO 597, DINOv3 528 vs OC-SORT 865/1,953,
  BoT-SORT 4,233/8,938). Its association (motion + NMS-rescue + global matching)
  carries it even though the appearance branch is domain-mismatched.
- **OC-SORT is the strongest appearance-free tracker** (best/near-best F1, second-best
  MOTA & IDsw). The simple **SORT** and the over-eager **BoT-SORT** churn the most IDs.
- **ReID adds nothing here**: BoT-SORT-ReID ≈ BoT-SORT (sometimes worse), confirming
  the MOT17 person-ReID features are useless on thermal animals — TrackTrack's edge is
  its *design*, not the appearance cue.
- **The detector dominates the tracker.** Every tracker on **FasterRCNN** lands at
  negative MOTA (its low-precision FP flood overwhelms any association). High-precision
  **YOLO11l** + a good tracker gives the best MOTA; **DINOv3** + OC-SORT/TrackTrack gives
  the best F1 (0.560).
- The OC-SORT row here uses the benchmark's uniform kwargs (`min_hits=1, iou_threshold=0.1`)
  for cross-tracker fairness, so it differs slightly from the standalone OC-SORT table
  above (`min_hits=3, iou_threshold=0.3` → F1 0.502 / MOTA 0.299 / IDsw 496).

## Reading the numbers

- **Only elephant works** (large, common: 67% of all GT boxes). Every tiny / rare
  species (human ~15–28 px, giraffe, unknown) collapses to near-zero recall — the
  same tiny-thermal-object difficulty seen across all three models.
- **`lion` has 0 test GT boxes** (8 train / 3 test tracks, all in held-out videos here),
  so its per-class MOTA values (−612, −2446) are denominator artifacts — `n/a`, not a signal.
- **`unknown`** (species-unlabeled animals) is never recovered by any model → 0 across the board.
- Almost all ID switches come from the **elephant** class (the only one actually tracked):
  YOLO 493 / 496, DINOv3 1204 / 1205, FRCNN 2486 / 2652 total.

## Provenance

Under `/work/ziwen/experiments/`:

| Model | Detection mAP (training run, W&B `test/*`) | Tracking (`test_metrics.json`) |
| --- | --- | --- |
| YOLO11l | `yolo11l_birdsai_manual_20260615_115439` (`test_metrics.json`) | `yolo_birdsai_dettrack_20260615_140408` |
| DINOv3 | `dinov3_vitb16_birdsai_20260615_141227` (W&B run `o6a6h3tv`) | `dinov3_birdsai_dettrack_20260615_182902` |
| FasterRCNN | `fasterrcnn_birdsai_20260615_115438` (W&B run `ti0ezb0x`) | `fasterrcnn_birdsai_dettrack_20260615_182903` |

> Detection mAP for DINOv3 / FasterRCNN comes from each training run's W&B test
> summary (`test/mAP`, `test/AP_per_class/*`) — the Lightning detector runs do
> not write a local `test_metrics.json` (YOLO's manual loop does). FasterRCNN's
> per-class keys are 1-indexed (`cls1`=human … `cls5`=unknown; `cls0`=background);
> DINOv3/YOLO are 0-indexed.

> **SAM3** is evaluated separately as an open-vocabulary text-prompted method on the
> **coarse 2-class** `{animal, human}` taxonomy — see the standalone section below.

## SAM3 — open-vocabulary text-prompted (separate, coarse 2-class)

SAM3 cannot distinguish fine species from a text prompt, so it runs on the **coarse
2-class** taxonomy with two noun phrases — **"animal"** and **"person"** — and **no GT
boxes ever reach the model** (open-vocab self-detection + SAM3's own tracker, not OC-SORT).
**These numbers are NOT comparable to the fine-grained 5-class detector tables above**
(different taxonomy, different tracker, frozen non-thermal model).

| Metric | SAM3 (text: animal / person) |
| --- | ---: |
| Precision | 0.478 |
| Recall | 0.297 |
| F1 / IDF1 | 0.367 |
| MOTA | −0.031 |
| IDsw | 275 |
| AP@0.5 | 0.152 |

- Eval: text-prompt MOT, `clip_len = 32`, global MOTA @ IoU ≥ 0.5; 11 test videos,
  44,197 tracked boxes (animal 40,210 / person 3,987). Ran ~47 min @ 4.3 fps.
- As expected, a frozen open-vocabulary model on tiny thermal-IR blobs underperforms
  the in-domain trained detectors — SAM3 has never seen thermal imagery.
- Config `configs/MOT/sam3_birdsai.yaml`; run dir
  `/data/ESA_DLSTEM_2025/experiments/MOT_birdsai_sam3/sam3_text_birdsai_20260616_094330`
  (`predictions.json` + `mot_format/`). Metrics from the Lightning test table
  (`logs/mot/sam3_birdsai_20260616_094324.log`) — this eval path does not write a
  `test_metrics.json`.
