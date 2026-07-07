# Vehicle Detection & Tracking — HiEUM Detection Results (Union ep8)

Use case: **moving-vehicle (car) detection and tracking** on the space-tracker car
MOT benchmark. This document reports the **detection** results of the checkpoint we
have committed to as the HiEUM detection row.

**Committed checkpoint: HiEUM union ep8** (`model_ep8.pth`). HiEUM was retrained
(continuing from the public author checkpoint) on the **union of all three car MOT
train sets** — RsCarData + SAT-MTB car + SDM-Car — supervised (`sup_mode 0`),
15 epochs. Epoch **8** was the global best on the union validation split
(best-F1 = 0.6007, argmax rule) and also won on the held-out test sets
(mean best-F1 +0.022 over the author checkpoint, +0.068 on SDM-Car); see
[`hieum_union_vs_author.md`](hieum_union_vs_author.md) for the full ep8 / ep12 /
author comparison. From here on, **ep8 is the detection result for this use case.**

## Protocol

- Single class **`car`**. Detector-only (no cross-frame identity), so ID switches
  are 0 by construction.
- Matching: HiEUM paper protocol — **centroid Euclidean distance ≤ 5 px** (not IoU).
- Input: full-resolution clips, `hieum_image_size = 1024²`, `seq_len = 20`.
- **best-F1** = each set evaluated at its own optimal score threshold via a shared
  sweep; best-Precision / best-Recall are that operating point. This is the primary
  detection metric. Two thresholds are involved:
  - `score_thresh = 0.01` — the **model output floor**: at inference HiEUM keeps
    every detection with score ≥ 0.01, so the candidate pool handed to the sweep is
    complete.
  - **sweep floor `0.02`** — the **lowest candidate cutoff the sweep searches**: F1
    is maximised only over thresholds ≥ 0.02, never below. It sits just above the
    model floor so that at every evaluated cutoff (≥ 0.02) the candidate pool
    (≥ 0.01) is still complete — the sweep can never brush the model's output floor.

  The winning per-set thresholds (`best-thr` below, 0.16–0.20) land well above the
  0.02 floor, so the floor does not affect the reported operating points.
- Object size: on the COCO area split, **every car box lands in the `small`
  bucket** on all three sets (there are effectively no large cars in satellite
  video), so a small-vs-large breakdown is degenerate here and is omitted.

## Detection results — best-F1 (own-optimal threshold)

| Test set | num GT | best-F1 | best-P | best-R | best-thr |
|---|---:|---:|---:|---:|---:|
| **RsCarData** | 153,428 | **0.8989** | 0.9486 | 0.8554 | 0.20 |
| **SAT-MTB** (car) | 255,817 | **0.7172** | 0.7801 | 0.6868 | 0.16 |
| **SDM-Car** | 290,010 | **0.6324** | 0.7563 | 0.5693 | 0.20 |
| **Mean** | — | **0.7495** | 0.8283 | 0.7038 | — |

## Detection counts (lowest-threshold operating point)

TP / FP / FN recorded at each run's default `per_category` operating point — the
sweep's **lowest cutoff (≈0.02)**, *not* the best-F1 threshold of the table above.
At this near-floor threshold recall is close to its ceiling (0.81–0.92) while FP is
large: the small-car detection-recall picture — the model finds most cars but
over-fires under the strict 5-px centroid match. Matching is centroid ≤ 5 px.

| Test set | num GT | TP | FP | FN |
|---|---:|---:|---:|---:|
| RsCarData | 153,428 | 141,800 | 142,310 | 11,628 |
| SAT-MTB (car) | 255,817 | 206,792 | 282,098 | 49,025 |
| SDM-Car | 290,010 | 179,727 | 247,598 | 110,283 |

(HiEUM is **detector-only** — no cross-frame identity — so MOT metrics like MOTA /
IDF1 / ID-switches degenerate to detection quantities here: IDsw is 0 by
construction, and MOTA reduces to `1 − (FP+FN)/GT`, adding nothing over the TP/FP/FN
above. They are therefore omitted. Only P/R/F1 at the swept best-thr — the primary
detection metric in the table above — should be read as the checkpoint's quality.)

## Efficiency

| Test set | FPS | Model size (MB) |
|---|---:|---:|
| RsCarData | 12.0 | 4.57 |
| SAT-MTB (car) | 12.1 | 4.57 |
| SDM-Car | 8.7 | 4.57 |

HiEUM is a compact 3D-sparse-conv detector (~4.6 MB); throughput is measured on
full-res 1024² clips, `seq_len = 20`.

## Takeaways

- **HiEUM union ep8 is detection/recall-bound on tiny cars.** Precision at the
  best-F1 point is strong (0.76–0.95) but recall falls from 0.855 (RsCarData) to
  0.687 (SAT-MTB) to 0.569 (SDM-Car) as the target scale and density get harder.
- best-F1 tracks dataset difficulty: RsCarData (0.899) ≫ SAT-MTB (0.717) >
  SDM-Car (0.632); **mean best-F1 = 0.750**.
- Every car is a `small` object under COCO's area split — the vehicle-detection
  challenge here is fundamentally a small-object recall problem.

## Provenance

- Checkpoint: `model_ep8.pth` from the 2026-06-29 supervised union retrain
  (RsCarData + SAT-MTB car + SDM-Car), continued from the public HiEUM author
  checkpoint. See memory `hieum_union_retrain.md`.
- Eval driver: `evaluation/eval_hieum.py`; per-set configs mirror
  `configs/MOT/hieum_{rscardata,satmtb,sdmcar}.yaml` with the ep8 checkpoint path,
  the model score floor (`score_thresh = 0.01`), and the score sweep (floor `0.02`)
  overridden.
- Raw outputs: `test_metrics.json` under
  `/work/ziwen/experiments/hieum_test_compare_20260702_v2/{rscardata,satmtb,sdmcar}_ep8/`.
</content>
</invoke>
