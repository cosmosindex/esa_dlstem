# HiEUM: Union-Trained vs. Author Checkpoint (Car MOT Test Sets)

Use case: **moving-car detection** on the space-tracker car MOT benchmark.

The public HiEUM checkpoint was trained on **RsCarData alone**. Our benchmark
carries more car data, so we retrained HiEUM (continuing from the author
checkpoint) on the **union of all three car MOT train sets**
(RsCarData + SAT-MTB car + SDM-Car), supervised (`sup_mode 0`), 15 epochs.
Epoch **8** was selected as the global best on the union validation split
(best-F1 = 0.6007); epoch **12** (converged low-LR tail) is reported as a
stability check.

This document compares the **author checkpoint** against the **union ep8 / ep12**
checkpoints on the held-out car test sets.

## Protocol (identical for all checkpoints)

- Metric: **best-F1** — each checkpoint evaluated at its own optimal score
  threshold via a shared sweep, plus F1 at a fixed threshold and MOTA for context.
- Matching: HiEUM paper protocol — **centroid Euclidean distance ≤ 5 px** (not IoU).
- Input: full-res clips, `hieum_image_size = 1024²`, `seq_len = 20`, single class `car`.
- Fairness fix: `score_thresh = 0.01` with the sweep floor at `0.02` so the
  union checkpoints' lower-scored detections are not pre-filtered (see caveat below).
- Detector-only (no cross-frame identity); ID switches are 0 by construction.

## Results — best-F1 (own-optimal threshold)

| Test set | num GT | Author | Union ep8 | Union ep12 | ep8 − Author |
|---|---:|---:|---:|---:|---:|
| **RsCarData** (in-domain for author) | 153,428 | **0.9058** | 0.8989 | 0.8985 | −0.007 |
| **SAT-MTB** (car) | 255,817 | 0.7120 | **0.7172** | 0.6899 | +0.005 |
| **SDM-Car** | 290,010 | 0.5646 | **0.6324** | 0.6313 | **+0.068** |
| **Mean** | — | 0.7275 | **0.7495** | 0.7399 | **+0.022** |

Best-threshold operating points (P / R / thr):

| Test set | Author | Union ep8 | Union ep12 |
|---|---|---|---|
| RsCarData | 0.942 / 0.873 @0.25 | 0.949 / 0.855 @0.20 | 0.945 / 0.858 @0.20 |
| SAT-MTB   | 0.796 / 0.672 @0.30 | 0.780 / 0.687 @0.16 | 0.759 / 0.661 @0.16 |
| SDM-Car   | 0.686 / 0.515 @0.35 | 0.756 / 0.569 @0.20 | 0.754 / 0.574 @0.20 |

## Verdict

**Union ep8 is the better checkpoint overall** (+0.022 mean best-F1).

- The author checkpoint wins only on **RsCarData** (+0.007), its own training
  domain — expected, and the margin is small.
- On **SAT-MTB** the two are tied (+0.005, noise-level).
- On **SDM-Car** union ep8 gains a substantial **+0.068**, i.e. union training
  trades a hair of in-domain RsCarData performance for a large out-of-domain
  generalization gain.
- **ep8 ≥ ep12 on all three sets** (notably SAT-MTB ep8 − ep12 = +0.027),
  confirming the union-validation argmax pick; the "stable tail" ep12 is not
  better on test.

Recommendation: use **union ep8** as the HiEUM row of the car MOT benchmark.

## Caveat — why the fairness fix matters

A first pass used the shipped configs (`score_thresh = 0.2` for SAT-MTB/SDM-Car,
sweep floor `0.10`). That pre-filter dropped every detection below 0.2 before the
metric, and the union checkpoints — whose score distribution shifted lower after
retraining — had their optimum **pegged at the sweep floor**, understating their
best-F1. Lowering `score_thresh` to 0.01 and extending the sweep down to 0.02
moved the union optima to interior thresholds (SAT-MTB 0.16, SDM-Car 0.20); the
numbers above are from that corrected (v2) run.

MOTA is negative on all runs (dense small cars → many FPs, and the strict 5-px
centroid match penalizes hard), but this is a detection-recall property of the
task and does not change the relative ranking between checkpoints.

## Provenance

- Author ckpt: `model_best.pth` (RsCarData-only, public release).
- Union ckpts: `model_ep{8,12}.pth` from the 2026-06-29 supervised union retrain.
- Eval driver: `evaluation/eval_hieum.py`; per-set configs mirror
  `configs/MOT/hieum_{rscardata,satmtb,sdmcar}.yaml` with the checkpoint path and
  score sweep overridden.
- Raw outputs: `test_metrics.json` under each
  `hieum_test_compare_20260702_v2/{dataset}_{tag}/` run directory.
