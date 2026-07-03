# BIRDSAI Detection+Tracking — supervised detectors vs SAM3 train-exemplar

> Overall detection+tracking comparison on the **same GT** (`annotations_sam3`,
> 16 test videos, 78,761 boxes, fine 5-class, IoU 0.5). Two paradigms at one
> operating point each:
> - **Supervised detector + OC-SORT** — in-domain trained detector (06-17), the
>   standalone pipeline (`track_thresh=0.3, min_hits=3`); high precision.
> - **SAM3 train-exemplar** — frozen open-vocab SAM3, one TRAIN exemplar per
>   species + transductive frame-0 selection + SOT + gated periodic re-detect
>   (`evaluation/eval_birdsai_sam3_xexemplar.py`); **no in-domain training, no
>   test GT** (fair, not an oracle). Box scores are constant 1.0 → no mAP, single
>   operating point only.
> Sources: detector+OC-SORT `birdsai_octrack_sam3gt.json`; SAM3 run
> `sam3_birdsai_xexemplar_full_20260622_142509/test_metrics.json`.
>
> **⚠️ Operating-point caveat — do NOT cross-read with `birdsai_detection_sam3gt_compare.md`.**
> The P/R here are measured **after OC-SORT** (`track_thresh=0.3, min_hits=3`), i.e.
> the detect+track pipeline; the detection doc measures the **raw detector** at
> `score ≥ 0.5` with no tracker. So the same detector shows different P/R in the two
> docs — `min_hits=3` confirmation drops transient detections (Recall ↓) and filters
> spurious ones (Precision ↑; most visible on FasterRCNN, P 0.465→0.647). They are
> **different stages / operating points, not a discrepancy.** SAM3 train-exemplar is
> identical in both docs because it is already an end-to-end tracker (its outputs are
> tracks).

## Overall

| Method | Precision | Recall | F1 | MOTA | IDsw |
| --- | ---: | ---: | ---: | ---: | ---: |
| FasterRCNN + OC-SORT | 0.647 | 0.330 | **0.437** | +0.130 | 1,549 |
| YOLO11l + OC-SORT | 0.806 | 0.196 | 0.315 | **+0.143** | **435** |
| DINOv3 + OC-SORT | 0.666 | 0.219 | 0.330 | +0.099 | 829 |
| **SAM3 train-exemplar** | 0.195 | **0.581** | 0.292 | −1.858 | 2,916 |

The two paradigms have **opposite operating-point character**: supervised
detectors are high-precision / low-recall (positive MOTA, few ID switches); SAM3
train-exemplar is high-recall / low-precision — its periodic re-detect floods
false positives, so MOTA goes sharply negative and IDsw is highest. On the pooled
F1 the best detector (FasterRCNN, 0.437) leads SAM3 (0.292), but that lead is
carried almost entirely by the common **elephant** class (61 % of GT).

## Per-class F1 — where each paradigm wins

| Class | #GT | FasterRCNN+OC | YOLO+OC | DINOv3+OC | **SAM3 train-exemplar** |
| --- | ---: | ---: | ---: | ---: | ---: |
| human | 21,853 | 0.125 | 0.078 | 0.030 | **0.128** |
| elephant | 48,055 | **0.618** | 0.443 | 0.460 | 0.565 |
| giraffe | 3,231 | 0.015 | 0.030 | 0.000 | **0.474** |
| lion | 351 | 0.000 | 0.000 | 0.000 | **0.176** |
| unknown | 5,271 | 0.013 | 0.000 | 0.000 | **0.057** |

**This is the headline.** The supervised detectors only track the large, common
classes (elephant, and partly human); every rare / tiny species —
**giraffe, lion, unknown — collapses to ≈ 0**. SAM3 train-exemplar is the **only**
method that tracks them: giraffe 0.474, lion 0.176, unknown 0.057, with per-class
recall up to lion 0.79 / giraffe 0.55. A single train exemplar recovers exactly
the long-tail species the in-domain detectors never learn to find.

## Talking points

1. **Overall MOTA/F1 favours supervised detectors — but only via elephant.**
   High precision + OC-SORT gives positive MOTA and the top pooled F1; strip the
   one common class and the detectors have almost nothing on the tail.
2. **SAM3 train-exemplar owns the long tail.** giraffe / lion / unknown are
   tracked only by SAM3 (detectors ≈ 0). One TRAIN exemplar per species, no
   in-domain training, no test GT — a fair result, not an oracle.
3. **Opposite failure modes.** Detectors miss (low recall → high MOTA but blind to
   rare species); SAM3 over-fires (high recall → FP flood → negative MOTA). Pick by
   need: census of a known common species → detector; find-anything incl. rare
   species → train-exemplar.
4. Consistent with the detection-level finding and the GT-oracle control
   (`birdsai_gt_oracle_results.md`): the bottleneck is **detection / finding the
   tiny rare animals**, and the train-exemplar is the one route that addresses it.
