# Fire Detection Comparison on RGBT-3M (FireRGBT)

Use case: **wildfire detection** (detection-only, no tracking).
Dataset `FireRGBT` (RGBT-3M) · fine 3-class {smoke, fire, person} · all images 640×480.
Identical train/val/test splits and transforms across all three models (apples-to-apples).
Test set = 3,366 frames · IoU = 0.5 · mAP@0.5 = VOC all-point, score-ranked.
Detection operating point = score ≥ 0.5.

Three detector paradigms:

| Method | Paradigm | Backbone | Trainable |
|---|---|---|---:|
| **Faster R-CNN** | fine-tuned two-stage | R50-FPN | 41.3 M |
| **YOLO11l** | fine-tuned one-stage | YOLO11l (manual-loop trained) | ~25 M |
| **DINOv3+FCOS** | frozen foundation features + dense head | DINOv3 ViT-B/16 (frozen) | 4.9 M head / 90.6 M total |

## Per-class AP@0.5

| Class | Faster R-CNN | YOLO11l | DINOv3+FCOS |
|---|---:|---:|---:|
| smoke | **0.950** | 0.929 | 0.932 |
| fire | **0.883** | 0.849 | 0.661 |
| person | **0.876** | 0.873 | 0.701 |
| **mAP** | **0.903** | 0.884 | 0.765 |

## Overall detection metrics

| Method | Precision | Recall | F1 | mAP@0.5 |
|---|---:|---:|---:|---:|
| Faster R-CNN | 0.709 | 0.939 | 0.808 | **0.903** |
| YOLO11l | 0.919 | 0.870 | 0.894 | 0.884 |
| DINOv3+FCOS | 0.685 | 0.832 | 0.751 | 0.765 |

## Per-class detection P / R / F1 (single operating point, score ≥ 0.5)

### Faster R-CNN
| Class | nGT | P | R | F1 |
|---|---:|---:|---:|---:|
| smoke | 4086 | 0.877 | 0.949 | 0.912 |
| fire | 3401 | 0.851 | 0.898 | 0.874 |
| person | 1740 | 0.849 | 0.917 | 0.882 |

### DINOv3+FCOS
| Class | nGT | P | R | F1 |
|---|---:|---:|---:|---:|
| smoke | 4086 | 0.965 | 0.806 | 0.879 |
| fire | 3401 | 0.935 | 0.441 | 0.599 |
| person | 1740 | 0.951 | 0.480 | 0.638 |

(YOLO11l per-class P/R not dumped at frame level; only per-class AP above.)

## Detection by object size (small vs large, COCO area split)

The "frozen foundation features collapse on tiny objects" finding — frozen
stride-16 DINOv3 features keep precision but lose almost all recall on small
fire/person, while the fine-tuned detector holds up.

| Method | size | P | R | F1 |
|---|---|---:|---:|---:|
| Faster R-CNN | small | 0.805 | 0.870 | 0.836 |
| Faster R-CNN | large | 0.896 | 0.955 | 0.924 |
| DINOv3+FCOS | small | 0.946 | **0.178** | 0.300 |
| DINOv3+FCOS | large | 0.956 | 0.858 | 0.904 |

(Per-size dump unavailable for YOLO11l.)

## Efficiency

| Method | Params | GFLOPs | FPS | Model MB |
|---|---:|---:|---:|---:|
| Faster R-CNN | 41.3 M | 181.7 | 97.6 | 158.0 |
| DINOv3+FCOS | 90.6 M | 385.3 | 117.0 | 345.6 |

(YOLO11l efficiency not captured in its manual-loop `test_metrics.json`.)

## Takeaways

- **Fine-tuned detectors win**: Faster R-CNN (0.903) ≈ YOLO11l (0.884) ≫ frozen
  DINOv3+FCOS (0.765).
- **Frozen DINOv3 matches on large/easy smoke** (0.93, on par with both fine-tuned
  detectors) but **lags badly on small fire/person** (~29 px median): per-class AP
  0.66 / 0.70 vs Faster R-CNN's 0.88 / 0.88.
- The size split makes the mechanism explicit: frozen DINOv3 small-object **recall
  collapses to 0.178** (vs Faster R-CNN 0.870) at near-perfect precision — frozen
  stride-16 features can localise big objects but miss tiny ones. This is the
  "frozen foundation features vs fine-tuned task-specific detector" result.

## Figure — performance vs object size

![Fire detection vs object size](figures/fire_size_trend.png)

Per-frame predictions were re-dumped (`evaluation/eval_fire_detect_dump.py`) and
re-binned by GT object size (sqrt-area, 640² eval space, 6 quantile bins ≈ 1538
boxes each). One panel per detector; lines = Precision / Recall / F1 at IoU 0.5,
score ≥ 0.5. Built by `tools/plot_fire_size_trend.py`
(→ `figures/fire_size_trend.{png,pdf,csv}`). Same layout as the BIRDSAI
size-trend plot.

Reading: **Faster R-CNN** and **YOLO11l** stay flat-high across all sizes (only a
mild dip at the smallest <18 px bin, F1 ≈ 0.71 / 0.73). **DINOv3+FCOS** keeps high
**precision** everywhere but its **recall collapses on small objects** — R = 0.05
at <18 px and 0.24 at 18–30 px, only recovering above ~42 px. The frozen stride-16
foundation features localise large smoke as well as the fine-tuned detectors but
miss tiny fire/person — exactly the per-size mechanism behind the 0.765 vs
0.90/0.88 overall mAP gap.

## Sources

- Faster R-CNN: `/work/ziwen/experiments/fasterrcnn_fire_20260525_202026/test_metrics.json`
- YOLO11l: `/work/ziwen/experiments/yolo11l_fire_manual_20260611_161635/test_metrics.json`
- DINOv3+FCOS: `/work/ziwen/experiments/dinov3_vitb16_fire_20260612_122759/test_metrics.json`
