# BIRDSAI Detection Comparison on SAM3-refined GT (`annotations_sam3`)

Frame set: identical 15,494 test frames · IoU = 0.5 · fine 5-class.
Detector operating point = score ≥ 0.5; SAM3-xexemplar boxes carry score 1.0.
mAP@0.5 = VOC all-point, score-ranked (only meaningful for the score-producing detectors).

## Per-class F1 (IoU 0.5, single operating point)

| Class | nGT | FasterRCNN | YOLO11l | DINOv3 | SAM3-xexemplar |
|---|---:|---:|---:|---:|---:|
| human | 21853 | 0.164 | 0.109 | 0.025 | 0.128 |
| elephant | 48055 | 0.673 | 0.496 | 0.474 | 0.565 |
| giraffe | 3231 | 0.022 | 0.032 | 0.000 | 0.474 |
| lion | 351 | 0.000 | 0.000 | 0.000 | 0.176 |
| unknown | 5271 | 0.018 | 0.001 | 0.000 | 0.057 |
| **OVERALL** | 78761 | **0.459** | **0.359** | **0.340** | **0.292** |

## Overall detection metrics

| Method | Precision | Recall | F1 | mAP@0.5 |
|---|---:|---:|---:|---:|
| FasterRCNN | 0.465 | 0.452 | 0.459 | 0.139 |
| YOLO11l | 0.772 | 0.234 | 0.359 | 0.101 |
| DINOv3 | 0.666 | 0.228 | 0.340 | 0.077 |
| SAM3-xexemplar | 0.195 | 0.581 | 0.292 |   —   |

## Per-class mAP@0.5 (detectors only)

| Class | nGT | FasterRCNN | YOLO11l | DINOv3 |
|---|---:|---:|---:|---:|
| human | 21853 | 0.044 | 0.084 | 0.026 |
| elephant | 48055 | 0.649 | 0.400 | 0.355 |
| giraffe | 3231 | 0.001 | 0.020 | 0.001 |
| lion | 351 | 0.000 | 0.000 | 0.000 |
| unknown | 5271 | 0.002 | 0.000 | 0.002 |
| **mean** | — | **0.139** | **0.101** | **0.077** |

## Appendix — per-class Precision / Recall (IoU 0.5)


### FasterRCNN

| Class | P | R | F1 |
|---|---:|---:|---:|
| human | 0.189 | 0.144 | 0.164 |
| elephant | 0.674 | 0.672 | 0.673 |
| giraffe | 0.022 | 0.022 | 0.022 |
| lion | 0.000 | 0.000 | 0.000 |
| unknown | 0.015 | 0.022 | 0.018 |

### YOLO11l

| Class | P | R | F1 |
|---|---:|---:|---:|
| human | 0.551 | 0.061 | 0.109 |
| elephant | 0.820 | 0.355 | 0.496 |
| giraffe | 0.788 | 0.016 | 0.032 |
| lion | 0.000 | 0.000 | 0.000 |
| unknown | 0.009 | 0.001 | 0.001 |

### DINOv3

| Class | P | R | F1 |
|---|---:|---:|---:|
| human | 0.753 | 0.013 | 0.025 |
| elephant | 0.665 | 0.368 | 0.474 |
| giraffe | 0.000 | 0.000 | 0.000 |
| lion | 0.000 | 0.000 | 0.000 |
| unknown | 0.000 | 0.000 | 0.000 |

### SAM3-xexemplar

| Class | P | R | F1 |
|---|---:|---:|---:|
| human | 0.080 | 0.314 | 0.128 |
| elephant | 0.463 | 0.724 | 0.565 |
| giraffe | 0.419 | 0.546 | 0.474 |
| lion | 0.099 | 0.792 | 0.176 |
| unknown | 0.031 | 0.393 | 0.057 |
