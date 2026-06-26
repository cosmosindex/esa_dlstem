# BIRDSAI Tracking Comparison on SAM3-refined GT (`annotations_sam3`)

6 TBD trackers × 3 detectors (cached detections → online tracking), re-scored on the **same GT as the detection table**.
Videos: 16/16 (sweep subset, all 18 runs identical set) · 15494 frames · IoU 0.5 · fine 5-class.
MOTA = 1−(FP+FN+IDsw)/GT · IDsw = identity switches · F1/P/R = detection level.

## MOTA (rows = tracker, cols = detector)

| Tracker | FasterRCNN | YOLO11l | DINOv3 |
|---|---:|---:|---:|
| SORT | -0.708 | **+0.120** | -0.297 |
| OC-SORT | -0.153 | **+0.168** | +0.087 |
| ByteTrack | +0.053 | **+0.127** | +0.055 |
| BoT-SORT | -0.237 | **+0.117** | +0.022 |
| BoT-SORT+ReID | -0.242 | **+0.150** | +0.057 |
| TrackTrack | +0.066 | **+0.187** | +0.105 |

## F1 (rows = tracker, cols = detector)

| Tracker | FasterRCNN | YOLO11l | DINOv3 |
|---|---:|---:|---:|
| SORT | 0.379 | **0.437** | 0.363 |
| OC-SORT | **0.454** | 0.392 | 0.382 |
| ByteTrack | **0.482** | 0.334 | 0.317 |
| BoT-SORT | **0.447** | 0.381 | 0.372 |
| BoT-SORT+ReID | **0.446** | 0.407 | 0.397 |
| TrackTrack | **0.483** | 0.398 | 0.385 |

## IDsw (rows = tracker, cols = detector)

| Tracker | FasterRCNN | YOLO11l | DINOv3 |
|---|---:|---:|---:|
| SORT | 4372 | **1484** | 4604 |
| OC-SORT | 3631 | **738** | 1585 |
| ByteTrack | 3854 | **2264** | 3214 |
| BoT-SORT | 6433 | **3938** | 6241 |
| BoT-SORT+ReID | 5425 | **2416** | 3672 |
| TrackTrack | 1686 | **429** | 833 |

## Per-class F1 — FasterRCNN backbone (IoU 0.5)

| Tracker | human | elephant | giraffe | lion | unknown |
|---|---:|---:|---:|---:|---:|
| SORT | 0.203 | 0.596 | 0.016 | 0.003 | 0.037 |
| OC-SORT | 0.176 | 0.662 | 0.020 | 0.000 | 0.018 |
| ByteTrack | 0.144 | 0.683 | 0.017 | 0.000 | 0.017 |
| BoT-SORT | 0.176 | 0.661 | 0.021 | 0.000 | 0.018 |
| BoT-SORT+ReID | 0.181 | 0.659 | 0.021 | 0.000 | 0.018 |
| TrackTrack | 0.177 | 0.683 | 0.021 | 0.000 | 0.017 |
