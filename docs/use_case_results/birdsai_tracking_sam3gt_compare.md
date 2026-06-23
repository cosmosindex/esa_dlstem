# BIRDSAI Tracking Comparison on SAM3-refined GT (`annotations_sam3`)

6 TBD trackers × 3 detectors (cached detections → online tracking), re-scored on the **same GT as the detection table**.
Videos: 11/16 (sweep subset, all 18 runs identical set) · 12236 frames · IoU 0.5 · fine 5-class.
MOTA = 1−(FP+FN+IDsw)/GT · IDsw = identity switches · F1/P/R = detection level.

## MOTA (rows = tracker, cols = detector)

| Tracker | FasterRCNN | YOLO11l | DINOv3 |
|---|---:|---:|---:|
| SORT | -3.579 | **+0.103** | -1.311 |
| OC-SORT | -0.888 | +0.243 | **+0.263** |
| ByteTrack | -0.299 | **+0.190** | +0.030 |
| BoT-SORT | -0.933 | **+0.186** | -0.001 |
| BoT-SORT+ReID | -1.055 | **+0.221** | +0.154 |
| TrackTrack | -0.510 | +0.256 | **+0.299** |

## F1 (rows = tracker, cols = detector)

| Tracker | FasterRCNN | YOLO11l | DINOv3 |
|---|---:|---:|---:|
| SORT | 0.232 | **0.488** | 0.364 |
| OC-SORT | 0.391 | 0.496 | **0.557** |
| ByteTrack | **0.463** | 0.433 | 0.391 |
| BoT-SORT | 0.389 | **0.476** | 0.462 |
| BoT-SORT+ReID | 0.378 | 0.505 | **0.543** |
| TrackTrack | 0.436 | 0.500 | **0.557** |

## IDsw (rows = tracker, cols = detector)

| Tracker | FasterRCNN | YOLO11l | DINOv3 |
|---|---:|---:|---:|
| SORT | 7142 | **1554** | 8283 |
| OC-SORT | 4997 | **870** | 2188 |
| ByteTrack | 4797 | **2823** | 6745 |
| BoT-SORT | 7151 | **4057** | 9265 |
| BoT-SORT+ReID | 5856 | **2451** | 5120 |
| TrackTrack | 2703 | **580** | 749 |

## Per-class F1 — FasterRCNN backbone (IoU 0.5)

| Tracker | human | elephant | giraffe | lion | unknown |
|---|---:|---:|---:|---:|---:|
| SORT | 0.201 | 0.431 | 0.064 | 0.000 | 0.012 |
| OC-SORT | 0.183 | 0.611 | 0.051 | 0.000 | 0.002 |
| ByteTrack | 0.122 | 0.671 | 0.046 | 0.000 | 0.001 |
| BoT-SORT | 0.174 | 0.613 | 0.053 | 0.000 | 0.002 |
| BoT-SORT+ReID | 0.206 | 0.597 | 0.052 | 0.000 | 0.002 |
| TrackTrack | 0.229 | 0.648 | 0.047 | 0.000 | 0.001 |
