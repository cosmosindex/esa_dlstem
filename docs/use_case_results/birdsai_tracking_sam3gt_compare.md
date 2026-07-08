# BIRDSAI Tracking Comparison on SAM3-refined GT (`annotations_sam3`)

6 TBD trackers × 3 detectors (cached detections → online tracking), re-scored on the **same GT as the detection table**.
Videos: 16/16 (sweep subset, all 18 runs identical set) · 15494 frames · fine 5-class.

Two scoring passes on the *same* tracks + *same* GT:
- **Full HOTA suite** via **TrackEval** — the identical pipeline used for the NeurIPS Space-Tracker-MOT (car) table (`compute_hota.py`), so these numbers are directly comparable to that benchmark. HOTA is α-averaged; CLEAR/Identity at IoU 0.5. Matching is class-restricted (see Provenance).
- **Detection-level P/R/F1** from the per-class greedy matcher (`_birdsai_tracking_compare.py`, IoU 0.5) — kept because the long-tail-species story is a detection-recall story.

## Full metric suite — TrackEval (rows = tracker)

### FasterRCNN detections

| Tracker | HOTA | DetA | AssA | MOTA | IDF1 | IDsw | MT | ML |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SORT | 0.149 | 0.184 | 0.123 | -0.696 | 0.118 | 3449 | **15** | **66** |
| OC-SORT | 0.154 | 0.230 | 0.105 | -0.143 | 0.139 | 2894 | 14 | 69 |
| ByteTrack | 0.179 | 0.250 | 0.131 | +0.058 | 0.163 | 3463 | 14 | 71 |
| BoT-SORT | 0.160 | 0.230 | 0.114 | -0.228 | 0.131 | 5798 | 14 | 69 |
| BoT-SORT+ReID | 0.186 | 0.229 | 0.153 | -0.231 | 0.159 | 4647 | 14 | 67 |
| TrackTrack | **0.218** | **0.252** | **0.191** | **+0.072** | **0.220** | **1248** | 14 | 69 |

### YOLO11l detections

| Tracker | HOTA | DetA | AssA | MOTA | IDF1 | IDsw | MT | ML |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SORT | 0.158 | **0.210** | 0.119 | +0.122 | 0.161 | 1280 | **3** | **70** |
| OC-SORT | 0.147 | 0.182 | 0.119 | +0.168 | 0.161 | 689 | 0 | 73 |
| ByteTrack | 0.087 | 0.151 | 0.050 | +0.127 | 0.083 | 2233 | 0 | 77 |
| BoT-SORT | 0.086 | 0.178 | 0.042 | +0.117 | 0.076 | 3882 | 0 | 74 |
| BoT-SORT+ReID | 0.134 | 0.193 | 0.093 | +0.151 | 0.132 | 2345 | 1 | 73 |
| TrackTrack | **0.175** | 0.187 | **0.163** | **+0.187** | **0.200** | **409** | **3** | 74 |

### DINOv3 detections

| Tracker | HOTA | DetA | AssA | MOTA | IDF1 | IDsw | MT | ML |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SORT | 0.114 | 0.195 | 0.066 | -0.277 | 0.077 | 2965 | **7** | **75** |
| OC-SORT | 0.144 | 0.211 | 0.099 | +0.092 | 0.113 | 1207 | 1 | 77 |
| ByteTrack | 0.083 | 0.169 | 0.041 | +0.056 | 0.063 | 3077 | 0 | 79 |
| BoT-SORT | 0.083 | 0.205 | 0.034 | +0.027 | 0.059 | 5841 | 1 | 77 |
| BoT-SORT+ReID | 0.163 | **0.225** | 0.118 | +0.064 | 0.133 | 3188 | 1 | 77 |
| TrackTrack | **0.178** | 0.218 | **0.146** | **+0.109** | **0.160** | **563** | 1 | 77 |

## Mean across the three detectors (ranking)

Macro mean over the 3 detector backbones (rate metrics averaged; IDsw/MT/ML summed) — a single ranking of the trackers.

| Tracker | HOTA | DetA | AssA | MOTA | IDF1 | ΣIDsw | ΣMT | ΣML |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TrackTrack | **0.190** | **0.219** | **0.167** | **+0.123** | **0.193** | **2,220** | 18 | **211** |
| BoT-SORT+ReID | 0.161 | 0.216 | 0.121 | -0.005 | 0.141 | 10,180 | 16 | 217 |
| OC-SORT | 0.149 | 0.208 | 0.108 | +0.039 | 0.137 | 4,790 | 15 | 219 |
| SORT | 0.140 | 0.196 | 0.103 | -0.283 | 0.119 | 7,694 | **25** | **211** |
| ByteTrack | 0.117 | 0.190 | 0.074 | +0.081 | 0.103 | 8,773 | 14 | 227 |
| BoT-SORT | 0.110 | 0.204 | 0.064 | -0.028 | 0.089 | 15,521 | 15 | 220 |

## Detection-level F1 — greedy per-class matcher (rows = tracker)

Detection precision/recall/F1 (identity-agnostic); complements the HOTA DetA above. Bold = best detector per tracker.

| Tracker | FasterRCNN | YOLO11l | DINOv3 |
|---|---:|---:|---:|
| SORT | 0.379 | **0.437** | 0.363 |
| OC-SORT | **0.454** | 0.392 | 0.382 |
| ByteTrack | **0.482** | 0.334 | 0.317 |
| BoT-SORT | **0.447** | 0.381 | 0.372 |
| BoT-SORT+ReID | **0.446** | 0.407 | 0.397 |
| TrackTrack | **0.483** | 0.398 | 0.385 |

## Per-class breakdown — FasterRCNN backbone

Association-aware **HOTA** (TrackEval) beside detection-level **F1** (greedy), per fine species. This is the headline: every rare/tiny species collapses to ≈0 on *both* metrics — the trackers only hold the large, common **elephant**.

| Tracker | human H/F1 | elephant H/F1 | giraffe H/F1 | lion H/F1 | unknown H/F1 |
|---|---:|---:|---:|---:|---:|
| SORT | 0.043/0.203 | 0.211/0.596 | 0.026/0.016 | 0.003/0.003 | 0.018/0.037 |
| OC-SORT | 0.047/0.176 | 0.205/0.662 | 0.040/0.020 | 0.000/0.000 | 0.016/0.018 |
| ByteTrack | 0.019/0.144 | 0.236/0.683 | 0.012/0.017 | 0.000/0.000 | 0.010/0.017 |
| BoT-SORT | 0.019/0.176 | 0.219/0.661 | 0.013/0.021 | 0.000/0.000 | 0.010/0.018 |
| BoT-SORT+ReID | 0.026/0.181 | 0.254/0.659 | 0.014/0.021 | 0.000/0.000 | 0.011/0.018 |
| TrackTrack | 0.055/0.177 | 0.287/0.683 | 0.043/0.021 | 0.000/0.000 | 0.013/0.017 |

## Takeaways

- **HOTA re-ranks the trackers vs. F1.** By detection-level F1 the score-thresholding trackers look fine, but under HOTA **TrackTrack wins on every detector** (mean HOTA 0.190) — it pairs the best AssA with by far the fewest ID switches via strict track initialisation. Same lesson as the Space-Tracker-MOT car table: once association is scored properly, strict-init ReID tracking leads.
- **ByteTrack and BoT-SORT collapse on association.** Their AssA falls to 0.03–0.13 and IDsw floods (BoT-SORT 3.9k–5.8k), exactly the score-threshold / ID-switch failure seen on satellite cars — their defaults assume high-scoring detections that thermal aerial video does not provide.
- **Detection is the ceiling.** HOTA stays ≤ 0.22 for the best tracker because DetA is 0.15–0.25; association can only recover so much when the detector misses most objects.
- **Long-tail collapse is the story, and HOTA confirms it.** Per class, only **elephant** clears HOTA 0.20–0.29; giraffe/lion/unknown sit at ≈0 on both HOTA and F1. The trackers inherit the detectors' inability to find rare, tiny thermal species — see `birdsai_tracking_vs_sam3xexemplar.md` for the SAM3 train-exemplar route that recovers them.

## Provenance

- Tracks: cached `mot_format/*.txt` of the 18-run sweep (`evaluation/eval_birdsai_track_sweep.py`), `/data/ESA_DLSTEM_2025/experiments/MOT_birdsai_sweep/`. Detectors never re-run.
- GT: `annotations_sam3` (SAM3-refined boxes), fine 5-class, same as the detection table.
- **Full suite**: `evaluation/compute_birdsai_hota.py` → TrackEval (HOTA + CLEAR + Identity). The sweep runs one tracker per class and bakes the class into the track id (`class = id // 1_000_000`); TrackEval's MotChallenge box dataset is single-foreground-class, so we translate every class-`c` box by `c·100000` px on x in **both** GT and tracker files. Translation preserves within-class IoU exactly while zeroing cross-class IoU, making matching class-restricted (as in the greedy eval) in a single pooled pass. Frames are re-indexed to 1 (BIRDSAI frame ids start ≫ 1). TrackEval MOTA reproduces the greedy MOTA within ~0.01; IDsw differs because TrackEval uses the standard HOTA/CLEAR switch definition, not the greedy last-mapping counter — prefer the TrackEval column.
- **Detection-level F1**: `evaluation/_birdsai_tracking_compare.py` (per-class greedy IoU 0.5), cached in `birdsai_tracking_sam3gt_compare.json`.
- Full HOTA suite JSON (overall + per-class): `birdsai_tracking_sam3gt_hota.json`.
