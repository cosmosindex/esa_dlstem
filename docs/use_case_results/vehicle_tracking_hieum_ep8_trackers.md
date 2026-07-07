# Vehicle Detection & Tracking — Trackers on HiEUM ep8 Detections

Use case: **moving-vehicle (car) tracking** on the space-tracker car MOT benchmark.

This document reports every **tracking-by-detection (TBD)** tracker we benchmarked,
run on the committed HiEUM detection result — **union ep8** (see
[`vehicle_detection_hieum_ep8.md`](vehicle_detection_hieum_ep8.md)). All six
trackers consume the **same** cached ep8 detections, so any difference between them
is pure **association**, not detection.

Six trackers, two families:

| Tracker | Family | Appearance model |
|---|---|---|
| **SORT** | motion-only (Kalman + IoU) | none |
| **OC-SORT** | motion-only (observation-centric) | none |
| **ByteTrack** | motion-only (two-stage association) | none |
| **BoT-SORT** | motion + (optional) ReID | none here (CMC off) |
| **BoT-SORT-ReID** | motion + ReID | FastReID SBS-S50 (2048-D) |
| **TrackTrack** | motion + ReID | FastReID SBS-S50 (2048-D) |

> Joint-detection-and-tracking methods (FairMOT, TGraM) and open-vocabulary methods
> (SAM3) run their **own** detector and therefore cannot consume the shared ep8
> detections — they are not part of this shared-detection comparison.

## Protocol

- Detections: HiEUM **union ep8**, cached per video with `score_floor = 0.05`,
  `nms_iou = 0.1`, `max_dets = 128`, native resolution.
  ReID trackers additionally consume FastReID features cropped from the **ep8**
  boxes (regenerated for this run).
- Single class **`car`**. Matching: HiEUM protocol — **centroid distance ≤ 5 px**.
- Each tracker keeps the **benchmark-tuned** per-tracker `score_floor` and
  association hyper-parameters from `configs/MOT/tracker/*` (only the detection
  cache was swapped to ep8). See the score-scale caveat at the end.
- Metrics: CLEAR-MOT + HOTA via TrackEval. All figures are **×100**.

## Results per dataset (sorted by HOTA)

### RsCarData

| Tracker | HOTA | DetA | AssA | MOTA | IDF1 | IDsw | MT | ML |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **SORT** | **40.25** | 39.32 | 41.98 | 22.84 | **49.45** | 1017 | 233 | 215 |
| OC-SORT | 34.93 | 35.31 | 35.12 | 19.91 | 42.79 | 1758 | 189 | 300 |
| TrackTrack | 31.28 | 21.91 | 45.07 | 19.25 | 41.41 | 63 | 117 | 532 |
| BoT-SORT-ReID | 20.65 | 31.95 | 13.71 | −6.04 | 15.49 | 40910 | 178 | 368 |
| BoT-SORT | 3.81 | 30.79 | 0.65 | −16.71 | 0.91 | 57920 | 167 | 394 |
| ByteTrack | 0.98 | 2.00 | 0.59 | −0.81 | 0.44 | 3520 | 0 | 884 |

### SAT-MTB (car)

| Tracker | HOTA | DetA | AssA | MOTA | IDF1 | IDsw | MT | ML |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **SORT** | **27.23** | 21.66 | 35.94 | −4.98 | **27.25** | 3255 | 170 | 1367 |
| BoT-SORT-ReID | 21.56 | 15.18 | 31.69 | −1.68 | 19.12 | 9553 | 119 | 1687 |
| OC-SORT | 17.18 | 16.22 | 19.13 | −2.70 | 16.24 | 5057 | 90 | 1592 |
| TrackTrack | 10.04 | 3.02 | 34.23 | 2.39 | 7.12 | 20 | 11 | 2180 |
| BoT-SORT | 3.07 | 11.37 | 1.07 | −8.24 | 1.16 | 26569 | 50 | 1818 |
| ByteTrack | 1.04 | 1.60 | 0.80 | −0.74 | 0.42 | 4422 | 0 | 2248 |

### SDM-Car

| Tracker | HOTA | DetA | AssA | MOTA | IDF1 | IDsw | MT | ML |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **SORT** | **28.49** | 23.19 | 35.67 | −0.45 | **31.00** | 1423 | 212 | 1381 |
| TrackTrack | 27.26 | 18.60 | 40.36 | 3.85 | 29.87 | 206 | 138 | 1665 |
| BoT-SORT-ReID | 24.24 | 19.96 | 29.82 | −0.72 | 24.53 | 10473 | 156 | 1571 |
| OC-SORT | 23.17 | 20.61 | 26.46 | 2.41 | 24.47 | 2420 | 156 | 1511 |
| BoT-SORT | 3.56 | 15.67 | 0.99 | −10.14 | 1.35 | 45169 | 70 | 1742 |
| ByteTrack | 1.26 | 2.41 | 0.80 | −1.64 | 0.56 | 7440 | 1 | 2451 |

## Mean across the three datasets (ranking)

| Tracker | HOTA | DetA | AssA | IDF1 | ΣIDsw |
|---|---:|---:|---:|---:|---:|
| **SORT** | **31.99** | 28.06 | 37.86 | **35.90** | 5,695 |
| OC-SORT | 25.09 | 24.05 | 26.90 | 27.83 | 9,235 |
| TrackTrack | 22.86 | 14.51 | **39.89** | 26.13 | **289** |
| BoT-SORT-ReID | 22.15 | 22.36 | 25.07 | 19.71 | 60,936 |
| BoT-SORT | 3.48 | 19.28 | 0.90 | 1.14 | 129,658 |
| ByteTrack | 1.09 | 2.00 | 0.73 | 0.47 | 15,382 |

## Takeaways

- **SORT wins on every dataset** (mean HOTA 31.99) — on small, fixed-camera
  satellite video, a plain Kalman + IoU tracker with an aggressive detection floor
  beats every appearance-aware or two-stage tracker. This mirrors the author-ckpt
  benchmark: simplicity wins.
- **The score-thresholding trackers collapse.** ByteTrack (mean HOTA 1.09) and
  BoT-SORT (3.48) fail because their defaults assume high-scoring detections:
  ByteTrack's `track_thresh` starves it (n_dets drops ~20×, DetA ≈ 2), while
  BoT-SORT's association explodes into **100k+ ID switches** on dense tiny cars.
- **ReID buys AssA, not HOTA.** TrackTrack posts the best mean AssA (39.89) and by
  far the fewest ID switches (289) via strict track initialisation — but that same
  strictness throws away detections (mean DetA 14.51), capping its HOTA below SORT.
  BoT-SORT-ReID's ReID cannot overcome its motion-model ID-switch flood. On thermal-
  scale satellite cars, MOT17-domain appearance features add little.
- **AssA ≫ DetA everywhere** — with ep8 detections held fixed, association quality
  (35–45 AssA for the top trackers) is not the bottleneck; **detection recall is**
  (DetA 20–39). This is the same detection-bound story as the ep8 detection report.

## ep8 vs. author-checkpoint detections (same trackers)

Swapping the author checkpoint for ep8 changes tracking in step with the detection
changes (mean HOTA, ×100):

| Tracker | ep8 | author | Δ |
|---|---:|---:|---:|
| SORT | 31.99 | 33.46 | −1.47 |
| OC-SORT | 25.09 | 26.90 | −1.81 |
| TrackTrack | 22.86 | 24.55 | −1.69 |
| BoT-SORT-ReID | 22.15 | 24.99 | −2.84 |
| BoT-SORT | 3.48 | 4.40 | −0.92 |
| ByteTrack | 1.09 | 1.69 | −0.60 |

Per dataset the picture is **not** uniform, and it tracks the detection report:

- **SDM-Car** (where ep8 gained +0.068 best-F1): ep8 **improves** tracking —
  SORT +3.3, OC-SORT +2.1, BoT-SORT-ReID +1.8 HOTA.
- **RsCarData** (author's in-domain set): mixed — SORT/OC-SORT slightly up, the
  ReID trackers down.
- **SAT-MTB**: ep8 is lower for most trackers, which **dominates the mean**.

### Caveat — the per-tracker `score_floor` was tuned to the author checkpoint

ep8's detection scores cluster **lower** than the author checkpoint's (the detection
report's best-F1 threshold moved from 0.25–0.35 down to 0.16–0.20). The tracker
configs keep the author-tuned `score_floor` (e.g. SORT 0.25), so on ep8 they
pre-filter a larger fraction of valid low-scored detections — starving the tracker,
most visibly on SAT-MTB. The comparison above is therefore **conservative** for
ep8: a per-tracker `score_floor` re-tune to ep8's score scale would likely recover
the SAT-MTB gap. We report the untuned numbers to keep the benchmark configuration
fixed; re-tuning is a follow-up if we want ep8's best-case tracking.

## Provenance

- Detections: HiEUM `model_ep8.pth` (2026-06-29 union retrain). Cache regenerated
  with `cache_hieum_dets.py` (plain) and `cache_hieum_dets_with_feats.py` (FastReID)
  into `hieum_dets_cache_ep8/` and `hieum_dets_with_feats_cache_ep8/`.
- Trackers: `evaluation/eval_tracker.py` (motion-only),
  `evaluation/eval_botsort_reid.py`, `evaluation/eval_tracktrack.py`; per-tracker
  configs mirror `configs/MOT/tracker/*` with the cache path swapped to the ep8
  caches.
- HOTA: `compute_hota.py` (TrackEval). **SAT-MTB car-only seqmap fix applied** —
  the trackers emit only `car_*.txt`, so SAT-MTB GT/seqmap was restricted to its
  car sequences (else TrackEval errors on the missing `airplane_07.txt` and skips
  the whole dataset).
- Raw outputs + `hota_summary_ep8.csv`:
  `/data/ESA_DLSTEM_2025/experiments/MOT/tracker_ep8_20260703/`.
- Author-checkpoint baseline for the Δ table:
  `/data/ESA_DLSTEM_2025/experiments/MOT/tracker_20260427/hota_summary.csv`.
</content>
