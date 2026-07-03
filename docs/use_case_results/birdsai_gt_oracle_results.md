# BIRDSAI GT-oracle Tracker Comparison — pure association vs object size

> **Control experiment** (mirrors the MOT Exp2 oracle). Every frame's detections
> are the **ground-truth boxes** (`annotations_sam3`, score = 1) fed to all 6 TBD
> trackers — the detector is removed (DetA ≈ 1), so what's measured is **pure
> association**. **NOT a headline MOT number**: GT boxes never reach the model in
> the real benchmark; this isolates "is the small-object failure detection or
> tracking?".
> 16 test videos · 78,761 GT boxes · IoU 0.5 · fine 5-class.
> Pipeline: `evaluation/_birdsai_build_gt_predictions.py` →
> `scripts/run_birdsai_gt_oracle.sh` (`eval_birdsai_track_sweep.py`) →
> `evaluation/_birdsai_gt_oracle_hota_by_size.py` (real TrackEval HOTA/AssA per
> size bin, whole tracks binned by median √area — same machinery as MOT Exp2);
> figure `tools/plot_birdsai_gt_oracle.py`.

## Headline: detection is the bottleneck

Feeding GT boxes lifts every tracker from F1 ≈ 0.3–0.5 (real detectors) to
**F1 ≈ 0.91–0.96**, with DetA ≈ 0.82–0.98 and tracked Recall ≈ 1.0 at *every*
size. The small-object collapse in the detector tables (giraffe / lion / unknown
≈ 0) **disappears** once detection is perfect → the wild-animal small-object
failure is a **detection** problem, not a tracking one.

| Tracker | F1 (GT-fed) | IDF1 | IDsw | DetA |
| --- | ---: | ---: | ---: | ---: |
| SORT | **0.964** | 56.7 | 1,780 | 0.862 |
| OC-SORT | 0.944 | 61.1 | 1,364 | **0.975** |
| ByteTrack | 0.913 | 30.0 | 5,680 | 0.816 |
| BoT-SORT | 0.957 | 29.1 | 7,266 | 0.934 |
| BoT-SORT+ReID | 0.957 | 30.5 | 7,237 | 0.933 |
| TrackTrack | 0.924 | 59.2 | **1,098** | 0.943 |

*(Recall is a detection-layer metric — here it only confirms detection is no
longer the loss; it does NOT rank association and is excluded from the
comparison below.)*

## Pure association: AssA vs object size

With detection oracled out, **AssA** (TrackEval association accuracy) is the
clean metric. It still drops on small objects — the genuine tracking finding —
and tracker choice matters most exactly where it is hardest. See
`figures/birdsai_gt_oracle_size.png`.

**AssA (%) by object size** (overall AssA in the last column):

| Tracker | <14 | 14–20 | 20–28 | 28–38 | 38–50 | ≥50 | **all** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SORT | 35.8 | 19.7 | 45.7 | 47.1 | 69.5 | 29.0 | 42.3 |
| OC-SORT | 49.4 | 30.4 | 53.5 | **65.7** | **72.1** | **31.2** | **46.9** |
| ByteTrack | 6.0 | 2.3 | 4.8 | 18.6 | 52.7 | 19.0 | 21.0 |
| BoT-SORT | 5.8 | 2.4 | 4.6 | 19.5 | 53.1 | 18.8 | 20.4 |
| BoT-SORT+ReID | 6.0 | 3.2 | 5.6 | 19.5 | 53.8 | 19.2 | 21.0 |
| TrackTrack | **52.3** | 27.0 | **54.9** | 52.9 | 68.8 | 30.3 | 46.5 |

## Talking points

1. **Detection, not tracking, is the wild-animal bottleneck.** GT-fed Recall ≈ 1.0
   at every size and F1 ≈ 0.96; the tiny-species collapse is entirely the
   detector's.
2. **Association still has a real small-object weakness.** Even with perfect
   boxes, AssA falls toward the small bins (e.g. OC-SORT 72 → 49 → 30 from
   38–50 px down to 14–20 px) — tiny dense thermal blobs are genuinely ambiguous
   to associate. AssA is also **non-monotonic**: it peaks at 38–50 px and dips
   again for the largest (>50 px) elephants (long trajectories + crossings).
3. **Two clear tiers, and they split by association design.** Motion-based
   **OC-SORT (46.9) / TrackTrack (46.5) / SORT (42.3)** keep identity across all
   sizes; the **ByteTrack / BoT-SORT family (≈ 20–21)** collapses, especially on
   small objects (AssA ≈ 5–6 on <28 px). On the smallest bin, OC-SORT (49) and
   TrackTrack (52) are ~9× the BoT-SORT family (≈ 6).
4. **ReID adds nothing.** BoT-SORT 20.4 ≈ BoT-SORT+ReID 21.0 — MOT17 appearance
   weights are useless on grayscale thermal animals; the edge of OC-SORT /
   TrackTrack is their *motion* association, not appearance.
