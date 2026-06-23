# BIRDSAI — SAM3 cross-image (train-set) exemplar MOT, the FAIR few-shot row

**2026-06-22.** Wild-animal thermal MOT. Follow-up to
[`birdsai_sam3_oracle_results.md`](birdsai_sam3_oracle_results.md), which proved
that detection (not tracking) is the bottleneck but only via **oracle** rows that
seed SAM3 from *test* GT. Question here: can we recover the same tiny thermal
species **fairly** — seeding SAM3 from *training* exemplars only, never touching
test GT?

This also overturns the oracle doc's caveat *"SAM3 has no cross-image exemplar API
… train-set exemplars are impossible"*: that was true only of the **video**
predictor's box API. The SAM3 **image** grounding head accepts cross-image
exemplars (its geometry encoder RoI-pools the exemplar's appearance from whatever
frame you give it, then grounds on a different frame), so a train exemplar can
drive detection on a test frame.

All numbers are on the **16 BIRDSAI test videos**, **fine 5-class, IoU 0.5,
class-aware greedy matching** — same scorer as `eval_birdsai_detect_track.py`.

## Method (`evaluation/eval_birdsai_sam3_xexemplar.py`)

Fair = no test GT ever reaches the model. Three user-confirmed design choices:

- **Prior = fine / known-species.** 15/16 BIRDSAI test videos are single-species
  (each animal video is purely one of lion/elephant/giraffe/unknown); only one is
  mixed. We read the species present in each video (the granted prior) and run only
  those species' exemplar banks → clean fine labels, isolates the recall gain.
- **Exemplar selection = transductive.** Per (video, class): gather 24 size-diverse
  TRAIN candidate boxes, score each by the detection confidence it produces on the
  **test video's own frame 0** (image only, no GT labels), keep the top **K=4**,
  pool with **M=2** train-background negatives into one prototype. Peeks at the
  unlabelled test frame 0 → *transductive*, still never uses test GT.
- **Tracking = SAM3 SOT propagation + gated periodic re-detect.** Frame-0 exemplar
  detections seed `SAM3Tracker` (SAM2-style mask memory), which propagates them.
  Every 8 frames the detector re-runs; a detection that matches no active track
  (generous center-distance gate) AND persists to the next keyframe is promoted to
  a new track. This catches objects that enter after frame 0 without re-spawning
  drifted existing tracks.

GT = `annotations_sam3` (SAM3 box-refined tight boxes) for **both** exemplars and
eval GT, per the project decision to use the refined labels.

## Results — F1 (per fine class)

| class (size) | P | R | **F1** | MOTA | nGT |
|---|---|---|---|---|---|
| human 35 px (fast) | 0.080 | 0.314 | 0.128 | −3.31 | 21 853 |
| elephant (big) | 0.463 | 0.724 | **0.565** | −0.16 | 48 055 |
| giraffe 18 px | 0.419 | 0.546 | **0.474** | −0.23 | 3 231 |
| lion 14 px | 0.099 | 0.792 | **0.176** | −6.49 | 351 |
| unknown ~17 px | 0.031 | 0.393 | 0.057 | −12.01 | 5 271 |
| **OVERALL** | 0.195 | 0.581 | **0.292** | **−1.86** | |

Run: 16/16 videos, ~144 min, GPU0, `SEED_CAP=24`, gated re-detect r=8. Output
`/work/ziwen/experiments/sam3_birdsai_xexemplar_full_20260622_142509`.

## Where it sits vs the detectors and the oracle rows

| class | best of 3 detectors | **xexemplar (FAIR)** | Exp2 (exemplar oracle) | Exp1 (GT-init oracle) |
|---|---|---|---|---|
| giraffe | ~0.03 | **0.474** | 0.616 | 0.894 |
| lion | 0.000 | **0.176** | 0.553 | 0.783 |
| elephant | 0.60 | 0.565 | 0.587 | 0.700 |
| human | 0.08 | 0.128 | 0.414 | 0.640 |
| unknown | 0.01 | 0.057 | 0.216 | 0.540 |

⚠️ **Not yet a clean table:** the detector/oracle columns were scored on the
**original** annotations, this row on the tighter **annotations_sam3** (IoU 0.5 is
harder against tight boxes), so this row is conservative relative to them. A fair
table requires re-scoring all methods on `annotations_sam3` from their cached
`predictions.json` (the reusable scorer is `score_video()` in this script).

## Conclusions

1. **The probe's promise holds at full video-MOT scale, fairly.** With one
   training exemplar (no test GT), giraffe goes **≈0 → 0.474** and lion
   **0.000 → 0.176** — species all three trained detectors miss entirely. The
   ordering is detector < **fair xexemplar** < Exp2 oracle < Exp1 oracle: fair
   train-exemplars sit between deployable detectors and the test-GT ceilings, and
   the gap to Exp2 is the price of fairness.
2. **Recall is the story; precision is the intrinsic limit.** OVERALL recall 0.581
   but precision 0.195 → MOTA negative. The exemplar prototype fires on thermal
   hot-spots that look like the target on a frozen RGB-trained model; this FP flood
   is not fixable by score thresholding (true detections live at score 0.2–0.4;
   raising the threshold removes TPs, not FPs).
3. **`unknown` is the weak point** (F1 0.057, MOTA −12): the "unknown animal"
   prototype is the least specific and floods the most. `human` stays ≈0 — a
   *tracking* ceiling (fast small objects drift SAM2 memory), not detection: even
   GT-init oracle only reached 0.640.

## Engineering notes

- **Seed-threshold trap** (`evaluation/_xexemplar_thr_probe.py`): per-frame
  detections that are correct sit at score 0.2–0.4 and are outscored by FP
  hot-spots, so there is no clean high-precision seed set.
- **Gated re-detect** fixed a P=0.001 catastrophe in the naive version: the flood
  came from *re-spawning* existing tracks whose mask drifted (they failed to match
  their own re-detection). A generous center-distance match + a persistence check
  removed the duplication while still catching genuinely new objects.
- **OOM fix:** `SAM3Tracker.propagate`'s memory-attention VRAM scales with
  (#objects × #memory frames); the human FP flood reached ~88 objects/clip → 28 GB
  → OOM. `SEED_CAP=24` / `PROMOTE_CAP=8` bound it (max real density ~7/frame, so no
  recall cost — and trimming FP lifts precision), plus `expandable_segments:True`.
- **Crash-safe / resumable:** each video writes `pred_<id>.json` before scoring;
  relaunch with `--resume-dir <exp>` reuses completed videos. Scoring is decoupled
  (`score_video()`), so partial runs still produce a table.

## Caveats

- **Transductive**, not inductive: exemplar *selection* uses the unlabelled test
  frame 0. No test GT is ever used, but it is not a pure train-only detector.
- GT = `annotations_sam3`; comparison to the detector/oracle rows is only
  suggestive until those are re-scored on the same GT (open NEXT step).
- Per-video variance is large (human: video 011 R 0.002 vs 355 R 0.70; elephant:
  058/349 R ~0.14 vs 352/353 R 0.69/0.95) — driven by whether the frame-0
  transductive template grounds on the actual animals or on background.
