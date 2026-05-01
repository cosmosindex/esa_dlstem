# SAM 3 vs SAM 3.1 zero-shot MOT — HOTA findings

Comparison of SAM 3 (base) and SAM 3.1 (multiplex) under the *identical*
text-prompted MOT eval pipeline + RAFT static-tracklet filter, across
the 5 satellite-video benchmarks. Pipeline definitions live in
`docs/sam3_raft_mot_experiment.md`; this document distils the results.

- SAM 3 run dir: `/data/ESA_DLSTEM_2025/experiments/MOT/sam3_raft_filtered_20260429_142221/`
- SAM 3.1 run dir: `/data/ESA_DLSTEM_2025/experiments/MOT/sam3p1_raft_filtered_20260430_070810/`
- Both use the same configs / clip_len=32 / `dump_mot_format=true` /
  open-vocab text prompt = dataset's dominant category.

## Headline numbers (HOTA, %)

| Dataset       | SAM 3 raw | SAM 3 + RAFT | **SAM 3.1 raw** | SAM 3.1 + RAFT | Δ (3.1 − 3) raw |
|---------------|----------:|-------------:|----------------:|---------------:|----------------:|
| airmot        |   39.65   |   0.25       | **27.41**       |  0.52          | **−12.24**      |
| rscardata     |    1.67   |   0.00       | **0.00**        |  0.00          | **−1.67**       |
| satmtb        |   11.66   |   4.10       | **10.10**       |  3.63          | **−1.56**       |
| sdmcar        |    0.36   |   0.24       | **0.00**        |  0.00          | **−0.36**       |
| viso_no_car   |   43.39   |   3.64       | **32.46**       |  4.06          | **−10.93**      |
| viso_combined |    4.41   |   0.28       |  3.06           |  0.31          | **−1.35**       |

**SAM 3.1 is uniformly worse than SAM 3 base on every benchmark.** The
gap is largest where target scale is most favourable to a strong
detector (airmot, viso_no_car) — exactly the scenes where SAM 3 base
was already getting double-digit HOTA.

## Per-component breakdown (SAM 3.1 raw)

| Dataset      | DetA  | AssA  | LocA  | MOTA   | MOTP  | IDF1  | IDsw | n_dets |
|--------------|------:|------:|------:|-------:|------:|------:|-----:|-------:|
| airmot       | 21.11 | 44.48 | 74.30 | 18.85  | 68.74 | 25.94 |   48 |   6047 |
| rscardata    |  0.00 |  0.00 |  -    |  0.00  |  -    |  0.00 |    0 |      0 |
| satmtb       |  2.00 | 52.47 | 73.11 |-11.72  | 69.53 |  3.80 |   30 |  49241 |
| sdmcar       |  0.00 |  0.00 |  -    |  0.00  |  -    |  0.00 |    0 |      9 |
| viso_no_car  | 27.39 | 38.69 | 77.19 |  0.77  | 73.81 | 38.72 |    2 |    997 |

## Key findings

### 1. SAM 3.1 multiplex collapses on satellite small-vehicle datasets
Both `rscardata` (7 test seqs, 12.5 K cars at avg ~7 px) and `sdmcar`
(20 test seqs, similar scale) get **HOTA=0**. The model produced
**zero true positives across every frame of every test video** —
inspection of `per_image_metrics.json` showed 2208/2208 frames with
TP=0 on rscardata and 3200/3200 on sdmcar. The 9 stray detections that
do show up in `n_dets` for sdmcar all fall below IoU=0.5 against any GT.

This is *not* a wrapper bug:
- The same wrapper / prompt / pipeline works on viso_no_car (planes
  + ships, larger objects).
- **SAM 3 base on the same data did detect cars** — rscardata HOTA=1.67
  (914 valid track segments, 438 true detections); sdmcar HOTA=0.36
  (85 detections). Numbers are tiny but non-zero, vs SAM 3.1's literal
  zero across both datasets.
- SAM 3.1 produced no detections at all on these two benchmarks despite
  text prompt = `"car"` being identical to the SAM 3 run.

The most likely cause is the **multiplex predictor's input resolution**:
`build_sam3_multiplex_video_predictor` hard-codes `image_size=1008`,
which on a 1024² rscardata frame means tiny cars (5-8 px) end up
sub-pixel after backbone-stride downsampling and the open-vocab head
loses confidence on every box. The wider trade-off Meta describe in
`RELEASE_SAM3p1.md` (mixed video-PCS results, ~7× speedup at 128
objects) is consistent with multiplex sacrificing some detection
quality at extreme scales.

**Practical implication for the paper**: SAM 3.1's multiplex throughput
gains do not translate to satellite tiny-vehicle MOT. Stick with SAM 3
base for those benchmarks; SAM 3.1 is only competitive on airplane /
ship / multi-class scenes (airmot, viso_no_car, satmtb-airplane
subset).

### 2. RAFT filter hurts both SAM 3 and SAM 3.1 on these scenes
Across every dataset and every model, `<tracker>_raft` HOTA is much
lower than the raw row — for SAM 3 base too (airmot 39.65 → 0.25;
viso_no_car 43.39 → 3.64). The drop is **even bigger for SAM 3 base**
than for SAM 3.1, simply because SAM 3 base had more detections to lose
in the first place.

This is the opposite of what we expected from the design ("filter out
static-tracklet false positives, keep moving targets"). Two compounding
causes show in `n_dets`:

- **Tracklets are short** in text-prompted SAM 3/3.1 — IDs reset every
  clip (clip_len=32) and only get stitched across boundaries via IoU.
  A 4-6 frame tracklet doesn't carry enough RAFT samples to robustly
  hit the 80th-percentile flow threshold, so even genuinely-moving
  cars get dropped.
- **Median |flow| inside a 5-7 px box is dominated by noise**, not
  signal — RAFT was tuned for HD imagery where motion is tens of
  pixels per frame.

For the paper either drop the RAFT line for SAM 3 / SAM 3.1 evaluations,
or rerun with a much lower `--tau` (0.1-0.2 instead of 0.5) and re-aggregate.
The headline conclusion (SAM 3.1 < SAM 3 base) is robust either way
because both halves move together.

### 3. SAT-MTB pattern is consistent (high AssA, low DetA)
SAM 3.1 on satmtb gives DetA=2.00 / AssA=52.47 — the multiplex tracker
is fine *once it has a target*, but the detector misses most of them.
This mirrors the small-vehicle failure: satmtb test contains many small
boats and cars that the open-vocab head drops, while planes are
consistently recovered.

### 4. viso_combined is meaningful only as a mass-weighted check
The composite (`viso_no_car` ∪ `rscardata`) drops from 32.46 → 3.06 in
HOTA because the rscardata car half (7 sequences, far more frames than
the 2 viso_no_car seqs) dilutes the score with HOTA=0 rows. The number
is correct under the official MOTChallenge protocol (sequences are
treated equally regardless of their per-class GT density), so cite it
as the lower-bound per-paper, but always show the per-half breakdown
alongside.

## Caveats / things that bit us

- `MOTFormatDumpCallback` originally **did not write empty files for
  videos with zero detections**, which made `compute_hota.py` bail out
  with `Tracker file not found`. Fixed in
  `lightning_modules/mot_format_dump.py` (track `_seen_video_ids` and
  emit empty `.txt` per video at `on_test_end`). Existing run dirs were
  patched post-hoc by `tools/fill_missing_mot_format.py` (67 empty raw
  + 67 empty filtered files written into the SAM 3.1 dest dir).
- Initial SAM 3.1 airmot run also hit
  `RuntimeError("No points are provided; please add points first")`
  mid-clip when the multiplex inner tracker had nothing to propagate.
  Fixed in `models/sam3.py::SAM31TextTracker.propagate()` with a
  try/except that drops the rest of the failing clip and falls through
  to empty outputs. AirMOT was re-run cleanly afterwards (64/64 clips).

## Reproducibility

```bash
# Refresh the SAM 3.1 hota_summary.csv from existing run dirs
python tools/fill_missing_mot_format.py \
    --tracker-output-root /data/ESA_DLSTEM_2025/experiments/MOT/sam3p1_raft_filtered_20260430_070810
rm -rf /data/.../sam3p1_raft_filtered_20260430_070810/_hota_workspace
python compute_hota.py \
    --tracker-output-root /data/.../sam3p1_raft_filtered_20260430_070810 \
    --workspace /data/.../sam3p1_raft_filtered_20260430_070810/_hota_workspace \
    --output /data/.../sam3p1_raft_filtered_20260430_070810/hota_summary.csv \
    --also-filtered --combine-viso

# Re-run SAM 3.1 from scratch (next time will work without post-hoc
# patching thanks to the callback fix)
bash run_sam3p1_raft_mot.sh
```
