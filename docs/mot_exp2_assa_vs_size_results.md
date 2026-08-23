# Exp2 — Association ability vs. object size (GT-box oracle)

> Fair-comparison **Experiment 2**: feed the *same GT boxes* (score = 1) to every
> tracker's **unchanged association**, then score HOTA/AssA/IDF1/IDsw stratified
> by object size. Detection is held perfect (oracle), so this isolates **pure
> association ability** from detection — see
> [`mot_fair_comparison_framework.md`](mot_fair_comparison_framework.md).

- **Run root:** `/data/ESA_DLSTEM_2025/experiments/MOT/exp2_oracle_20260608`
- **Source table:** `assa_vs_size_le32.csv` (270 rows; `compute_hota_by_size.py
  --bins 0,5,8,12,20,32`). The original full-range table (`assa_vs_size.csv`,
  edges `0,5,8,12,20,40,inf`, 6 methods) is kept for reference.
- **Completed:** 2026-06-09 (motion TBD + JDT), extended 2026-08-06 with the two
  appearance-aware TBDs (`_EXP2_REID_COMPLETE`)
- **Datasets pooled (5):** rscardata, satmtb, sdmcar, airmot, viso_no_car
- **Methods (9) — every association method in the benchmark:**
  - **TBD** — SORT / ByteTrack / OC-SORT / BoT-SORT / **BoT-SORT+ReID** / **TrackTrack**
    (the last two consume a FastReID SBS-S50 embedding computed on the GT boxes,
    `evaluation/cache_gt_feats_mot.py`)
  - **JDT** — FairMOT & TGraM (4-class union `model_best`; only detection swapped
    for GT, association untouched)
  - **query-based** — MOTRv2 (GT boxes enter as its proposal port)
- **Size range: ≤32 px.** Above ~32 px every method saturates (AssA ≈ 0.99) and
  the bins hold only a few dozen tracks, so the figure stops at 32 px; bins are
  `<5 / 5–8 / 8–12 / 12–20 / 20–32`.
- **Pooling:** across all 5 datasets per size bin — AssA/IDF1/DetA weighted by
  `n_gt_tracks`, IDsw summed (it is a count); empty bins excluded.
- **Reported metrics: AssA, IDF1, IDsw — *not* HOTA/MOTA** (see "Why" below).
- **Figure:** `docs/figures/exp2_assa_vs_size.{pdf,png}` (AssA / IDF1 / IDsw vs
  size; TBD solid, JDT dashed, query-based dotted) — `tools/plot_assa_vs_size.py`.

> ⚠️ **MOTRv2 caveat.** Under the GT-box oracle MOTRv2 emits almost no tracks at
> all (DetA 0.005–0.06 below 20 px, vs 0.69–0.99 for every other method), so its
> AssA/IDF1 are computed over a near-empty output. This is the query-based
> paradigm genuinely failing on sub-20 px objects — its track queries never
> reach the output score threshold — not a scoring bug. Read its curve as
> "collapses", not as a calibrated association score.

> ⚠️ **TrackTrack oracle setting.** TrackTrack's `min_box_area` (default 100 px²)
> is an *output* filter that would delete every satellite car (≈25 px²) before
> association is scored, so it is set to 0 for the oracle. Every other
> hyperparameter is the wrapper default, matching the other TBD runs.

> ✅ **OC-SORT fixed (2026-06-09).** The first scorer pass returned `NaN` for
> OC-SORT because upstream's observation-centric recovery re-emits a track's
> last observation *on top of* its matched detection → duplicate `(frame, id)`
> in a timestep, which TrackEval rejects. Fixed at the wrapper layer
> (`models/trackers/ocsort.py::_dedup_by_id`, keep highest-score row per id),
> re-ran all 5 datasets (0 duplicates), and recomputed the table. OC-SORT
> numbers below are valid.

## Why AssA / IDF1 / IDsw — not HOTA or MOTA

The whole point of the oracle is to remove detection from the comparison, so the
reported metric must not fold detection back in. **HOTA and MOTA do:**

- **HOTA** $= \sqrt{\mathrm{DetA}\cdot\mathrm{AssA}}$ — half of it *is* detection.
- **MOTA** $= 1 - (\mathrm{FN}+\mathrm{FP}+\mathrm{IDsw})/\mathrm{GT}$ — dominated by
  FN/FP (detection side); ID switches are only a small term.

One might assume that feeding GT boxes makes `DetA ≈ 1`, so HOTA would reduce to
`√AssA` and be harmless. **It does not** — a tracker only *outputs* a box once
its track is confirmed, so DetA measures output completeness, not box quality:

| method | DetA (oracle, pooled) | min bin |
|---|---|---|
| SORT | 0.866 | 0.749 |
| ByteTrack | 0.881 | 0.780 |
| OC-SORT | 0.928 | 0.819 |
| BoT-SORT | 0.942 | 0.884 |
| **FairMOT** | **0.778** | 0.555 |
| **TGraM** | **0.778** | 0.560 |

DetA spans **0.78–0.94** even with perfect input boxes: TBD trackers withhold the
first `min_hits` frames of every track (and trim a `max_age` tail); JDT trackers
drop more still, because GT boxes must additionally survive their heatmap-peak /
confidence-threshold / track-confirmation pipeline. Comparing HOTA (or MOTA)
would therefore conflate *"how many boxes a tracker chose to emit during
warm-up/confirmation"* — a detection/output-side property — with association.

**AssA is scored only over the association of the boxes that *are* output**, so
it is independent of output completeness and isolates association cleanly. IDF1
is the next-purest joint metric and IDsw the raw switch count. All four columns
(HOTA/DetA/AssA/MOTA) remain in `assa_vs_size.csv` for reference.

## AssA by object size (higher = better)

| size (px) | #tracks | SORT | ByteTrack | OC-SORT | BoT-SORT | BoT-SORT+ReID | TrackTrack | FairMOT | TGraM | MOTRv2 |
|---|---|---|---|---|---|---|---|---|---|---|
| **<5** | 2133 | 0.813 | 0.869 | 0.887 | 0.836 | 0.836 | 0.924 | 0.589 | 0.589 | 0.012 |
| **5–8** | 2745 | 0.794 | 0.855 | 0.904 | 0.884 | 0.884 | 0.934 | 0.655 | 0.654 | 0.023 |
| 8–12 | 833 | 0.887 | 0.883 | 0.940 | 0.909 | 0.907 | 0.953 | 0.793 | 0.797 | 0.010 |
| 12–20 | 114 | 0.929 | 0.916 | 0.961 | 0.949 | 0.949 | 0.962 | 0.878 | 0.885 | 0.077 |
| 20–32 | 36 | 0.982 | 0.981 | 0.988 | 0.987 | 0.987 | 0.991 | 0.985 | 0.983 | 0.390 |

## IDF1 by object size (higher = better)

| size (px) | #tracks | SORT | ByteTrack | OC-SORT | BoT-SORT | BoT-SORT+ReID | TrackTrack | FairMOT | TGraM | MOTRv2 |
|---|---|---|---|---|---|---|---|---|---|---|
| **<5** | 2133 | 0.826 | 0.920 | 0.859 | 0.897 | 0.897 | 0.855 | 0.570 | 0.570 | 0.005 |
| **5–8** | 2745 | 0.884 | 0.952 | 0.926 | 0.951 | 0.950 | 0.804 | 0.731 | 0.730 | 0.005 |
| 8–12 | 833 | 0.944 | 0.963 | 0.949 | 0.958 | 0.957 | 0.943 | 0.856 | 0.860 | 0.004 |
| 12–20 | 114 | 0.967 | 0.974 | 0.970 | 0.975 | 0.975 | 0.895 | 0.913 | 0.918 | 0.081 |
| 20–32 | 36 | 0.994 | 0.994 | 0.994 | 0.995 | 0.995 | 0.968 | 0.992 | 0.991 | 0.384 |

## IDsw by object size (sum; lower = better)

| size (px) | #tracks | SORT | ByteTrack | OC-SORT | BoT-SORT | BoT-SORT+ReID | TrackTrack | FairMOT | TGraM | MOTRv2 |
|---|---|---|---|---|---|---|---|---|---|---|
| **<5** | 2133 | 1617 | 1065 | 1318 | 14357 | 14336 | 400 | 55683 | 55266 | 1920 |
| **5–8** | 2745 | 1714 | 871 | 1131 | 3379 | 3401 | 437 | 58697 | 59587 | 1930 |
| 8–12 | 833 | 226 | 135 | 203 | 1262 | 1269 | 59 | 6514 | 6284 | 328 |
| 12–20 | 114 | 16 | 15 | 12 | 21 | 21 | 6 | 595 | 601 | 145 |
| 20–32 | 36 | 1 | 2 | 1 | 2 | 2 | 0 | 52 | 54 | 44 |

## DetA by object size (output completeness, NOT box quality)

| size (px) | #tracks | SORT | ByteTrack | OC-SORT | BoT-SORT | BoT-SORT+ReID | TrackTrack | FairMOT | TGraM | MOTRv2 |
|---|---|---|---|---|---|---|---|---|---|---|
| **<5** | 2133 | 0.787 | 0.853 | 0.840 | 0.941 | 0.941 | 0.874 | 0.687 | 0.685 | 0.010 |
| **5–8** | 2745 | 0.894 | 0.883 | 0.965 | 0.935 | 0.935 | 0.797 | 0.805 | 0.807 | 0.008 |
| 8–12 | 833 | 0.911 | 0.896 | 0.965 | 0.954 | 0.954 | 0.945 | 0.851 | 0.850 | 0.005 |
| 12–20 | 114 | 0.954 | 0.938 | 0.988 | 0.980 | 0.980 | 0.881 | 0.905 | 0.906 | 0.060 |
| 20–32 | 36 | 0.990 | 0.989 | 0.997 | 0.996 | 0.996 | 0.950 | 0.983 | 0.983 | 0.242 |

## Findings

1. **JDT association collapses on small objects (<8 px).** With identical GT
   boxes, FairMOT/TGraM reach only AssA 0.59–0.66 and IDF1 0.57–0.73, while
   motion-based TBD hold AssA 0.79–0.93. Satellite micro-objects carry almost no
   appearance signal, so the stride-4 centre-sampled ReID embedding JDT relies on
   has nothing to separate identities with.
2. **TrackTrack is the strongest associator at every size** (AssA 0.924 at <5 px,
   0.934 at 5–8 px) and has by far the fewest ID switches (400 vs 1065–14357 for
   the other TBDs at <5 px). Its track-perspective association and track-aware
   initialisation are what pay off, *not* its appearance branch — see 3. Note its
   IDF1 is mid-pack (0.80–0.86) because it emits fewer boxes (DetA 0.80–0.87,
   `min_len=3` plus track-aware init): IDF1 charges for the withheld boxes, AssA
   does not.
3. **ReID adds nothing.** BoT-SORT+ReID matches plain BoT-SORT to within 0.002
   AssA in every bin (0.836 vs 0.836 at <5 px, IDsw 14336 vs 14357). The MOT17
   RGB-pedestrian FastReID embedding carries no usable signal on 5–30 px
   satellite crops. Same result as the BIRDSAI GT-oracle on thermal imagery.
4. **IDsw differs by orders of magnitude.** In the <8 px bins FairMOT/TGraM
   produce **55k–59k** ID switches vs 400–1700 for the motion TBDs. BoT-SORT
   (both variants) sits in between at 14k, an outlier among TBDs driven by its
   low `new_track_thresh` re-initialisation behaviour.
5. **MOTRv2 does not survive the oracle.** Even fed perfect boxes as proposals it
   outputs almost nothing below 20 px (DetA ≤0.06) — the query-based paradigm's
   track queries do not latch onto sub-20 px targets. See the caveat above.
6. **The gap closes as size grows.** By 20–32 px every method except MOTRv2
   reaches AssA ≈ 0.98–0.99; association is no longer the bottleneck.
7. **FairMOT ≈ TGraM.** Under the GT-box oracle TGraM's graph spatio-temporal
   reasoning shows no advantage over FairMOT's centre-sampled ReID.

## Reproduce

```bash
# 1. FastReID features on the GT boxes + the two appearance-aware oracle runs,
#    then re-score everything with the <=32 px bins (all three phases):
bash scripts/run_exp2_reid_oracle.sh

# 2. or just re-score from the existing oracle run dirs (reads mot_format only):
micromamba run -n esa_dlstem python evaluation/compute_hota_by_size.py \
  --oracle-root /data/ESA_DLSTEM_2025/experiments/MOT/exp2_oracle_20260608 \
  --workspace /tmp/hota_size_ws32 --bins 0,5,8,12,20,32 \
  --output /data/ESA_DLSTEM_2025/experiments/MOT/exp2_oracle_20260608/assa_vs_size_le32.csv

# 3. figure
micromamba run -n esa_dlstem python tools/plot_assa_vs_size.py
```

Size bins are `bin_idx` 0..4 = `<5 / 5–8 / 8–12 / 12–20 / 20–32` px;
`bin_idx = -1` (`all`) is the un-stratified row over the whole size range.
