# Exp2 — Association ability vs. object size (GT-box oracle)

> Fair-comparison **Experiment 2**: feed the *same GT boxes* (score = 1) to every
> tracker's **unchanged association**, then score HOTA/AssA/IDF1/IDsw stratified
> by object size. Detection is held perfect (oracle), so this isolates **pure
> association ability** from detection — see
> [`mot_fair_comparison_framework.md`](mot_fair_comparison_framework.md).

- **Run root:** `/data/ESA_DLSTEM_2025/experiments/MOT/exp2_oracle_20260608`
- **Source table:** `assa_vs_size.csv` (210 rows; produced by `compute_hota_by_size.py`)
- **Completed:** 2026-06-09 11:39 (`_EXP2_FULLY_COMPLETE`)
- **Datasets pooled (5):** rscardata, satmtb, sdmcar, airmot, viso_no_car
  (together span car ~5 px → ship/airplane → train, by object size)
- **Methods (6):** TBD = SORT / ByteTrack / OC-SORT / BoT-SORT;
  JDT = **FairMOT** & **TGraM** (4-class union `model_best`, the all-dataset
  checkpoints; only detection swapped for GT, association untouched)
- **Pooling:** across all 5 datasets per size bin, weighted by `n_gt_tracks`;
  empty bins (no GT track of that size in a dataset) excluded.
- **Reported metrics: AssA, IDF1, IDsw only — *not* HOTA/MOTA** (see "Why" below).
- **Figure:** `docs/figures/exp2_assa_vs_size.{pdf,png}` (AssA & IDF1 vs size,
  TBD solid / JDT dashed) — `tools/plot_assa_vs_size.py`.

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

| size (px) | #tracks | SORT | ByteTrack | OC-SORT | BoT-SORT | **FairMOT** | **TGraM** |
|---|---|---|---|---|---|---|---|
| **<5**   | 2133 | 0.813 | 0.869 | **0.887** | 0.836 | **0.589** | **0.589** |
| **5–8**  | 2745 | 0.794 | 0.855 | **0.904** | 0.884 | **0.655** | **0.654** |
| 8–12     | 833  | 0.887 | 0.883 | **0.940** | 0.909 | 0.793 | 0.797 |
| 12–20    | 114  | 0.929 | 0.916 | **0.961** | 0.949 | 0.878 | 0.885 |
| 20–40    | 51   | 0.986 | 0.984 | 0.991 | 0.990 | 0.987 | 0.986 |
| ≥40      | 61   | 0.996 | 0.997 | 0.999 | 0.999 | 0.996 | 0.996 |

## IDF1 by object size (higher = better)

| size (px) | SORT | ByteTrack | OC-SORT | BoT-SORT | FairMOT | TGraM |
|---|---|---|---|---|---|---|
| **<5**  | 0.826 | 0.920 | 0.859 | 0.897 | **0.570** | **0.570** |
| **5–8** | 0.884 | 0.952 | 0.926 | 0.951 | **0.731** | **0.730** |
| 8–12    | 0.944 | 0.963 | 0.949 | 0.958 | 0.856 | 0.860 |
| 12–20   | 0.967 | 0.974 | 0.970 | 0.975 | 0.913 | 0.918 |
| 20–40   | 0.996 | 0.996 | 0.995 | 0.996 | 0.994 | 0.993 |
| ≥40     | 0.999 | 1.000 | 0.999 | 1.000 | 0.997 | 0.997 |

## IDsw by object size (sum; lower = better)

| size (px) | SORT | ByteTrack | OC-SORT | BoT-SORT | FairMOT | TGraM |
|---|---|---|---|---|---|---|
| **<5**  | 1617 | 1065 | 1318 | 14357 | **55683** | **55266** |
| **5–8** | 1714 | 871  | 1131 | 3379  | **58697** | **59587** |
| 8–12    | 226  | 135  | 203  | 1262  | 6514  | 6284  |
| 12–20   | 16   | 15   | 12   | 21    | 595   | 601   |
| 20–40   | 1    | 2    | 1    | 2     | 52    | 54    |
| ≥40     | 1    | 0    | 0    | 1     | 23    | 23    |

## Findings

1. **JDT association collapses on small objects (<8 px).** With identical GT
   boxes, FairMOT/TGraM reach only AssA 0.59–0.65 and IDF1 0.57, while
   motion-based TBD hold AssA 0.79–0.90 / IDF1 0.83–0.92. **OC-SORT is the
   strongest TBD on small objects** (AssA 0.887 at <5 px, 0.904 at 5–8 px) —
   its observation-centric, velocity-consistent association handles micro-object
   motion best.
2. **IDsw differs by an order of magnitude.** In the <8 px bins, FairMOT/TGraM
   produce **55k–59k** ID switches vs. thousands for TBD. Satellite micro-objects
   carry almost no appearance signal, so the learned appearance-ReID association
   that JDT relies on breaks down.
3. **The gap closes as size grows.** By ≥20 px all methods reach AssA ≈ 0.98–0.99;
   association is no longer the bottleneck.
4. **FairMOT ≈ TGraM.** Under the GT-box oracle, TGraM's graph spatio-temporal
   reasoning shows no advantage over FairMOT's center-sampled ReID.

Consistent with the prior expectation that *JDT AssA collapses <8 px while
motion-TBD holds*.

## Reproduce

```bash
# re-score from existing oracle run dirs (safe, reads mot_format only)
micromamba run -n esa_dlstem python compute_hota_by_size.py \
  --oracle-root /data/ESA_DLSTEM_2025/experiments/MOT/exp2_oracle_20260608 \
  --workspace /tmp/hota_size_ws_final \
  --output /data/ESA_DLSTEM_2025/experiments/MOT/exp2_oracle_20260608/assa_vs_size.csv
```

Size bins are `bin_idx` 0..5 = `<5 / 5–8 / 8–12 / 12–20 / 20–40 / ≥40` px;
`bin_idx = -1` (`all`) is the un-stratified row.
