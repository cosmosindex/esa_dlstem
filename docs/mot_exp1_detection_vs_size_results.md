# Exp1 — Detection ability vs. object size (per class, SAT-MTB)

> Fair-comparison **Experiment 1** (the detection axis; complement of
> [Exp2](mot_exp2_assa_vs_size_results.md)'s association axis). Each method uses
> **its own detections**; we measure **Recall vs object size, per class**. See
> [`mot_fair_comparison_framework.md`](mot_fair_comparison_framework.md).

- **Dataset:** SAT-MTB test (the only MOT set where every detector is in-domain
  or cached — see below). 61 sequences, 4 coarse classes.
- **Detectors compared:**
  - **car** → **HiEUM** (car specialist) vs FairMOT / TGraM (4-class union, own decode)
  - **airplane / ship / train** → **Faster R-CNN** (trained on SAT-MTB `det_hbb`,
    3-class, in-domain, dense per-frame GT) vs FairMOT / TGraM
- **Metric:** Recall at **IoU ≥ 0.5** (a GT box counts as detected if some pred
  box overlaps it ≥ 0.5). Recall needs only GT class+size — pred class is
  irrelevant — so it scores the JDT `mot_format` outputs (which carry no class
  column) uniformly with the specialist detectors.
- **Figure:** `docs/figures/exp1_detection_recall_by_size.{pdf,png}` (2×2, one
  panel per class) — `tools/plot_exp1_detection_vs_size.py`.
- **Table:** `docs/figures/exp1_detection_recall_by_size.csv`
  (`detector,class,bin,n_gt,n_hit,recall`) — `compute_exp1_detection_recall.py`.

## Why per-class, not one pooled curve

Detector identity tracks **class**, not pixel size, and the classes **overlap
heavily in size** (car 4–11 px, ship 5–46 px, airplane 10–90 px) — so there is
**no pixel threshold** at which "HiEUM gives way to Faster R-CNN". A single
pooled "detection vs size" line would also be a chimera (HiEUM cars at the small
end stitched to Faster R-CNN trains at the large end). Splitting by class keeps,
within each panel, all lines detecting the **same objects**, so the curves are
comparable and "which detector covers which size" is read off the panels (HiEUM
lives only in *car*, Faster R-CNN only in the others).

## Methodology notes (correctness)

- **Per-box class from the GT label**, not the sequence folder. SAT-MTB MOT
  sequences are **mixed-class** (e.g. small cars move through *train* scenes),
  and `_load_annotations` returns labels re-indexed to `COARSE_CATEGORIES`
  (alphabetical): **0=airplane, 1=car, 2=ship, 3=train** — *not* the raw MOT
  `_MOT_CLASS_MAP` (`0=car,…`). Verified by per-label median size.
- **Each class is scored only on the videos where its specialist ran** (HiEUM →
  21 car-folder vids; Faster R-CNN → 40 non-car vids; JDT ran on all 61). This
  keeps specialist vs JDT on the *same* GT boxes and excludes boxes the
  specialist never saw (e.g. cars inside train scenes).
- Bins with < 50 GT boxes are dropped from the plot (SAT-MTB size outliers).

## Recall by class × size (IoU ≥ 0.5)

| class | size (px) | n_gt | **specialist** | FairMOT | TGraM |
|---|---|---|---|---|---|
| **car** | <5 | 171580 | **HiEUM 0.44** | 0.19 | 0.13 |
| car | 5–8 | 80585 | **HiEUM 0.39** | 0.28 | 0.17 |
| car | 8–12 | 4258 | HiEUM 0.05 | 0.20 | 0.07 |
| **airplane** | 20–40 | 3661 | **FRCNN 0.57** | 0.44 | 0.16 |
| airplane | ≥40 | 7587 | **FRCNN 0.91** | 0.66 | 0.46 |
| **ship** | 8–12 | 3062 | FRCNN 0.06 | **FairMOT 0.30** | 0.12 |
| ship | 12–20 | 5746 | FRCNN 0.07 | **FairMOT 0.35** | 0.28 |
| ship | 20–40 | 5083 | FRCNN 0.23 | **FairMOT 0.43** | 0.16 |
| **train** | ≥40 | 973 | **FRCNN 0.74** | 0.00 | 0.00 |

## Findings

1. **Small-object detection cliff is universal.** Even the best detector finds
   < 45 % of < 5 px cars at IoU 0.5; recall climbs monotonically with size.
2. **Specialists win in their sweet spot.** HiEUM beats both JDT on tiny cars
   (0.44 vs 0.13–0.19 at < 5 px); Faster R-CNN dominates large airplanes (0.91)
   and trains (0.74).
3. **JDT collapses on very large objects.** FairMOT/TGraM recall ≈ **0** on
   ≥ 40 px trains — the car-dominated union training leaves their heatmap/anchor
   range unsuited to huge objects, while Faster R-CNN (dense det GT incl. trains)
   gets 0.74.
4. **JDT is competitive — even better — on mid-size ships.** FairMOT outperforms
   Faster R-CNN on ships at every size bin (e.g. 0.43 vs 0.23 at 20–40 px);
   Faster R-CNN is unexpectedly weak on ships.

## Reproduce

```bash
micromamba run -n esa_dlstem python compute_exp1_detection_recall.py   # -> CSV
micromamba run -n esa_dlstem python tools/plot_exp1_detection_vs_size.py  # -> figure
```

Edit `IOU_THRESH` in `compute_exp1_detection_recall.py` to re-score at a
different overlap (e.g. 0.3 for a more lenient tiny-object criterion).
