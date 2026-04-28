# HiEUM on RsCarData — paper-protocol reproduction

> Final run: `/data/ESA_DLSTEM_2025/experiments/Detection/hieum_20260427/hieum_rscardata_20260427_143339/`
> Result: **F1 0.906 / Pr 0.942 / Re 0.873** at score_thresh 0.25, vs paper **F1 0.897 / Pr 0.961 / Re 0.842**.

---

## What it took to match the paper

Four protocol pieces had to be aligned with HiEUM's official `evaluation_final.py`:

| Item | Paper | Where in our code |
|---|---|---|
| TP/FP criterion | centroid distance ≤ 5 px (Sec 4.2) | `match_metric: centroid` + `centroid_dist_thresh: 5.0` |
| NMS | Soft-NMS linear, Nt = 0.1 | `_soft_nms_linear` in `models/hieum.py` |
| Score threshold | sweep [0.10..0.35], report best F1 | `score_sweep:` in `configs/MOT/hieum_rscardata.yaml` |
| **GT labels** | `labeleddata20230227/<seq>/img1/<frame>.xml` (re-curated, 156k boxes) | `_override_test_anns_from_xml` in `datasets/rscardata.py` — auto-prefers the new XMLs over the COCO MOT JSON |

The final fix that closed the gap was the GT label switch. The COCO MOT JSON (`test1024_mot.json`) ships the **older** annotations with 93 491 boxes; the paper evaluates against `labeleddata20230227/` with 155 987 boxes — 66 % more. Many of the predictions we were counting as false positives were actually unlabeled cars in the older GT.

---

## Result table (centroid 5 px, Soft-NMS, NEW labels, macro across 7 seqs)

| score thr | Pr | Re | F1 |
|---:|---:|---:|---:|
| 0.10 | 0.671 | 0.923 | 0.773 |
| 0.15 | 0.813 | 0.913 | 0.860 |
| 0.20 | 0.898 | 0.896 | 0.897 |
| **0.25** | **0.942** | **0.873** | **0.906** |
| 0.30 | 0.962 | 0.839 | 0.896 |
| 0.32 | 0.967 | 0.820 | 0.887 |
| 0.34 | 0.972 | 0.795 | 0.874 |
| 0.35 | 0.974 | 0.779 | 0.864 |
| **paper** | **0.961** | **0.842** | **0.897** |

F1 peaks at **0.25** (not the upper bound 0.35), with **Pr 94.2 / Re 87.3 / F1 90.6**.

Per-video macro at the best threshold:

| seq | Pr | Re | F1 |
|---|---:|---:|---:|
| test1024/002 | 0.950 | 0.828 | 0.885 |
| test1024/003 | 0.951 | 0.874 | 0.911 |
| test1024/005 | 0.950 | 0.857 | 0.901 |
| test1024/006 | 0.937 | 0.883 | 0.909 |
| test1024/008 | 0.938 | 0.944 | 0.941 |
| test1024/009 | 0.907 | 0.866 | 0.886 |
| test1024/010 | 0.959 | 0.861 | 0.907 |

Variation between sequences is small (F1 0.89–0.94). Micro F1 = 0.905 (within 0.05 pt of macro), so the averaging convention does not matter on this dataset.

---

## Verdict

The pretrained HiEUM checkpoint reproduces the paper's reported numbers under the paper's own protocol once the four protocol items above are aligned. We end up 0.9 pt above paper's F1, which is well within run-to-run / numerical-precision variance for this kind of detector.

Speed on RTX 5000 Ada with `ConvAlgo.Native` (sm_89 has no precompiled cumm GEMM kernels, so the native path is mandatory): **12.7 fps** end-to-end vs. paper's 98.8 fps on RTX 2080 Ti with the implicit_gemm tuner. The 8× slowdown is the algo-fallback cost; not a model issue.

---

## Output files

`test_metrics.json` (compact, 12 keys):

```
test/Precision, test/Recall, test/F1                 # at the wrapper's score_thresh (0.05 floor)
test/best_F1, test/best_Precision, test/best_Recall  # best operating point from sweep
test/best_score_thresh                               # the score that hit best F1
test/fps, test/total_time_s, test/model_size_MB
per_category, per_size                               # det breakdown from viz callback
```

`sweep_results.json` (full per-threshold breakdown):

```json
{
  "thresholds": [0.10, ..., 0.35],
  "micro": {"Pr": [...], "Re": [...], "F1": [...]},
  "macro": {"Pr": [...], "Re": [...], "F1": [...]},
  "best_micro": {"score_thresh": 0.25, "Precision": ..., "Recall": ..., "F1": ...},
  "best_macro": {"score_thresh": 0.25, "Precision": ..., "Recall": ..., "F1": ...},
  "per_video_macro": {video_id: {"score_thresh", "Precision", "Recall", "F1", "tp", "fp", "fn"}}
}
```

`per_image_metrics.json` (per-frame TP/FP/FN at the visualization callback's `score_thresh`).

---

## Reproducing this run

```bash
DEST_ROOT=/data/ESA_DLSTEM_2025/experiments/Detection/hieum_20260427 \
    bash run_all_mot_hieum.sh --datasets rscardata
```

Configuration: `configs/MOT/hieum_rscardata.yaml`. The dataset class auto-detects `labeleddata20230227/` and prefers it over the COCO MOT JSON for the test split.
