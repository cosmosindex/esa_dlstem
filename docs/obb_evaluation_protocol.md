# OBB Evaluation Protocol on OOTB

> Two evaluation modes co-exist in `lightning_modules/sot_metrics.py`, chosen per-config via `obb_eval_mode`.
> This doc explains what they do and when to use which.

---

## Why two modes

OOTB's ground truth is a rotated 4-corner polygon (8 floats per frame). Trackers in our benchmark come in two output families:

1. **Mask-based trackers** (SAM 2, SAM 3, SAM 3.1, SAMURAI) — produce a segmentation mask → we fit `cv2.minAreaRect` → real rotated bounding rectangle.
2. **HBB-only trackers** (OSTrack-384, ODTrack, classical Siamese) — produce axis-aligned `(x, y, w, h)` only. No rotation information.

Forcing a single metric on both families is unfair in different directions depending on which you pick. Instead we run each family under the protocol that matches its output.

---

## Mode 1: `obb_eval_mode: polygon` (mask-based trackers)

Faithful polygon-level evaluation.

| Step | Implementation |
|---|---|
| IoU | `obb_iou_1_vs_n` — `cv2.intersectConvexConvex` between GT and pred polygons |
| Centre | Polygon centroid: mean of the 4 corner coordinates |
| Precision threshold | 20 px (OTB convention) |
| Success thresholds | `[0.00, 0.05, …, 1.00]` (21 points) |

Used by: `configs/sam2_ootb.yaml`, `configs/sam3_ootb.yaml`, `configs/samurai_ootb.yaml`.

This is **stricter** than OOTB's official protocol — IoU numbers are lower than what the OOTB paper reports, because the metric punishes rotation mismatch. It's the right metric to compare a truly OBB-aware tracker against itself on rotated targets.

---

## Mode 2: `obb_eval_mode: ootb_aabb` (HBB-only trackers)

Matches the official **OOTB v1.0 MATLAB toolkit** at `github.com/YZCU/OOTB`, specifically `tracker_benchmark_v1.0/rstEval/`:

| Source MATLAB | Our implementation | What it does |
|---|---|---|
| `corner2rect.m` | `obb_utils.obb_to_aabb` | 4 corners → `(min x, min y, max x, max y)` AABB |
| `calcRectInt.m` | `_iou_1_vs_n` | Plain rectangle IoU |
| `calcSeqErrRobust.m` centre | AABB centre `(x + w/2, y + h/2)` | Mean of the bounding rect |
| `thresholdSetError = 0:30`, `rankIdx=6` | `OOTB_PRECISION_THRESHOLDS`, P@5 | Precision at 5 px |
| `thresholdSetNorm_Precision = 0:0.05:1`, `rankIdx=11` | `OOTB_NORM_PRECISION_THRESHOLDS`, NP@0.5 | Normalised precision at 0.5 |

Key insight: the OOTB toolkit **collapses polygon GT to AABB before computing IoU**. It does NOT use polygon IoU anywhere in its evaluation code. So "OOTB-compatible" means "AABB-of-polygon IoU", not "polygon IoU".

Used by: `configs/ostrack_ootb.yaml`, `configs/odtrack_ootb.yaml`, and any future HBB-only tracker evaluated on OOTB.

Under this mode an HBB tracker can be directly compared with the 33 trackers reported in the OOTB ISPRS 2024 paper — same metric, same thresholds.

---

## What gets reported

`SOTMetrics.compute()` always emits **both** precision values regardless of mode, so a single run can be read under either protocol:

```json
{
  "success_auc":       0.xxx,   // IoU-based AUC (mode-dependent IoU definition)
  "precision_20":      0.xxx,   // OTB protocol (20 px)
  "precision_5":       0.xxx,   // OOTB protocol (5 px)
  "norm_precision_05": 0.xxx,   // OOTB protocol (normalised, 0.5)
  "mean_iou":          0.xxx,
  "success_plot":           {...},
  "precision_plot":         {...},  // OTB (0..50)
  "ootb_precision_plot":    {...},  // OOTB (0..30)
  "ootb_norm_precision_plot": {...} // OOTB (0..1)
}
```

The `success_auc` value depends on mode — polygon mode uses polygon IoU, `ootb_aabb` mode uses AABB-of-polygon IoU. Because of this the `success_auc` numbers from the two modes on the same dataset are **not directly comparable** across modes; they measure different things.

---

## How to choose in the paper

| Table | Metric | Rationale |
|---|---|---|
| OOTB main results (all 7 of our SOT models) | `success_auc` under each model's natural mode, `precision_5`, `norm_precision_05` | Each model evaluated under the protocol matching its output type. Cite OOTB paper at the table caption. |
| OOTB ablation — "Does real OBB output help?" | Re-run mask-based models under `ootb_aabb` mode for side-by-side vs HBB baselines | Controls for metric difference — isolates output-type contribution. |

---

## Data flow summary

```
OOTB GT polygon (8 floats)
  │
  ├─ ootb.py loader emits  ── boxes=AABB-of-polygon (xyxy)
  │                       └─ obb=(1, 8) raw polygon corners
  │
  ├─ Tracker runs:
  │    SAM family   → mask → mask_to_obb → pred_obb = rotated 4-corner
  │    OSTrack/ODTrack → HBB → pred_obb = 4-corner of the axis-aligned box
  │
  └─ SAM2SOTEvalCallback → SOTMetrics(obb_eval_mode=...)
       ├─ "polygon"   → obb_iou_1_vs_n(gt_obb, pred_obb) + polygon centroid
       └─ "ootb_aabb" → obb_to_aabb(both) → plain AABB IoU + AABB centre
```
