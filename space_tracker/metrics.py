"""Per-sequence SR / NPR / PR / P@5 — the protocol used by the paper.

* **SR**  = AUC of the success curve over IoU ∈ [0, 1].
* **PR**  = AUC of the precision curve over CLE ∈ [0, 50] px (OTB convention).
* **NPR** = AUC of the precision curve over normalised CLE ∈ [0, 0.5]
            (TrackingNet / LaSOT convention).
* **P@5** = fraction of frames with CLE < 5 px.

All curves are computed *per sequence*, then averaged with equal weight
across sequences. This matches LaSOT / GOT-10k / TrackingNet / OOTB.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


SUCCESS_THRESHOLDS = np.linspace(0.0, 1.0, 21)
PRECISION_THRESHOLDS = np.arange(0, 51, dtype=float)
NORM_PRECISION_THRESHOLDS = np.linspace(0.0, 0.5, 21)


def _iou_aabb(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two xyxy boxes."""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    area_a = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
    area_b = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _center(b: np.ndarray) -> tuple[float, float]:
    return float(0.5 * (b[0] + b[2])), float(0.5 * (b[1] + b[3]))


def per_frame_records(
    gt_boxes: list[np.ndarray | None],
    pred_boxes: list[np.ndarray | None],
) -> list[dict]:
    """Compute per-frame (best_iou, center_dist, norm_center_dist) records.

    Frames with no GT (target absent / fully invisible) are skipped — they
    don't enter the metrics. Frames where the tracker reports no prediction
    on a visible target count as full failure (IoU=0, CLE=inf).
    """
    out = []
    for gt, pred in zip(gt_boxes, pred_boxes):
        if gt is None:
            continue
        if pred is None:
            out.append({"best_iou": 0.0, "center_dist": float("inf"),
                        "norm_center_dist": float("inf")})
            continue
        iou = _iou_aabb(gt, pred)
        gx, gy = _center(gt); px, py = _center(pred)
        cdist = float(np.hypot(gx - px, gy - py))
        gw, gh = gt[2] - gt[0], gt[3] - gt[1]
        gd = float(np.hypot(gw, gh))
        ncdist = cdist / max(gd, 1e-6)
        out.append({"best_iou": iou, "center_dist": cdist,
                    "norm_center_dist": ncdist})
    return out


def per_sequence_metrics(records: list[dict]) -> dict:
    """SR / NPR / PR / P@5 + curves for ONE sequence."""
    if not records:
        return {"SR": 0.0, "NPR": 0.0, "PR": 0.0, "P@5": 0.0,
                "n_frames": 0, "success_curve": np.zeros_like(SUCCESS_THRESHOLDS),
                "precision_curve": np.zeros_like(PRECISION_THRESHOLDS),
                "norm_precision_curve": np.zeros_like(NORM_PRECISION_THRESHOLDS)}

    ious  = np.array([r["best_iou"]         for r in records], dtype=np.float64)
    cd    = np.array([r["center_dist"]      for r in records], dtype=np.float64)
    ncd   = np.array([r["norm_center_dist"] for r in records], dtype=np.float64)

    s_curve = np.array([(ious >= t).mean() for t in SUCCESS_THRESHOLDS])
    p_curve = np.array([(cd   <= t).mean() for t in PRECISION_THRESHOLDS])
    n_curve = np.array([(ncd  <= t).mean() for t in NORM_PRECISION_THRESHOLDS])

    return {
        "n_frames":             len(records),
        "SR":                   float(s_curve.mean()),
        "PR":                   float(p_curve.mean()),
        "NPR":                  float(n_curve.mean()),
        "P@5":                  float((cd <= 5.0).mean()),
        "success_curve":        s_curve,
        "precision_curve":      p_curve,
        "norm_precision_curve": n_curve,
    }


def aggregate(per_seq_results: Iterable[dict]) -> dict:
    """Equal-weight mean across sequences (the paper protocol)."""
    summaries = [s for s in per_seq_results if s and s["n_frames"]]
    if not summaries:
        return {"SR": float("nan"), "NPR": float("nan"),
                "PR": float("nan"), "P@5": float("nan"),
                "n_sequences": 0, "n_frames": 0}
    return {
        "n_sequences": len(summaries),
        "n_frames":    int(sum(s["n_frames"] for s in summaries)),
        "SR":          float(np.mean([s["SR"]   for s in summaries])),
        "NPR":         float(np.mean([s["NPR"]  for s in summaries])),
        "PR":          float(np.mean([s["PR"]   for s in summaries])),
        "P@5":         float(np.mean([s["P@5"]  for s in summaries])),
    }
