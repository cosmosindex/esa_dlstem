"""
SOT (Single Object Tracking) Metrics
=====================================
Standalone metrics class for SOT evaluation.
No Lightning dependency — can be used anywhere.

Standard SOT metrics:
  - Success Plot (SR):    fraction of frames with IoU >= threshold, threshold in [0, 1].
                          Reported as AUC of the success curve.
  - Precision Plot (PR):  fraction of frames with centre distance <= threshold.
                          Reported as AUC of the precision curve over [0, 30] px,
                          following OOTB's protocol for satellite video (Chen
                          et al., ISPRS 2024), which argues that the generic-
                          vision 20 px threshold corresponds to satellite-
                          video's 5 px due to the lower spatial resolution and
                          smaller objects, so the AUC is integrated over the
                          pixel-accuracy regime that actually matters for
                          satellite targets.
  - Normalised Precision (NPR): fraction of frames with centre distance / GT
                                diagonal <= threshold. Reported as AUC over
                                [0, 0.5], following the GOT-10k / LaSOT-ext
                                normalised-precision convention.
  - P@5 (diagnostic):     fraction of frames with CLE < 5 px, single-threshold
                          scalar, mirroring OOTB's primary precision metric.

References: OTB (Wu et al., 2015); OOTB (Chen et al., ISPRS 2024).

OBB evaluation modes
--------------------
When OBB (8-corner polygon) annotations are present, IoU and centre distance
are computed under one of two modes, selected via `obb_eval_mode`:

* `"polygon"` (default) — polygon IoU via `cv2.intersectConvexConvex` +
  polygon centroid (mean of 4 corners). Faithful to rotated mask outputs;
  used for SAM 2 / SAM 3 / SAMURAI.

* `"ootb_aabb"` — matches the official OOTB v1.0 MATLAB toolkit
  (github.com/YZCU/OOTB, `tracker_benchmark_v1.0/rstEval/`). Both GT and
  prediction polygons are first collapsed to their axis-aligned bounding
  rectangles via min/max of corners (`corner2rect.m`), then standard AABB
  IoU is computed (`calcRectInt.m`). Precision uses OOTB's `thresholdSetError`
  `0:30 px` reported at 5 px (`rankIdx=6`); normalised precision uses
  `0:0.05:1` reported at 0.5 (`rankIdx=11`). Use this mode for HBB-only
  trackers (OSTrack, ODTrack, classical Siamese) so their numbers are
  directly comparable with the 33 trackers reported in the OOTB paper.

The mode does NOT change anything when OBB is not provided — plain AABB IoU
from the `boxes` field is used in either case.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from obb_utils import obb_iou_1_vs_n, obb_to_aabb


# Evaluation thresholds.
#
# SR (success): AUC over IoU ∈ [0, 1].
# PR (precision): AUC over centre-location error ∈ [0, 30] px — follows the
#     OOTB v1.0 MATLAB toolkit (perfPlot.m) and the satellite-video protocol
#     argued in Chen et al., ISPRS 2024 (GV's 20 px ≈ SV's 5 px).
# NPR (normalised precision): AUC over normalised centre distance ∈ [0, 0.5],
#     matching GOT-10k / LaSOT-ext normalised-precision convention.
# `PRECISION_THRESHOLDS_OTB` is kept only for back-compat plots / the legacy
# P@20 number reported alongside the OOTB protocol for readers used to OTB.
SUCCESS_THRESHOLDS = np.linspace(0, 1, 21)                 # 0.00, 0.05, …, 1.00
PRECISION_THRESHOLDS = np.arange(0, 31, dtype=float)        # 0, 1, …, 30 px (PR)
NORM_PRECISION_THRESHOLDS = np.linspace(0, 0.5, 21)         # 0.000, 0.025, …, 0.500 (NPR)

# Back-compat aliases (legacy OTB plot range / full OOTB norm range 0–1).
PRECISION_THRESHOLDS_OTB = np.arange(0, 51, dtype=float)    # 0, 1, …, 50 px
OOTB_PRECISION_THRESHOLDS = PRECISION_THRESHOLDS            # alias — OOTB = primary
OOTB_NORM_PRECISION_THRESHOLDS = np.linspace(0, 1, 21)      # 0.00, 0.05, …, 1.00

OBB_EVAL_MODES = ("polygon", "ootb_aabb")

# Size threshold: small < 32x32 = 1024 px²
_SMALL_AREA_THRESH = 1024


@dataclass
class SOTRecord:
    """One record per GT object per frame."""
    video_id: str
    frame_id: int
    gt_class: str
    gt_size: str                # "small" or "large"
    best_iou: float
    center_dist: float          # in pixels
    norm_center_dist: float     # normalised by GT diagonal


class SOTMetrics:
    """Accumulate per-frame SOT records and compute Success/Precision metrics.

    Args:
        class_names: Optional id → display name map.
        obb_eval_mode: How to handle OBB annotations when present.
            * ``"polygon"`` (default): polygon IoU + polygon centroid.
              Use for SAM 2 / SAM 3 / SAMURAI and other mask-based trackers.
            * ``"ootb_aabb"``: collapse both GT and prediction polygons to
              their min/max AABB, then use plain AABB IoU + AABB centre.
              Matches the OOTB v1.0 MATLAB toolkit; use for HBB-only trackers
              (OSTrack, ODTrack, classical Siamese) on OOTB.
    """

    def __init__(
        self,
        class_names: dict[int, str] | None = None,
        obb_eval_mode: str = "polygon",
        sequence_attributes: dict[str, list[str]] | None = None,
    ):
        if obb_eval_mode not in OBB_EVAL_MODES:
            raise ValueError(
                f"obb_eval_mode must be one of {OBB_EVAL_MODES}, got {obb_eval_mode!r}"
            )
        self.class_names = class_names or {}
        self.obb_eval_mode = obb_eval_mode
        # video_id → list of sequence-level attribute names that are active.
        # Multi-label: a single sequence can have several attributes, so this
        # creates overlapping (non-partition) groups when aggregating metrics.
        self.sequence_attributes: dict[str, list[str]] = sequence_attributes or {}
        self.records: list[SOTRecord] = []

    def reset(self):
        self.records.clear()

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(
        self,
        gt_boxes: np.ndarray,       # (M, 4) xyxy
        gt_labels: np.ndarray,      # (M,)
        pred_boxes: np.ndarray,     # (N, 4) xyxy
        pred_scores: np.ndarray,    # (N,)
        pred_labels: np.ndarray,    # (N,)
        video_id: str,
        frame_id: int,
        gt_obb: np.ndarray | None = None,    # (M, 8) OBB corners
        pred_obb: np.ndarray | None = None,  # (N, 8) OBB corners
    ):
        """Process one frame: for each GT, find the best same-class prediction.

        When gt_obb and pred_obb are provided, IoU is computed using OBB
        (oriented bounding box) intersection. Center distance uses the OBB
        centroid (mean of 4 corners).
        """
        obb_present = (gt_obb is not None and pred_obb is not None
                       and len(gt_obb) > 0 and len(pred_obb) > 0)
        # "polygon" → polygon IoU + polygon centroid.
        # "ootb_aabb" → AABB-of-polygon IoU + AABB centre (OOTB v1.0 protocol).
        use_polygon = obb_present and self.obb_eval_mode == "polygon"
        use_ootb_aabb = obb_present and self.obb_eval_mode == "ootb_aabb"

        # Pre-compute AABB-from-polygon for both GT and preds when needed
        # (cheap min/max per polygon).
        gt_aabb_from_obb = None
        pred_aabb_from_obb = None
        if use_ootb_aabb:
            gt_aabb_from_obb = np.stack([obb_to_aabb(o) for o in gt_obb])
            pred_aabb_from_obb = np.stack([obb_to_aabb(o) for o in pred_obb]) \
                if len(pred_obb) else np.zeros((0, 4), dtype=np.float32)

        for gi in range(len(gt_boxes)):
            gt_box = gt_boxes[gi]
            gt_cls = int(gt_labels[gi])
            gt_name = self.class_names.get(gt_cls, f"cls{gt_cls}")

            # Filter to same class
            same_cls = np.where(pred_labels == gt_cls)[0]

            # Size bucket: use the AABB-from-OBB under OOTB protocol so that
            # "small vs large" matches what the OOTB toolkit sees.
            if use_ootb_aabb:
                gt_box_for_size = gt_aabb_from_obb[gi]
            else:
                gt_box_for_size = gt_box
            gt_area = float(
                (gt_box_for_size[2] - gt_box_for_size[0])
                * (gt_box_for_size[3] - gt_box_for_size[1])
            )
            gt_size = "small" if gt_area < _SMALL_AREA_THRESH else "large"

            if len(same_cls) == 0:
                self.records.append(SOTRecord(
                    video_id=video_id, frame_id=frame_id, gt_class=gt_name,
                    gt_size=gt_size,
                    best_iou=0.0, center_dist=float("inf"),
                    norm_center_dist=float("inf"),
                ))
                continue

            # Best same-class prediction by IoU
            if use_polygon:
                sc_obb = pred_obb[same_cls]
                ious = obb_iou_1_vs_n(gt_obb[gi], sc_obb)
            elif use_ootb_aabb:
                sc_aabb = pred_aabb_from_obb[same_cls]
                ious = _iou_1_vs_n(gt_aabb_from_obb[gi], sc_aabb)
            else:
                sc_boxes = pred_boxes[same_cls]
                ious = _iou_1_vs_n(gt_box, sc_boxes)
            best_idx = int(ious.argmax())
            best_iou = float(ious[best_idx])

            # Centre distance — OBB centroid in polygon mode, AABB centre
            # (both in the non-OBB and OOTB-AABB modes).
            if use_polygon:
                gt_pts = gt_obb[gi].reshape(4, 2)
                gt_cx, gt_cy = float(gt_pts[:, 0].mean()), float(gt_pts[:, 1].mean())
                pr_pts = pred_obb[same_cls[best_idx]].reshape(4, 2)
                pr_cx, pr_cy = float(pr_pts[:, 0].mean()), float(pr_pts[:, 1].mean())
            elif use_ootb_aabb:
                gt_a = gt_aabb_from_obb[gi]
                pr_a = pred_aabb_from_obb[same_cls[best_idx]]
                gt_cx, gt_cy = (gt_a[0] + gt_a[2]) / 2, (gt_a[1] + gt_a[3]) / 2
                pr_cx, pr_cy = (pr_a[0] + pr_a[2]) / 2, (pr_a[1] + pr_a[3]) / 2
            else:
                best_box = pred_boxes[same_cls[best_idx]]
                gt_cx, gt_cy = (gt_box[0] + gt_box[2]) / 2, (gt_box[1] + gt_box[3]) / 2
                pr_cx, pr_cy = (best_box[0] + best_box[2]) / 2, (best_box[1] + best_box[3]) / 2
            cdist = float(np.sqrt((gt_cx - pr_cx) ** 2 + (gt_cy - pr_cy) ** 2))

            # Normalised by GT diagonal
            gt_w, gt_h = gt_box[2] - gt_box[0], gt_box[3] - gt_box[1]
            gt_diag = float(np.sqrt(gt_w ** 2 + gt_h ** 2))
            ncdist = cdist / max(gt_diag, 1e-6)

            self.records.append(SOTRecord(
                video_id=video_id, frame_id=frame_id, gt_class=gt_name,
                gt_size=gt_size,
                best_iou=best_iou, center_dist=cdist,
                norm_center_dist=ncdist,
            ))

    # ------------------------------------------------------------------
    # Compute metrics
    # ------------------------------------------------------------------

    def compute(self) -> dict:
        """Compute overall + per-category + per-size + per-attribute metrics."""
        if not self.records:
            return {}

        result = _compute_from_records(self.records)

        # Per category
        by_class: dict[str, list[SOTRecord]] = {}
        for r in self.records:
            by_class.setdefault(r.gt_class, []).append(r)
        result["per_category"] = {
            name: _compute_from_records(recs) for name, recs in sorted(by_class.items())
        }

        # Per size (small < 32x32 = 1024 px², large >= 1024 px²)
        by_size: dict[str, list[SOTRecord]] = {}
        for r in self.records:
            by_size.setdefault(r.gt_size, []).append(r)
        result["per_size"] = {
            name: _compute_from_records(recs) for name, recs in sorted(by_size.items())
        }

        # Per sequence attribute (multi-label: a record may appear in several
        # attribute groups, so groups do NOT partition the record set).
        if self.sequence_attributes:
            by_attr = _group_by_attribute(self.records, self.sequence_attributes)
            if by_attr:
                result["per_sequence_attribute"] = {
                    name: _compute_from_records(recs)
                    for name, recs in sorted(by_attr.items())
                }

        return result

    # ------------------------------------------------------------------
    # Plot generation
    # ------------------------------------------------------------------

    def plot(self, output_dir: str | Path) -> dict[str, Path]:
        """Generate Success Plot and Precision Plot as PNG files.

        Returns dict mapping plot name to file path.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}

        if not self.records:
            return paths

        # Group by category and size
        by_class: dict[str, list[SOTRecord]] = {}
        by_size: dict[str, list[SOTRecord]] = {}
        for r in self.records:
            by_class.setdefault(r.gt_class, []).append(r)
            by_size.setdefault(r.gt_size, []).append(r)

        # --- Success Plot (per category) ---
        paths["success_plot"] = _plot_success(
            self.records, by_class, output_dir / "success_plot.png",
            "Success Plot (per category)",
        )

        # --- Success Plot (per size) ---
        paths["success_plot_size"] = _plot_success(
            self.records, by_size, output_dir / "success_plot_size.png",
            "Success Plot (per size)",
        )

        # --- Precision Plot (per category) ---
        paths["precision_plot"] = _plot_precision(
            self.records, by_class, output_dir / "precision_plot.png",
            "Precision Plot (per category)",
        )

        # --- Precision Plot (per size) ---
        paths["precision_plot_size"] = _plot_precision(
            self.records, by_size, output_dir / "precision_plot_size.png",
            "Precision Plot (per size)",
        )

        # --- Per sequence attribute (OOTB: 12 attrs, SV248S: 10, SatSOT: 11) ---
        if self.sequence_attributes:
            by_attr = _group_by_attribute(self.records, self.sequence_attributes)
            if by_attr:
                paths["success_plot_attr"] = _plot_success(
                    self.records, by_attr, output_dir / "success_plot_attr.png",
                    "Success Plot (per sequence attribute)",
                )
                paths["precision_plot_attr"] = _plot_precision(
                    self.records, by_attr, output_dir / "precision_plot_attr.png",
                    "Precision Plot (per sequence attribute)",
                )

        return paths


# ======================================================================
# Private helpers
# ======================================================================

def _group_by_attribute(
    records: list[SOTRecord],
    sequence_attributes: dict[str, list[str]],
) -> dict[str, list[SOTRecord]]:
    """Multi-label groupings by sequence-level attribute.

    For each record, look up its ``video_id`` in ``sequence_attributes`` and
    append it to every attribute group named in that list. Records whose
    video carries no attributes (or is missing from the mapping) don't
    contribute to any attribute group.
    """
    groups: dict[str, list[SOTRecord]] = {}
    for r in records:
        attrs = sequence_attributes.get(r.video_id, ())
        for attr in attrs:
            groups.setdefault(attr, []).append(r)
    return groups


def _plot_success(
    all_records: list[SOTRecord],
    groups: dict[str, list[SOTRecord]],
    save_path: Path,
    title: str,
) -> Path:
    """Generate a Success Plot with overall + per-group curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))

    ious_all = np.array([r.best_iou for r in all_records])
    success_all = np.array([np.mean(ious_all >= t) for t in SUCCESS_THRESHOLDS])
    auc_all = float(np.mean(success_all))
    ax.plot(SUCCESS_THRESHOLDS, success_all, "k-", linewidth=2,
            label=f"Overall [{auc_all:.3f}]")

    for name, recs in sorted(groups.items()):
        ious = np.array([r.best_iou for r in recs])
        curve = np.array([np.mean(ious >= t) for t in SUCCESS_THRESHOLDS])
        auc = float(np.mean(curve))
        ax.plot(SUCCESS_THRESHOLDS, curve, "--", linewidth=1.5,
                label=f"{name} [{auc:.3f}]")

    ax.set_xlabel("Overlap threshold")
    ax.set_ylabel("Success rate")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def _plot_precision(
    all_records: list[SOTRecord],
    groups: dict[str, list[SOTRecord]],
    save_path: Path,
    title: str,
) -> Path:
    """Generate a Precision Plot with overall + per-group curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))

    cdists_all = np.array([r.center_dist for r in all_records])
    prec_all = np.array([np.mean(cdists_all <= t) for t in PRECISION_THRESHOLDS])
    pr_auc_all = float(prec_all.mean())
    ax.plot(PRECISION_THRESHOLDS, prec_all, "k-", linewidth=2,
            label=f"Overall [PR={pr_auc_all:.3f}]")

    for name, recs in sorted(groups.items()):
        cdists = np.array([r.center_dist for r in recs])
        curve = np.array([np.mean(cdists <= t) for t in PRECISION_THRESHOLDS])
        pr_auc = float(curve.mean())
        ax.plot(PRECISION_THRESHOLDS, curve, "--", linewidth=1.5,
                label=f"{name} [PR={pr_auc:.3f}]")

    ax.set_xlabel("Center location error (pixels)")
    ax.set_ylabel("Precision")
    ax.set_title(f"{title} — PR AUC over 0–30 px (OOTB protocol)")
    ax.set_xlim(0, PRECISION_THRESHOLDS[-1])
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def _compute_from_records(records: list[SOTRecord]) -> dict:
    """Compute SOT summary from a list of records.

    Primary metrics (reported in the benchmark table):

    * `success_auc`        — SR: AUC of the success curve over IoU ∈ [0, 1].
    * `precision_auc`      — PR: AUC of the precision curve over CLE ∈ [0, 30] px
                             (OOTB's satellite-video protocol; GV's 20 px ≈ SV's 5 px).
    * `norm_precision_auc` — NPR: AUC of the normalised precision curve over
                             normalised CLE ∈ [0, 0.5].
    * `precision_5`        — Diagnostic P@5: fraction of frames with CLE < 5 px
                             (OOTB's primary precision scalar).

    Legacy / compatibility fields (kept so older readers still work):
    `precision_20` (OTB P@20), `norm_precision_05` (single-threshold NP@0.5),
    `precision_plot_otb` (0–50 px curve), `precision_plot` (0–30 px curve),
    `norm_precision_plot` (0–0.5 curve), `ootb_norm_precision_plot` (0–1 curve).
    """
    n = len(records)
    ious = np.array([r.best_iou for r in records])
    cdists = np.array([r.center_dist for r in records])
    ncdists = np.array([r.norm_center_dist for r in records])

    # --- SR: success AUC over IoU ∈ [0, 1] ---
    success_curve = [float(np.mean(ious >= t)) for t in SUCCESS_THRESHOLDS]
    success_auc = float(np.mean(success_curve))

    # --- PR: precision AUC over CLE ∈ [0, 30] px (OOTB protocol) ---
    precision_curve = [float(np.mean(cdists <= t)) for t in PRECISION_THRESHOLDS]
    precision_auc = float(np.mean(precision_curve))
    precision_5 = float(np.mean(cdists <= 5))

    # --- NPR: normalised precision AUC over norm-CLE ∈ [0, 0.5] ---
    nprec_curve = [float(np.mean(ncdists <= t)) for t in NORM_PRECISION_THRESHOLDS]
    norm_precision_auc = float(np.mean(nprec_curve))

    # --- Legacy / back-compat (OTB P@20 range, OOTB 0–1 NP range) ---
    precision_curve_otb = [float(np.mean(cdists <= t)) for t in PRECISION_THRESHOLDS_OTB]
    precision_20 = float(np.mean(cdists <= 20))
    nprec_curve_full = [float(np.mean(ncdists <= t)) for t in OOTB_NORM_PRECISION_THRESHOLDS]
    norm_precision_05 = float(np.mean(ncdists <= 0.5))

    return {
        "n_frames": n,
        # primary (ranked + diagnostic)
        "success_auc": round(success_auc, 4),
        "precision_auc": round(precision_auc, 4),
        "norm_precision_auc": round(norm_precision_auc, 4),
        "precision_5": round(precision_5, 4),
        "mean_iou": round(float(np.mean(ious)), 4),
        # legacy / compatibility
        "precision_20": round(precision_20, 4),
        "norm_precision_05": round(norm_precision_05, 4),
        # curves
        "success_plot": {f"{t:.2f}": round(v, 4) for t, v in zip(SUCCESS_THRESHOLDS, success_curve)},
        "precision_plot": {str(int(t)): round(v, 4) for t, v in zip(PRECISION_THRESHOLDS, precision_curve)},
        "norm_precision_plot": {f"{t:.3f}": round(v, 4) for t, v in zip(NORM_PRECISION_THRESHOLDS, nprec_curve)},
        "precision_plot_otb": {str(int(t)): round(v, 4) for t, v in zip(PRECISION_THRESHOLDS_OTB, precision_curve_otb)},
        "ootb_precision_plot": {str(int(t)): round(v, 4) for t, v in zip(PRECISION_THRESHOLDS, precision_curve)},
        "ootb_norm_precision_plot": {f"{t:.2f}": round(v, 4) for t, v in zip(OOTB_NORM_PRECISION_THRESHOLDS, nprec_curve_full)},
    }


def _iou_1_vs_n(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of one box (4,) against N boxes (N, 4). All xyxy format."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_a + area_b - inter
    return inter / np.clip(union, 1e-6, None)
