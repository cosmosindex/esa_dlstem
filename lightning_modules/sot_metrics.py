"""
SOT (Single Object Tracking) Metrics
=====================================
Standalone metrics class for SOT evaluation.
No Lightning dependency — can be used anywhere.

Standard SOT metrics (all aggregated **per-sequence** — see below):
  - Success Plot (SR):    fraction of frames with IoU >= threshold, threshold in [0, 1].
                          Reported as AUC of the success curve.
  - Precision Plot (PR):  fraction of frames with centre distance <= threshold.
                          Reported as AUC of the precision curve over [0, 50] px,
                          following the OTB precision-plot convention (Wu et al.,
                          TPAMI 2015). We deviate from OOTB's [0, 30] px range
                          to (a) avoid PR saturation on relatively larger targets
                          in OOTB and (b) keep PR numerically comparable across
                          datasets that span heterogeneous target-size
                          distributions. A dedicated tiny-object subset analysis
                          (see ``docs/sot_benchmark_tables.md``) reports P@5 as
                          the primary precision metric for sub-8-px targets.
  - Normalised Precision (NPR): fraction of frames with centre distance / GT
                                diagonal <= threshold. Reported as AUC over
                                [0, 0.5], following the GOT-10k / LaSOT-ext /
                                TrackingNet normalised-precision convention.
  - P@5 (diagnostic):     fraction of frames with CLE < 5 px, single-threshold
                          scalar, mirroring OOTB's primary precision metric.

Aggregation: **per-sequence**
-----------------------------
All scalars and curves are computed independently for each sequence, then
arithmetic-mean across sequences (equal weight per sequence). This matches the
OTB / LaSOT / GOT-10k / TrackingNet / VOT / OOTB convention and avoids long
sequences dominating the headline numbers (which would be the effect of
per-frame pooling). See ``docs/sot_benchmark_tables.md`` and
``tools/reaggregate_sot_per_sequence.py`` for the rationale and an offline
re-aggregation tool that applies the same protocol to historical
``per_image_metrics.json`` files.

References: OTB (Wu et al., 2015); OOTB (Chen et al., ISPRS 2024); LaSOT
(Fan et al., CVPR 2019); GOT-10k (Huang et al., TPAMI 2021); TrackingNet
(Müller et al., ECCV 2018).

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
# PR (precision): AUC over centre-location error ∈ [0, 50] px — follows the
#     OTB precision-plot convention (Wu et al., TPAMI 2015), which is the
#     range adopted by LaSOT / GOT-10k / TrackingNet for cross-dataset
#     comparability. See ``docs/sot_benchmark_tables.md``.
# NPR (normalised precision): AUC over normalised centre distance ∈ [0, 0.5],
#     matching the GOT-10k / LaSOT-ext / TrackingNet normalised-precision
#     convention.
SUCCESS_THRESHOLDS = np.linspace(0, 1, 21)                 # 0.00, 0.05, …, 1.00
PRECISION_THRESHOLDS = np.arange(0, 51, dtype=float)        # 0, 1, …, 50 px (PR)
NORM_PRECISION_THRESHOLDS = np.linspace(0, 0.5, 21)         # 0.000, 0.025, …, 0.500 (NPR)

# Back-compat aliases (OOTB's [0, 30] PR range / OOTB's [0, 1] NP range).
PRECISION_THRESHOLDS_OOTB = np.arange(0, 31, dtype=float)   # 0, 1, …, 30 px (legacy)
PRECISION_THRESHOLDS_OTB = PRECISION_THRESHOLDS              # alias — OTB = primary
OOTB_PRECISION_THRESHOLDS = PRECISION_THRESHOLDS_OOTB       # legacy alias
OOTB_NORM_PRECISION_THRESHOLDS = np.linspace(0, 1, 21)      # 0.00, 0.05, …, 1.00

# Sequence-level tiny-object cutoff: median sqrt(GT area) < 8 px (≡ median
# area < 64 px²). Used to surface a tiny-object subset alongside the overall
# numbers — see ``docs/sot_benchmark_tables.md``.
TINY_SQRT_AREA_THRESH = 8.0

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
        tiny_video_ids: set[str] | None = None,
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
        # video_ids whose median sqrt(GT area) is below TINY_SQRT_AREA_THRESH.
        # If provided, ``compute()`` adds a ``tiny`` subgroup alongside the
        # overall and per-category breakdowns. Caller computes this from the
        # dataset's GT (the metrics class itself sees only post-IoU records).
        self.tiny_video_ids: set[str] = set(tiny_video_ids or ())
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
        """Compute overall + per-category + per-size + per-attribute + tiny.

        All groupings are aggregated **per-sequence**: each sequence in the
        group contributes one curve / one scalar, and the group reports the
        equal-weight arithmetic mean across sequences.
        """
        if not self.records:
            return {}

        result = _compute_from_records(self.records)

        # Per category — sequence's class is its dominant per-frame label.
        by_class_seqs = _group_sequences_by_dominant_class(self.records)
        result["per_category"] = {
            name: _compute_from_sequences(seqs)
            for name, seqs in sorted(by_class_seqs.items())
        }

        # Per size (small < 32x32 = 1024 px², large >= 1024 px²) — sequence
        # inherits its first-record size key (size is essentially constant
        # within a sequence in this codebase's datasets).
        by_size_seqs = _group_sequences_by_size(self.records)
        result["per_size"] = {
            name: _compute_from_sequences(seqs)
            for name, seqs in sorted(by_size_seqs.items())
        }

        # Per sequence attribute (multi-label: a sequence may appear in
        # several attribute groups, so groups do NOT partition the records).
        if self.sequence_attributes:
            by_attr_seqs = _group_sequences_by_attribute(
                self.records, self.sequence_attributes,
            )
            if by_attr_seqs:
                result["per_sequence_attribute"] = {
                    name: _compute_from_sequences(seqs)
                    for name, seqs in sorted(by_attr_seqs.items())
                }

        # Tiny subset (sequence-level: median sqrt(GT area) < 8 px). Caller
        # supplies the video_ids — see ``tools/reaggregate_sot_per_sequence.py``
        # for an example computation.
        if self.tiny_video_ids:
            tiny_recs = [r for r in self.records if r.video_id in self.tiny_video_ids]
            if tiny_recs:
                result["tiny"] = _compute_from_records(tiny_recs)

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

def _group_records_by_video(
    records: list[SOTRecord],
) -> dict[str, list[SOTRecord]]:
    """Group records by video_id, preserving order within each sequence."""
    by_seq: dict[str, list[SOTRecord]] = {}
    for r in records:
        by_seq.setdefault(r.video_id, []).append(r)
    return by_seq


def _group_sequences_by_dominant_class(
    records: list[SOTRecord],
) -> dict[str, list[list[SOTRecord]]]:
    """Bucket whole sequences by each sequence's most-frequent gt_class.

    Returns ``{class_name: [sequence_records, ...]}``. Ties on dominant class
    are broken alphabetically.
    """
    by_seq = _group_records_by_video(records)
    out: dict[str, list[list[SOTRecord]]] = {}
    for vid, recs in by_seq.items():
        votes: dict[str, int] = {}
        for r in recs:
            votes[r.gt_class] = votes.get(r.gt_class, 0) + 1
        if not votes:
            continue
        dominant = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        out.setdefault(dominant, []).append(recs)
    return out


def _group_sequences_by_size(
    records: list[SOTRecord],
) -> dict[str, list[list[SOTRecord]]]:
    """Bucket whole sequences by the first record's gt_size key."""
    by_seq = _group_records_by_video(records)
    out: dict[str, list[list[SOTRecord]]] = {}
    for vid, recs in by_seq.items():
        if not recs:
            continue
        out.setdefault(recs[0].gt_size, []).append(recs)
    return out


def _group_sequences_by_attribute(
    records: list[SOTRecord],
    sequence_attributes: dict[str, list[str]],
) -> dict[str, list[list[SOTRecord]]]:
    """Multi-label sequence groupings by sequence-level attribute.

    For each sequence, look up its ``video_id`` in ``sequence_attributes`` and
    append the whole sequence to every attribute group named in that list.
    Sequences whose video carries no attributes don't contribute.
    """
    by_seq = _group_records_by_video(records)
    out: dict[str, list[list[SOTRecord]]] = {}
    for vid, recs in by_seq.items():
        for attr in sequence_attributes.get(vid, ()):
            out.setdefault(attr, []).append(recs)
    return out


def _group_by_attribute(
    records: list[SOTRecord],
    sequence_attributes: dict[str, list[str]],
) -> dict[str, list[SOTRecord]]:
    """Flat per-attribute record groupings (multi-label).

    Sister of ``_group_sequences_by_attribute`` returning a flat list of
    records per attribute — the shape expected by ``_plot_success`` /
    ``_plot_precision`` (``dict[str, list[SOTRecord]]``).
    """
    out: dict[str, list[SOTRecord]] = {}
    for r in records:
        for attr in sequence_attributes.get(r.video_id, ()):
            out.setdefault(attr, []).append(r)
    return out


def _per_seq_curve(
    records: list[SOTRecord],
    metric: str,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Per-sequence-averaged curve for ``records``.

    For each sequence in ``records`` (grouped by ``video_id``), compute the
    threshold curve (``IoU >= t`` for success, ``CLE <= t`` for precision /
    norm-precision), then average across sequences.
    """
    by_seq = _group_records_by_video(records)
    if not by_seq:
        return np.zeros_like(thresholds)
    per_seq = []
    for recs in by_seq.values():
        if metric == "success":
            vals = np.array([r.best_iou for r in recs])
            per_seq.append(np.array([float(np.mean(vals >= t)) for t in thresholds]))
        elif metric == "precision":
            vals = np.array([r.center_dist for r in recs])
            per_seq.append(np.array([float(np.mean(vals <= t)) for t in thresholds]))
        elif metric == "norm_precision":
            vals = np.array([r.norm_center_dist for r in recs])
            per_seq.append(np.array([float(np.mean(vals <= t)) for t in thresholds]))
        else:
            raise ValueError(f"unknown metric {metric!r}")
    return np.mean(per_seq, axis=0)


def _plot_success(
    all_records: list[SOTRecord],
    groups: dict[str, list[SOTRecord]],
    save_path: Path,
    title: str,
) -> Path:
    """Generate a Success Plot with overall + per-group curves (per-sequence)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))

    success_all = _per_seq_curve(all_records, "success", SUCCESS_THRESHOLDS)
    auc_all = float(success_all.mean())
    ax.plot(SUCCESS_THRESHOLDS, success_all, "k-", linewidth=2,
            label=f"Overall [{auc_all:.3f}]")

    for name, recs in sorted(groups.items()):
        curve = _per_seq_curve(recs, "success", SUCCESS_THRESHOLDS)
        ax.plot(SUCCESS_THRESHOLDS, curve, "--", linewidth=1.5,
                label=f"{name} [{float(curve.mean()):.3f}]")

    ax.set_xlabel("Overlap threshold")
    ax.set_ylabel("Success rate")
    ax.set_title(f"{title} — per-sequence average")
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
    """Generate a Precision Plot with overall + per-group curves (per-sequence)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))

    prec_all = _per_seq_curve(all_records, "precision", PRECISION_THRESHOLDS)
    pr_auc_all = float(prec_all.mean())
    ax.plot(PRECISION_THRESHOLDS, prec_all, "k-", linewidth=2,
            label=f"Overall [PR={pr_auc_all:.3f}]")

    for name, recs in sorted(groups.items()):
        curve = _per_seq_curve(recs, "precision", PRECISION_THRESHOLDS)
        ax.plot(PRECISION_THRESHOLDS, curve, "--", linewidth=1.5,
                label=f"{name} [PR={float(curve.mean()):.3f}]")

    ax.set_xlabel("Center location error (pixels)")
    ax.set_ylabel("Precision")
    ax.set_title(
        f"{title} — per-sequence average, PR AUC over 0–{int(PRECISION_THRESHOLDS[-1])} px"
    )
    ax.set_xlim(0, PRECISION_THRESHOLDS[-1])
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def _seq_curves(records: list[SOTRecord]) -> dict:
    """Per-sequence summary: curves + scalars on this sequence alone."""
    ious = np.array([r.best_iou for r in records], dtype=np.float64)
    cdists = np.array([r.center_dist for r in records], dtype=np.float64)
    ncdists = np.array([r.norm_center_dist for r in records], dtype=np.float64)

    success_curve = np.array([float(np.mean(ious >= t)) for t in SUCCESS_THRESHOLDS])
    precision_curve = np.array([float(np.mean(cdists <= t)) for t in PRECISION_THRESHOLDS])
    precision_curve_ootb = np.array([float(np.mean(cdists <= t)) for t in PRECISION_THRESHOLDS_OOTB])
    nprec_curve = np.array([float(np.mean(ncdists <= t)) for t in NORM_PRECISION_THRESHOLDS])
    nprec_curve_full = np.array([float(np.mean(ncdists <= t)) for t in OOTB_NORM_PRECISION_THRESHOLDS])

    return {
        "n_frames": len(records),
        "success_curve": success_curve,
        "precision_curve": precision_curve,
        "precision_curve_ootb": precision_curve_ootb,
        "norm_precision_curve": nprec_curve,
        "norm_precision_curve_full": nprec_curve_full,
        "success_auc": float(success_curve.mean()),
        "precision_auc": float(precision_curve.mean()),
        "precision_auc_ootb": float(precision_curve_ootb.mean()),
        "norm_precision_auc": float(nprec_curve.mean()),
        "precision_5": float(np.mean(cdists <= 5.0)),
        "precision_20": float(np.mean(cdists <= 20.0)),
        "norm_precision_05": float(np.mean(ncdists <= 0.5)),
        "mean_iou": float(np.mean(ious)),
    }


def _compute_from_sequences(seq_records: list[list[SOTRecord]]) -> dict:
    """Per-sequence aggregation entry point.

    Each item in ``seq_records`` is the record list for one sequence. We
    compute the per-sequence summary, then arithmetic-mean across sequences.
    """
    if not seq_records:
        return {}
    summaries = [_seq_curves(recs) for recs in seq_records if recs]
    if not summaries:
        return {}

    def _scalar_mean(key: str) -> float:
        return float(np.mean([s[key] for s in summaries]))

    success_curve = np.mean([s["success_curve"] for s in summaries], axis=0)
    precision_curve = np.mean([s["precision_curve"] for s in summaries], axis=0)
    precision_curve_ootb = np.mean([s["precision_curve_ootb"] for s in summaries], axis=0)
    nprec_curve = np.mean([s["norm_precision_curve"] for s in summaries], axis=0)
    nprec_curve_full = np.mean([s["norm_precision_curve_full"] for s in summaries], axis=0)

    n_frames = sum(s["n_frames"] for s in summaries)

    return {
        "n_frames": n_frames,
        "n_sequences": len(summaries),
        "aggregation": "per_sequence",
        # primary (ranked + diagnostic)
        "success_auc":        round(_scalar_mean("success_auc"), 4),
        "precision_auc":      round(_scalar_mean("precision_auc"), 4),
        "norm_precision_auc": round(_scalar_mean("norm_precision_auc"), 4),
        "precision_5":        round(_scalar_mean("precision_5"), 4),
        "mean_iou":           round(_scalar_mean("mean_iou"), 4),
        # legacy / compatibility scalars
        "precision_20":       round(_scalar_mean("precision_20"), 4),
        "norm_precision_05":  round(_scalar_mean("norm_precision_05"), 4),
        "precision_auc_ootb": round(_scalar_mean("precision_auc_ootb"), 4),
        # curves (means of per-sequence curves)
        "success_plot": {f"{t:.2f}": round(float(v), 4)
                         for t, v in zip(SUCCESS_THRESHOLDS, success_curve)},
        "precision_plot": {str(int(t)): round(float(v), 4)
                           for t, v in zip(PRECISION_THRESHOLDS, precision_curve)},
        "norm_precision_plot": {f"{t:.3f}": round(float(v), 4)
                                for t, v in zip(NORM_PRECISION_THRESHOLDS, nprec_curve)},
        "precision_plot_ootb": {str(int(t)): round(float(v), 4)
                                for t, v in zip(PRECISION_THRESHOLDS_OOTB, precision_curve_ootb)},
        "ootb_norm_precision_plot": {f"{t:.2f}": round(float(v), 4)
                                     for t, v in zip(OOTB_NORM_PRECISION_THRESHOLDS, nprec_curve_full)},
    }


def _compute_from_records(records: list[SOTRecord]) -> dict:
    """Per-sequence aggregation from a flat list of records.

    Groups by ``video_id`` and delegates to ``_compute_from_sequences``.
    """
    if not records:
        return {}
    by_seq = _group_records_by_video(records)
    return _compute_from_sequences(list(by_seq.values()))


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
