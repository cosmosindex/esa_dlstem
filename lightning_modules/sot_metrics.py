"""
SOT (Single Object Tracking) Metrics
=====================================
Standalone metrics class for SOT evaluation.
No Lightning dependency — can be used anywhere.

Standard SOT metrics:
  - Success Plot:    fraction of frames with IoU >= threshold, for threshold in [0, 1]
                     Reported as AUC (area under the success curve).
  - Precision Plot:  fraction of frames with center distance <= threshold, for threshold in [0, 50] px
                     Reported as Precision@20 (value at 20 px threshold).

Reference: OTB benchmark (Wu et al., 2015)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


# Evaluation thresholds
SUCCESS_THRESHOLDS = np.linspace(0, 1, 21)         # 0.00, 0.05, ..., 1.00
PRECISION_THRESHOLDS = np.arange(0, 51, dtype=float)  # 0, 1, ..., 50 px

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
    """Accumulate per-frame SOT records and compute Success/Precision metrics."""

    def __init__(self, class_names: dict[int, str] | None = None):
        self.class_names = class_names or {}
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
    ):
        """Process one frame: for each GT, find the best same-class prediction."""
        for gi in range(len(gt_boxes)):
            gt_box = gt_boxes[gi]
            gt_cls = int(gt_labels[gi])
            gt_name = self.class_names.get(gt_cls, f"cls{gt_cls}")

            # Filter to same class
            same_cls = np.where(pred_labels == gt_cls)[0]

            gt_area = float((gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1]))
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
            sc_boxes = pred_boxes[same_cls]
            ious = _iou_1_vs_n(gt_box, sc_boxes)
            best_idx = int(ious.argmax())
            best_iou = float(ious[best_idx])
            best_box = sc_boxes[best_idx]

            # Center distance
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
        """Compute overall + per-category + per-size SOT metrics."""
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

        return paths


# ======================================================================
# Private helpers
# ======================================================================

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
    p20_all = float(np.mean(cdists_all <= 20))
    ax.plot(PRECISION_THRESHOLDS, prec_all, "k-", linewidth=2,
            label=f"Overall [{p20_all:.3f}]")

    for name, recs in sorted(groups.items()):
        cdists = np.array([r.center_dist for r in recs])
        curve = np.array([np.mean(cdists <= t) for t in PRECISION_THRESHOLDS])
        p20 = float(np.mean(cdists <= 20))
        ax.plot(PRECISION_THRESHOLDS, curve, "--", linewidth=1.5,
                label=f"{name} [{p20:.3f}]")

    ax.set_xlabel("Center location error (pixels)")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def _compute_from_records(records: list[SOTRecord]) -> dict:
    """Compute SOT summary from a list of records."""
    n = len(records)
    ious = np.array([r.best_iou for r in records])
    cdists = np.array([r.center_dist for r in records])

    # Success plot + AUC
    success_curve = [float(np.mean(ious >= t)) for t in SUCCESS_THRESHOLDS]
    success_auc = float(np.mean(success_curve))

    # Precision plot + P@20
    precision_curve = [float(np.mean(cdists <= t)) for t in PRECISION_THRESHOLDS]
    precision_20 = float(np.mean(cdists <= 20))

    return {
        "n_frames": n,
        "success_auc": round(success_auc, 4),
        "precision_20": round(precision_20, 4),
        "mean_iou": round(float(np.mean(ious)), 4),
        "success_plot": {f"{t:.2f}": round(v, 4) for t, v in zip(SUCCESS_THRESHOLDS, success_curve)},
        "precision_plot": {str(int(t)): round(v, 4) for t, v in zip(PRECISION_THRESHOLDS, precision_curve)},
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
