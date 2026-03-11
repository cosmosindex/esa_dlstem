"""
SOT Evaluation Callbacks
========================
Lightning Callbacks that evaluate Single Object Tracking metrics
(Success AUC, Precision@20) during testing.

Fully independent from ObjectDetectionModule and SAM2EvaluationModule —
just add the callback to the Trainer's callback list.

Two variants:
  SOTEvalCallback      — for FasterRCNN / YOLO (per-frame detection batches)
  SAM2SOTEvalCallback  — for SAM2 (video clip batches, reads test_step outputs)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import lightning as L

from .sot_metrics import SOTMetrics


class SOTEvalCallback(L.Callback):
    """
    SOT evaluation callback for detection models (FasterRCNN, YOLO).

    Runs model inference on each test batch, computes per-frame SOT metrics,
    and at epoch end logs Success AUC / Precision@20 and generates plots.

    Args:
        class_names:    Dict mapping class id -> display name, e.g. {0: "car"}.
        output_dir:     Directory to save plots and metrics JSON.
        score_thresh:   Only consider predictions with score >= this value.
    """

    def __init__(
        self,
        class_names: dict[int, str],
        output_dir: str | Path = "experiments",
        score_thresh: float = 0.5,
    ):
        super().__init__()
        self.class_names = class_names
        self.output_dir = Path(output_dir)
        self.score_thresh = score_thresh
        self.sot = SOTMetrics(class_names=class_names)

    def on_test_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        self.sot.reset()

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        images, targets = batch

        pl_module.model.eval()
        with torch.no_grad():
            preds = pl_module.model(images)

        for pred, tgt in zip(preds, targets):
            video_id = tgt.get("video_id", f"batch{batch_idx}")
            frame_id = tgt.get("frame_id", 0)

            # To numpy
            gt_boxes = tgt["boxes"].cpu().numpy()
            gt_labels = tgt["labels"].cpu().numpy()

            pred_boxes = pred["boxes"].cpu().numpy()
            pred_scores = pred["scores"].cpu().numpy()
            pred_labels = pred["labels"].cpu().numpy()

            # Filter low confidence
            keep = pred_scores >= self.score_thresh
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]
            pred_labels = pred_labels[keep]

            self.sot.update(
                gt_boxes, gt_labels,
                pred_boxes, pred_scores, pred_labels,
                video_id=str(video_id),
                frame_id=int(frame_id),
            )

    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        result = self.sot.compute()
        if not result:
            return

        _log_and_save(trainer, pl_module, result, self.sot, self.output_dir)


class SAM2SOTEvalCallback(L.Callback):
    """
    SOT evaluation callback for SAM2 (video clip batches).

    Reads per-frame results returned by SAM2EvaluationModule.test_step()
    and computes SOT metrics.

    Args:
        class_names:    Dict mapping class id -> display name.
        output_dir:     Directory to save plots and metrics JSON.
        score_thresh:   Only consider predictions with score >= this value.
    """

    def __init__(
        self,
        class_names: dict[int, str],
        output_dir: str | Path = "experiments",
        score_thresh: float = 0.5,
    ):
        super().__init__()
        self.class_names = class_names
        self.output_dir = Path(output_dir)
        self.score_thresh = score_thresh
        self.sot = SOTMetrics(class_names=class_names)

    def on_test_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        self.sot.reset()

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        if outputs is None:
            return

        for frame_result in outputs:
            pred = frame_result["pred"]
            tgt = frame_result["target"]
            video_id = frame_result["video_id"]
            frame_id = frame_result["frame_id"]

            gt_boxes = _to_numpy(tgt["boxes"])
            gt_labels = _to_numpy(tgt["labels"])

            pred_boxes = _to_numpy(pred["boxes"])
            pred_scores = _to_numpy(pred["scores"])
            pred_labels = _to_numpy(pred["labels"])

            # Filter low confidence
            keep = pred_scores >= self.score_thresh
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]
            pred_labels = pred_labels[keep]

            self.sot.update(
                gt_boxes, gt_labels,
                pred_boxes, pred_scores, pred_labels,
                video_id=str(video_id),
                frame_id=int(frame_id),
            )

    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        result = self.sot.compute()
        if not result:
            return

        _log_and_save(trainer, pl_module, result, self.sot, self.output_dir)


# ======================================================================
# Helpers
# ======================================================================

def _log_and_save(
    trainer: L.Trainer,
    pl_module: L.LightningModule,
    result: dict,
    sot: SOTMetrics,
    output_dir: Path,
):
    """Shared logging, JSON export, and plot generation for both callbacks."""
    # Overall
    pl_module.log("test/sot_success_auc", result["success_auc"], prog_bar=True)
    pl_module.log("test/sot_precision_20", result["precision_20"], prog_bar=True)
    pl_module.log("test/sot_mean_iou", result["mean_iou"])

    # Per-category
    for name, cat_result in result.get("per_category", {}).items():
        pl_module.log(f"test/sot_success_auc_{name}", cat_result["success_auc"])
        pl_module.log(f"test/sot_precision_20_{name}", cat_result["precision_20"])

    # Per-size
    for name, size_result in result.get("per_size", {}).items():
        pl_module.log(f"test/sot_success_auc_{name}", size_result["success_auc"])
        pl_module.log(f"test/sot_precision_20_{name}", size_result["precision_20"])

    # Save JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "sot_metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    # Generate plots
    plot_paths = sot.plot(output_dir)

    # Log plots to W&B if available
    _log_plots_wandb(trainer, plot_paths)


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return np.asarray(x)


def _log_plots_wandb(trainer: L.Trainer, plot_paths: dict[str, Path]):
    """Log plot images to W&B if a WandbLogger is attached."""
    logger = trainer.logger
    if logger is None:
        return
    if not hasattr(logger, "experiment") or not hasattr(logger.experiment, "log"):
        return

    try:
        import wandb
        for name, path in plot_paths.items():
            logger.experiment.log({
                f"test/{name}": wandb.Image(str(path)),
            })
    except ImportError:
        pass
