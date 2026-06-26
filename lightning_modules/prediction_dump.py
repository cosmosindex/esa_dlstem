"""
VideoPredictionDumpCallback
===========================
Dumps every test frame's tracker predictions to a single ``predictions.json``,
matching the schema produced by ``evaluation/eval_birdsai_detect_track.py`` so
the JSON aligns across models for the later combined visualization.

Subscribes to ``on_test_batch_end``, pulls predictions out of the
``frame_result`` dicts produced by ``VideoTrackerEvaluationModule``, buffers
them per video, and at ``on_test_end`` writes one ``predictions.json``::

    {
      "model": ..., "dataset": ..., "split": ..., "class_names": {id: name},
      "videos": {
        "<video_id>": {
          "image_dir": ...,
          "frames": {
            "<frame_id>": {
              "image_path": ...,
              "detections": {"boxes": [[x1,y1,x2,y2],...], "scores": [...], "labels": [...]},
              "tracks":     {"boxes": [...], "scores": [...], "labels": [...], "track_ids": [...]}
            }
          }
        }
      }
    }

Detector+tracker models (e.g. SAM3 text tracker) emit tracked detections
directly — there is no separate raw-detection stage — so ``detections`` and
``tracks`` carry the same boxes; ``tracks`` additionally records ``track_ids``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import lightning as L
from torch.utils.data import ConcatDataset


def _to_np(x) -> np.ndarray:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class VideoPredictionDumpCallback(L.Callback):
    """Dump per-frame tracker predictions to predictions.json.

    Args:
        output_dir:        Run directory; predictions.json is written here.
        class_names:       {class_id: name} recorded in the JSON header so the
                           dump is self-describing (SAM3 coarse 2-class differs
                           from the detectors' fine 5-class taxonomy).
        model_name:        Recorded under "model".
        dataset / split:   Recorded in the JSON header.
        score_thresh:      Drop predictions with score < threshold before dump.
        frame_filename_fmt:Pattern to rebuild each frame's image_path under the
                           video's image_dir (BIRDSAI: "{video_id}_{frame_id:010d}.jpg").
    """

    def __init__(
        self,
        output_dir: str | Path,
        class_names: dict,
        model_name: str = "sam3",
        dataset: str = "BIRDSAI",
        split: str = "test",
        score_thresh: float = 0.0,
        frame_filename_fmt: str = "{video_id}_{frame_id:010d}.jpg",
    ):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.class_names = {int(k): v for k, v in class_names.items()}
        self.model_name = model_name
        self.dataset = dataset
        self.split = split
        self.score_thresh = float(score_thresh)
        self.frame_filename_fmt = frame_filename_fmt

        # video_id -> {frame_id(str) -> frame_dict}
        self._buf: dict[str, dict[str, dict]] = {}
        # video_id -> image_dir str (resolved from the dataset)
        self._image_dirs: dict[str, str] = {}

    # ------------------------------------------------------------------

    def on_test_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        self._buf.clear()
        self._image_dirs.clear()
        # Resolve per-video image dirs from the underlying dataset(s).
        dm = getattr(trainer, "datamodule", None)
        ds = getattr(dm, "test_dataset", None) if dm is not None else None
        if ds is None:
            return
        parts = ds.datasets if isinstance(ds, ConcatDataset) else [ds]
        for part in parts:
            cache = getattr(part, "_img_dir_cache", None)
            if cache:
                for vid, d in cache.items():
                    self._image_dirs[vid] = str(d)

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
        for fr in outputs:
            video_id = fr.get("video_id")
            if video_id is None:
                continue
            frame_id = int(fr["frame_id"])
            pred = fr["pred"]

            boxes = _to_np(pred.get("boxes"))
            scores = _to_np(pred.get("scores"))
            labels = _to_np(pred.get("labels"))
            tids = _to_np(pred.get("track_ids"))

            if boxes is None or len(boxes) == 0:
                boxes = np.zeros((0, 4), dtype=np.float32)
                scores = np.zeros((0,), dtype=np.float32)
                labels = np.zeros((0,), dtype=np.int64)
                tids = np.zeros((0,), dtype=np.int64)
            else:
                n = len(boxes)
                if scores is None:
                    scores = np.ones((n,), dtype=np.float32)
                if labels is None:
                    labels = np.zeros((n,), dtype=np.int64)
                if tids is None:
                    tids = np.arange(n, dtype=np.int64)
                keep = scores >= self.score_thresh
                boxes, scores, labels, tids = boxes[keep], scores[keep], labels[keep], tids[keep]

            boxes_l = np.asarray(boxes, dtype=np.float32).round(2).tolist()
            scores_l = np.asarray(scores, dtype=np.float32).round(4).tolist()
            labels_l = np.asarray(labels, dtype=np.int64).tolist()
            tids_l = np.asarray(tids, dtype=np.int64).tolist()

            img_dir = self._image_dirs.get(video_id)
            image_path = (
                str(Path(img_dir) / self.frame_filename_fmt.format(
                    video_id=video_id, frame_id=frame_id))
                if img_dir else ""
            )

            self._buf.setdefault(video_id, {})[str(frame_id)] = {
                "image_path": image_path,
                "detections": {"boxes": boxes_l, "scores": scores_l, "labels": labels_l},
                "tracks": {
                    "boxes": boxes_l, "scores": scores_l,
                    "labels": labels_l, "track_ids": tids_l,
                },
            }

    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        predictions = {
            "model": self.model_name,
            "dataset": self.dataset,
            "split": self.split,
            "class_names": self.class_names,
            "videos": {
                vid: {
                    "image_dir": self._image_dirs.get(vid, ""),
                    "frames": frames,
                }
                for vid, frames in self._buf.items()
            },
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / "predictions.json", "w") as f:
            json.dump(predictions, f)
        print(f"predictions.json → {self.output_dir / 'predictions.json'} "
              f"({len(self._buf)} videos)")
