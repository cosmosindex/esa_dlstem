"""
MOT-format dump callback for video tracker eval (SAM2 / SAM3).

Subscribes to ``on_test_batch_end``, pulls predictions out of the
``frame_result`` dicts produced by ``VideoTrackerEvaluationModule``,
buffers them per video, and at ``on_test_end`` writes one MOTChallenge
text file per video to ``<output_dir>/mot_format/<safe_video_id>.txt``::

    frame, id, x, y, w, h, conf, -1, -1, -1   (xywh, top-left)

Output is consumed by ``compute_hota.py`` (TrackEval). The frame-id
namespace stays whatever the dataset emits — ``compute_hota.py`` already
applies any 1-indexing offset on its side.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import lightning as L
import numpy as np
import torch


def _to_np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _safe_video_id(video_id: str) -> str:
    return video_id.replace("/", "_")


class MOTFormatDumpCallback(L.Callback):
    """Dump per-video predictions as MOTChallenge text files.

    Args:
        output_dir:  Run directory; files written to ``output_dir/mot_format/``.
        score_thresh: Drop predictions with score < threshold before dumping.
                      Default 0.0 (keep all SAM3 emissions, since the RAFT
                      filter is the next stage and benefits from full input).
    """

    def __init__(self, output_dir: str | Path, score_thresh: float = 0.0):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.score_thresh = float(score_thresh)
        # video_id -> list of (frame_id, track_id, x1, y1, x2, y2, score)
        self._buf: dict[str, list[tuple]] = defaultdict(list)
        # Track every video_id that has been *seen* in test_step results,
        # even if it produced zero predictions. Lets us write empty
        # mot_format files so compute_hota doesn't bail with
        # "Tracker file not found" when the model misses an entire
        # video (e.g. SAM 3.1 multiplex on small-vehicle datasets).
        self._seen_video_ids: set[str] = set()

    def on_test_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        self._buf.clear()
        self._seen_video_ids.clear()

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
            pred = fr.get("pred")
            video_id = fr.get("video_id")
            if video_id is not None:
                self._seen_video_ids.add(video_id)
            if pred is None:
                continue
            frame_id = int(fr["frame_id"])

            boxes = _to_np(pred.get("boxes", np.zeros((0, 4))))
            scores = _to_np(pred.get("scores", np.zeros((0,))))
            tids = pred.get("track_ids")
            if tids is None:
                # No association layer ran (det-only). Skip — HOTA needs ids.
                continue
            tids = _to_np(tids)

            if len(boxes) == 0:
                continue

            keep = scores >= self.score_thresh
            if not keep.any():
                continue
            boxes = boxes[keep]
            scores = scores[keep]
            tids = tids[keep]

            for (x1, y1, x2, y2), sc, tid in zip(boxes, scores, tids):
                self._buf[video_id].append(
                    (frame_id, int(tid), float(x1), float(y1),
                     float(x2), float(y2), float(sc))
                )

    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        out_dir = self.output_dir / "mot_format"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Ensure every seen video_id has a file, even if empty — covers
        # videos where the model produced zero detections in any frame.
        for vid in self._seen_video_ids:
            self._buf.setdefault(vid, [])
        for video_id, rows in self._buf.items():
            rows.sort(key=lambda r: (r[0], r[1]))
            seen: set[tuple[int, int]] = set()
            lines = []
            for frame_id, tid, x1, y1, x2, y2, sc in rows:
                # TrackEval rejects duplicate (frame, id). Keep the higher-
                # confidence row and drop the rest if SAM3's text mode ever
                # emits two boxes for the same global id in one frame.
                key = (frame_id, tid)
                if key in seen:
                    continue
                seen.add(key)
                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    continue
                lines.append(
                    f"{frame_id},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},"
                    f"{sc:.4f},-1,-1,-1"
                )
            (out_dir / f"{_safe_video_id(video_id)}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else "")
            )
