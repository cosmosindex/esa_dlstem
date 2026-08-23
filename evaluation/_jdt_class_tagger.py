"""Recover a per-track class label from a class-agnostic JDT association.

Shared by ``eval_fairmot.py`` and ``eval_tgram.py``.

The 4-class union models *are* class-aware on the detection side —
``mot_decode``'s ``_topk`` returns the heatmap channel index per peak, which
arrives as column 5 of the decoded ``[K, 6]`` array. Association, however, is
class-agnostic **by design**: FairMOT and TGraM share ONE re-ID embedding across
all classes and never gate matching on class, so the label is not carried on the
track and cannot be read off afterwards.

So we re-attach it: every frame each online track is matched back to the
detection it was updated from (IoU over the decoded boxes — the Kalman update
copies the matched detection's box verbatim, so the winner is a near-exact
overlap), and each track's final label is the **majority vote** over its
lifetime. Coasted tracks (predicted, no detection this frame) match nothing and
simply cast no vote.
"""
from __future__ import annotations

import numpy as np

# The 4-class union models' heatmap channel order
# (`{FairMOT,TGraM}/src/lib/cfg/union_all.json` → `tools/export_mot_jde.py`).
MODEL_CLASS_NAMES = {0: "car", 1: "airplane", 2: "ship", 3: "train"}

# Model class name → a dataset's own category name (VISO calls an airplane "plane").
MODEL_TO_DATASET_CLASS = {
    "viso_nocar":  {"airplane": "plane"},
    "viso_no_car": {"airplane": "plane"},
}


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


class ClassTagger:
    """Majority-vote a class label per track id. One instance per sequence."""

    def __init__(self, iou_thr: float = 0.5):
        self.iou_thr = iou_thr
        self.votes: dict[int, dict[int, int]] = {}

    def vote(self, pred_boxes, pred_ids, det_rows, conf_thres: float) -> None:
        """``det_rows`` is the stashed ``[K, 6]`` decode (col 4 score, col 5 class)."""
        if det_rows is None or len(det_rows) == 0 or len(pred_boxes) == 0:
            return
        det_rows = det_rows[det_rows[:, 4] > conf_thres]
        if len(det_rows) == 0:
            return
        ious = _iou_matrix(np.asarray(pred_boxes, dtype=np.float32),
                           det_rows[:, :4].astype(np.float32))
        best = ious.argmax(axis=1)
        for i, tid in enumerate(pred_ids):
            if ious[i, best[i]] < self.iou_thr:
                continue
            cls_id = int(det_rows[best[i], 5])
            v = self.votes.setdefault(int(tid), {})
            v[cls_id] = v.get(cls_id, 0) + 1

    def label(self, track_id: int) -> str | None:
        v = self.votes.get(int(track_id))
        if not v:
            return None
        # Ties broken by the lower class id, for determinism.
        cls_id = sorted(v.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        return MODEL_CLASS_NAMES.get(cls_id)
