"""
Run a tracker over cached HiEUM detections + score against GT.

Pipeline (one tracker × one dataset)::

    for video in dataset.test_videos:
        cache  = load_cache(video.video_id)        # HiEUM dets per frame
        gt     = load_gt_per_frame(dataset, video) # GT (boxes, track_ids)
        tracker.reset()
        for fid in video.frame_ids:
            dets   = cache[fid]                     # [N, 5] xyxy + score
            tracks = tracker.update(dets, fid)      # [M, 6] xyxy+score+id
            score_per_frame(tracks, gt[fid])

Outputs to ``EXPERIMENT_DIR``::

    test_metrics.json          micro Pr/Re/F1 (det), MOTA/IDF1/IDsw (track)
    per_video_metrics.json     same metrics per video, for macro aggregation
    mot_format/<video>.txt     MOTChallenge-format output for TrackEval/HOTA

Usage::

    python eval_tracker.py --config configs/MOT/tracker/sort_rscardata.yaml
"""

from __future__ import annotations

# --- repo root on path so top-level modules (transforms, obb_utils, ...) import ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from datasets.rscardata import RsCarDataset
from datasets.satmtb import SATMTBDataset
from datasets.sdmcar import SDMCarDataset
from models.trackers import build_tracker


_DATASET_TABLE = {
    "rscardata": (
        "RsCarData", RsCarDataset,
        "/data/ESA_DLSTEM_2025/data/trafic/RsCarData",
        {},
    ),
    "satmtb": (
        "SAT-MTB", SATMTBDataset,
        "/data/ESA_DLSTEM_2025/data/trafic/SAT-MTB",
        {"task": "mot", "categories": ["car"]},
    ),
    "sdmcar": (
        "SDM-Car", SDMCarDataset,
        "/data/ESA_DLSTEM_2025/data/trafic/SDM-Car",
        {},
    ),
}


# ----------------------------------------------------------------------
# IO helpers
# ----------------------------------------------------------------------

def _safe_video_id(video_id: str) -> str:
    return video_id.replace("/", "_")


def _load_cache(cache_dir: Path, video_id: str) -> dict:
    path = cache_dir / f"{_safe_video_id(video_id)}.json"
    with open(path) as f:
        return json.load(f)


def _build_dataset(name: str, split: str = "test"):
    if name not in _DATASET_TABLE:
        raise ValueError(f"unknown dataset {name!r}, choose from {list(_DATASET_TABLE)}")
    _, cls, root, extra = _DATASET_TABLE[name]
    return cls(
        root=root, split=split, mode="detection",
        class_map={"car": 0}, **extra,
    )


def _gt_per_frame(dataset, video) -> dict[int, dict]:
    """Pull GT (boxes, track_ids) for every frame of a video.

    Returns ``{frame_id: {"boxes": Nx4 float32, "track_ids": N int64}}``.
    """
    out: dict[int, dict] = {}
    for fid in video.frame_ids:
        ann = dataset._load_annotations(video, fid)
        out[fid] = {
            "boxes":     np.asarray(ann["boxes"], dtype=np.float32),
            "track_ids": np.asarray(ann["track_ids"], dtype=np.int64),
        }
    return out


# ----------------------------------------------------------------------
# Per-frame matching (centroid OR IoU, mirrors VideoTrackerEvaluationModule)
# ----------------------------------------------------------------------

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


def _centroid_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    cx_a = (a[:, 0] + a[:, 2]) * 0.5; cy_a = (a[:, 1] + a[:, 3]) * 0.5
    cx_b = (b[:, 0] + b[:, 2]) * 0.5; cy_b = (b[:, 1] + b[:, 3]) * 0.5
    dx = cx_a[:, None] - cx_b[None, :]; dy = cy_a[:, None] - cy_b[None, :]
    return np.sqrt(dx * dx + dy * dy).astype(np.float32)


def _greedy_match(
    gt_boxes: np.ndarray, pred_boxes: np.ndarray,
    metric: str, iou_thr: float, dist_thr: float,
) -> tuple[list[tuple[int, int]], int]:
    """Greedy GT → pred matching. Returns (matches, n_matched)."""
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return [], 0
    if metric == "centroid":
        dist = _centroid_dist(gt_boxes, pred_boxes)
        accept = dist <= dist_thr
        score = dist
        descending = False
    else:
        iou = _iou_matrix(gt_boxes, pred_boxes)
        accept = iou >= iou_thr
        score = iou
        descending = True

    rows, cols = np.where(accept)
    if len(rows) == 0:
        return [], 0
    order = score[rows, cols].argsort()
    if descending:
        order = order[::-1]
    rows, cols = rows[order], cols[order]
    matched_gt: set[int] = set(); matched_pred: set[int] = set()
    matches: list[tuple[int, int]] = []
    for r, c in zip(rows.tolist(), cols.tolist()):
        if r in matched_gt or c in matched_pred:
            continue
        matched_gt.add(r); matched_pred.add(c)
        matches.append((r, c))
    return matches, len(matches)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dataset_key   = cfg["dataset"]
    tracker_name  = cfg["tracker"]
    tracker_cfg   = cfg.get("tracker_kwargs", {})
    cache_dir     = Path(cfg["cache_dir"])
    metric        = cfg.get("match_metric", "centroid")
    iou_thr       = float(cfg.get("iou_thresh", 0.5))
    dist_thr      = float(cfg.get("centroid_dist_thresh", 5.0))
    score_floor   = float(cfg.get("score_floor", 0.0))

    exp_root = os.environ.get("EXPERIMENT_ROOT", "/work/ziwen/experiments")
    run_name = f"{tracker_name}_{dataset_key}"
    experiment_dir = Path(f"{exp_root}/{run_name}_{datetime.now():%Y%m%d_%H%M%S}")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    mot_dir = experiment_dir / "mot_format"
    mot_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print(f"Tracker eval: {tracker_name} on {dataset_key}")
    print(f"Cache:    {cache_dir}")
    print(f"Output:   {experiment_dir}")
    print(f"Match:    {metric} (iou≥{iou_thr} / dist≤{dist_thr}px)")
    print("=" * 60)

    dataset = _build_dataset(dataset_key)
    tracker = build_tracker(tracker_name, **tracker_cfg)

    # Global accumulators (micro)
    det_tp = det_fp = det_fn = 0
    tr_tp = tr_fp = tr_fn = id_switches = 0
    total_gt = 0
    last_gt_to_pred: dict[int, int] = {}

    per_video: dict[str, dict] = {}
    t_total_start = time.perf_counter()

    for v_idx, video in enumerate(dataset.videos, 1):
        cache = _load_cache(cache_dir, video.video_id)
        gt = _gt_per_frame(dataset, video)
        tracker.reset()
        last_gt_to_pred = {}   # tracking IDs reset per video

        # Per-video accumulators (macro source)
        v_det_tp = v_det_fp = v_det_fn = 0
        v_tr_tp = v_tr_fp = v_tr_fn = v_id_sw = 0
        v_num_gt = 0

        # MOTChallenge file
        mot_lines: list[str] = []

        # Cache stores parallel lists indexed by per-video frame position.
        cache_frame_ids = cache["frame_ids"]
        cache_boxes = cache["boxes"]
        cache_scores = cache["scores"]
        idx_by_fid = {fid: i for i, fid in enumerate(cache_frame_ids)}

        for fid in video.frame_ids:
            i = idx_by_fid.get(fid)
            if i is None:
                dets = np.zeros((0, 5), dtype=np.float32)
            else:
                if cache_boxes[i]:
                    dets = np.column_stack([
                        np.asarray(cache_boxes[i], dtype=np.float32),
                        np.asarray(cache_scores[i], dtype=np.float32),
                    ])
                else:
                    dets = np.zeros((0, 5), dtype=np.float32)
                if score_floor > 0 and len(dets):
                    dets = dets[dets[:, 4] >= score_floor]

            tracks = tracker.update(dets, frame_id=fid)
            pred_boxes = tracks[:, :4] if len(tracks) else np.zeros((0, 4), dtype=np.float32)
            pred_ids   = tracks[:, 5].astype(np.int64) if len(tracks) else np.zeros(0, dtype=np.int64)
            pred_scores = tracks[:, 4] if len(tracks) else np.zeros(0, dtype=np.float32)

            gt_boxes = gt[fid]["boxes"]
            gt_tids  = gt[fid]["track_ids"]

            # Detection accumulator
            matches, _ = _greedy_match(gt_boxes, pred_boxes, metric, iou_thr, dist_thr)
            tp_d = len(matches)
            v_det_tp += tp_d
            v_det_fp += len(pred_boxes) - tp_d
            v_det_fn += len(gt_boxes) - tp_d

            # Tracking accumulator (same matches, plus ID-switch tally)
            v_num_gt += len(gt_boxes)
            for r, c in matches:
                gt_id = int(gt_tids[r])
                pr_id = int(pred_ids[c])
                prev = last_gt_to_pred.get(gt_id)
                if prev is not None and prev != pr_id:
                    v_id_sw += 1
                last_gt_to_pred[gt_id] = pr_id
                v_tr_tp += 1
            v_tr_fn += len(gt_boxes) - tp_d
            v_tr_fp += len(pred_boxes) - tp_d

            # MOTChallenge dump (frame is 1-indexed, ignore class column)
            for j in range(len(tracks)):
                x1, y1, x2, y2 = pred_boxes[j]
                w, h = float(x2 - x1), float(y2 - y1)
                mot_lines.append(
                    f"{int(fid)},{int(pred_ids[j])},{float(x1):.2f},{float(y1):.2f},"
                    f"{w:.2f},{h:.2f},{float(pred_scores[j]):.4f},-1,-1,-1"
                )

        # Save MOT-format file for this video
        with open(mot_dir / f"{_safe_video_id(video.video_id)}.txt", "w") as f:
            f.write("\n".join(mot_lines))

        # Per-video metrics
        v_prec = v_det_tp / max(v_det_tp + v_det_fp, 1)
        v_rec  = v_det_tp / max(v_det_tp + v_det_fn, 1)
        v_f1   = 2 * v_prec * v_rec / max(v_prec + v_rec, 1e-9)
        v_mota = 1.0 - (v_tr_fp + v_tr_fn + v_id_sw) / max(v_num_gt, 1)
        v_tp = v_tr_tp; v_fp = v_tr_fp; v_fn = v_tr_fn
        v_idp = v_tp / max(v_tp + v_fp, 1)
        v_idr = v_tp / max(v_tp + v_fn, 1)
        v_idf1 = 2 * v_idp * v_idr / max(v_idp + v_idr, 1e-9)

        per_video[video.video_id] = {
            "det": {"tp": v_det_tp, "fp": v_det_fp, "fn": v_det_fn,
                    "Precision": v_prec, "Recall": v_rec, "F1": v_f1},
            "track": {"tp": v_tr_tp, "fp": v_tr_fp, "fn": v_tr_fn,
                      "id_switches": v_id_sw, "num_gt": v_num_gt,
                      "MOTA": v_mota, "IDF1": v_idf1},
        }

        # Accumulate to global
        det_tp += v_det_tp; det_fp += v_det_fp; det_fn += v_det_fn
        tr_tp += v_tr_tp; tr_fp += v_tr_fp; tr_fn += v_tr_fn
        id_switches += v_id_sw; total_gt += v_num_gt

        print(f"[{v_idx}/{len(dataset.videos)}] {video.video_id}  "
              f"F1={v_f1:.3f}  MOTA={v_mota:.3f}  IDF1={v_idf1:.3f}  "
              f"IDsw={v_id_sw}")

    t_total = time.perf_counter() - t_total_start

    # Global (micro) summary
    prec = det_tp / max(det_tp + det_fp, 1)
    rec  = det_tp / max(det_tp + det_fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    mota = 1.0 - (tr_fp + tr_fn + id_switches) / max(total_gt, 1)
    idp  = tr_tp / max(tr_tp + tr_fp, 1)
    idr  = tr_tp / max(tr_tp + tr_fn, 1)
    idf1 = 2 * idp * idr / max(idp + idr, 1e-9)

    # Macro: per-video means (skip empty videos)
    macro_f1 = float(np.mean([m["det"]["F1"]  for m in per_video.values()])) if per_video else 0.0
    macro_pr = float(np.mean([m["det"]["Precision"] for m in per_video.values()])) if per_video else 0.0
    macro_re = float(np.mean([m["det"]["Recall"]    for m in per_video.values()])) if per_video else 0.0
    macro_mota = float(np.mean([m["track"]["MOTA"] for m in per_video.values()])) if per_video else 0.0
    macro_idf1 = float(np.mean([m["track"]["IDF1"] for m in per_video.values()])) if per_video else 0.0

    summary = {
        "tracker": tracker_name,
        "dataset": dataset_key,
        "match_metric": metric,
        "iou_thresh": iou_thr,
        "centroid_dist_thresh": dist_thr,
        "score_floor": score_floor,
        "total_videos": len(dataset.videos),
        "total_time_s": t_total,
        # Micro (pooled)
        "Precision": prec, "Recall": rec, "F1": f1,
        "MOTA": mota, "IDF1": idf1, "ID_switches": id_switches,
        "num_gt": total_gt,
        # Macro (mean over videos)
        "macro_Precision": macro_pr, "macro_Recall": macro_re, "macro_F1": macro_f1,
        "macro_MOTA": macro_mota, "macro_IDF1": macro_idf1,
    }
    with open(experiment_dir / "test_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(experiment_dir / "per_video_metrics.json", "w") as f:
        json.dump(per_video, f, indent=2)

    print()
    print("=" * 60)
    print("micro:  Pr={:.3f}  Re={:.3f}  F1={:.3f}  MOTA={:.3f}  "
          "IDF1={:.3f}  IDsw={}".format(prec, rec, f1, mota, idf1, id_switches))
    print("macro:  Pr={:.3f}  Re={:.3f}  F1={:.3f}  MOTA={:.3f}  "
          "IDF1={:.3f}".format(macro_pr, macro_re, macro_f1, macro_mota, macro_idf1))
    print(f"time:   {t_total:.1f}s")
    print(f"output: {experiment_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
