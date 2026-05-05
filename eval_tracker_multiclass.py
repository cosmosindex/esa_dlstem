"""Multi-class variant of eval_tracker.py.

Same design and outputs as eval_tracker.py but adapted to FasterRCNN
detections that carry per-detection class labels (1=airplane, 2=ship,
3=train).  For each (video, class) pair we run an *independent* tracker
instance, then write the resulting tracks to a class-segregated
mot_format directory.  HOTA is therefore computable per class via
``compute_hota.py`` and can be macro-averaged downstream.

Cache schema expected (produced by ``tools/cache_fasterrcnn_dets.py``):

    {
      "video_id":  "...",
      "frame_ids": [int, ...],
      "boxes":     [ [[x1,y1,x2,y2], ...], ... ],   # per-frame
      "scores":    [ [float, ...],          ... ],
      "labels":    [ [int, ...],            ... ],   # per-frame class ids
      ...
    }

Config schema::

    dataset:        airmot | viso_nocar | satmtb_nocar
    tracker:        sort | bytetrack | ocsort | botsort | botsort_reid | tracktrack
    cache_dir:      /path/to/.../<dataset>/
    classes:        {airplane: 1, ship: 2, train: 3}
    score_floor:    0.25
    match_metric:   centroid | iou
    iou_thresh:     0.5
    centroid_dist_thresh: 5.0
    tracker_kwargs: {...}
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from datasets.airmot import AIRMOTDataset
from datasets.satmtb import SATMTBDataset
from datasets.viso import VISODataset
from models.trackers import build_tracker


# Default class-id schema, matching FasterRCNN's training config.
_FRCN_CLASS_MAP_AIRPLANE_SHIP_TRAIN = {"airplane": 1, "ship": 2, "train": 3}
_FRCN_CLASS_MAP_VISO              = {"plane": 1, "ship": 2, "train": 3}
_FRCN_CLASS_MAP_AIRMOT            = {"airplane": 1, "ship": 2}

_DATASET_TABLE = {
    # name → (display, cls, root, build_kwargs, class_map)
    "satmtb_nocar": (
        "SAT-MTB", SATMTBDataset,
        "/data/ESA_DLSTEM_2025/data/trafic/SAT-MTB",
        {"task": "mot", "categories": ["airplane", "ship", "train"]},
        _FRCN_CLASS_MAP_AIRPLANE_SHIP_TRAIN,
    ),
    "viso_nocar": (
        "VISO", VISODataset,
        "/data/ESA_DLSTEM_2025/data/trafic/VISO",
        {"categories": ["plane", "ship", "train"]},
        _FRCN_CLASS_MAP_VISO,
    ),
    "airmot": (
        "AIR-MOT-100", AIRMOTDataset,
        "/data/ESA_DLSTEM_2025/data/trafic/AIR-MOT-100",
        {},
        _FRCN_CLASS_MAP_AIRMOT,
    ),
}

# Default split policy: only sequences FasterRCNN has *not* seen during
# training. SAT-MTB → test; VISO/AIR-MOT → all (no_split).
_DATASET_SPLIT = {
    "satmtb_nocar": "test",
    "viso_nocar":   "no_split",
    "airmot":       "no_split",
}


def _safe_video_id(video_id: str) -> str:
    return video_id.replace("/", "_")


def _load_cache(cache_dir: Path, video_id: str) -> dict:
    path = cache_dir / f"{_safe_video_id(video_id)}.json"
    with open(path) as f:
        return json.load(f)


def _build_dataset(name: str, mode: str = "detection"):
    if name not in _DATASET_TABLE:
        raise ValueError(f"unknown dataset {name!r}")
    _, cls, root, extra, class_map = _DATASET_TABLE[name]
    return cls(
        root=root,
        split=_DATASET_SPLIT[name],
        mode=mode,
        class_map=dict(class_map),
        **extra,
    )


def _gt_per_frame(dataset, video) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for fid in video.frame_ids:
        ann = dataset._load_annotations(video, fid)
        out[fid] = {
            "boxes":     np.asarray(ann["boxes"],     dtype=np.float32),
            "labels":    np.asarray(ann["labels"],    dtype=np.int64),
            "track_ids": np.asarray(ann["track_ids"], dtype=np.int64),
        }
    return out


# ---- match helpers (copied verbatim from eval_tracker.py) -------------

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


def _greedy_match(gt_boxes, pred_boxes, metric, iou_thr, dist_thr):
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return [], 0
    if metric == "centroid":
        score = _centroid_dist(gt_boxes, pred_boxes)
        accept = score <= dist_thr
        descending = False
    else:
        score = _iou_matrix(gt_boxes, pred_boxes)
        accept = score >= iou_thr
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


# ---- main -------------------------------------------------------------


def _empty_acc():
    return {"det_tp": 0, "det_fp": 0, "det_fn": 0,
            "tr_tp": 0,  "tr_fp": 0,  "tr_fn": 0,
            "id_sw": 0,  "num_gt": 0, "n_videos": 0,
            "n_videos_with_gt": 0}


def _summarize(acc):
    prec = acc["det_tp"] / max(acc["det_tp"] + acc["det_fp"], 1)
    rec  = acc["det_tp"] / max(acc["det_tp"] + acc["det_fn"], 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    mota = 1.0 - (acc["tr_fp"] + acc["tr_fn"] + acc["id_sw"]) / max(acc["num_gt"], 1)
    idp  = acc["tr_tp"] / max(acc["tr_tp"] + acc["tr_fp"], 1)
    idr  = acc["tr_tp"] / max(acc["tr_tp"] + acc["tr_fn"], 1)
    idf1 = 2 * idp * idr / max(idp + idr, 1e-9)
    return {"Precision": prec, "Recall": rec, "F1": f1,
            "MOTA": mota, "IDF1": idf1, "ID_switches": acc["id_sw"],
            "num_gt": acc["num_gt"], **{k: acc[k] for k in
                ("det_tp", "det_fp", "det_fn", "tr_tp", "tr_fp", "tr_fn",
                 "n_videos", "n_videos_with_gt")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dataset_key  = cfg["dataset"]
    tracker_name = cfg["tracker"]
    tracker_cfg  = cfg.get("tracker_kwargs", {})
    cache_dir    = Path(cfg["cache_dir"])
    metric       = cfg.get("match_metric", "centroid")
    iou_thr      = float(cfg.get("iou_thresh", 0.5))
    dist_thr     = float(cfg.get("centroid_dist_thresh", 5.0))
    score_floor  = float(cfg.get("score_floor", 0.0))

    classes      = cfg.get("classes") or _DATASET_TABLE[dataset_key][4]   # name → id
    name_by_id   = {v: k for k, v in classes.items()}

    exp_root = os.environ.get("EXPERIMENT_ROOT", "/work/ziwen/experiments")
    run_name = f"{tracker_name}_{dataset_key}"
    experiment_dir = Path(f"{exp_root}/{run_name}_{datetime.now():%Y%m%d_%H%M%S}")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    mot_root = experiment_dir / "mot_format"
    mot_root.mkdir(exist_ok=True)
    for cname in classes.keys():
        (mot_root / cname).mkdir(exist_ok=True)

    print("=" * 60)
    print(f"Multi-class tracker eval: {tracker_name} on {dataset_key}")
    print(f"Classes:  {classes}")
    print(f"Cache:    {cache_dir}")
    print(f"Output:   {experiment_dir}")
    print(f"Match:    {metric} (iou≥{iou_thr} / dist≤{dist_thr}px)")
    print("=" * 60)

    dataset = _build_dataset(dataset_key)
    print(f"[dataset] {len(dataset.videos)} videos")

    # Per-(video, class) and aggregate accumulators
    per_video_class: dict[str, dict[str, dict]] = {}
    per_class_acc: dict[str, dict] = {c: _empty_acc() for c in classes.keys()}
    overall = _empty_acc()
    t0_total = time.perf_counter()

    for v_idx, video in enumerate(dataset.videos, 1):
        cache = _load_cache(cache_dir, video.video_id)
        gt    = _gt_per_frame(dataset, video)

        cache_frame_ids = cache["frame_ids"]
        cache_boxes  = cache["boxes"]
        cache_scores = cache["scores"]
        cache_labels = cache["labels"]
        idx_by_fid = {fid: i for i, fid in enumerate(cache_frame_ids)}

        per_video_class[video.video_id] = {}

        for cname, cid in classes.items():
            tracker = build_tracker(tracker_name, **tracker_cfg)
            tracker.reset()
            mot_lines: list[str] = []

            v_acc = _empty_acc()
            v_acc["n_videos"] = 1
            last_gt_to_pred: dict[int, int] = {}
            has_any_gt = False

            for fid in video.frame_ids:
                # --- detections for this class ---
                i = idx_by_fid.get(fid)
                if i is None or not cache_boxes[i]:
                    dets = np.zeros((0, 5), dtype=np.float32)
                else:
                    bs = np.asarray(cache_boxes[i], dtype=np.float32)
                    sc = np.asarray(cache_scores[i], dtype=np.float32)
                    lb = np.asarray(cache_labels[i], dtype=np.int64)
                    keep = lb == cid
                    if score_floor > 0:
                        keep &= sc >= score_floor
                    bs, sc = bs[keep], sc[keep]
                    if len(sc):
                        dets = np.column_stack([bs, sc])
                    else:
                        dets = np.zeros((0, 5), dtype=np.float32)

                tracks = tracker.update(dets, frame_id=fid)
                pred_boxes  = tracks[:, :4] if len(tracks) else np.zeros((0, 4), dtype=np.float32)
                pred_ids    = tracks[:, 5].astype(np.int64) if len(tracks) else np.zeros(0, dtype=np.int64)
                pred_scores = tracks[:, 4] if len(tracks) else np.zeros(0, dtype=np.float32)

                # --- GT for this class ---
                gt_lab = gt[fid]["labels"]
                gt_keep = gt_lab == cid
                gt_boxes = gt[fid]["boxes"][gt_keep]
                gt_tids  = gt[fid]["track_ids"][gt_keep]
                if len(gt_boxes) > 0:
                    has_any_gt = True

                matches, _ = _greedy_match(gt_boxes, pred_boxes, metric, iou_thr, dist_thr)
                tp = len(matches)
                v_acc["det_tp"] += tp
                v_acc["det_fp"] += len(pred_boxes) - tp
                v_acc["det_fn"] += len(gt_boxes) - tp
                v_acc["num_gt"] += len(gt_boxes)
                for r, c in matches:
                    g_id = int(gt_tids[r]); p_id = int(pred_ids[c])
                    prev = last_gt_to_pred.get(g_id)
                    if prev is not None and prev != p_id:
                        v_acc["id_sw"] += 1
                    last_gt_to_pred[g_id] = p_id
                    v_acc["tr_tp"] += 1
                v_acc["tr_fn"] += len(gt_boxes) - tp
                v_acc["tr_fp"] += len(pred_boxes) - tp

                # MOTChallenge format (per-class file, ignore the "class" col)
                for j in range(len(tracks)):
                    x1, y1, x2, y2 = pred_boxes[j]
                    w, h = float(x2 - x1), float(y2 - y1)
                    mot_lines.append(
                        f"{int(fid)},{int(pred_ids[j])},{float(x1):.2f},{float(y1):.2f},"
                        f"{w:.2f},{h:.2f},{float(pred_scores[j]):.4f},-1,-1,-1"
                    )

            # write per-(class, video) MOT file
            with open(mot_root / cname / f"{_safe_video_id(video.video_id)}.txt", "w") as f:
                f.write("\n".join(mot_lines))

            v_acc["n_videos_with_gt"] = 1 if has_any_gt else 0
            per_video_class[video.video_id][cname] = _summarize(v_acc)

            # accumulate
            for k in ("det_tp", "det_fp", "det_fn", "tr_tp", "tr_fp", "tr_fn",
                     "id_sw", "num_gt"):
                per_class_acc[cname][k] += v_acc[k]
                overall[k]              += v_acc[k]
            per_class_acc[cname]["n_videos"] += 1
            per_class_acc[cname]["n_videos_with_gt"] += v_acc["n_videos_with_gt"]
            overall["n_videos"] += 1
            overall["n_videos_with_gt"] += v_acc["n_videos_with_gt"]

        # video-level summary line
        v_total = sum(per_video_class[video.video_id][c]["num_gt"] for c in classes)
        v_idf1  = np.mean([per_video_class[video.video_id][c]["IDF1"]
                           for c in classes
                           if per_video_class[video.video_id][c]["num_gt"] > 0]) if v_total > 0 else 0.0
        print(f"[{v_idx}/{len(dataset.videos)}] {video.video_id:30s} "
              f"GT={v_total:>5d}  mean-IDF1={v_idf1:.3f}")

    elapsed = time.perf_counter() - t0_total

    summary = {
        "tracker": tracker_name,
        "dataset": dataset_key,
        "classes": classes,
        "match_metric": metric,
        "iou_thresh": iou_thr,
        "centroid_dist_thresh": dist_thr,
        "score_floor": score_floor,
        "total_time_s": elapsed,
        "overall": _summarize(overall),
        "per_class": {c: _summarize(per_class_acc[c]) for c in classes},
    }
    with open(experiment_dir / "test_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(experiment_dir / "per_video_metrics.json", "w") as f:
        json.dump(per_video_class, f, indent=2)

    print()
    print("=" * 60)
    for c in classes:
        s = summary["per_class"][c]
        print(f"[{c:10s}]  Pr={s['Precision']:.3f}  Re={s['Recall']:.3f}  "
              f"F1={s['F1']:.3f}  MOTA={s['MOTA']:.3f}  IDF1={s['IDF1']:.3f}  "
              f"IDsw={s['ID_switches']}  GT={s['num_gt']}")
    o = summary["overall"]
    print(f"[overall  ]  Pr={o['Precision']:.3f}  Re={o['Recall']:.3f}  "
          f"F1={o['F1']:.3f}  MOTA={o['MOTA']:.3f}  IDF1={o['IDF1']:.3f}")
    print(f"time: {elapsed:.1f}s   output: {experiment_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
