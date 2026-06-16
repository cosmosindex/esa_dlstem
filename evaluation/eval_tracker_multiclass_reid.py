"""Multi-class variant of eval_botsort_reid.py / eval_tracktrack.py for the
non-car FasterRCNN pipeline.

Reads the .npz feature cache produced by ``cache_fasterrcnn_dets_with_feats.py``
(extra ``labels`` array carries the class id per detection), runs an
independent tracker instance per (video, class), and writes the resulting
tracks to a class-segregated mot_format directory:
``mot_format/<class_name>/<safe_video_id>.txt``.

Dispatches between BoT-SORT-ReID and TrackTrack based on the
``tracker:`` field of the config; their ``update_with_feats`` signatures
differ (see ``eval_botsort_reid.py`` and ``eval_tracktrack.py``), so we
build the call frame for each separately here.
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

from datasets.airmot import AIRMOTDataset
from datasets.satmtb import SATMTBDataset
from datasets.viso import VISODataset
from models.trackers import build_tracker
from models.trackers.botsort_reid import BoTSortReIDTracker
# eval_tracktrack.py uses ``build_tracker("tracktrack", ...)`` which returns
# the upstream Tracker instance; we mirror that.


_FRCN_AIRPLANE = {"airplane": 1, "ship": 2, "train": 3}
_FRCN_VISO     = {"plane": 1,    "ship": 2, "train": 3}
_FRCN_AIRMOT   = {"airplane": 1, "ship": 2}

_DATASET_TABLE = {
    "satmtb_nocar": (SATMTBDataset, "/data/ESA_DLSTEM_2025/data/trafic/SAT-MTB",
                     {"task": "mot", "categories": ["airplane", "ship", "train"]},
                     "test", _FRCN_AIRPLANE),
    "viso_nocar":   (VISODataset, "/data/ESA_DLSTEM_2025/data/trafic/VISO",
                     {"categories": ["plane", "ship", "train"]},
                     "no_split", _FRCN_VISO),
    "airmot":       (AIRMOTDataset, "/data/ESA_DLSTEM_2025/data/trafic/AIR-MOT-100",
                     {}, "no_split", _FRCN_AIRMOT),
}


def _safe_video_id(video_id: str) -> str:
    return video_id.replace("/", "_")


def _build_dataset(name: str):
    cls, root, extra, split, cmap = _DATASET_TABLE[name]
    if cls is VISODataset or cls is AIRMOTDataset:
        return cls(root=root, split=split, mode="detection",
                   class_map=dict(cmap), **extra) \
            if "mode" in cls.__init__.__code__.co_varnames \
            else cls(root=root, split=split, class_map=dict(cmap), **extra)
    return cls(root=root, split=split, mode="detection",
               class_map=dict(cmap), **extra)


def _load_feat_cache(cache_dir: Path, video_id: str) -> dict:
    """Return per-frame {boxes, scores, feats, labels} dicts."""
    path = cache_dir / f"{_safe_video_id(video_id)}.npz"
    z = np.load(path)
    frame_ids = z["frame_ids"].tolist()
    flat_frame = z["flat_frame"]
    boxes = z["boxes"]
    scores = z["scores"]
    feats = z["feats"].astype(np.float32)
    labels = z["labels"]

    boxes_pf:  dict[int, np.ndarray] = {}
    scores_pf: dict[int, np.ndarray] = {}
    feats_pf:  dict[int, np.ndarray] = {}
    labels_pf: dict[int, np.ndarray] = {}
    for fid in frame_ids:
        boxes_pf[fid]  = np.zeros((0, 4),            dtype=np.float32)
        scores_pf[fid] = np.zeros(0,                 dtype=np.float32)
        feats_pf[fid]  = np.zeros((0, feats.shape[1]), dtype=np.float32)
        labels_pf[fid] = np.zeros(0,                 dtype=np.int64)

    if len(flat_frame):
        order = np.argsort(flat_frame, kind="stable")
        ff = flat_frame[order]; bb = boxes[order]; ss = scores[order]
        fe = feats[order]; ll = labels[order]
        starts = np.concatenate(([0], np.where(np.diff(ff) != 0)[0] + 1, [len(ff)]))
        for s, e in zip(starts[:-1], starts[1:]):
            fid = int(ff[s])
            boxes_pf[fid]  = bb[s:e]
            scores_pf[fid] = ss[s:e]
            feats_pf[fid]  = fe[s:e]
            labels_pf[fid] = ll[s:e].astype(np.int64)

    return {"frame_ids": frame_ids, "boxes": boxes_pf, "scores": scores_pf,
            "feats": feats_pf, "labels": labels_pf,
            "feat_dim": int(feats.shape[1])}


def _gt_per_frame(dataset, video):
    out = {}
    for fid in video.frame_ids:
        ann = dataset._load_annotations(video, fid)
        out[fid] = {
            "boxes":     np.asarray(ann["boxes"],     dtype=np.float32),
            "labels":    np.asarray(ann["labels"],    dtype=np.int64),
            "track_ids": np.asarray(ann["track_ids"], dtype=np.int64),
        }
    return out


# match helpers
def _iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ar_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = ar_a[:, None] + ar_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def _centroid_dist(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    cxa = (a[:, 0] + a[:, 2]) * 0.5; cya = (a[:, 1] + a[:, 3]) * 0.5
    cxb = (b[:, 0] + b[:, 2]) * 0.5; cyb = (b[:, 1] + b[:, 3]) * 0.5
    dx = cxa[:, None] - cxb[None, :]; dy = cya[:, None] - cyb[None, :]
    return np.sqrt(dx * dx + dy * dy).astype(np.float32)


def _greedy_match(gt, pr, metric, iou_thr, dist_thr):
    if len(gt) == 0 or len(pr) == 0:
        return [], 0
    if metric == "centroid":
        score = _centroid_dist(gt, pr); accept = score <= dist_thr; desc = False
    else:
        score = _iou_matrix(gt, pr); accept = score >= iou_thr; desc = True
    rs, cs = np.where(accept)
    if len(rs) == 0:
        return [], 0
    order = score[rs, cs].argsort()
    if desc: order = order[::-1]
    rs, cs = rs[order], cs[order]
    used_g, used_p, matches = set(), set(), []
    for r, c in zip(rs.tolist(), cs.tolist()):
        if r in used_g or c in used_p: continue
        used_g.add(r); used_p.add(c); matches.append((r, c))
    return matches, len(matches)


def _make_tracker(tracker_name: str, tracker_cfg: dict):
    if tracker_name == "botsort_reid":
        return BoTSortReIDTracker(**tracker_cfg)
    return build_tracker(tracker_name, **tracker_cfg)


def _update_tracker(tracker_name, tracker, boxes, scores, feats, fid, feat_dim):
    if tracker_name == "botsort_reid":
        return tracker.update_with_feats(boxes, scores, feats, frame_id=fid)
    # tracktrack: pack into [N, 6+D]
    if len(boxes):
        pad = np.zeros((len(boxes), 1), dtype=np.float32)
        dets_arr = np.concatenate(
            [boxes.astype(np.float32),
             scores.astype(np.float32)[:, None],
             pad,
             feats.astype(np.float32)], axis=1)
    else:
        dets_arr = np.zeros((0, 6 + feat_dim), dtype=np.float32)
    return tracker.update_with_feats(dets_arr, dets_arr)


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
    return {"Precision": prec, "Recall": rec, "F1": f1, "MOTA": mota,
            "IDF1": idf1, "ID_switches": acc["id_sw"], "num_gt": acc["num_gt"],
            **{k: acc[k] for k in
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

    classes = cfg.get("classes") or _DATASET_TABLE[dataset_key][4]

    exp_root = os.environ.get("EXPERIMENT_ROOT", "/work/ziwen/experiments")
    run_name = f"{tracker_name}_{dataset_key}"
    experiment_dir = Path(f"{exp_root}/{run_name}_{datetime.now():%Y%m%d_%H%M%S}")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    mot_root = experiment_dir / "mot_format"
    mot_root.mkdir(exist_ok=True)
    for c in classes:
        (mot_root / c).mkdir(exist_ok=True)

    print("=" * 60)
    print(f"Multi-class ReID tracker eval: {tracker_name} on {dataset_key}")
    print(f"Classes: {classes}")
    print(f"Cache:   {cache_dir}")
    print(f"Output:  {experiment_dir}")
    print("=" * 60)

    dataset = _build_dataset(dataset_key)
    print(f"[dataset] {len(dataset.videos)} videos")

    per_video_class: dict[str, dict] = {}
    per_class_acc = {c: _empty_acc() for c in classes}
    overall = _empty_acc()
    t0 = time.perf_counter()

    for v_idx, video in enumerate(dataset.videos, 1):
        feat_cache = _load_feat_cache(cache_dir, video.video_id)
        gt = _gt_per_frame(dataset, video)
        feat_dim = feat_cache["feat_dim"]

        per_video_class[video.video_id] = {}
        for cname, cid in classes.items():
            tracker = _make_tracker(tracker_name, tracker_cfg)
            if hasattr(tracker, "reset"):
                try:
                    tracker.reset(vid_name=f"{_safe_video_id(video.video_id)}__{cname}")
                except TypeError:
                    tracker.reset()

            v_acc = _empty_acc(); v_acc["n_videos"] = 1
            last_gt_to_pred: dict[int, int] = {}
            mot_lines: list[str] = []
            has_gt = False

            for fid in video.frame_ids:
                boxes  = feat_cache["boxes"][fid]
                scores = feat_cache["scores"][fid]
                feats  = feat_cache["feats"][fid]
                labels = feat_cache["labels"][fid]

                keep = labels == cid
                if score_floor > 0:
                    keep &= scores >= score_floor
                boxes_c = boxes[keep]
                scores_c = scores[keep]
                feats_c = feats[keep]

                tracks = _update_tracker(tracker_name, tracker,
                                         boxes_c, scores_c, feats_c, fid, feat_dim)

                pred_boxes = tracks[:, :4] if len(tracks) else np.zeros((0, 4), dtype=np.float32)
                pred_ids   = tracks[:, 5].astype(np.int64) if len(tracks) else np.zeros(0, dtype=np.int64)
                pred_scores = tracks[:, 4] if len(tracks) else np.zeros(0, dtype=np.float32)

                gt_lab = gt[fid]["labels"]
                gt_keep = gt_lab == cid
                gt_boxes = gt[fid]["boxes"][gt_keep]
                gt_tids  = gt[fid]["track_ids"][gt_keep]
                if len(gt_boxes): has_gt = True

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

                for j in range(len(tracks)):
                    x1, y1, x2, y2 = pred_boxes[j]
                    w, h = float(x2 - x1), float(y2 - y1)
                    mot_lines.append(
                        f"{int(fid)},{int(pred_ids[j])},{float(x1):.2f},{float(y1):.2f},"
                        f"{w:.2f},{h:.2f},{float(pred_scores[j]):.4f},-1,-1,-1")

            with open(mot_root / cname / f"{_safe_video_id(video.video_id)}.txt", "w") as f:
                f.write("\n".join(mot_lines))

            v_acc["n_videos_with_gt"] = 1 if has_gt else 0
            per_video_class[video.video_id][cname] = _summarize(v_acc)

            for k in ("det_tp", "det_fp", "det_fn", "tr_tp", "tr_fp", "tr_fn",
                     "id_sw", "num_gt"):
                per_class_acc[cname][k] += v_acc[k]
                overall[k]              += v_acc[k]
            per_class_acc[cname]["n_videos"] += 1
            per_class_acc[cname]["n_videos_with_gt"] += v_acc["n_videos_with_gt"]
            overall["n_videos"] += 1
            overall["n_videos_with_gt"] += v_acc["n_videos_with_gt"]

        v_total = sum(per_video_class[video.video_id][c]["num_gt"] for c in classes)
        v_idf1  = np.mean([per_video_class[video.video_id][c]["IDF1"]
                           for c in classes
                           if per_video_class[video.video_id][c]["num_gt"] > 0]) if v_total > 0 else 0.0
        print(f"[{v_idx}/{len(dataset.videos)}] {video.video_id:30s} "
              f"GT={v_total:>5d}  mean-IDF1={v_idf1:.3f}")

    elapsed = time.perf_counter() - t0
    summary = {
        "tracker": tracker_name, "dataset": dataset_key, "classes": classes,
        "match_metric": metric, "iou_thresh": iou_thr,
        "centroid_dist_thresh": dist_thr, "score_floor": score_floor,
        "total_time_s": elapsed,
        "overall":  _summarize(overall),
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
