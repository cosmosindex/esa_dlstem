"""
eval_birdsai_track_sweep.py
===========================
Run ONE TBD tracker over a detector's cached BIRDSAI detections and score it
against GT — the multi-tracker companion to ``eval_birdsai_detect_track.py``.

Where ``eval_birdsai_detect_track.py`` runs a detector then OC-SORT, this script
reads the already-dumped ``predictions.json`` (per-frame raw ``detections`` at
original resolution, canonical 0-indexed labels) and feeds them to any tracker
in the benchmark — so all 3 detectors × all 6 TBD trackers share identical
detections (the detector is never re-run).

Trackers:
  appearance-free : sort, ocsort, bytetrack, botsort   (need only boxes+scores)
  appearance-aware: botsort_reid, tracktrack           (need --feat-cache, built
                    by cache_birdsai_feats.py)

Scoring is identical to eval_birdsai_detect_track.py: per-class greedy IoU
matching (IoU>=0.5), class-aware MOTA / IDF1 / IDsw + detection Pr/Re/F1.

Usage::

    python evaluation/eval_birdsai_track_sweep.py \\
        --predictions /work/.../yolo_birdsai_dettrack_*/predictions.json \\
        --tracker bytetrack --out-root /data/.../MOT_birdsai_sweep

    python evaluation/eval_birdsai_track_sweep.py \\
        --predictions .../predictions.json --tracker tracktrack \\
        --feat-cache /data/.../MOT_birdsai_sweep/feats/yolo \\
        --out-root /data/.../MOT_birdsai_sweep
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from datasets.birdsai_mot import BIRDSAIMOTDataset
from models.trackers import build_tracker

BIRDSAI_ROOT = "/data/ESA_DLSTEM_2025/data/wild_animal/BIRDSAI"
CANON_NAMES = {0: "human", 1: "elephant", 2: "giraffe", 3: "lion", 4: "unknown"}

APPEARANCE_FREE = {"sort", "ocsort", "bytetrack", "botsort"}
REID_TRACKERS = {"botsort_reid", "tracktrack"}

# Tracker kwargs mirror the benchmark's tiny-object (rscardata/sdmcar) tuning so
# BIRDSAI stays consistent with the satellite MOT sweep.
TRACKER_KWARGS = {
    "sort":      {"max_age": 30, "min_hits": 1, "iou_threshold": 0.1, "score_thresh": 0.0},
    "ocsort":    {"det_thresh": 0.30, "max_age": 30, "min_hits": 1, "iou_threshold": 0.1,
                  "delta_t": 3, "asso_func": "iou", "inertia": 0.2, "use_byte": False},
    "bytetrack": {"track_thresh": 0.30, "track_buffer": 30, "match_thresh": 0.5,
                  "mot20": False, "frame_rate": 30},
    "botsort":   {"track_high_thresh": 0.30, "track_low_thresh": 0.10, "new_track_thresh": 0.35,
                  "track_buffer": 30, "match_thresh": 0.5, "proximity_thresh": 0.5,
                  "appearance_thresh": 0.25, "cmc_method": "none", "mot20": False, "frame_rate": 30},
    "botsort_reid": {"feat_dim": 2048, "track_high_thresh": 0.30, "track_low_thresh": 0.10,
                     "new_track_thresh": 0.35, "track_buffer": 30, "match_thresh": 0.5,
                     "proximity_thresh": 0.5, "appearance_thresh": 0.25, "cmc_method": "none",
                     "mot20": False, "frame_rate": 30},
    "tracktrack": {"feat_dim": 2048, "det_thr": 0.30, "init_thr": 0.40, "match_thr": 0.70,
                   "tai_thr": 0.45, "penalty_p": 0.20, "penalty_q": 0.40, "reduce_step": 0.05,
                   "min_len": 3, "min_box_area": 25.0, "max_time_lost": 30},
}
# Drop very-low-score dets before feeding (cache floor is 0.05). Mirror benchmark.
SCORE_FLOOR = {"sort": 0.0, "ocsort": 0.0, "bytetrack": 0.10,
               "botsort": 0.10, "botsort_reid": 0.10, "tracktrack": 0.10}


# ---------------------------------------------------------------------------
# Matching (class-aware greedy IoU) — copied from eval_birdsai_detect_track.py
# ---------------------------------------------------------------------------

def _iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def _greedy_match(gt_boxes, pred_boxes, iou_thr):
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return []
    iou = _iou_matrix(gt_boxes, pred_boxes)
    rows, cols = np.where(iou >= iou_thr)
    if len(rows) == 0:
        return []
    order = iou[rows, cols].argsort()[::-1]
    rows, cols = rows[order], cols[order]
    mg, mp, matches = set(), set(), []
    for r, c in zip(rows.tolist(), cols.tolist()):
        if r in mg or c in mp:
            continue
        mg.add(r); mp.add(c); matches.append((r, c))
    return matches


def _safe_video_id(video_id: str) -> str:
    return video_id.replace("/", "_")


def _load_feats(npz_path: Path):
    """Return {frame_id -> feats[N, D]} in the order detections were cached."""
    z = np.load(npz_path)
    flat_frame = z["flat_frame"]
    feats = z["feats"].astype(np.float32)
    per_frame: dict[int, np.ndarray] = {}
    if len(flat_frame) == 0:
        return per_frame, (feats.shape[1] if feats.ndim == 2 else 2048)
    order = np.argsort(flat_frame, kind="stable")
    ff = flat_frame[order]; fe = feats[order]
    uniq, starts = np.unique(ff, return_index=True)
    starts = list(starts) + [len(ff)]
    for k, fid in enumerate(uniq):
        per_frame[int(fid)] = fe[starts[k]:starts[k + 1]]
    return per_frame, feats.shape[1]


def _new_trackers(tracker_name, classes):
    """One independent tracker per class (mirrors the OC-SORT eval)."""
    kw = TRACKER_KWARGS[tracker_name]
    if tracker_name in REID_TRACKERS:
        return {c: build_tracker(tracker_name, **kw) for c in classes}
    return {c: build_tracker(tracker_name, **kw) for c in classes}


def _update_tracker(tracker, name, boxes, scores, feats, fid, feat_dim=2048):
    """Unified call across the appearance-free / ReID APIs. Returns [M, 6]."""
    n = len(boxes)
    if name in APPEARANCE_FREE:
        dets = (np.column_stack([boxes, scores]).astype(np.float32)
                if n else np.zeros((0, 5), dtype=np.float32))
        return tracker.update(dets, frame_id=fid)

    # ReID paths — normalize shapes (handle empty frames explicitly).
    D = feats.shape[1] if (feats is not None and getattr(feats, "ndim", 0) == 2
                           and feats.shape[1] > 0) else feat_dim
    boxes = np.asarray(boxes, dtype=np.float32).reshape(n, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(n)
    feats = (np.asarray(feats, dtype=np.float32).reshape(n, D) if n
             else np.zeros((0, D), dtype=np.float32))

    if name == "botsort_reid":
        return tracker.update_with_feats(boxes, scores, feats, frame_id=fid)
    # tracktrack: [N, 6+D] = [x1,y1,x2,y2,score,_pad,*feat]
    dets = (np.concatenate(
        [boxes, scores.reshape(n, 1), np.zeros((n, 1), dtype=np.float32), feats], axis=1)
        if n else np.zeros((0, 6 + D), dtype=np.float32))
    return tracker.update_with_feats(dets, dets)


def metrics_from(d, t):
    prec = d["tp"] / max(d["tp"] + d["fp"], 1)
    rec = d["tp"] / max(d["tp"] + d["fn"], 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    mota = 1.0 - (t["fp"] + t["fn"] + t["idsw"]) / max(t["ngt"], 1)
    idp = t["tp"] / max(t["tp"] + t["fp"], 1)
    idr = t["tp"] / max(t["tp"] + t["fn"], 1)
    idf1 = 2 * idp * idr / max(idp + idr, 1e-9)
    return {"Precision": prec, "Recall": rec, "F1": f1, "MOTA": mota,
            "IDF1": idf1, "IDsw": t["idsw"], "num_gt": t["ngt"]}


def main():
    ap = argparse.ArgumentParser(description="BIRDSAI TBD tracker sweep from cached detections")
    ap.add_argument("--predictions", required=True, help="detector predictions.json")
    ap.add_argument("--tracker", required=True,
                    choices=sorted(APPEARANCE_FREE | REID_TRACKERS))
    ap.add_argument("--feat-cache", default=None, help="dir of per-video .npz (ReID trackers)")
    ap.add_argument("--out-root", default="/data/ESA_DLSTEM_2025/experiments/MOT_birdsai_sweep")
    ap.add_argument("--model-name", default=None, help="override; else read from predictions.json")
    ap.add_argument("--split", default=None, help="override; else read from predictions.json")
    ap.add_argument("--iou-thresh", type=float, default=0.5)
    args = ap.parse_args()

    if args.tracker in REID_TRACKERS and not args.feat_cache:
        ap.error(f"--feat-cache is required for {args.tracker}")

    with open(args.predictions) as f:
        preds = json.load(f)
    videos = preds["videos"]
    model_name = args.model_name or preds.get("model", "model")
    split = args.split or preds.get("split", "test")
    score_floor = SCORE_FLOOR[args.tracker]

    run_name = f"{model_name}_{args.tracker}_birdsai_track"
    experiment_dir = Path(args.out_root) / f"{run_name}_{datetime.now():%Y%m%d_%H%M%S}"
    (experiment_dir / "mot_format").mkdir(parents=True, exist_ok=True)

    # GT (fine 5-class, canonical 0-indexed) — matches the predictions taxonomy.
    canon_map = {v: k for k, v in CANON_NAMES.items()}
    dataset = BIRDSAIMOTDataset(root=BIRDSAI_ROOT, split=split,
                                granularity="fine", class_map=canon_map)
    classes = sorted(CANON_NAMES)

    print("=" * 64)
    print(f"BIRDSAI track sweep: model={model_name} tracker={args.tracker} split={split}")
    print(f"  preds:  {args.predictions}")
    print(f"  output: {experiment_dir}")
    print(f"  kwargs: {TRACKER_KWARGS[args.tracker]}  score_floor={score_floor}")
    print("=" * 64, flush=True)

    det = {c: {"tp": 0, "fp": 0, "fn": 0} for c in classes}
    trk = {c: {"tp": 0, "fp": 0, "fn": 0, "idsw": 0, "ngt": 0} for c in classes}
    per_video = {}
    t0 = time.perf_counter()

    for v_idx, video in enumerate(dataset.videos, 1):
        vid = video.video_id
        if vid not in videos:
            print(f"  [{v_idx}] {vid} — not in predictions.json, skipping")
            continue
        frames_pred = videos[vid]["frames"]

        feats_per_frame = {}
        if args.tracker in REID_TRACKERS:
            npz = Path(args.feat_cache) / f"{_safe_video_id(vid)}.npz"
            feats_per_frame, _ = _load_feats(npz)

        trackers = _new_trackers(args.tracker, classes)
        for t in trackers.values():
            if hasattr(t, "reset"):
                try:
                    t.reset(vid_name=_safe_video_id(vid))
                except TypeError:
                    t.reset()

        last_gt_to_pred = {c: {} for c in classes}
        mot_lines = []
        vdet = {c: {"tp": 0, "fp": 0, "fn": 0} for c in classes}
        vtrk = {c: {"tp": 0, "fp": 0, "fn": 0, "idsw": 0, "ngt": 0} for c in classes}

        for fid in video.frame_ids:
            fr = frames_pred.get(str(int(fid)))
            if fr is None:
                boxes = np.zeros((0, 4), np.float32); scores = np.zeros(0, np.float32)
                labels = np.zeros(0, np.int64); feats_f = np.zeros((0, 2048), np.float32)
            else:
                boxes = np.asarray(fr["detections"]["boxes"], dtype=np.float32).reshape(-1, 4)
                scores = np.asarray(fr["detections"]["scores"], dtype=np.float32).reshape(-1)
                labels = np.asarray(fr["detections"]["labels"], dtype=np.int64).reshape(-1)
                feats_f = feats_per_frame.get(int(fid)) if args.tracker in REID_TRACKERS else None

            ann = dataset._load_annotations(video, fid)
            gt_boxes = np.asarray(ann["boxes"], dtype=np.float32)
            gt_labels = np.asarray(ann["labels"], dtype=np.int64)
            gt_tids = np.asarray(ann["track_ids"], dtype=np.int64)

            for c in classes:
                cd = labels == c
                if score_floor > 0:
                    cd = cd & (scores >= score_floor)
                cb, cs = boxes[cd], scores[cd]
                cf = feats_f[cd] if (feats_f is not None and len(feats_f) == len(boxes)) else None

                tracks = _update_tracker(trackers[c], args.tracker, cb, cs, cf, int(fid))
                tracks = np.asarray(tracks, dtype=np.float32).reshape(-1, tracks.shape[1] if len(tracks) else 6)
                tb = tracks[:, :4] if len(tracks) else np.zeros((0, 4), np.float32)
                ts = tracks[:, 4] if len(tracks) else np.zeros(0, np.float32)
                tid = (tracks[:, 5].astype(np.int64) + c * 1_000_000
                       if len(tracks) else np.zeros(0, np.int64))

                gm = gt_labels == c
                gtb_c = gt_boxes[gm]; gtid_c = gt_tids[gm]
                matches = _greedy_match(gtb_c, tb, args.iou_thresh)
                tp = len(matches)
                vdet[c]["tp"] += tp; vdet[c]["fp"] += len(tb) - tp; vdet[c]["fn"] += len(gtb_c) - tp
                vtrk[c]["ngt"] += len(gtb_c); vtrk[c]["tp"] += tp
                vtrk[c]["fp"] += len(tb) - tp; vtrk[c]["fn"] += len(gtb_c) - tp
                for r, cc in matches:
                    g = int(gtid_c[r]); p = int(tid[cc])
                    prev = last_gt_to_pred[c].get(g)
                    if prev is not None and prev != p:
                        vtrk[c]["idsw"] += 1
                    last_gt_to_pred[c][g] = p

                for j in range(len(tb)):
                    x1, y1, x2, y2 = tb[j]
                    mot_lines.append(
                        f"{int(fid)},{int(tid[j])},{x1:.2f},{y1:.2f},"
                        f"{x2 - x1:.2f},{y2 - y1:.2f},{float(ts[j]):.4f},-1,-1,-1")

        with open(experiment_dir / "mot_format" / f"{_safe_video_id(vid)}.txt", "w") as f:
            f.write("\n".join(mot_lines))

        def _pool(d, keys):
            return {k: sum(d[c][k] for c in classes) for k in keys}
        pt_ = _pool(vtrk, ["tp", "fp", "fn", "idsw", "ngt"])
        pd_ = _pool(vdet, ["tp", "fp", "fn"])
        per_video[vid] = metrics_from(pd_, pt_)

        for c in classes:
            for k in ("tp", "fp", "fn"):
                det[c][k] += vdet[c][k]
            for k in ("tp", "fp", "fn", "idsw", "ngt"):
                trk[c][k] += vtrk[c][k]
        m = per_video[vid]
        print(f"[{v_idx}/{len(dataset.videos)}] {vid}  F1={m['F1']:.3f} "
              f"MOTA={m['MOTA']:.3f} IDF1={m['IDF1']:.3f} IDsw={m['IDsw']}", flush=True)

    per_class = {CANON_NAMES[c]: metrics_from(det[c], trk[c]) for c in classes}
    pooled_d = {k: sum(det[c][k] for c in classes) for k in ("tp", "fp", "fn")}
    pooled_t = {k: sum(trk[c][k] for c in classes) for k in ("tp", "fp", "fn", "idsw", "ngt")}
    overall = metrics_from(pooled_d, pooled_t)

    summary = {
        "model": model_name, "dataset": "BIRDSAI", "split": split,
        "tracker": args.tracker, "tracker_kwargs": TRACKER_KWARGS[args.tracker],
        "score_floor": score_floor, "iou_thresh": args.iou_thresh,
        "total_videos": len(dataset.videos), "total_time_s": time.perf_counter() - t0,
        "overall": overall, "per_class": per_class,
    }
    with open(experiment_dir / "test_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(experiment_dir / "per_video_metrics.json", "w") as f:
        json.dump(per_video, f, indent=2)

    print("\n" + "=" * 64)
    print(f"{model_name} + {args.tracker}  OVERALL  "
          f"Pr={overall['Precision']:.3f} Re={overall['Recall']:.3f} F1={overall['F1']:.3f} | "
          f"MOTA={overall['MOTA']:.3f} IDF1={overall['IDF1']:.3f} IDsw={overall['IDsw']}")
    print(f"test_metrics.json → {experiment_dir}")


if __name__ == "__main__":
    main()
