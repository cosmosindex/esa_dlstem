"""
Score a directory of MOTChallenge track files with the SAME metric code the
in-process tracker driver uses.

Some trackers cannot run inside the project environment -- MASA, for instance,
needs mmdet 3.3 / torch 2.1 and lives in its own env -- so they emit
MOTChallenge text instead of being scored inline. Re-implementing the metric for
those would make their numbers quietly incomparable to the rest of the table, so
this imports `_greedy_match`, `_empty_acc` and `_summarize` from
``eval_tracker_multiclass.py`` verbatim and only replaces the "where do the
predictions come from" step.

Expected input: one ``<video_id>.txt`` per sequence, rows of
``frame,id,x,y,w,h,score,class,-1`` with class as the 1-based detector label.

Usage:
    python evaluation/score_mot_format.py \
        --tracks /path/to/masa_spacetracker_nocar/mot_format \
        --dataset spacetracker_nocar --name MASA
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.eval_tracker_multiclass import (  # noqa: E402
    _build_dataset, _empty_acc, _gt_per_frame, _greedy_match, _safe_video_id,
    _summarize, _DATASET_TABLE,
)


def _load_tracks(path: Path) -> dict[int, dict]:
    """frame id -> {boxes (N,4) xyxy, ids (N,), labels (N,)}"""
    per: dict[int, list] = defaultdict(list)
    if not path.exists():
        return {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        f = line.split(",")
        fid, tid = int(float(f[0])), int(float(f[1]))
        x, y, w, h = (float(f[2]), float(f[3]), float(f[4]), float(f[5]))
        cls = int(float(f[7])) if len(f) > 7 else -1
        per[fid].append((x, y, x + w, y + h, tid, cls))
    out = {}
    for fid, rows in per.items():
        arr = np.asarray([r[:4] for r in rows], dtype=np.float32)
        out[fid] = {
            "boxes": arr,
            "ids": np.asarray([r[4] for r in rows], dtype=np.int64),
            "labels": np.asarray([r[5] for r in rows], dtype=np.int64),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracks", required=True, help="dir of <video>.txt files")
    ap.add_argument("--dataset", default="spacetracker_nocar")
    ap.add_argument("--name", default="tracker")
    ap.add_argument("--match-metric", default="centroid", choices=["centroid", "iou"])
    ap.add_argument("--centroid-dist-thresh", type=float, default=5.0)
    ap.add_argument("--iou-thresh", type=float, default=0.5)
    ap.add_argument("--class-agnostic", action="store_true",
                    help="Ignore class labels: pool all GT and all predictions "
                         "into one foreground class. Required for trackers that "
                         "do not emit a class at all (MOTRv2 writes -1), and it "
                         "is the same pooling the paper's HOTA uses. Apply it to "
                         "EVERY tracker being compared, not just the agnostic "
                         "one, or the numbers are not comparable.")
    ap.add_argument("--per-class-dirs", action="store_true",
                    help="Tracks laid out as <tracks>/<class>/<video>.txt (what the "
                         "in-process driver writes) instead of one flat file per "
                         "video with a class column. Used to validate this scorer "
                         "against a tracker whose numbers are already known.")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    classes = _DATASET_TABLE[args.dataset][4]          # name -> id
    if args.class_agnostic:
        classes = {"foreground": None}                 # None = match everything
    dataset = _build_dataset(args.dataset)
    tracks_dir = Path(args.tracks)
    print(f"[{args.name}] {len(dataset.videos)} videos, classes={classes}")

    per_class = {c: _empty_acc() for c in classes}
    overall = _empty_acc()

    for video in dataset.videos:
        gt = _gt_per_frame(dataset, video)
        if args.per_class_dirs and args.class_agnostic:
            merged: dict[int, dict] = {}
            for sub in sorted(d for d in tracks_dir.iterdir() if d.is_dir()):
                part = _load_tracks(sub / f"{_safe_video_id(video.video_id)}.txt")
                for fid, rec in part.items():
                    if fid not in merged:
                        merged[fid] = {k: v.copy() for k, v in rec.items()}
                    else:
                        for k in ("boxes", "ids", "labels"):
                            merged[fid][k] = np.concatenate([merged[fid][k], rec[k]])
            pred_by_class = {None: merged}
        elif args.per_class_dirs:
            pred_by_class = {
                cid: _load_tracks(tracks_dir / cname / f"{_safe_video_id(video.video_id)}.txt")
                for cname, cid in classes.items()
            }
            pred = None
        else:
            pred = _load_tracks(tracks_dir / f"{_safe_video_id(video.video_id)}.txt")

        for cname, cid in classes.items():
            acc = per_class[cname]
            last_gt_to_pred: dict[int, int] = {}
            for fid in video.frame_ids:
                g = gt.get(fid)
                if g is None:
                    gt_boxes = np.zeros((0, 4), dtype=np.float32)
                    gt_tids = np.zeros(0, dtype=np.int64)
                elif cid is None:
                    gt_boxes, gt_tids = g["boxes"], g["track_ids"]
                else:
                    keep = g["labels"] == cid
                    gt_boxes, gt_tids = g["boxes"][keep], g["track_ids"][keep]

                if args.per_class_dirs:
                    p = pred_by_class[cid].get(fid)
                    if p is None:
                        pred_boxes = np.zeros((0, 4), dtype=np.float32)
                        pred_ids = np.zeros(0, dtype=np.int64)
                    else:
                        # already class-segregated by directory
                        pred_boxes, pred_ids = p["boxes"], p["ids"]
                else:
                    p = pred.get(fid)
                    if p is None:
                        pred_boxes = np.zeros((0, 4), dtype=np.float32)
                        pred_ids = np.zeros(0, dtype=np.int64)
                    elif cid is None:
                        pred_boxes, pred_ids = p["boxes"], p["ids"]
                    else:
                        keep = p["labels"] == cid
                        pred_boxes, pred_ids = p["boxes"][keep], p["ids"][keep]

                matches, _ = _greedy_match(gt_boxes, pred_boxes, args.match_metric,
                                           args.iou_thresh, args.centroid_dist_thresh)
                tp = len(matches)
                acc["det_tp"] += tp
                acc["det_fp"] += len(pred_boxes) - tp
                acc["det_fn"] += len(gt_boxes) - tp
                acc["num_gt"] += len(gt_boxes)
                for r, c in matches:
                    g_id, p_id = int(gt_tids[r]), int(pred_ids[c])
                    prev = last_gt_to_pred.get(g_id)
                    if prev is not None and prev != p_id:
                        acc["id_sw"] += 1
                    last_gt_to_pred[g_id] = p_id
                    acc["tr_tp"] += 1
                acc["tr_fn"] += len(gt_boxes) - tp
                acc["tr_fp"] += len(pred_boxes) - tp

    for cname in classes:
        for k in overall:
            overall[k] += per_class[cname][k]

    results = {}
    print("=" * 60)
    for cname in classes:
        s = _summarize(per_class[cname])
        results[cname] = s
        print(f"[{cname:10s}]  Pr={s['Precision']:.3f}  Re={s['Recall']:.3f}  "
              f"F1={s['F1']:.3f}  MOTA={s['MOTA']:.3f}  IDF1={s['IDF1']:.3f}  "
              f"IDsw={s['ID_switches']}  GT={s['num_gt']}")
    s = _summarize(overall)
    results["overall"] = s
    print(f"[{'overall':10s}]  Pr={s['Precision']:.3f}  Re={s['Recall']:.3f}  "
          f"F1={s['F1']:.3f}  MOTA={s['MOTA']:.3f}  IDF1={s['IDF1']:.3f}")
    print("=" * 60)

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
