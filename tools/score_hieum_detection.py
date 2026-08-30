"""
Score cached HiEUM detections with the SAME detection metrics as Faster R-CNN.

`tables/mot_detection.tex` reports mAP_50, class-agnostic AP_overall and
P/R/F1 at score 0.5. For Faster R-CNN those come from `pr_curve.json`, written
by `lightning_modules/module.py::_save_pr_curves` at test time. HiEUM is not a
Lightning model and produces no such file, so this reproduces that computation
exactly -- greedy IoU>=0.5 matching, PASCAL 11-point interpolated AP, cumulative
P/R over score-sorted predictions -- and writes a `pr_curve.json` with the same
schema, so the table generator consumes both detectors through one code path.

Car is a single foreground class, so mAP_50 and AP_overall coincide by
definition; both are emitted for schema compatibility.

Usage:
    python tools/score_hieum_detection.py \
        --det-cache /path/to/hieum_dets_cache_st/spacetracker_car \
        --out /work/anon/experiments/hieum_car_st/pr_curve.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from project_paths import DATA_ROOT

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N,4) xyxy, b: (M,4) xyxy -> (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    ar_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    ar_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = ar_a[:, None] + ar_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def _pr_and_ap(records: list[tuple[float, int]], n_gt: int) -> dict:
    """Identical to lightning_modules/module.py::_save_pr_curves::_pr_and_ap."""
    records_sorted = sorted(records, key=lambda x: x[0], reverse=True)
    tp_cum = fp_cum = 0
    precs, recs = [], []
    for _, is_tp in records_sorted:
        if is_tp:
            tp_cum += 1
        else:
            fp_cum += 1
        precs.append(tp_cum / max(tp_cum + fp_cum, 1))
        recs.append(tp_cum / max(n_gt, 1))
    ap = 0.0
    for t in [i / 10 for i in range(11)]:
        p_above = [p for p, r in zip(precs, recs) if r >= t]
        ap += max(p_above) if p_above else 0.0
    ap /= 11.0
    return {
        "n_gt": int(n_gt),
        "n_pred": len(records),
        "AP_11pt": round(float(ap), 4),
        "scores": [round(s, 6) for s, _ in records_sorted],
        "precision": [round(p, 6) for p in precs],
        "recall": [round(r, 6) for r in recs],
    }


def main():
    ap_ = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--det-cache", required=True)
    ap_.add_argument("--release", default=f"{DATA_ROOT}/release/space_tracker")
    ap_.add_argument("--split", default="test")
    ap_.add_argument("--iou", type=float, default=0.5)
    ap_.add_argument("--seq-subset", default=None,
                     help="File of release sequence names, one per line. Scores "
                          "only those. Used for the author-unseen subset: the "
                          "HiEUM initialisation is the authors' RsCarData "
                          "checkpoint, and 14 of the 48 car test sequences come "
                          "from RsCarData's native train partition, so those 14 "
                          "cannot be considered held out no matter how we split.")
    ap_.add_argument("--out", required=True)
    args = ap_.parse_args()

    from datasets.space_tracker_mot import SpaceTrackerMOTDataset

    ds = SpaceTrackerMOTDataset(root=args.release, split=args.split,
                                class_map={"car": 0}, categories=["car"],
                                complete_only=False)
    ann = json.loads(Path(args.release).joinpath(
        "mot", "annotations", "space_tracker_mot.json").read_text())
    keep = {v["name"] for v in ann["videos"] if v["category"] == "car"}
    videos = [v for v in ds.videos if v.video_id in keep]
    if args.seq_subset:
        want = {l.strip() for l in Path(args.seq_subset).read_text().splitlines()
                if l.strip()}
        before = len(videos)
        videos = [v for v in videos if v.video_id in want]
        missing = want - {v.video_id for v in videos}
        if missing:
            raise SystemExit(f"subset names not in the split: {sorted(missing)[:5]}")
        print(f"subset {Path(args.seq_subset).name}: {before} -> {len(videos)} sequences")
    print(f"{len(videos)} car-category {args.split} sequences")

    det_dir = Path(args.det_cache)
    records: list[tuple[float, int]] = []
    n_gt_total = 0
    n_missing = 0

    for vi, video in enumerate(videos, 1):
        f = det_dir / f"{video.video_id}.json"
        if not f.exists():
            n_missing += 1
            # Still count its GT: a sequence the detector produced nothing for
            # is recall loss, not an excusable absence.
            for fid in video.frame_ids:
                n_gt_total += len(np.asarray(
                    ds._load_annotations(video, fid)["boxes"], np.float32).reshape(-1, 4))
            continue
        d = json.loads(f.read_text())
        by_frame = {int(fid): i for i, fid in enumerate(d["frame_ids"])}
        for fid in video.frame_ids:
            gt = np.asarray(ds._load_annotations(video, fid)["boxes"],
                            np.float32).reshape(-1, 4)
            n_gt_total += len(gt)
            i = by_frame.get(int(fid))
            if i is None:
                continue
            boxes = np.asarray(d["boxes"][i], np.float32).reshape(-1, 4)
            scores = np.asarray(d["scores"][i], np.float32).reshape(-1)
            if len(boxes) == 0:
                continue
            order = np.argsort(-scores)
            boxes, scores = boxes[order], scores[order]
            ious = _iou_matrix(boxes, gt)
            taken = np.zeros(len(gt), dtype=bool)
            for k in range(len(boxes)):
                j, best = -1, args.iou
                for m in range(len(gt)):
                    if not taken[m] and ious[k, m] >= best:
                        best, j = ious[k, m], m
                if j >= 0:
                    taken[j] = True
                    records.append((float(scores[k]), 1))
                else:
                    records.append((float(scores[k]), 0))
        if vi % 10 == 0 or vi == len(videos):
            print(f"  [{vi}/{len(videos)}] preds={len(records)} gt={n_gt_total}",
                  flush=True)

    if n_missing:
        print(f"WARNING: {n_missing} sequences had no cached detections")
    curve = _pr_and_ap(records, n_gt_total)
    out = {
        "mAP_11pt": curve["AP_11pt"],
        "AP_overall_11pt": curve["AP_11pt"],   # single class: identical
        "per_class": {"cls1": curve},
        "overall": curve,
        "_meta": {"detector": "HiEUM", "split": args.split,
                  "n_sequences": len(videos), "iou_thresh": args.iou,
                  "sequences_without_detections": n_missing},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    print(f"\nAP_11pt={curve['AP_11pt']:.4f}  n_gt={curve['n_gt']}  "
          f"n_pred={curve['n_pred']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
