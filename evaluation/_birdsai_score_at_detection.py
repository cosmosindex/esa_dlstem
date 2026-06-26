"""
Quick stats (no inference): for each GT box, find the best-IoU same-class RAW
detection (scores dumped down to 0.05 floor) and record that detection's score.
Tells us at what confidence small species actually get detected.

    python evaluation/_birdsai_score_at_detection.py <predictions.json> [iou_thr]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets.birdsai_mot import BIRDSAIMOTDataset

BIRDSAI_ROOT = "/data/ESA_DLSTEM_2025/data/wild_animal/BIRDSAI"
CANON_NAMES = {0: "human", 1: "elephant", 2: "giraffe", 3: "lion", 4: "unknown"}


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    a = a[:, None, :]; b = b[None, :, :]
    x1 = np.maximum(a[..., 0], b[..., 0]); y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 2], b[..., 2]); y2 = np.minimum(a[..., 3], b[..., 3])
    iw = np.clip(x2 - x1, 0, None); ih = np.clip(y2 - y1, 0, None)
    inter = iw * ih
    ar = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    br = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / np.clip(ar + br - inter, 1e-9, None)


def main():
    pred_path = Path(sys.argv[1])
    iou_thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    preds = json.load(open(pred_path))

    canon_map = {v: k for k, v in CANON_NAMES.items()}
    ds = BIRDSAIMOTDataset(root=BIRDSAI_ROOT, split="test",
                           granularity="fine", class_map=canon_map)

    # per class: list of (matched_det_score or None, gt_diag_px)
    rec = {c: [] for c in CANON_NAMES}

    for video in ds.videos:
        vid = video.video_id
        vp = preds["videos"].get(vid)
        if vp is None:
            continue
        for fid in video.frame_ids:
            ann = ds._load_annotations(video, fid)
            gtb = np.asarray(ann["boxes"], np.float32).reshape(-1, 4)
            gtl = np.asarray(ann["labels"], np.int64).reshape(-1)
            fr = vp["frames"].get(str(int(fid)))
            if fr is None:
                pb = np.zeros((0, 4), np.float32); ps = np.zeros(0); pl = np.zeros(0, np.int64)
            else:
                d = fr["detections"]
                pb = np.asarray(d["boxes"], np.float32).reshape(-1, 4)
                ps = np.asarray(d["scores"], np.float32).reshape(-1)
                pl = np.asarray(d["labels"], np.int64).reshape(-1)
            for c in CANON_NAMES:
                gm = gtl == c
                if not gm.any():
                    continue
                gb = gtb[gm]
                pm = pl == c
                cb = pb[pm]; cs = ps[pm]
                wh = gb[:, 2:] - gb[:, :2]
                diag = np.sqrt((wh ** 2).sum(1))
                if len(cb) == 0:
                    for dg in diag:
                        rec[c].append((None, float(dg)))
                    continue
                iou = iou_matrix(gb, cb)
                for i in range(len(gb)):
                    j = int(iou[i].argmax())
                    if iou[i, j] >= iou_thr:
                        rec[c].append((float(cs[j]), float(diag[i])))
                    else:
                        rec[c].append((None, float(diag[i])))

    print(f"\n=== {pred_path.parent.name}  (IoU>={iou_thr}, raw-det score floor 0.05) ===")
    print(f"{'class':9s} {'nGT':>7s} {'det%':>6s} | matched-det score (on detected GTs)        | GT diag px")
    print(f"{'':9s} {'':>7s} {'':>6s} |  p10   p25   p50   p75   p90   mean  | det@.3 .5 | p50  med-undet")
    for c in CANON_NAMES:
        items = rec[c]
        n = len(items)
        if n == 0:
            print(f"{CANON_NAMES[c]:9s} {0:7d}    n/a")
            continue
        scores = np.array([s for s, _ in items if s is not None])
        diag = np.array([dg for _, dg in items])
        det_diag = np.array([dg for s, dg in items if s is not None])
        undet_diag = np.array([dg for s, dg in items if s is None])
        detpct = 100 * len(scores) / n
        if len(scores):
            p = np.percentile(scores, [10, 25, 50, 75, 90])
            frac3 = 100 * (scores >= 0.3).mean()
            frac5 = 100 * (scores >= 0.5).mean()
            med_det = np.median(det_diag)
            med_undet = np.median(undet_diag) if len(undet_diag) else float("nan")
            print(f"{CANON_NAMES[c]:9s} {n:7d} {detpct:5.1f}% | "
                  f"{p[0]:.2f}  {p[1]:.2f}  {p[2]:.2f}  {p[3]:.2f}  {p[4]:.2f}  {scores.mean():.2f} | "
                  f"{frac3:4.0f}% {frac5:3.0f}% | {med_det:4.1f} {med_undet:5.1f}")
        else:
            print(f"{CANON_NAMES[c]:9s} {n:7d}  0.0% | (never detected at this IoU)        "
                  f"           | med GT diag {np.median(diag):.1f}")


if __name__ == "__main__":
    main()
