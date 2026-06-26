#!/usr/bin/env python
"""Offline re-score of the standalone OC-SORT (min_hits=3, iou=0.3) tracks on the
SAM3-refined GT (annotations_sam3), for the standalone OC-SORT main table in
docs/use_case_results/birdsai_tracking_compare_ppt.md.

The 06-17 retrained-detector dettrack runs already produced per-class OC-SORT
tracks in mot_format/*.txt (class encoded as track_id // 1_000_000). This script
re-scores those cached tracks against annotations_sam3 using the EXACT metric
formulas of eval_birdsai_detect_track.py (greedy IoU match, MOTA, IDF1, IDsw),
so the numbers are directly comparable to that pipeline — only the GT changed.

Pure offline: parses .txt tracks, no GPU / no re-tracking.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from datasets.birdsai_mot import BIRDSAIMOTDataset

BIRDSAI_ROOT = "/data/ESA_DLSTEM_2025/data/wild_animal/BIRDSAI"
ANN = "annotations_sam3"
IOU_THR = 0.5
CANON = {0: "human", 1: "elephant", 2: "giraffe", 3: "lion", 4: "unknown"}
CLASSES = sorted(CANON)

# 06-17 retrained BEST-ckpt standalone OC-SORT runs (same as detection table)
RUNS = {
    "DINOv3": "/work/ziwen/experiments/dinov3_birdsai_dettrack_20260617_170549",
    "YOLO11l": "/work/ziwen/experiments/yolo_birdsai_dettrack_20260617_214738",
    "FasterRCNN": "/work/ziwen/experiments/fasterrcnn_birdsai_dettrack_20260617_185512",
}
OUT_JSON = Path("docs/use_case_results/birdsai_octrack_sam3gt.json")


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    A = a[:, None, :]; B = b[None, :, :]
    x1 = np.maximum(A[..., 0], B[..., 0]); y1 = np.maximum(A[..., 1], B[..., 1])
    x2 = np.minimum(A[..., 2], B[..., 2]); y2 = np.minimum(A[..., 3], B[..., 3])
    iw = np.clip(x2 - x1, 0, None); ih = np.clip(y2 - y1, 0, None)
    inter = iw * ih
    ar = (A[..., 2] - A[..., 0]) * (A[..., 3] - A[..., 1])
    br = (B[..., 2] - B[..., 0]) * (B[..., 3] - B[..., 1])
    return inter / np.clip(ar + br - inter, 1e-9, None)


def greedy_match(g, p, thr):
    if len(g) == 0 or len(p) == 0:
        return []
    iou = iou_matrix(g, p); rs, cs = np.where(iou >= thr)
    order = iou[rs, cs].argsort()[::-1]
    mg, mp, out = set(), set(), []
    for k in order:
        r, c = rs[k], cs[k]
        if r in mg or c in mp:
            continue
        mg.add(r); mp.add(c); out.append((r, c))
    return out


def parse_tracks(txt):
    by_frame = {}
    if not txt.exists():
        return by_frame
    for line in open(txt):
        p = line.strip().split(",")
        if len(p) < 6:
            continue
        f = int(float(p[0])); tid = int(float(p[1]))
        x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
        cls = tid // 1_000_000
        d = by_frame.setdefault(f, {"boxes": [], "labels": [], "ids": []})
        d["boxes"].append([x, y, x + w, y + h]); d["labels"].append(cls); d["ids"].append(tid)
    return by_frame


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


def score_run(run_dir, ds):
    det = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CLASSES}
    trk = {c: {"tp": 0, "fp": 0, "fn": 0, "idsw": 0, "ngt": 0} for c in CLASSES}
    for video in ds.videos:
        tracks = parse_tracks(Path(run_dir) / "mot_format" / f"{video.video_id}.txt")
        last = {c: {} for c in CLASSES}
        for fid in video.frame_ids:
            a = ds._load_annotations(video, fid)
            gb = np.asarray(a["boxes"], np.float32).reshape(-1, 4)
            gl = np.asarray(a["labels"], np.int64).reshape(-1)
            gid = np.asarray(a["track_ids"], np.int64).reshape(-1)
            t = tracks.get(int(fid), {"boxes": [], "labels": [], "ids": []})
            pb = np.asarray(t["boxes"], np.float32).reshape(-1, 4)
            pl = np.asarray(t["labels"], np.int64).reshape(-1)
            pid = np.asarray(t["ids"], np.int64).reshape(-1)
            for c in CLASSES:
                gm = gl == c; pm = pl == c
                gbc = gb[gm]; gic = gid[gm]; cbc = pb[pm]; cic = pid[pm]
                ms = greedy_match(gbc, cbc, IOU_THR)
                tp = len(ms)
                det[c]["tp"] += tp; det[c]["fp"] += len(cbc) - tp; det[c]["fn"] += len(gbc) - tp
                trk[c]["tp"] += tp; trk[c]["fp"] += len(cbc) - tp; trk[c]["fn"] += len(gbc) - tp
                trk[c]["ngt"] += len(gbc)
                for r, cc in ms:
                    g = int(gic[r]); pp = int(cic[cc])
                    prev = last[c].get(g)
                    if prev is not None and prev != pp:
                        trk[c]["idsw"] += 1
                    last[c][g] = pp
    per_class = {CANON[c]: metrics_from(det[c], trk[c]) for c in CLASSES}
    pd_ = {k: sum(det[c][k] for c in CLASSES) for k in ("tp", "fp", "fn")}
    pt_ = {k: sum(trk[c][k] for c in CLASSES) for k in ("tp", "fp", "fn", "idsw", "ngt")}
    return metrics_from(pd_, pt_), per_class


def main():
    ds = BIRDSAIMOTDataset(root=BIRDSAI_ROOT, split="test", granularity="fine",
                           annotations_dirname=ANN,
                           class_map={v: k for k, v in CANON.items()})
    nfr = sum(len(v.frame_ids) for v in ds.videos)
    print(f"GT={ANN}  videos={len(ds.videos)}  frames={nfr}  IoU={IOU_THR}\n")
    out = {}
    for label, run in RUNS.items():
        overall, per_class = score_run(run, ds)
        out[label] = {"overall": overall, "per_class": per_class}
        o = overall
        print(f"{label:11s} P={o['Precision']:.3f} R={o['Recall']:.3f} F1={o['F1']:.3f} "
              f"MOTA={o['MOTA']:+.3f} IDF1={o['IDF1']:.3f} IDsw={o['IDsw']}")
        for cn, m in per_class.items():
            print(f"   {cn:8s} P={m['Precision']:.3f} R={m['Recall']:.3f} F1={m['F1']:.3f} "
                  f"MOTA={m['MOTA']:+.3f} IDsw={m['IDsw']} nGT={m['num_gt']}")
        print()
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
