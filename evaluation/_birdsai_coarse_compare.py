"""
Coarse 2-class (person/animal) comparison of the 3 fine-class detectors vs SAM3,
scored identically on the COMMON test videos. No inference -- reads predictions.json
(track boxes) + GT only. person=human, animal={elephant,giraffe,lion,unknown}.

    python evaluation/_birdsai_coarse_compare.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets.birdsai_mot import BIRDSAIMOTDataset

BIRDSAI_ROOT = "/data/ESA_DLSTEM_2025/data/wild_animal/BIRDSAI"
CANON = {0: "human", 1: "elephant", 2: "giraffe", 3: "lion", 4: "unknown"}
COARSE = {"person": 0, "animal": 1}

RUNS = {
    "DINOv3":     "/work/ziwen/experiments/dinov3_birdsai_dettrack_20260617_170549/predictions.json",
    "FasterRCNN": "/work/ziwen/experiments/fasterrcnn_birdsai_dettrack_20260617_185512/predictions.json",
    "YOLO11l":    "/work/ziwen/experiments/yolo_birdsai_dettrack_20260617_214738/predictions.json",
    "SAM3":       "/data/ESA_DLSTEM_2025/experiments/MOT_birdsai_sam3/sam3_text_birdsai_20260616_094330/predictions.json",
}
IOU_THR = 0.5


def fine_to_coarse(lbl):       # detector fine 0-4 -> coarse
    return 0 if lbl == 0 else 1


def sam3_to_coarse(lbl):       # SAM3 {0:animal,1:human} -> coarse
    return 0 if lbl == 1 else 1


def iou_mat(a, b):
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


def greedy(g, p, thr):
    if len(g) == 0 or len(p) == 0:
        return []
    iou = iou_mat(g, p)
    rs, cs = np.where(iou >= thr)
    order = iou[rs, cs].argsort()[::-1]
    mg, mp, out = set(), set(), []
    for k in order:
        r, c = rs[k], cs[k]
        if r in mg or c in mp:
            continue
        mg.add(r); mp.add(c); out.append((r, c))
    return out


def main():
    canon_map = {v: k for k, v in CANON.items()}
    ds = BIRDSAIMOTDataset(root=BIRDSAI_ROOT, split="test",
                           granularity="fine", class_map=canon_map)
    preds = {m: json.load(open(p)) for m, p in RUNS.items()}

    # common video set
    vsets = [set(d["videos"].keys()) for d in preds.values()]
    common = sorted(set.intersection(*vsets))
    print(f"common test videos: {len(common)} "
          f"(detectors had {len(vsets[0])}, SAM3 {len(preds['SAM3']['videos'])})")

    for m in RUNS:
        mapper = sam3_to_coarse if m == "SAM3" else fine_to_coarse
        acc = {c: {"tp": 0, "fp": 0, "fn": 0, "idsw": 0, "ngt": 0} for c in (0, 1)}
        last = {c: {} for c in (0, 1)}
        for video in ds.videos:
            vid = video.video_id
            if vid not in common:
                continue
            fr = preds[m]["videos"][vid]["frames"]
            for fid in video.frame_ids:
                ann = ds._load_annotations(video, fid)
                gtb = np.asarray(ann["boxes"], np.float32).reshape(-1, 4)
                gtl = np.array([fine_to_coarse(l) for l in ann["labels"]], np.int64)
                gtid = np.asarray(ann["track_ids"], np.int64).reshape(-1)
                f = fr.get(str(int(fid)), {})
                tr = f.get("tracks", {}) or {}
                pb = np.asarray(tr.get("boxes", []), np.float32).reshape(-1, 4)
                pl = np.array([mapper(l) for l in tr.get("labels", [])], np.int64)
                ptid = np.asarray(tr.get("track_ids", []), np.int64).reshape(-1)
                for c in (0, 1):
                    gm = gtl == c; pm = pl == c
                    gb = gtb[gm]; gi = gtid[gm]
                    cb = pb[pm]; ci = ptid[pm]
                    ms = greedy(gb, cb, IOU_THR)
                    tp = len(ms)
                    acc[c]["tp"] += tp
                    acc[c]["fp"] += len(cb) - tp
                    acc[c]["fn"] += len(gb) - tp
                    acc[c]["ngt"] += len(gb)
                    for r, cc in ms:
                        g = int(gi[r]); pp = int(ci[cc])
                        prev = last[c].get(g)
                        if prev is not None and prev != pp:
                            acc[c]["idsw"] += 1
                        last[c][g] = pp

        def met(d):
            pr = d["tp"] / max(d["tp"] + d["fp"], 1)
            re = d["tp"] / max(d["tp"] + d["fn"], 1)
            f1 = 2 * pr * re / max(pr + re, 1e-9)
            mota = 1 - (d["fp"] + d["fn"] + d["idsw"]) / max(d["ngt"], 1)
            return pr, re, f1, mota, d["idsw"], d["ngt"]

        pool = {k: sum(acc[c][k] for c in (0, 1)) for k in ("tp", "fp", "fn", "idsw", "ngt")}
        print(f"\n### {m}")
        print(f"{'':9s} {'Pr':>6s} {'Re':>6s} {'F1':>6s} {'MOTA':>7s} {'IDsw':>6s} {'nGT':>7s}")
        for c, name in ((0, "person"), (1, "animal")):
            pr, re, f1, mota, idsw, ng = met(acc[c])
            print(f"{name:9s} {pr:6.3f} {re:6.3f} {f1:6.3f} {mota:7.3f} {idsw:6d} {ng:7d}")
        pr, re, f1, mota, idsw, ng = met(pool)
        print(f"{'OVERALL':9s} {pr:6.3f} {re:6.3f} {f1:6.3f} {mota:7.3f} {idsw:6d} {ng:7d}")


if __name__ == "__main__":
    main()
