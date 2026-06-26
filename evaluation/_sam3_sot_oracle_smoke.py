"""
SMOKE (Path B', user idea): SOT-style ORACLE -- init SAM3 with frame-0 GT boxes
(one tracked object each) and propagate via SAM2-style mask propagation. Tests
whether SAM3's *tracker* (not its text detector) can hold small thermal objects.
Uses test GT init -> oracle / upper bound, NOT a fair MOT row.

    CUDA_VISIBLE_DEVICES=0 python evaluation/_sam3_sot_oracle_smoke.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets.birdsai_mot import BIRDSAIMOTDataset
from models.sam3 import SAM3Tracker

BIRDSAI_ROOT = "/data/ESA_DLSTEM_2025/data/wild_animal/BIRDSAI"
CANON = {0: "human", 1: "elephant", 2: "giraffe", 3: "lion", 4: "unknown"}
CLIP = 60
CASES = [("0000000012_0000000000", "lion"), ("0000000065_0000000000", "giraffe"),
         ("0000000060_0000000000", "unknown"), ("0000000352_0000000000", "elephant")]


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
        return 0
    iou = iou_mat(g, p); rs, cs = np.where(iou >= thr)
    order = iou[rs, cs].argsort()[::-1]; mg, mp = set(), set(); tp = 0
    for k in order:
        r, c = rs[k], cs[k]
        if r in mg or c in mp:
            continue
        mg.add(r); mp.add(c); tp += 1
    return tp


def main():
    ds = BIRDSAIMOTDataset(root=BIRDSAI_ROOT, split="test", granularity="fine",
                           class_map={v: k for k, v in CANON.items()})
    vmap = {v.video_id: v for v in ds.videos}
    trk = SAM3Tracker()

    for vid, name in CASES:
        video = vmap[vid]
        fids = video.frame_ids[:CLIP]
        frames = [ds._load_frame(video, f) for f in fids]
        a0 = ds._load_annotations(video, fids[0])
        gb0 = np.asarray(a0["boxes"], np.float32).reshape(-1, 4)
        gl0 = np.asarray(a0["labels"], np.int64).reshape(-1)
        gid0 = np.asarray(a0["track_ids"], np.int64).reshape(-1)
        if len(gb0) == 0:
            print(f"[{name}] {vid}: no frame0 GT, skip"); continue
        diag = np.median(np.hypot(gb0[:, 2] - gb0[:, 0], gb0[:, 3] - gb0[:, 1]))

        trk.init_video(frames)
        trk.add_prompts(0, gb0, gl0, obj_ids=[int(i) for i in gid0])
        outs = trk.propagate()

        # per-frame recall@.3/.5 over objects that were init'd at frame 0
        tp3 = tp5 = ngt = npred = 0
        for j, fid in enumerate(fids):
            ann = ds._load_annotations(video, fid)
            gtb = np.asarray(ann["boxes"], np.float32).reshape(-1, 4)
            o = outs[j] if j < len(outs) else None
            pb = o["boxes"].numpy() if o is not None and len(o["boxes"]) else np.zeros((0, 4), np.float32)
            ngt += len(gtb); npred += len(pb)
            tp3 += greedy(gtb, pb, 0.3); tp5 += greedy(gtb, pb, 0.5)
        rec3 = tp3 / max(ngt, 1); rec5 = tp5 / max(ngt, 1)
        prec5 = tp5 / max(npred, 1)
        print(f"[{name:8s}] {vid}  diag={diag:3.0f}px  init_objs={len(gb0)}  "
              f"frames={len(fids)}  | Recall@.3={rec3:.3f}  Recall@.5={rec5:.3f}  "
              f"Prec@.5={prec5:.3f}  (nGT={ngt} nPred={npred})")
        trk.reset_state()


if __name__ == "__main__":
    main()
