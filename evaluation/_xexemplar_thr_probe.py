"""Diagnostic: per-frame exemplar-detection count + recall/precision vs score
threshold, to choose the SOT seed threshold. Reuses the eval's detector+banks.

    CUDA_VISIBLE_DEVICES=0 python evaluation/_xexemplar_thr_probe.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from datasets.birdsai_mot import BIRDSAIMOTDataset
import evaluation.eval_birdsai_sam3_xexemplar as X

BIRDSAI_ROOT = "/data/ESA_DLSTEM_2025/data/wild_animal/BIRDSAI"
CANON = X.CANON
cmap = {v: k for k, v in CANON.items()}
THRS = [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
IOU_HIT = 0.5

# (video_id, expected class) — giraffe/lion first (method should work here)
TARGETS = [("0000000065_0000000000", 2),   # giraffe
           ("0000000358_0000000000", 2),   # giraffe (2nd)
           ("0000000012_0000000000", 3)]   # lion


def main():
    np.random.seed(0)
    ds = BIRDSAIMOTDataset(root=BIRDSAI_ROOT, split="test", granularity="fine",
                           annotations_dirname="annotations_sam3", class_map=cmap)
    ds_tr = BIRDSAIMOTDataset(root=BIRDSAI_ROOT, split="train", granularity="fine",
                              annotations_dirname="annotations_sam3", class_map=cmap)
    by_id = {v.video_id: v for v in ds.videos}
    det = X.XExemplarDetector()
    rng = np.random.default_rng(0)

    for vid, cls in TARGETS:
        if vid not in by_id:
            print(f"{vid}: not in test set, skip"); continue
        v = by_id[vid]
        fids = v.frame_ids
        f0 = Image.fromarray(ds._load_frame(v, fids[0]))
        banks = X.select_banks(det, ds_tr, {cls}, f0, rng, verbose=True)
        if cls not in banks:
            print(f"{vid}: no bank"); continue
        bank = banks[cls]
        sample = fids[::max(1, len(fids) // 10)][:10]
        agg = {t: [0, 0, 0] for t in THRS}   # dets, dets-hitting-gt, gt-matched
        ngt = 0
        for fid in sample:
            q = Image.fromarray(ds._load_frame(v, fid))
            ann = ds._load_annotations(v, fid)
            gt = np.asarray([b for b, l in zip(ann["boxes"], ann["labels"]) if int(l) == cls],
                            np.float32).reshape(-1, 4)
            ngt += len(gt)
            bx, sc = det.detect(q, bank)
            for t in THRS:
                keep = sc >= t
                b = bx[keep]
                agg[t][0] += len(b)
                if len(b) and len(gt):
                    iou = X.iou_matrix(b, gt)
                    agg[t][1] += int((iou >= IOU_HIT).any(1).sum())
                    agg[t][2] += int((iou >= IOU_HIT).any(0).sum())
        print(f"\n=== {vid} [{CANON[cls]}] {len(sample)} frames, {ngt} GT ===")
        print(f"  {'thr':>4} | {'dets':>5} {'d/frm':>6} | recall  prec")
        for t in THRS:
            d, hit, matched = agg[t]
            rec = matched / max(ngt, 1); prec = hit / max(d, 1)
            print(f"  {t:>4} | {d:5d} {d/len(sample):6.1f} | {rec:.3f}  {prec:.3f}")


if __name__ == "__main__":
    main()
