"""Diagnostic: dump the FULL output-track score distribution for the first N
frames of one rscardata eval sequence, with NO score filtering, so we can see
where the bulk of the (dropped at thr=0.5) tracks actually sit.

Bounded to N frames -> stale tracks can't accumulate -> no QIM OOM.
"""
import argparse, json, os
from copy import deepcopy
import torch
from torch.utils.data import DataLoader
from main import get_args_parser
from models import build_model
from util.tool import load_model
from submit_dance import ListImgDataset, RuntimeTrackerBase


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(parents=[get_args_parser()])
    p.add_argument("--vid_rel", default="eval/rscardata/test1024_009")
    p.add_argument("--n_frames", default=30, type=int)
    p.add_argument("--miss_tolerance", default=20, type=int)
    p.add_argument("--thr", default=0.3, type=float, help="real pruning threshold")
    args = p.parse_args()

    detr, _, _ = build_model(args)
    # use the REAL pruning threshold so stale tracks are removed like eval
    detr.track_embed.score_thr = args.thr
    detr.track_base = RuntimeTrackerBase(args.thr, args.thr, args.miss_tolerance)
    detr = load_model(detr, args.resume)
    detr.eval().cuda()

    det_db = json.load(open(os.path.join(args.mot_path, args.det_db)))
    img_dir = os.path.join(args.mot_path, args.vid_rel, "img1")
    fnames = sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg"))[:args.n_frames]
    img_list = [os.path.join(args.vid_rel, "img1", f) for f in fnames]

    loader = DataLoader(ListImgDataset(args.mot_path, img_list, det_db), 1, num_workers=0)
    detr.track_base.clear()
    track_instances = None
    all_scores = []
    n_prop_total = 0
    for i, data in enumerate(loader):
        cur_img, ori_img, proposals = [d[0] for d in data]
        cur_img, proposals = cur_img.cuda(), proposals.cuda()
        n_prop_total += proposals.shape[0]
        if track_instances is not None:
            track_instances.remove("boxes")
            track_instances.remove("labels")
        seq_h, seq_w, _ = ori_img.shape
        res = detr.inference_single_image(cur_img, (seq_h, seq_w), track_instances, proposals)
        track_instances = res["track_instances"]
        dt = deepcopy(track_instances)
        keep = (dt.obj_idxes >= 0) & (dt.scores > args.thr)
        all_scores.append(dt.scores[keep].tolist())

    import numpy as np
    per_frame = [len(x) for x in all_scores]
    s = np.array([v for x in all_scores for v in x])
    print(f"vid={args.vid_rel} frames={len(fnames)} thr={args.thr} "
          f"miss_tol={args.miss_tolerance}")
    print(f"proposals fed: avg {n_prop_total/len(fnames):.1f}/frame (= GT objects/frame)")
    print(f"OUTPUT tracks (obj_idx>=0 & score>thr): total={s.size} "
          f"avg {np.mean(per_frame):.1f}/frame  max {max(per_frame)}/frame")
    print(f"  ratio output/GT per frame: {np.mean(per_frame)/(n_prop_total/len(fnames)):.2f}x")
    if s.size:
        print(f"  surviving score: min={s.min():.3f} p50={np.median(s):.3f} max={s.max():.3f}")


if __name__ == "__main__":
    main()
