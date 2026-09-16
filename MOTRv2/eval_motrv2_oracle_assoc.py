"""GT-box association ORACLE for MOTRv2 — fair-comparison Experiment 2,
PROJECT-TO-GT readout (replaces the naive score-threshold readout).

Why this exists
---------------
MOTRv2 is the query-based end-to-end transformer paradigm row. Unlike the
tracking-by-detection methods (SORT/ByteTrack/.../FairMOT/TGraM) where the fed
GT boxes pass straight through to the output, MOTRv2's GT boxes are only
*proposals* (anchors): the transformer still emits its OWN boxes scored by its
learned confidence head, and ``RuntimeTrackerBase`` keeps only ``score > thr``.

On dense small-object satellite scenes the confidence head collapses into a
razor-thin band (~[0.30, 0.46]) with no real/spurious separation, so a single
score threshold either drops ~99 % of objects (thr 0.5) or floods 100-150x
spurious tracks (thr 0.3). Either way the AssA reflects a detection/confidence
artifact, NOT association — defeating the whole point of Exp2.

Fix (PROJECT-TO-GT): hold detection literally perfect by OUTPUTTING the GT boxes
themselves, and read MOTRv2's *association* off them. We run the model with a
low id-assignment threshold so it propagates track ids normally, then per frame
match each GT box to the model's track query of highest IoU and stamp that
query's track id onto the GT box. Detection = the GT box set (DetA perfect vs
gt.txt); association = MOTRv2's propagated track id. Spurious / over-emitted
tracks that don't best-match any GT are simply discarded by the projection, so
the 100x redundancy collapses to exactly |GT| outputs per frame. A GT box with
no track query above the IoU gate gets a fresh unique id (= MOTRv2 failed to
track it -> a faithful fragmentation that lowers AssA).

Emits the standard scorer contract (matched by compute_hota_by_size.py):
    {EXPERIMENT_ROOT}/motrv2_oracle_{dataset}_{TS}/mot_format/<safe_id>.txt
        frame,id,x,y,w,h,1.0,-1,-1,-1     (frame = NATIVE fid from filename)
    .../test_metrics.json

Run from inside MOTRv2/ with CUDA_VISIBLE_DEVICES=0.
"""
import argparse
import json
import os
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from main import get_args_parser
from models import build_model
from util.tool import load_model
from submit_dance import ListImgDataset, RuntimeTrackerBase

_DATASETS = ["rscardata", "satmtb", "sdmcar", "airmot", "viso_no_car"]
_FRESH_ID_BASE = 1_000_000          # unmatched-GT id namespace (never collides)


def _box_iou(a, b):
    """a:[N,4] b:[M,4] xyxy abs -> IoU [N,M]."""
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-6)


@torch.no_grad()
def run_video(detr, mot_path, det_db, vid_rel, iou_gate, fresh_counter):
    """vid_rel = 'eval/<dataset>/<safe_id>'. Returns (mot_lines, n_frames, n_out, fresh_counter)."""
    img_dir = os.path.join(mot_path, vid_rel, "img1")
    fnames = sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg"))
    img_list = [os.path.join(vid_rel, "img1", f) for f in fnames]
    fids = [int(f[:-4]) for f in fnames]                       # NATIVE frame ids

    loader = DataLoader(ListImgDataset(mot_path, img_list, det_db), 1, num_workers=0)
    detr.track_base.clear()
    track_instances = None
    mot_lines, n_out = [], 0
    for i, data in enumerate(loader):
        cur_img, ori_img, proposals = [d[0] for d in data]
        cur_img, proposals = cur_img.cuda(), proposals.cuda()
        if track_instances is not None:
            track_instances.remove("boxes")
            track_instances.remove("labels")
        seq_h, seq_w, _ = ori_img.shape
        res = detr.inference_single_image(cur_img, (seq_h, seq_w), track_instances, proposals)
        track_instances = res["track_instances"]

        # GT boxes for THIS frame = the fed proposals, in abs ltwh (== gt.txt set).
        key = img_list[i][:-4] + ".txt"
        gt_ltwh = [list(map(float, ln.split(",")))[:4] for ln in det_db.get(key, [])]
        fid = fids[i]
        if not gt_ltwh:
            continue
        gt_t = torch.tensor(gt_ltwh)                            # [G,4] l,t,w,h (abs)
        gt_xyxy = torch.stack([gt_t[:, 0], gt_t[:, 1],
                               gt_t[:, 0] + gt_t[:, 2],
                               gt_t[:, 1] + gt_t[:, 3]], 1)

        # model's tracked queries that own an id this frame
        dt = deepcopy(track_instances)
        keep = dt.obj_idxes >= 0
        tr_xyxy = dt.boxes[keep].cpu()                          # [T,4] xyxy abs
        tr_ids = dt.obj_idxes[keep].cpu().tolist()

        if len(tr_ids):
            iou = _box_iou(gt_xyxy, tr_xyxy)                    # [G,T]
            best_iou, best_j = iou.max(dim=1)
        else:
            best_iou = torch.zeros(len(gt_ltwh))
            best_j = torch.zeros(len(gt_ltwh), dtype=torch.long)

        for g, (l, t, w, h) in enumerate(gt_ltwh):
            if len(tr_ids) and best_iou[g].item() >= iou_gate:
                tid = tr_ids[best_j[g].item()]                  # MOTRv2's association
            else:
                tid = fresh_counter                             # untracked GT -> unique id
                fresh_counter += 1
            mot_lines.append(f"{fid},{int(tid)},{l:.2f},{t:.2f},{w:.2f},{h:.2f},1.0,-1,-1,-1")
            n_out += 1
    return mot_lines, len(fids), n_out, fresh_counter


def main():
    parser = argparse.ArgumentParser("MOTRv2 GT-box oracle (Exp2, project-to-GT)",
                                     parents=[get_args_parser()])
    parser.add_argument("--id_score_threshold", default=0.3, type=float,
                        help="RuntimeTrackerBase new-id / keep threshold; low so "
                             "the model assigns & propagates track ids on dense scenes")
    parser.add_argument("--miss_tolerance", default=20, type=int)
    parser.add_argument("--iou_gate", default=0.5, type=float,
                        help="min IoU to bind a GT box to a model track query")
    parser.add_argument("--dataset", default="all",
                        help="one of the 5 datasets, or 'all'")
    args = parser.parse_args()

    datasets = _DATASETS if args.dataset == "all" else [args.dataset]
    exp_root = os.environ.get("EXPERIMENT_ROOT", "/work/anon/experiments")

    detr, _, _ = build_model(args)
    thr = args.id_score_threshold
    detr.track_embed.score_thr = thr
    detr.track_base = RuntimeTrackerBase(thr, thr, args.miss_tolerance)
    detr = load_model(detr, args.resume)
    detr.eval().cuda()

    with open(os.path.join(args.mot_path, args.det_db)) as f:
        det_db = json.load(f)

    for dataset in datasets:
        ds_root = os.path.join(args.mot_path, "eval", dataset)
        if not os.path.isdir(ds_root):
            print(f"!! skip {dataset}: {ds_root} missing")
            continue
        run_name = f"motrv2_oracle_{dataset}"
        exp_dir = Path(f"{exp_root}/{run_name}_{datetime.now():%Y%m%d_%H%M%S}")
        mot_dir = exp_dir / "mot_format"
        mot_dir.mkdir(parents=True, exist_ok=True)
        print("=" * 60)
        print(f"MOTRv2 GT-box ORACLE (project-to-GT): {dataset}  ->  {exp_dir}")
        print(f"  id_thr={thr} iou_gate={args.iou_gate} miss_tol={args.miss_tolerance}")
        print("=" * 60)

        safe_ids = sorted(d for d in os.listdir(ds_root)
                          if os.path.isdir(os.path.join(ds_root, d)))
        t0 = time.perf_counter()
        n_frames_total = n_out_total = 0
        fresh = _FRESH_ID_BASE
        for k, sid in enumerate(safe_ids, 1):
            lines, nf, no, fresh = run_video(detr, args.mot_path, det_db,
                                             f"eval/{dataset}/{sid}",
                                             args.iou_gate, fresh)
            (mot_dir / f"{sid}.txt").write_text("\n".join(lines))
            n_frames_total += nf
            n_out_total += no
            print(f"  [{k}/{len(safe_ids)}] {sid}: {nf} frames, {no} out-tracks")

        dt = time.perf_counter() - t0
        (exp_dir / "test_metrics.json").write_text(json.dumps({
            "tracker": "motrv2_oracle", "dataset": dataset,
            "mode": "gt_box_oracle_project_to_gt",
            "checkpoint": args.resume, "num_videos": len(safe_ids),
            "total_frames": n_frames_total, "total_out_tracks": n_out_total,
            "id_score_threshold": thr, "iou_gate": args.iou_gate,
            "miss_tolerance": args.miss_tolerance,
            "n_untracked_gt": fresh - _FRESH_ID_BASE,
            "total_time_s": dt, "fps": n_frames_total / max(dt, 1e-9),
        }, indent=2))
        print(f"  -> frames={n_frames_total} out={n_out_total} "
              f"untracked_gt={fresh - _FRESH_ID_BASE} time={dt:.1f}s  {exp_dir}")


if __name__ == "__main__":
    main()
