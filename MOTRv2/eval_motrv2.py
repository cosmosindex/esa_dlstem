"""Evaluate MOTRv2 from an on-disk proposal database.

Upstream
--------
MOTRv2 -- https://github.com/megvii-research/MOTRv2, commit ``1aac7c3``.
This tree is that clone with the changes below; its upstream history is parked
at ``MOTRv2/.git-upstream``, so
``git --git-dir=.git-upstream diff`` reproduces this list exactly.

* ``datasets/dance.py`` -- the training split directory is an argument
  (``--mot_train_name``) instead of the literal ``DanceTrack/train``; and
  ``targets['labels']`` is built with ``dtype=torch.long``. The second is
  load-bearing: a satellite frame can hold zero ground-truth objects, and
  ``torch.as_tensor([])`` is float32, which promotes the concatenated per-frame
  labels to float and breaks the matcher's ``cost[:, tgt_ids]`` indexing.
  DanceTrack never has an empty frame, so upstream cannot hit it.
* ``main.py``, ``submit_dance.py``, ``util/tool.py`` -- ``torch.load(...,
  weights_only=False)``. The checkpoints carry an ``argparse.Namespace``, which
  torch 2.6 refuses to unpickle by default.
* ``models/ops/`` -- CUDA sources updated for the current toolkit.
* ``configs/motrv2_union.args``, ``configs/motrv2_st_nocar.args``,
  ``eval_motrv2.py``, ``eval_motrv2_oracle_assoc.py`` -- ours, not upstream's.

The ``.args`` files are read with ``args=$(cat configs/x.args)``, which does not
expand shell variables, so pass them through ``envsubst`` first::

    args=$(envsubst < configs/motrv2_st_nocar.args)

``$DATA_ROOT`` and ``$CHECKPOINT_ROOT`` are the only machine-specific values in
them.

What this script is
-------------------
MOTRv2 is the query-based end-to-end transformer paradigm row of the MOT study
(track queries propagated by transformer self-attention; no center-sampled ReID).
It does not detect on its own: its "detection" front-end is a proposal channel
(YOLOX in the paper). **This script is agnostic to where those proposals come
from** -- whatever `--det_db` points at is what MOTRv2 sees. That makes it serve
two different experiments, and the distinction matters:

* `--det_db` = the Faster R-CNN detections every TBD tracker consumes -> this is
  MOTRv2's ordinary self-detection row, directly comparable to the rest of the
  main table.
* `--det_db` = GROUND-TRUTH boxes at score 1.0 -> the association oracle of
  fair-comparison Experiment 2, where detection is held perfect for every method
  so any difference is pure association ability
  (Experiment 2 in the paper's supplementary).

Pass `--proposal-tag` to record which of the two a run was, since nothing in the
output is otherwise able to tell them apart. It was previously hard-coded to
"gt_box_oracle", which mislabelled the Faster R-CNN runs.

This reads ONLY the on-disk MOTRv2 eval layout produced by
``tools/export_motrv2.py --mode eval`` (so it never imports the project's
``datasets``/``models`` packages, which collide with MOTRv2's own):

    <mot_path>/eval/<dataset>/<safe_id>/img1/{fid:08d}.jpg
    <mot_path>/det_db_eval_gt.json   key "eval/<dataset>/<safe_id>/img1/{fid:08d}.txt"

and emits the standard scorer contract (matched by compute_hota_by_size.py):

    {EXPERIMENT_ROOT}/motrv2_{proposal_tag}_{dataset}_{TS}/mot_format/<safe_id>.txt
        frame,id,x,y,w,h,conf,-1,-1,-1     (frame = NATIVE fid from the filename)
    .../test_metrics.json

Run from inside MOTRv2/ with CUDA_VISIBLE_DEVICES=0::

    python eval_motrv2.py --resume exps/.../checkpointNN.pth \
        --mot_path ${DATA_ROOT}/data/motrv2 \
        --det_db det_db_eval_gt.json --dataset all
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


def _filter_by_score(dt, thr):
    keep = (dt.scores > thr) & (dt.obj_idxes >= 0)
    return dt[keep]


def _filter_by_area(dt, thr):
    wh = dt.boxes[:, 2:4] - dt.boxes[:, 0:2]
    return dt[(wh[:, 0] * wh[:, 1]) > thr]


@torch.no_grad()
def run_video(detr, mot_path, det_db, vid_rel, prob_thr, area_thr):
    """vid_rel = 'eval/<dataset>/<safe_id>'. Returns (mot_lines, n_frames, n_out)."""
    img_dir = os.path.join(mot_path, vid_rel, "img1")
    fnames = sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg"))
    img_list = [os.path.join(vid_rel, "img1", f) for f in fnames]
    fids = [int(f[:-4]) for f in fnames]                    # NATIVE frame ids

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

        dt = deepcopy(track_instances)
        dt = _filter_by_score(dt, prob_thr)
        dt = _filter_by_area(dt, area_thr)
        fid = fids[i]
        for (x1, y1, x2, y2), tid, sc in zip(dt.boxes.tolist(),
                                             dt.obj_idxes.tolist(),
                                             dt.scores.tolist()):
            if tid < 0:
                continue
            mot_lines.append(f"{fid},{int(tid)},{x1:.2f},{y1:.2f},"
                             f"{x2 - x1:.2f},{y2 - y1:.2f},{sc:.4f},-1,-1,-1")
            n_out += 1
    return mot_lines, len(fids), n_out


def main():
    parser = argparse.ArgumentParser("MOTRv2 evaluation from a proposal DB",
                                     parents=[get_args_parser()])
    parser.add_argument("--proposal-tag", default="gt_oracle",
                        choices=["gt_oracle", "frcnn"],
                        help="What --det_db actually holds. Names the output "
                             "directory and is recorded in test_metrics.json; "
                             "the two are NOT comparable to each other.")
    parser.add_argument("--score_threshold", default=0.5, type=float)
    parser.add_argument("--update_score_threshold", default=0.5, type=float)
    parser.add_argument("--miss_tolerance", default=20, type=int)
    parser.add_argument("--area_threshold", default=0.0, type=float,
                        help="min box area (px^2); 0 keeps satellite micro-objects")
    parser.add_argument("--dataset", default="all",
                        help="one of the 5 datasets, or 'all'")
    args = parser.parse_args()

    datasets = _DATASETS if args.dataset == "all" else [args.dataset]
    exp_root = os.environ.get("EXPERIMENT_ROOT", "/work/anon/experiments")

    detr, _, _ = build_model(args)
    detr.track_embed.score_thr = args.update_score_threshold
    detr.track_base = RuntimeTrackerBase(args.score_threshold,
                                         args.score_threshold, args.miss_tolerance)
    detr = load_model(detr, args.resume)
    detr.eval().cuda()

    with open(os.path.join(args.mot_path, args.det_db)) as f:
        det_db = json.load(f)

    for dataset in datasets:
        ds_root = os.path.join(args.mot_path, "eval", dataset)
        if not os.path.isdir(ds_root):
            print(f"!! skip {dataset}: {ds_root} missing")
            continue
        run_name = f"motrv2_{args.proposal_tag}_{dataset}"
        exp_dir = Path(f"{exp_root}/{run_name}_{datetime.now():%Y%m%d_%H%M%S}")
        mot_dir = exp_dir / "mot_format"
        mot_dir.mkdir(parents=True, exist_ok=True)
        print("=" * 60)
        print(f"MOTRv2 [proposals={args.proposal_tag}]: {dataset}  ->  {exp_dir}")
        print("=" * 60)

        safe_ids = sorted(d for d in os.listdir(ds_root)
                          if os.path.isdir(os.path.join(ds_root, d)))
        t0 = time.perf_counter()
        n_frames_total = n_out_total = 0
        for k, sid in enumerate(safe_ids, 1):
            lines, nf, no = run_video(detr, args.mot_path, det_db,
                                      f"eval/{dataset}/{sid}",
                                      args.score_threshold, args.area_threshold)
            (mot_dir / f"{sid}.txt").write_text("\n".join(lines))
            n_frames_total += nf
            n_out_total += no
            print(f"  [{k}/{len(safe_ids)}] {sid}: {nf} frames, {no} out-tracks")

        dt = time.perf_counter() - t0
        (exp_dir / "test_metrics.json").write_text(json.dumps({
            "tracker": "motrv2", "dataset": dataset,
            "proposal_source": args.proposal_tag, "det_db": args.det_db,
            "checkpoint": args.resume, "num_videos": len(safe_ids),
            "total_frames": n_frames_total, "total_out_tracks": n_out_total,
            "score_threshold": args.score_threshold,
            "area_threshold": args.area_threshold,
            "total_time_s": dt, "fps": n_frames_total / max(dt, 1e-9),
        }, indent=2))
        print(f"  -> frames={n_frames_total} out={n_out_total} "
              f"time={dt:.1f}s  {exp_dir}")


if __name__ == "__main__":
    main()
