#!/usr/bin/env python
"""Export our 5 MOT datasets to the MOTRv2 on-disk layout (+ GT-box det_db).

MOTRv2 (the query-based end-to-end transformer paradigm row of the MOT study)
trains/infers off a MOT-Challenge-style tree it cannot get from our Python
dataset classes:

    <ROOT>/<split>/<vid>/img1/{t:08d}.jpg     (frame, t = native frame_id)
    <ROOT>/<split>/<vid>/gt/gt.txt            (t,i,x,y,w,h,1,1  — xywh top-left)

and a ``det_db`` JSON of per-frame *proposals* (anchor queries):

    { "<split>/<vid>/img1/{t:08d}.txt": ["x,y,w,h,score", ...], ... }

For fair-comparison **Experiment 2** (association vs object size, GT-box oracle)
the proposals ARE the ground-truth boxes (score=1) at BOTH train and eval, so
detection is held perfect and only the learned track-query association is
measured — the same isolation the JDT/TBD oracles provide (see
``docs/mot_fair_comparison_framework.md``). MOTRv2 has no native detector and
the cached HiEUM detections are car-only, so GT proposals are also the only
source that uniformly covers all classes (car -> airplane/ship/train). The
proposal source is reported as a caveat, mirroring the "ReID training data"
column for the other paradigms.

This REUSES ``tools/export_mot_jde.py`` (same dataset builders, the
``source_image_path`` resolver and the sdmcar .avi decode path) so splits,
boxes and track_ids are byte-identical to the rest of the benchmark.

Two layouts (separate dir naming, one tree):
  * train: union of every dataset's TRAIN split, flattened to one split dir.
           vid dir = ``{dataset}__{safe_id}`` (globbed together by dance.py).
  * eval:  per-dataset TEST split. vid dir = ``{safe_id}`` (no prefix) so the
           size-stratified scorer maps ``mot_format/<safe_id>.txt`` straight to
           ``_safe_video_id(video.video_id)``.

Images: rscardata/satmtb/airmot/viso frames exist on disk -> symlinked (PIL/cv2
read by content, so a ``.jpg`` symlink to a ``.png`` is fine). sdmcar frames
live in .avi -> decoded sequentially to jpg.

Usage::

    python tools/export_motrv2.py --mode train          # union train tree
    python tools/export_motrv2.py --mode eval            # per-dataset test trees
    python tools/export_motrv2.py --mode train --limit-videos 1   # smoke
"""
import argparse
import json
import os
import sys

import cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.export_mot_jde import DATASETS, source_image_path  # noqa: E402

MOTR_ROOT = os.environ.get("MOTR_ROOT", "/data/ESA_DLSTEM_2025/data/motrv2")


def safe_id(video_id: str) -> str:
    return video_id.replace("/", "_")


def _open_avi(ds, video):
    """Open the sdmcar .avi for sequential decode (per-frame reopen is O(n^2))."""
    vid = video.video_id
    seq = vid.split("/")[-1]
    split_name = vid.split("/")[0]
    split_dir = getattr(ds, "_SPLIT_DIRS", {}).get(split_name, split_name)
    avi = os.path.join(ds.root, split_dir, f"{seq}.avi")
    cap = cv2.VideoCapture(avi)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {avi}")
    return cap


def export_video(name, ds, video, vid_dir, det_db, det_key_prefix):
    """Materialise one video's img1/ + gt/gt.txt; append its proposals to det_db.

    Returns (n_frames, n_boxes). Proposals == GT boxes (score=1); gt.txt targets
    use the SAME filtered box set (label in class_map, track_id >= 0) so the
    oracle is internally consistent at train and eval.
    """
    img_dir = os.path.join(vid_dir, "img1")
    gt_dir = os.path.join(vid_dir, "gt")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    fids = list(video.frame_ids)
    cap = None
    if name == "sdmcar":
        assert fids == list(range(len(fids))), \
            f"{video.video_id}: expected contiguous 0-indexed frame_ids"
        cap = _open_avi(ds, video)

    gt_lines = []
    n_boxes = 0
    W = H = None
    for fid in fids:
        ann = ds._load_annotations(video, fid)
        dst_img = os.path.join(img_dir, f"{fid:08d}.jpg")

        if cap is not None:                       # sdmcar: decode to jpg
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"{video.video_id}: read failed at frame {fid}")
            H, W = frame.shape[:2]
            if not os.path.exists(dst_img):
                cv2.imwrite(dst_img, frame)
        else:                                     # others: symlink to raw frame
            src = source_image_path(name, ds, video, fid)
            if not os.path.exists(src):
                raise FileNotFoundError(f"{video.video_id} frame {fid}: missing {src}")
            if W is None:
                W, H = Image.open(src).size       # header read, no decode
            if not os.path.lexists(dst_img):
                os.symlink(os.path.abspath(src), dst_img)

        # filtered GT boxes -> gt.txt rows (targets) AND det_db (proposals)
        boxes = ann["boxes"]
        labels = ann["labels"]
        tids = ann["track_ids"]
        prop_lines = []
        for (x1, y1, x2, y2), lab, tid in zip(boxes, labels, tids):
            if lab < 0:                           # category not in class_map
                continue
            tid = int(tid)
            if tid < 0:                           # untracked box -> not usable
                continue
            x, y = float(x1), float(y1)
            w, h = float(x2 - x1), float(y2 - y1)
            if w <= 0 or h <= 0:
                continue
            gt_lines.append(f"{fid},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,1")
            prop_lines.append(f"{x:.2f},{y:.2f},{w:.2f},{h:.2f},1.0")
            n_boxes += 1
        det_db[f"{det_key_prefix}/img1/{fid:08d}.txt"] = prop_lines

    if cap is not None:
        cap.release()
    with open(os.path.join(gt_dir, "gt.txt"), "w") as f:
        f.write("\n".join(gt_lines) + ("\n" if gt_lines else ""))
    return len(fids), n_boxes


def export_split(mode, datasets, split, limit_videos):
    """mode 'train' -> union tree under <ROOT>/train; 'eval' -> per-dataset
    trees under <ROOT>/eval/<name>. Writes the matching det_db JSON."""
    det_db = {}
    summary = []
    for name in datasets:
        ds = DATASETS[name]["build"](split)
        videos = ds.videos if limit_videos is None else ds.videos[:limit_videos]
        n_frames = n_boxes = 0
        for vi, video in enumerate(videos, 1):
            sid = safe_id(video.video_id)
            if mode == "train":
                vid_name = f"{name}__{sid}"
                vid_dir = os.path.join(MOTR_ROOT, "train", vid_name)
                key_prefix = f"train/{vid_name}"
            else:
                vid_dir = os.path.join(MOTR_ROOT, "eval", name, sid)
                key_prefix = f"eval/{name}/{sid}"
            nf, nb = export_video(name, ds, video, vid_dir, det_db, key_prefix)
            n_frames += nf
            n_boxes += nb
            print(f"  [{name}] [{vi}/{len(videos)}] {video.video_id}: "
                  f"{nf} frames, {nb} boxes")
        summary.append((name, len(videos), n_frames, n_boxes))

    db_name = "det_db_train_gt.json" if mode == "train" else "det_db_eval_gt.json"
    db_path = os.path.join(MOTR_ROOT, db_name)
    os.makedirs(MOTR_ROOT, exist_ok=True)
    with open(db_path, "w") as f:
        json.dump(det_db, f)
    print(f"\n=== {mode} summary (det_db: {db_path}, {len(det_db)} frames) ===")
    for name, nv, nf, nb in summary:
        print(f"  {name:12s} videos={nv:3d} frames={nf:6d} boxes={nb:8d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["train", "eval"])
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS),
                    choices=list(DATASETS))
    ap.add_argument("--limit-videos", type=int, default=None,
                    help="export only the first N videos per dataset (smoke)")
    args = ap.parse_args()

    split = "train" if args.mode == "train" else "test"
    print(f"MOTR_ROOT = {MOTR_ROOT}   mode={args.mode}  split={split}")
    export_split(args.mode, args.datasets, split, args.limit_videos)


if __name__ == "__main__":
    main()
