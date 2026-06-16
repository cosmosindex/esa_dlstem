#!/usr/bin/env python
"""Export our MOT car datasets to FairMOT/JDE training format.

FairMOT's JointDataset (FairMOT/src/lib/datasets/dataset/jde.py) trains from an
on-disk layout it cannot get from our Python dataset classes:

    <JDE_ROOT>/<dataset>/images/<video_id>/<frame>.<ext>          (image)
    <JDE_ROOT>/<dataset>/labels_with_ids/<video_id>/<frame>.txt   (label)

with a `.txt` list of image paths and a data-cfg JSON. Label rows are
    `class id cx cy w h`
where cx,cy,w,h are normalized to the ORIGINAL image W,H (verified against
jde.get_data: line 183 reads original h,w; lines 192-195 map normalized->pixel).

This script REUSES the project dataset classes (RsCarDataset / SDMCarDataset /
SATMTBDataset) so splits, boxes and track_ids are identical to the rest of the
benchmark. Per-video track_ids are remapped to dataset-global contiguous IDs
(0..nID-1), which is what FairMOT's ReID head needs.

Images: rscardata/satmtb frames exist on disk -> symlinked (no copy).
        sdmcar frames live inside .avi -> decoded sequentially to jpg.

Usage:
    python tools/export_mot_jde.py                          # all 3, split=train
    python tools/export_mot_jde.py --datasets rscardata     # one
    python tools/export_mot_jde.py --limit-videos 1         # smoke (1 vid each)
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.rscardata import RsCarDataset       # noqa: E402
from datasets.sdmcar import SDMCarDataset          # noqa: E402
from datasets.satmtb import SATMTBDataset          # noqa: E402
from datasets.airmot import AIRMOTDataset          # noqa: E402
from datasets.viso import VISODataset              # noqa: E402

# Default car-only root kept for back-compat; the all-class export should be
# run with JDE_ROOT=/data/ESA_DLSTEM_2025/data/fairmot_jde_all so it does NOT
# clobber the car-only export that backs the completed FairMOT/TGraM models.
JDE_ROOT = os.environ.get("JDE_ROOT", "/data/ESA_DLSTEM_2025/data/fairmot_jde")
CFG_DIR = "/home/ziwen/code/esa_dlstem/FairMOT/src/lib/cfg"

# Canonical unified Space-tracker MOT class indices (manifest order).
# Note VISO names planes "plane"; AIR-MOT/SAT-MTB use "airplane" — both map to 1.
CANON = {"car": 0, "airplane": 1, "ship": 2, "train": 3}
CLASS_NAMES = ["car", "airplane", "ship", "train"]

# name -> (builder, image-ext, fairmot data-cfg key)
# class_map passed to each dataset maps its NATIVE category names -> CANON index;
# a category absent from the map is dropped (_map_label returns -1). VISO's car
# is intentionally excluded (replaced by RsCarData, the curated re-annotation),
# matching the `viso_no_car` convention used everywhere else in the repo.
DATASETS = {
    "rscardata": dict(
        ext="jpg", key="rscardata",
        build=lambda split: RsCarDataset(
            root="/data/ESA_DLSTEM_2025/data/trafic/RsCarData",
            split=split, mode="detection", class_map={"car": 0}),
    ),
    "sdmcar": dict(
        ext="jpg", key="sdmcar",
        build=lambda split: SDMCarDataset(
            root="/data/ESA_DLSTEM_2025/data/trafic/SDM-Car",
            split=split, mode="detection", class_map={"car": 0}),
    ),
    "satmtb": dict(
        ext="png", key="satmtb",
        build=lambda split: SATMTBDataset(
            root="/data/ESA_DLSTEM_2025/data/trafic/SAT-MTB",
            split=split, mode="detection", task="mot",
            categories=["car", "airplane", "ship", "train"],
            class_map=CANON),
    ),
    "airmot": dict(
        ext="jpg", key="airmot",
        build=lambda split: AIRMOTDataset(
            root="/data/ESA_DLSTEM_2025/data/trafic/AIR-MOT-100",
            split=split, class_map={"airplane": 1, "ship": 2}),
    ),
    "viso_no_car": dict(
        ext="jpg", key="viso_no_car",
        build=lambda split: VISODataset(
            root="/data/ESA_DLSTEM_2025/data/trafic/VISO",
            split=split, categories=["plane", "ship", "train"],
            class_map={"plane": 1, "ship": 2, "train": 3}),
    ),
}


def source_image_path(name, ds, video, fid):
    """Absolute path of the real on-disk frame, or None if it must be decoded."""
    root = str(ds.root)
    vid = video.video_id
    if name == "rscardata":
        # video_id == "<official_split>/<seq>"; frames at images/<...>/img1/%06d.jpg
        return os.path.join(root, "images", vid, "img1", f"{fid:06d}.jpg")
    if name == "satmtb":
        # video_id == "<cat>/<seq>"; frames at SAT-MTB_Dataset/<cat>/<seq>/img/%06d.png
        return os.path.join(root, "SAT-MTB_Dataset", vid, "img", f"{fid:06d}.png")
    if name == "airmot":
        # video_id == "<seq>"; frames at <seq>/img/%06d[_suffix].jpg (seq 100 uses _8)
        suffix = getattr(video, "_img_suffix", "")
        fname = f"{fid:06d}_{suffix}.jpg" if suffix else f"{fid:06d}.jpg"
        return os.path.join(root, vid, "img", fname)
    if name == "viso_no_car":
        # video_id == "<cat>/<seq>"; frames at mot/<cat>/<seq>/img/%06d.jpg
        cat, seq = vid.split("/")
        return os.path.join(root, "mot", cat, seq, "img", f"{fid:06d}.jpg")
    return None  # sdmcar -> decode from .avi


def write_label(path, rows):
    """rows: list of (cls, gid, cx, cy, w, h). Always write a file (even empty):
    FairMOT's JointDataset.__init__ np.loadtxt's every listed frame's label path
    with no existence check, so a frame with no cars still needs an empty file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for cls, gid, cx, cy, w, h in rows:
            f.write(f"{cls} {gid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def boxes_to_rows(ann, W, H, id_map, vid, next_id):
    """xyxy abs boxes + track_ids -> JDE rows with dataset-global IDs."""
    rows = []
    boxes = ann["boxes"]
    labels = ann["labels"]
    tids = ann["track_ids"]
    for (x1, y1, x2, y2), lab, tid in zip(boxes, labels, tids):
        if lab < 0:  # category not in this dataset's class_map -> drop
            continue
        tid = int(tid)
        if tid >= 0:
            key = (vid, tid)
            if key not in id_map:
                id_map[key] = next_id[0]
                next_id[0] += 1
            gid = id_map[key]
        else:
            gid = -1
        cx = ((x1 + x2) / 2.0) / W
        cy = ((y1 + y2) / 2.0) / H
        bw = (x2 - x1) / W
        bh = (y2 - y1) / H
        if bw <= 0 or bh <= 0:
            continue
        rows.append((int(lab), gid, cx, cy, bw, bh))  # canonical class index
    return rows


def export_dataset(name, split, limit_videos=None, write_cfg=False):
    spec = DATASETS[name]
    ds = spec["build"](split)
    ext = spec["ext"]
    videos = ds.videos if limit_videos is None else ds.videos[:limit_videos]
    print(f"\n=== {name} [{split}] : {len(videos)} videos "
          f"(of {len(ds.videos)}) ===")

    img_root = os.path.join(JDE_ROOT, name, "images")
    lbl_root = os.path.join(JDE_ROOT, name, "labels_with_ids")
    id_map = {}            # (video_id, local_tid) -> global_id
    next_id = [0]          # mutable counter
    txt_lines = []
    n_frames = n_boxes = 0
    cls_counts = {c: 0 for c in range(len(CLASS_NAMES))}  # per-class box counts

    for vi, video in enumerate(videos, 1):
        vid = video.video_id
        fids = list(video.frame_ids)
        # sdmcar: open the .avi once and read sequentially (per-frame _load_frame
        # reopens the file, which is O(n^2) with seeks).
        cap = None
        if name == "sdmcar":
            assert fids == list(range(len(fids))), \
                f"{vid}: expected contiguous 0-indexed frame_ids"
            seq = vid.split("/")[-1]
            # video_id is "<split_name>/<seq>"; the on-disk dir may differ
            # (val -> validation). Map via the dataset's own table.
            split_name = vid.split("/")[0]
            split_dir = getattr(ds, "_SPLIT_DIRS", {}).get(split_name, split_name)
            avi = os.path.join(ds.root, split_dir, f"{seq}.avi")
            cap = cv2.VideoCapture(avi)
            if not cap.isOpened():
                raise FileNotFoundError(f"cannot open {avi}")

        # image size (constant within a video for all 3 datasets)
        W = H = None
        for fid in fids:
            ann = ds._load_annotations(video, fid)
            rel = os.path.join(name, "images", vid, f"{fid:06d}.{ext}")
            dst_img = os.path.join(JDE_ROOT, rel)
            os.makedirs(os.path.dirname(dst_img), exist_ok=True)

            if cap is not None:                       # sdmcar: decode
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"{vid}: read failed at frame {fid}")
                H, W = frame.shape[:2]
                if not os.path.exists(dst_img):
                    cv2.imwrite(dst_img, frame)
            else:                                     # rscardata/satmtb: symlink
                src = source_image_path(name, ds, video, fid)
                if not os.path.exists(src):
                    raise FileNotFoundError(f"{vid} frame {fid}: missing {src}")
                if W is None:
                    W, H = Image.open(src).size       # header read, no decode
                if not os.path.lexists(dst_img):
                    os.symlink(os.path.abspath(src), dst_img)

            rows = boxes_to_rows(ann, W, H, id_map, vid, next_id)
            dst_lbl = os.path.join(lbl_root, vid, f"{fid:06d}.txt")
            write_label(dst_lbl, rows)
            txt_lines.append(rel)
            n_frames += 1
            n_boxes += len(rows)
            for cls, *_ in rows:
                cls_counts[cls] = cls_counts.get(cls, 0) + 1

        if cap is not None:
            cap.release()
        print(f"  [{vi}/{len(videos)}] {vid}: {len(fids)} frames, "
              f"running nID={next_id[0]}")

    nID = next_id[0]
    # write image-list txt
    txt_dir = os.path.join(JDE_ROOT, "txt")
    os.makedirs(txt_dir, exist_ok=True)
    txt_path = os.path.join(txt_dir, f"{name}.{split}")
    with open(txt_path, "w") as f:
        f.write("\n".join(txt_lines) + "\n")

    # write per-dataset FairMOT data-cfg JSON (single-dataset training).
    # OFF by default for the all-class union export: the cfg filename
    # {name}.json lives in the fixed FairMOT CFG_DIR and would clobber the
    # car-only per-dataset cfgs. Union training uses union_all.json instead.
    if split == "train" and write_cfg:
        cfg = {
            "root": JDE_ROOT,
            "train": {spec["key"]: txt_path},
            "test_emb": {spec["key"]: txt_path},
            "test": {spec["key"]: txt_path},
        }
        os.makedirs(CFG_DIR, exist_ok=True)
        cfg_path = os.path.join(CFG_DIR, f"{name}.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"  -> cfg:   {cfg_path}")

    per_cls = {CLASS_NAMES[c]: n for c, n in sorted(cls_counts.items()) if n}
    print(f"  -> {n_frames} frames, {n_boxes} boxes, nID={nID}")
    print(f"  -> per-class boxes: {per_cls}")
    print(f"  -> list:  {txt_path}")
    return dict(dataset=name, split=split, videos=len(videos),
                frames=n_frames, boxes=n_boxes, nID=nID, txt=txt_path,
                per_cls=per_cls)


def write_union_cfg(datasets, union_name="union_all"):
    """Write <union_name>.json with a 'train' dict (one list per dataset) and a
    'val' dict. FairMOT's JointDataset offsets each dataset's contiguous IDs
    into a global space, so no re-export is needed. The val lists are used only
    for validation loss (id loss ignored in selection)."""
    txt_dir = os.path.join(JDE_ROOT, "txt")
    train_paths, val_paths = {}, {}
    for name in datasets:
        key = DATASETS[name]["key"]
        tr = os.path.join(txt_dir, f"{name}.train")
        va = os.path.join(txt_dir, f"{name}.val")
        if not os.path.exists(tr):
            raise FileNotFoundError(f"missing train list {tr} (run --split train first)")
        if not os.path.exists(va):
            raise FileNotFoundError(f"missing val list {va} (run --split val first)")
        train_paths[key] = tr
        val_paths[key] = va
    cfg = {"root": JDE_ROOT, "train": train_paths, "val": val_paths,
           "test_emb": dict(train_paths), "test": dict(train_paths)}
    os.makedirs(CFG_DIR, exist_ok=True)
    cfg_path = os.path.join(CFG_DIR, f"{union_name}.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\n=== union cfg ===\n  -> {cfg_path}")
    print(f"     train: {list(train_paths)}  val: {list(val_paths)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["rscardata", "sdmcar", "satmtb",
                             "airmot", "viso_no_car"],
                    choices=list(DATASETS))
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit-videos", type=int, default=None,
                    help="export only the first N videos per dataset (smoke)")
    ap.add_argument("--write-union", action="store_true",
                    help="(re)write <union-name>.json from existing train+val "
                         "lists and exit (no frame export)")
    ap.add_argument("--union-name", default="union_all",
                    help="basename of the union data-cfg JSON written to CFG_DIR")
    ap.add_argument("--per-dataset-cfg", action="store_true",
                    help="also write a single-dataset {name}.json cfg "
                         "(off by default; clobbers car-only cfgs)")
    args = ap.parse_args()

    print(f"JDE_ROOT = {JDE_ROOT}")
    if args.write_union:
        write_union_cfg(args.datasets, args.union_name)
        return

    summary = [export_dataset(n, args.split, args.limit_videos,
                              write_cfg=args.per_dataset_cfg)
               for n in args.datasets]
    print("\n=== summary ===")
    for s in summary:
        print(f"  {s['dataset']:12s} {s['split']:6s} "
              f"videos={s['videos']:3d} frames={s['frames']:6d} "
              f"boxes={s['boxes']:7d} nID={s['nID']:5d}  {s.get('per_cls', {})}")


if __name__ == "__main__":
    main()
