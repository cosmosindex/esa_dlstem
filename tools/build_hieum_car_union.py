"""Assemble the WHOLE space-tracker car-class MOT training set into HiEUM's
COCO-MOT layout so HiEUM (Xiao et al., TPAMI 2024) can be retrained on it.

The author's checkpoint was trained on RsCarData alone. Our space-tracker
benchmark adds two more car MOT datasets (SAT-MTB car, SDM-Car). This script
materialises every car train/val frame across all three via each dataset's own
loader (uniform `_load_frame` / `_load_annotations`, so AVI vs PNG vs JPG and
the RsCarData XML-test override are all handled transparently) and writes:

    OUT/
      images/<dataset>_<seq>/img1/<frame:06d>.jpg     (all frames as jpg)
      annotations/train_mot.json                      (union train, HiEUM schema)
      annotations/test1024_mot.json                   (union VAL — used only as
                                                       HiEUM's in-training val set)

HiEUM consumes `train_mot.json` with `--sup_mode 0` (supervised, GT boxes).
Image ids are globally consecutive *within* each video and videos are contiguous
blocks, which is what HiEUM's `get_im_ids` (im_ids = [img_id+i for i in range(T)])
requires. Videos shorter than `--seq-len` frames are dropped (cannot form a clip).

Usage:
    python tools/build_hieum_car_union.py --out /work/ziwen/data/hieum_car_union \
        --splits train val --seq-len 20 [--jpg-quality 95]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from datasets.rscardata import RsCarDataset
from datasets.satmtb import SATMTBDataset
from datasets.sdmcar import SDMCarDataset

# (key, class, root, extra-kwargs) — the project's car MOT benchmark (eval_tracker._DATASET_TABLE)
SPECS = [
    ("rscardata", RsCarDataset, "/data/ESA_DLSTEM_2025/data/trafic/RsCarData", {}),
    ("satmtb",    SATMTBDataset, "/data/ESA_DLSTEM_2025/data/trafic/SAT-MTB", {"task": "mot", "categories": ["car"]}),
    ("sdmcar",    SDMCarDataset, "/data/ESA_DLSTEM_2025/data/trafic/SDM-Car", {}),
]


def build_split(out: Path, split: str, json_name: str, seq_len: int, jpg_q: int):
    img_root = out / "images"
    ann_dir = out / "annotations"
    img_root.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    images, annotations, videos = [], [], []
    img_id = 0          # global, consecutive within each video
    ann_id = 0
    video_id = 0
    n_skipped = 0
    n_box = 0

    for dskey, cls, root, extra in SPECS:
        ds = cls(root=root, split=split, class_map={"car": 0}, **extra)
        for v in ds.videos:
            if v.num_frames < seq_len:
                n_skipped += 1
                continue
            video_id += 1
            tag = f"{dskey}_{v.video_id.replace('/', '_')}"
            seq_img_dir = img_root / tag / "img1"
            seq_img_dir.mkdir(parents=True, exist_ok=True)
            videos.append({"id": video_id, "file_name": tag})

            fids = list(v.frame_ids)
            vlen = len(fids)
            base_id = img_id
            for pos, fid in enumerate(fids):          # pos: 0-based position in video
                rgb = ds._load_frame(v, fid)
                h, w = rgb.shape[:2]
                fpath = seq_img_dir / f"{fid:06d}.jpg"
                if not fpath.exists():
                    cv2.imwrite(str(fpath), rgb[..., ::-1],
                                [cv2.IMWRITE_JPEG_QUALITY, jpg_q])
                img_id += 1
                file_name = f"images/{tag}/img1/{fid:06d}.jpg"
                images.append({
                    "id": img_id, "file_name": file_name,
                    "video_id": video_id,
                    "video_frame_id": pos + 1,        # 1-indexed position (HiEUM get_im_ids)
                    "video_len": vlen,
                    "height": int(h), "width": int(w),
                    "frame_id": pos + 1,
                    "prev_image_id": img_id - 1 if pos > 0 else -1,
                    "next_image_id": img_id + 1 if pos < vlen - 1 else -1,
                })
                ann = ds._load_annotations(v, fid)
                boxes = np.asarray(ann["boxes"], np.float32).reshape(-1, 4)
                tids = np.asarray(ann.get("track_ids", []), np.int64).reshape(-1)
                for j, (x1, y1, x2, y2) in enumerate(boxes):
                    bw, bh = float(x2 - x1), float(y2 - y1)
                    if bw <= 0 or bh <= 0:
                        continue
                    ann_id += 1
                    tid = int(tids[j]) if j < len(tids) else -1
                    annotations.append({
                        "id": ann_id, "image_id": img_id, "category_id": 1,
                        "track_id": video_id * 100000 + (tid if tid >= 0 else 0),
                        "bbox": [float(x1), float(y1), bw, bh],
                        "conf": 1.0, "area": bw * bh, "iscrowd": 0,
                    })
                    n_box += 1
            assert img_id - base_id == vlen
        print(f"  [{split}] {dskey}: cumulative videos={video_id} images={img_id} boxes={n_box}", flush=True)

    coco = {"images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": "car"}], "videos": videos}
    out_json = ann_dir / json_name
    json.dump(coco, open(out_json, "w"))
    print(f"[{split}] wrote {out_json}  videos={video_id} images={len(images)} "
          f"boxes={len(annotations)} skipped_short_videos={n_skipped}", flush=True)
    return len(images), len(annotations), video_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/work/ziwen/data/hieum_car_union")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--seq-len", type=int, default=20)
    ap.add_argument("--jpg-quality", type=int, default=95)
    args = ap.parse_args()

    out = Path(args.out)
    # train -> train_mot.json ; val -> test1024_mot.json (HiEUM's in-training val slot)
    name_map = {"train": "train_mot.json", "val": "test1024_mot.json", "test": "test_mot.json"}
    for split in args.splits:
        build_split(out, split, name_map[split], args.seq_len, args.jpg_quality)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
