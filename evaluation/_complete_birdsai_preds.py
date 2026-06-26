"""Step-A helper: complete raw detections for the 5 TestReal videos that the
detect-track predictions.json omitted (the old-val videos, which hold the only
`lion` samples). Reuses the exact builders/checkpoints from
eval_birdsai_detect_track.py and dumps the same per-frame schema so the offline
mAP recomputation can cover the full 16-video TestReal set.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from datasets.birdsai_mot import BIRDSAIMOTDataset
import evaluation.eval_birdsai_detect_track as E

OUT_DIR = Path("/work/ziwen/experiments/birdsai_resplit_A")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# old-val videos missing from the cached predictions (= the 5 we must infer)
MISSING = ['0000000011_0000000000', '0000000012_0000000000',
           '0000000061_0000000000', '0000000361_0000000000',
           '0000000362_0000000000']

MODELS = {
    "fasterrcnn": dict(
        config="configs/Detection/fasterrcnn_birdsai.yaml",
        ckpt="/work/ziwen/experiments/fasterrcnn_birdsai_20260615_115438/checkpoints/best-epoch=2-val_mAP=0.115.ckpt"),
    "yolo": dict(
        config="configs/Detection/yolo11_birdsai.yaml",
        ckpt="/work/ziwen/experiments/yolo11l_birdsai_manual_20260615_115439/checkpoints/best.pt"),
    "dinov3": dict(
        config="configs/Detection/dinov3_birdsai.yaml",
        ckpt="/work/ziwen/experiments/dinov3_vitb16_birdsai_20260615_141227/checkpoints/best-epoch=1-val_mAP=0.043.ckpt"),
}

DUMP_FLOOR = 0.05


def run_model(name, spec):
    out_path = OUT_DIR / f"missing_preds_{name}.json"
    if out_path.exists():
        print(f"[{name}] cache exists -> {out_path}; skip")
        return
    with open(REPO / spec["config"]) as f:
        cfg = yaml.safe_load(f)
    img_size = cfg.get("img_size", 640)
    amp_dtype = torch.bfloat16 if "bf16" in str(cfg.get("precision", "bf16-mixed")) else torch.float16
    torch.set_float32_matmul_precision("high")

    model, to_canon = E.build_model(name, cfg, spec["ckpt"])
    model.eval().to(E.DEVICE)

    canon_map = {v: k for k, v in E.CANON_NAMES.items()}
    ds = BIRDSAIMOTDataset(root=E.BIRDSAI_ROOT, split="no_split",
                           granularity="fine", class_map=canon_map)
    by_id = {v.video_id: v for v in ds.videos}

    videos_out = {}
    t0 = time.perf_counter()
    for vid in MISSING:
        video = by_id[vid]
        frames = {}
        for fid in video.frame_ids:
            rgb = ds._load_frame(video, fid)
            boxes, scores, labels = E.detect_frame(
                model, rgb, img_size, amp_dtype, to_canon, DUMP_FLOOR)
            img_path = str(ds._img_dir_cache[vid] / f"{vid}_{fid:010d}.jpg")
            frames[str(fid)] = {
                "image_path": img_path,
                "detections": {
                    "boxes": boxes.tolist(),
                    "scores": scores.tolist(),
                    "labels": labels.tolist(),
                },
            }
        videos_out[vid] = {"image_dir": str(ds._img_dir_cache[vid]), "frames": frames}
        print(f"[{name}] {vid}: {len(frames)} frames", flush=True)

    json.dump({"model": name, "img_size": img_size, "class_names": E.CANON_NAMES,
               "checkpoint": spec["ckpt"], "videos": videos_out}, open(out_path, "w"))
    print(f"[{name}] done in {time.perf_counter()-t0:.1f}s -> {out_path}", flush=True)

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for name, spec in MODELS.items():
        if only and name != only:
            continue
        print("=" * 60, name)
        run_model(name, spec)
