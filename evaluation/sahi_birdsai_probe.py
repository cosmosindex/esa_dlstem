"""SAHI (sliced) inference probe for BIRDSAI small-object detection.

Tests the *resolution hypothesis* WITHOUT retraining: the three already-trained
detectors (FasterRCNN / YOLO11l / DINOv3) are run on overlapping crops of each
frame (each crop upscaled to img_size², ~2.5x zoom) plus the full frame, then the
per-tile detections are mapped back to original pixels and merged with class-aware
NMS. If small-object recall jumps, feature-stride / resolution is the bottleneck and
retraining at higher res is worth it.

Runs only the 3 qualitative-video sequences. Writes a predictions.json per model
(same schema as eval_birdsai_detect_track) under --out-dir, and prints a TP/recall
table at score 0.2 / 0.5 to compare against the full-frame baseline.

    python evaluation/sahi_birdsai_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import time

import cv2
import numpy as np
import torch
import yaml
from torchvision.ops import batched_nms

from datasets.birdsai_mot import BIRDSAIMOTDataset
from eval_birdsai_detect_track import BIRDSAI_ROOT, CANON_NAMES, DEVICE, build_model

VIDEOS = ["0000000349_0000000000", "0000000352_0000000000", "0000000371_0000000000"]
MODELS = [
    ("fasterrcnn", "configs/Detection/fasterrcnn_birdsai.yaml",
     "/work/ziwen/experiments/fasterrcnn_birdsai_20260615_115438/checkpoints/best-epoch=2-val_mAP=0.115.ckpt"),
    ("yolo", "configs/Detection/yolo11_birdsai.yaml",
     "/work/ziwen/experiments/yolo11l_birdsai_manual_20260615_115439/checkpoints/best.pt"),
    ("dinov3", "configs/Detection/dinov3_birdsai.yaml",
     "/work/ziwen/experiments/dinov3_vitb16_birdsai_20260615_141227/checkpoints/best-epoch=1-val_mAP=0.043.ckpt"),
]
MODEL_TITLE = {"fasterrcnn": "FasterRCNN", "yolo": "YOLO11l", "dinov3": "DINOv3"}

GRID = (3, 3)      # (ncols, nrows) — tiles keep the frame's aspect (4:3) so the
                   # resize-to-square reproduces the exact train/eval anisotropy
OVERLAP = 0.2      # tile expansion as a fraction of the base cell
MERGE_IOU = 0.4    # cross-tile dedup NMS
DUMP_FLOOR = 0.05
MAX_FRAMES = 600


def make_tiles(W, H, grid, overlap):
    """Overlapping grid of tiles, each with the frame's aspect ratio, + full frame.
    Aspect-preserving tiles mean resize-to-640² reproduces the same 1.33x vertical
    stretch the detectors were trained/evaluated with (A.Resize(640,640))."""
    nx, ny = grid
    import math
    tw = min(W, math.ceil(W / nx * (1 + overlap)))
    th = min(H, math.ceil(H / ny * (1 + overlap)))

    def starts(n, tile, L):
        if n <= 1 or tile >= L:
            return [0]
        return [round(i * (L - tile) / (n - 1)) for i in range(n)]

    tiles = []
    for y0 in starts(ny, th, H):
        for x0 in starts(nx, tw, W):
            tiles.append((x0, y0, min(x0 + tw, W), min(y0 + th, H)))
    tiles = sorted(set(tiles))
    tiles.append((0, 0, W, H))   # full frame as the standard pred
    return tiles


@torch.no_grad()
def sahi_detect(model, rgb, img_size, amp_dtype, to_canon):
    """Sliced inference on one HxWx3 uint8 RGB frame. Returns merged
    (boxes_xyxy_orig, scores, labels_canonical)."""
    H, W = rgb.shape[:2]
    tiles = make_tiles(W, H, GRID, OVERLAP)
    inp = []
    for (x0, y0, x1, y1) in tiles:
        crop = rgb[y0:y1, x0:x1]
        rz = cv2.resize(crop, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        inp.append(torch.from_numpy(rz).permute(2, 0, 1).float().div_(255.0).to(DEVICE))
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=DEVICE == "cuda"):
        outs = model(inp)
    all_b, all_s, all_l = [], [], []
    for (x0, y0, x1, y1), out in zip(tiles, outs):
        b = out["boxes"].float().cpu().numpy()
        if not len(b):
            continue
        s = out["scores"].float().cpu().numpy()
        l = np.array([to_canon(x) for x in out["labels"].cpu().numpy()], np.int64)
        sx, sy = (x1 - x0) / img_size, (y1 - y0) / img_size
        b[:, [0, 2]] = b[:, [0, 2]] * sx + x0
        b[:, [1, 3]] = b[:, [1, 3]] * sy + y0
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, W)
        b[:, [1, 3]] = b[:, [1, 3]].clip(0, H)
        all_b.append(b); all_s.append(s); all_l.append(l)
    if not all_b:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)
    B = np.concatenate(all_b); S = np.concatenate(all_s); L = np.concatenate(all_l)
    keep = batched_nms(torch.from_numpy(B), torch.from_numpy(S),
                       torch.from_numpy(L), MERGE_IOU).numpy()
    B, S, L = B[keep], S[keep], L[keep]
    fl = S >= DUMP_FLOOR
    return B[fl], S[fl], L[fl]


# ---- class-aware greedy IoU match (mirrors viz/eval) ----
def _iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def count_tpfpfn(gtb, gtl, db, dl, iou_thr=0.5):
    tp = 0; matched = set()
    for c in np.unique(dl) if len(dl) else []:
        di = np.where(dl == c)[0]; gi = np.where(gtl == c)[0]
        if not len(gi):
            continue
        iou = _iou(db[di], gtb[gi]); rows, cols = np.where(iou >= iou_thr)
        if not len(rows):
            continue
        order = iou[rows, cols].argsort()[::-1]
        md, mg = set(), set()
        for r, cc in zip(rows[order], cols[order]):
            if r in md or cc in mg:
                continue
            md.add(r); mg.add(cc); tp += 1; matched.add((c, gi[cc]))
    return tp, len(db) - tp, len(gtb) - tp


def main():
    out_dir = Path("/work/ziwen/experiments/birdsai_sahi_probe")
    out_dir.mkdir(parents=True, exist_ok=True)
    canon_map = {v: k for k, v in CANON_NAMES.items()}
    ds = BIRDSAIMOTDataset(root=BIRDSAI_ROOT, split="no_split",
                           granularity="fine", class_map=canon_map)
    videos = {v.video_id: v for v in ds.videos if v.video_id in VIDEOS}

    torch.set_float32_matmul_precision("high")
    summary = {}
    for mtype, cfg_path, ckpt in MODELS:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        img_size = cfg.get("img_size", 640)
        amp_dtype = torch.bfloat16 if "bf16" in str(cfg.get("precision", "bf16-mixed")) else torch.float16
        print(f"\n[{MODEL_TITLE[mtype]}] loading {ckpt}", flush=True)
        model, to_canon = build_model(mtype, cfg, ckpt)
        model.eval().to(DEVICE)

        preds = {"model": mtype, "dataset": "BIRDSAI", "img_size": img_size,
                 "class_names": CANON_NAMES, "method": "sahi",
                 "grid": list(GRID), "overlap": OVERLAP, "videos": {}}
        # accumulators at two score thresholds
        acc = {0.2: [0, 0, 0], 0.5: [0, 0, 0]}
        t0 = time.perf_counter()
        for vid in VIDEOS:
            video = videos[vid]
            frames_out = {}
            for fid in video.frame_ids[:MAX_FRAMES]:
                rgb = ds._load_frame(video, fid)
                b, s, l = sahi_detect(model, rgb, img_size, amp_dtype, to_canon)
                frames_out[str(int(fid))] = {"detections": {
                    "boxes": b.round(2).tolist(), "scores": s.round(4).tolist(),
                    "labels": l.tolist()}}
                ann = ds._load_annotations(video, fid)
                gtb = np.asarray(ann["boxes"], np.float32).reshape(-1, 4)
                gtl = np.asarray(ann["labels"], np.int64).reshape(-1)
                for thr in (0.2, 0.5):
                    k = s >= thr
                    tp, fp, fn = count_tpfpfn(gtb, gtl, b[k], l[k])
                    acc[thr][0] += tp; acc[thr][1] += fp; acc[thr][2] += fn
            preds["videos"][vid] = {"frames": frames_out}
            print(f"  {vid}  done ({len(frames_out)} frames)", flush=True)

        with open(out_dir / f"{mtype}.json", "w") as f:
            json.dump(preds, f)
        dt = time.perf_counter() - t0
        summary[mtype] = {thr: tuple(acc[thr]) for thr in (0.2, 0.5)}
        print(f"  [{MODEL_TITLE[mtype]}] {dt:.0f}s -> {out_dir/f'{mtype}.json'}", flush=True)
        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("SAHI (sliced, 2.5x zoom) — aggregated over 3 viz sequences, iou=0.5")
    print("=" * 70)
    for thr in (0.5, 0.2):
        print(f"\n--- score_thr={thr} ---")
        for mtype, _, _ in MODELS:
            tp, fp, fn = summary[mtype][thr]
            rec = tp / max(tp + fn, 1); prec = tp / max(tp + fp, 1)
            print(f"  {MODEL_TITLE[mtype]:11s} TP={tp:6d} FP={fp:6d} FN={fn:6d}  "
                  f"recall={rec:.3f} prec={prec:.3f}")
    json.dump(summary, open(out_dir / "sahi_metrics.json", "w"), indent=2)
    print(f"\nmetrics -> {out_dir/'sahi_metrics.json'}")


if __name__ == "__main__":
    main()
