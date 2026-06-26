"""2x2 synchronized-panel qualitative video for BIRDSAI detection.

Same frame in every panel — GT | FasterRCNN | YOLO11l | DINOv3 — so the three
detectors can be compared side by side without overlay clutter (boxes are tiny
on thermal imagery).

Each model panel is colored by detection quality (class-aware greedy IoU match
at --iou, same protocol as the mAP eval):
    TP (green)   detection matched to a same-class GT
    FP (red)     detection with no matching GT  (includes class confusions)
    FN (orange)  GT box that no detection covered (missed)
The GT panel shows every ground-truth box (white) as reference.

Reads the cached per-frame detections (`predictions.json`, boxes already in
original-image pixels) and the fine-grained GT from BIRDSAIMOTDataset.

Usage:
    python evaluation/viz_birdsai_panel.py --video 0000000371_0000000000 \
        --score 0.5 --iou 0.5 --fps 12 [--max-frames 800] [--gif]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from datasets.birdsai_mot import BIRDSAIMOTDataset

ROOT = "/data/ESA_DLSTEM_2025/data/wild_animal/BIRDSAI"
CANON = {0: "human", 1: "elephant", 2: "giraffe", 3: "lion", 4: "unknown"}

# Detection-quality BGR colors (cv2 uses BGR).
C_TP = (90, 210, 90)    # green
C_FP = (60, 60, 235)    # red
C_FN = (40, 170, 250)   # orange
C_GT = (235, 235, 235)  # white (GT reference panel)

# 06-17 RETRAINED detectors — the same predictions the detection/tracking tables
# (annotations_sam3) consume, so the qualitative panels match the numbers.
PRED_PATHS = {
    "FasterRCNN": "/work/ziwen/experiments/fasterrcnn_birdsai_dettrack_20260617_185512/predictions.json",
    "YOLO11l": "/work/ziwen/experiments/yolo_birdsai_dettrack_20260617_214738/predictions.json",
    "DINOv3": "/work/ziwen/experiments/dinov3_birdsai_dettrack_20260617_170549/predictions.json",
}
MODEL_ORDER = ["FasterRCNN", "YOLO11l", "DINOv3"]
HEADER_H = 26


import os

# Optional override: point at a dir holding {fasterrcnn,yolo,dinov3}.json dumps
# (e.g. the SAHI probe output) instead of the baseline full-frame predictions.
_MKEY = {"FasterRCNN": "fasterrcnn", "YOLO11l": "yolo", "DINOv3": "dinov3"}


def load_video_preds(model, vid):
    """frame_id(str) -> {boxes,scores,labels}. Falls back to the step-A completion
    cache for the 5 old-val videos not present in the detect-track dump."""
    override = os.environ.get("BIRDSAI_PRED_DIR")
    if override:
        d = json.load(open(Path(override) / f"{_MKEY[model]}.json"))
        if vid in d["videos"]:
            return {f: e["detections"] for f, e in d["videos"][vid]["frames"].items()}
        return {}
    d = json.load(open(PRED_PATHS[model]))
    if vid in d["videos"]:
        return {f: e["detections"] for f, e in d["videos"][vid]["frames"].items()}
    # try the completion cache
    mkey = {"FasterRCNN": "fasterrcnn", "YOLO11l": "yolo", "DINOv3": "dinov3"}[model]
    alt = Path(f"/work/ziwen/experiments/birdsai_resplit_A/missing_preds_{mkey}.json")
    if alt.exists():
        d2 = json.load(open(alt))
        if vid in d2["videos"]:
            return {f: e["detections"] for f, e in d2["videos"][vid]["frames"].items()}
    return {}


def _iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def classify_dets(gtb, gtl, db, dl, iou_thr):
    """Class-aware greedy IoU match. Returns (tp_det_mask, fp_det_mask, fn_gt_mask)."""
    nd, ng = len(db), len(gtb)
    tp = np.zeros(nd, bool); gt_matched = np.zeros(ng, bool)
    for c in np.unique(dl) if nd else []:
        di = np.where(dl == c)[0]
        gi = np.where(gtl == c)[0]
        if len(gi) == 0:
            continue
        iou = _iou_matrix(db[di], gtb[gi])
        rows, cols = np.where(iou >= iou_thr)
        if len(rows) == 0:
            continue
        order = iou[rows, cols].argsort()[::-1]
        md, mg = set(), set()
        for r, col in zip(rows[order], cols[order]):
            if r in md or col in mg:
                continue
            md.add(r); mg.add(col)
            tp[di[r]] = True; gt_matched[gi[col]] = True
    fp = (~tp) if nd else np.zeros(0, bool)
    fn = ~gt_matched if ng else np.zeros(0, bool)
    return tp, fp, fn


def _rects(img, boxes, color):
    for x1, y1, x2, y2 in boxes.astype(int):
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)


def draw_gt_panel(rgb, gtb):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    _rects(bgr, gtb, C_GT)
    head = np.zeros((HEADER_H, bgr.shape[1], 3), np.uint8)
    cv2.putText(head, f"Ground Truth  (GT={len(gtb)})", (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([head, bgr])


def draw_model_panel(rgb, gtb, gtl, db, dl, ds_, title, score_thr, iou_thr):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    keep = ds_ >= score_thr if len(ds_) else np.zeros(0, bool)
    db, dl = db[keep], dl[keep]
    tp, fp, fn = classify_dets(gtb, gtl, db, dl, iou_thr)
    # draw FN first (so TP/FP overlay on top), then FP, then TP
    _rects(bgr, gtb[fn], C_FN)
    _rects(bgr, db[fp], C_FP)
    _rects(bgr, db[tp], C_TP)
    head = np.zeros((HEADER_H, bgr.shape[1], 3), np.uint8)
    txt = f"{title}   TP={int(tp.sum())}  FP={int(fp.sum())}  FN={int(fn.sum())}"
    cv2.putText(head, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([head, bgr])


def legend_strip(W):
    strip = np.zeros((24, W, 3), np.uint8)
    x = 8
    for color, name in [(C_TP, "TP (matched)"), (C_FP, "FP (false / wrong-class)"),
                        (C_FN, "FN (missed GT)"), (C_GT, "GT")]:
        cv2.rectangle(strip, (x, 6), (x + 16, 18), color, -1)
        cv2.putText(strip, name, (x + 20, 17), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
        x += 20 + 9 * len(name) + 16
    return strip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="0000000371_0000000000")
    ap.add_argument("--score", type=float, default=0.5)
    ap.add_argument("--iou", type=float, default=0.5, help="TP match IoU")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=600, help="0 = all")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out-dir", default="/work/ziwen/experiments/birdsai_qualitative")
    ap.add_argument("--annotations", default="annotations",
                    help="GT annotation subdir (use annotations_sam3 for tight SAM3-refined GT)")
    ap.add_argument("--gif", action="store_true", help="also write a downscaled GIF")
    args = ap.parse_args()

    vid = args.video
    canon_map = {v: k for k, v in CANON.items()}
    ds = BIRDSAIMOTDataset(root=ROOT, split="no_split", granularity="fine",
                           annotations_dirname=args.annotations, class_map=canon_map)
    video = next(v for v in ds.videos if v.video_id == vid)

    preds = {m: load_video_preds(m, vid) for m in MODEL_ORDER}

    frame_ids = video.frame_ids[args.start::args.stride]
    if args.max_frames:
        frame_ids = frame_ids[:args.max_frames]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = out_dir / f"{vid}_panel_s{args.score}_iou{args.iou}.mp4"

    # probe size
    rgb0 = ds._load_frame(video, frame_ids[0])
    H, W = rgb0.shape[:2]
    panel_h, panel_w = H + HEADER_H, W
    leg_h = 24
    canvas_w = panel_w * 2
    canvas_h = panel_h * 2 + leg_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(mp4_path), fourcc, args.fps, (canvas_w, canvas_h))
    gif_frames = []

    for fid in frame_ids:
        rgb = ds._load_frame(video, fid)
        ann = ds._load_annotations(video, fid)
        gtb = np.asarray(ann["boxes"], np.float32).reshape(-1, 4)
        gtl = np.asarray(ann["labels"], np.int64).reshape(-1)

        panels = [draw_gt_panel(rgb, gtb)]
        for m in MODEL_ORDER:
            det = preds[m].get(str(fid), {"boxes": [], "scores": [], "labels": []})
            db = np.asarray(det["boxes"], np.float32).reshape(-1, 4)
            dl = np.asarray(det["labels"], np.int64).reshape(-1)
            dsc = np.asarray(det["scores"], np.float32).reshape(-1)
            panels.append(draw_model_panel(
                rgb, gtb, gtl, db, dl, dsc, m, args.score, args.iou))

        top = np.hstack([panels[0], panels[1]])
        bot = np.hstack([panels[2], panels[3]])
        grid = np.vstack([top, bot, legend_strip(canvas_w)])
        vw.write(grid)
        if args.gif:
            small = cv2.resize(grid, (canvas_w // 2, canvas_h // 2))
            gif_frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

    vw.release()
    print(f"wrote raw {mp4_path}  ({len(frame_ids)} frames @ {args.fps}fps)")

    # Transcode the bulky mp4v to H.264 (much smaller); keep H.264 as the artifact.
    try:
        import subprocess
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        h264 = mp4_path.with_name(mp4_path.stem + "_h264.mp4")
        subprocess.run([ff, "-y", "-i", str(mp4_path), "-c:v", "libx264", "-crf", "24",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(h264)],
                       check=True, capture_output=True)
        mp4_path.unlink()
        mp4_path = h264
        print(f"transcoded -> {h264}  ({h264.stat().st_size/1e6:.1f} MB)")
    except Exception as e:
        print(f"H.264 transcode skipped ({e}); kept mp4v")

    if args.gif and gif_frames:
        try:
            import imageio
            gif_path = out_dir / f"{vid}_panel_s{args.score}.gif"
            imageio.mimsave(str(gif_path), gif_frames[::2], fps=max(6, args.fps // 2))
            print(f"wrote {gif_path}")
        except Exception as e:
            print(f"gif skipped: {e}")


if __name__ == "__main__":
    main()
