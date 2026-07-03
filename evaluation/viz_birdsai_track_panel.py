"""2x2 synchronized-panel qualitative video for BIRDSAI TRACKING + DETECTION QUALITY.

Combined ("二合一") view: each model panel shows BOTH
  * detection correctness of the tracked boxes — BORDER colour:
        TP (green)   tracked box matched to a same-class GT (class-aware IoU>=--iou)
        FP (red)     tracked box with no matching GT
        FN (orange)  GT box no track covered (a miss) — drawn as the GT box
  * tracking / identity — a trailing line (the object's centre path over the last
    TRAIL_LEN frames) coloured by track id, plus the id number. A continuous path
    that keeps its colour = one identity kept; the id number changing = an ID switch.

The id-trail palette deliberately avoids green / red / orange so the two colour
schemes never collide: green/red/orange ALWAYS mean TP/FP/FN, every other colour
is a track identity.

Layout: GT | FasterRCNN | YOLO11l | DINOv3 (GT panel = white boxes + id trails as
the identity reference). Same frame in every panel.

Usage:
    python evaluation/viz_birdsai_track_panel.py --video 0000000371_0000000000 \
        --annotations annotations_sam3 --iou 0.5 --fps 12 [--max-frames 800] [--gif]
"""
import argparse
import json
import os
import sys
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from datasets.birdsai_mot import BIRDSAIMOTDataset

ROOT = "/data/ESA_DLSTEM_2025/data/wild_animal/BIRDSAI"
CANON = {0: "human", 1: "elephant", 2: "giraffe", 3: "lion", 4: "unknown"}

# how many past frames of each track's centre to keep as a motion trail
TRAIL_LEN = 60

# detection-quality BGR colours (same semantics as the detection panel)
C_TP = (90, 210, 90)    # green
C_FP = (60, 60, 235)    # red
C_FN = (40, 170, 250)   # orange
C_GT = (235, 235, 235)  # white (GT reference panel)

PRED_PATHS = {
    "FasterRCNN": "/work/ziwen/experiments/fasterrcnn_birdsai_dettrack_20260617_185512/predictions.json",
    "YOLO11l": "/work/ziwen/experiments/yolo_birdsai_dettrack_20260617_214738/predictions.json",
    "DINOv3": "/work/ziwen/experiments/dinov3_birdsai_dettrack_20260617_170549/predictions.json",
}
MODEL_ORDER = ["FasterRCNN", "YOLO11l", "DINOv3"]
HEADER_H = 26

# Identity palette for trails — blues / cyans / magentas / purples / yellows ONLY,
# so it never clashes with TP-green / FP-red / FN-orange.
_PALETTE = [
    (255, 255, 0), (255, 128, 0), (255, 0, 200), (200, 100, 255), (255, 200, 0),
    (180, 0, 200), (255, 0, 120), (220, 220, 0), (255, 150, 80), (200, 0, 255),
    (255, 100, 160), (150, 80, 255), (255, 220, 120), (200, 0, 140), (120, 200, 255),
]


def _color(tid: int):
    return _PALETTE[int(tid) % len(_PALETTE)]


def load_video_tracks(model, vid):
    """frame_id(str) -> {boxes, labels, track_ids} from the `tracks` dump."""
    override = os.environ.get("BIRDSAI_PRED_DIR")
    path = (Path(override) / f"{ {'FasterRCNN': 'fasterrcnn', 'YOLO11l': 'yolo', 'DINOv3': 'dinov3'}[model] }.json"
            if override else PRED_PATHS[model])
    d = json.load(open(path))
    if vid not in d["videos"]:
        return {}
    return {f: e.get("tracks", {"boxes": [], "labels": [], "track_ids": []})
            for f, e in d["videos"][vid]["frames"].items()}


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


def _draw_trails(bgr, boxes, ids, hist):
    """Update per-id centre history and draw each active id's path (id-coloured)."""
    boxes_i = boxes.astype(int).reshape(-1, 4)
    ids_i = ids.astype(int).reshape(-1)
    for (x1, y1, x2, y2), tid in zip(boxes_i, ids_i):
        hist[int(tid)].append(((x1 + x2) // 2, (y1 + y2) // 2))
    for tid in ids_i:
        pts = list(hist[int(tid)])
        if len(pts) < 2:
            continue
        col = _color(int(tid))
        for i in range(1, len(pts)):
            cv2.line(bgr, pts[i - 1], pts[i], col, 1, cv2.LINE_AA)
        cv2.circle(bgr, pts[-1], 2, col, -1, cv2.LINE_AA)  # head marker


def _id_labels(bgr, boxes, ids):
    """Draw the compact local id number above each box (in its identity colour)."""
    for (x1, y1, x2, y2), tid in zip(boxes.astype(int).reshape(-1, 4), ids.astype(int).reshape(-1)):
        cv2.putText(bgr, str(int(tid) % 1_000_000), (int(x1), max(9, int(y1) - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, _color(int(tid)), 1, cv2.LINE_AA)


def _rects(bgr, boxes, color):
    for x1, y1, x2, y2 in boxes.astype(int).reshape(-1, 4):
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)


def _gt_panel(rgb, gtb, gt_ns, hist):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    _draw_trails(bgr, gtb, gt_ns, hist)
    _rects(bgr, gtb, C_GT)
    _id_labels(bgr, gtb, gt_ns)
    head = np.zeros((HEADER_H, bgr.shape[1], 3), np.uint8)
    n_id = len(np.unique(gt_ns)) if len(gt_ns) else 0
    cv2.putText(head, f"Ground Truth   GT={len(gtb)}  ids={n_id}", (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([head, bgr])


def _model_panel(rgb, gtb, gtl, tb, tl, tids, title, hist, iou_thr):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    tp, fp, fn = classify_dets(gtb, gtl, tb, tl, iou_thr)
    # trails first (under the boxes), then FN GT, then FP / TP boxes on top
    _draw_trails(bgr, tb, tids, hist)
    _rects(bgr, gtb[fn], C_FN)
    _rects(bgr, tb[fp], C_FP)
    _rects(bgr, tb[tp], C_TP)
    _id_labels(bgr, tb, tids)
    head = np.zeros((HEADER_H, bgr.shape[1], 3), np.uint8)
    txt = (f"{title}   TP={int(tp.sum())} FP={int(fp.sum())} FN={int(fn.sum())}"
           f"  ids={len(np.unique(tids)) if len(tids) else 0}")
    cv2.putText(head, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([head, bgr])


def _legend(W):
    strip = np.zeros((24, W, 3), np.uint8)
    x = 8
    for color, name in [(C_TP, "TP"), (C_FP, "FP"), (C_FN, "FN (missed GT)")]:
        cv2.rectangle(strip, (x, 6), (x + 16, 18), color, -1)
        cv2.putText(strip, name, (x + 20, 17), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
        x += 20 + 9 * len(name) + 16
    cv2.putText(strip, "| trail + number = track id (path=motion, same number=kept identity)",
                (x, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
    return strip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="0000000371_0000000000")
    ap.add_argument("--iou", type=float, default=0.5, help="TP match IoU")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=600, help="0 = all")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out-dir", default="/work/ziwen/experiments/birdsai_qualitative")
    ap.add_argument("--annotations", default="annotations_sam3")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()

    vid = args.video
    ds = BIRDSAIMOTDataset(root=ROOT, split="no_split", granularity="fine",
                           annotations_dirname=args.annotations,
                           class_map={v: k for k, v in CANON.items()})
    video = next(v for v in ds.videos if v.video_id == vid)
    tracks = {m: load_video_tracks(m, vid) for m in MODEL_ORDER}

    frame_ids = video.frame_ids[args.start::args.stride]
    if args.max_frames:
        frame_ids = frame_ids[:args.max_frames]

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = out_dir / f"{vid}_track_panel.mp4"

    rgb0 = ds._load_frame(video, frame_ids[0]); H, W = rgb0.shape[:2]
    canvas_w, canvas_h = W * 2, (H + HEADER_H) * 2 + 24
    vw = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"),
                         args.fps, (canvas_w, canvas_h))
    gif_frames = []

    # per-panel trail history: id -> deque of recent centres
    hist = {k: defaultdict(lambda: deque(maxlen=TRAIL_LEN))
            for k in ["GT"] + MODEL_ORDER}

    for fid in frame_ids:
        rgb = ds._load_frame(video, fid)
        ann = ds._load_annotations(video, fid)
        gtb = np.asarray(ann["boxes"], np.float32).reshape(-1, 4)
        gtid = np.asarray(ann["track_ids"], np.int64).reshape(-1)
        gtl = np.asarray(ann["labels"], np.int64).reshape(-1)
        # namespace GT ids by class so colours are stable per (class,id)
        gt_ns = gtid + gtl * 1_000_000

        panels = [_gt_panel(rgb, gtb, gt_ns, hist["GT"])]
        for m in MODEL_ORDER:
            t = tracks[m].get(str(fid), {"boxes": [], "labels": [], "track_ids": []})
            tb = np.asarray(t["boxes"], np.float32).reshape(-1, 4)
            tl = np.asarray(t.get("labels", []), np.int64).reshape(-1)
            tid = np.asarray(t.get("track_ids", []), np.int64).reshape(-1)
            panels.append(_model_panel(rgb, gtb, gtl, tb, tl, tid,
                                       f"{m} + OC-SORT", hist[m], args.iou))

        grid = np.vstack([np.hstack([panels[0], panels[1]]),
                          np.hstack([panels[2], panels[3]]),
                          _legend(canvas_w)])
        vw.write(grid)
        if args.gif:
            small = cv2.resize(grid, (canvas_w // 2, canvas_h // 2))
            gif_frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    vw.release()
    print(f"wrote raw {mp4_path}  ({len(frame_ids)} frames @ {args.fps}fps)")

    try:
        import subprocess
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        h264 = mp4_path.with_name(mp4_path.stem + "_h264.mp4")
        subprocess.run([ff, "-y", "-i", str(mp4_path), "-c:v", "libx264", "-crf", "24",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(h264)],
                       check=True, capture_output=True)
        mp4_path.unlink(); mp4_path = h264
        print(f"transcoded -> {h264}  ({h264.stat().st_size/1e6:.1f} MB)")
    except Exception as e:
        print(f"H.264 transcode skipped ({e}); kept mp4v")

    if args.gif and gif_frames:
        try:
            import imageio
            gif_path = out_dir / f"{vid}_track_panel.gif"
            imageio.mimsave(str(gif_path), gif_frames[::2], fps=max(6, args.fps // 2))
            print(f"wrote {gif_path}")
        except Exception as e:
            print(f"gif skipped: {e}")


if __name__ == "__main__":
    main()
