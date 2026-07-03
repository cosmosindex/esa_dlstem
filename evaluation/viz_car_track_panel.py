"""2-panel synchronized qualitative video for the space-tracker CAR MOT pipeline.

Left panel  = Ground Truth (white boxes + id-coloured motion trails).
Right panel = HiEUM detection + a TBD tracker (SORT by default), showing BOTH
  * detection correctness of each tracked box — BORDER colour:
        TP (green)   tracked box matched to a GT box (IoU >= --iou)
        FP (red)     tracked box with no matching GT
        FN (orange)  GT box that no track covered (a miss), drawn as the GT box
  * tracking / identity — a trailing line (the object's centre path over the last
    TRAIL_LEN frames) coloured by track id, plus the id number. A continuous path
    that keeps its colour = one identity kept; the id number changing = an ID switch.

The id-trail palette deliberately avoids green / red / orange so the two colour
schemes never collide.

Cars in RsCarData (VISO subset) are tiny (~5-8 px at 1024x1024), so each box also
gets a small ring marker to stay visible, and the whole panel can be upscaled
with --scale for legibility.

Usage:
    python evaluation/viz_car_track_panel.py --video test1024/008 \
        --tracker sort --mot-file <run>/mot_format/test1024_008.txt \
        --iou 0.5 --fps 12 [--max-frames 600] [--gif]
"""
import argparse
import os
import sys
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from datasets.rscardata import RsCarDataset
from datasets.satmtb import SATMTBDataset
from datasets.sdmcar import SDMCarDataset

# dataset key -> (class, root, extra-kwargs) — the project's car MOT benchmark
_DATASETS = {
    "rscardata": (RsCarDataset, "/data/ESA_DLSTEM_2025/data/trafic/RsCarData", {}),
    "satmtb":    (SATMTBDataset, "/data/ESA_DLSTEM_2025/data/trafic/SAT-MTB", {"task": "mot", "categories": ["car"]}),
    "sdmcar":    (SDMCarDataset, "/data/ESA_DLSTEM_2025/data/trafic/SDM-Car", {}),
}

TRAIL_LEN = 60

# detection-quality BGR colours
C_TP = (90, 210, 90)    # green
C_FP = (60, 60, 235)    # red
C_FN = (40, 170, 250)   # orange
C_GT = (235, 235, 235)  # white (GT reference panel)
HEADER_H = 28

# Identity palette for trails — blues / cyans / magentas / purples / yellows ONLY,
# so it never clashes with TP-green / FP-red / FN-orange.
_PALETTE = [
    (255, 255, 0), (255, 128, 0), (255, 0, 200), (200, 100, 255), (255, 200, 0),
    (180, 0, 200), (255, 0, 120), (220, 220, 0), (255, 150, 80), (200, 0, 255),
    (255, 100, 160), (150, 80, 255), (255, 220, 120), (200, 0, 140), (120, 200, 255),
]


def _color(tid: int):
    return _PALETTE[int(tid) % len(_PALETTE)]


def load_mot_tracks(path):
    """MOTChallenge file -> {frame_id(int) -> {boxes(xyxy), track_ids}}."""
    out = defaultdict(lambda: {"boxes": [], "track_ids": []})
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        p = line.split(",")
        f = int(float(p[0])); tid = int(float(p[1]))
        x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
        out[f]["boxes"].append([x, y, x + w, y + h])
        out[f]["track_ids"].append(tid)
    return {f: {"boxes": np.asarray(v["boxes"], np.float32).reshape(-1, 4),
                "track_ids": np.asarray(v["track_ids"], np.int64).reshape(-1)}
            for f, v in out.items()}


def _iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def classify_dets(gtb, db, iou_thr):
    """Greedy IoU match (single class). Returns (tp_mask, fp_mask, fn_gt_mask)."""
    nd, ng = len(db), len(gtb)
    tp = np.zeros(nd, bool); gt_matched = np.zeros(ng, bool)
    if nd and ng:
        iou = _iou_matrix(db, gtb)
        rows, cols = np.where(iou >= iou_thr)
        if len(rows):
            order = iou[rows, cols].argsort()[::-1]
            md, mg = set(), set()
            for r, col in zip(rows[order], cols[order]):
                if r in md or col in mg:
                    continue
                md.add(r); mg.add(col)
                tp[r] = True; gt_matched[col] = True
    fp = (~tp) if nd else np.zeros(0, bool)
    fn = ~gt_matched if ng else np.zeros(0, bool)
    return tp, fp, fn


def _draw_trails(bgr, boxes, ids, hist):
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
        cv2.circle(bgr, pts[-1], 2, col, -1, cv2.LINE_AA)


def _rects(bgr, boxes, color, ring=True):
    """Box + (optional) ring marker so tiny cars stay visible."""
    for x1, y1, x2, y2 in boxes.astype(int).reshape(-1, 4):
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        if ring:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(bgr, (cx, cy), 7, color, 1, cv2.LINE_AA)


def _id_labels(bgr, boxes, ids):
    for (x1, y1, x2, y2), tid in zip(boxes.astype(int).reshape(-1, 4), ids.astype(int).reshape(-1)):
        cv2.putText(bgr, str(int(tid) % 1_000_000), (int(x1), max(9, int(y1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, _color(int(tid)), 1, cv2.LINE_AA)


def _header(bgr, text):
    head = np.zeros((HEADER_H, bgr.shape[1], 3), np.uint8)
    cv2.putText(head, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([head, bgr])


def _gt_panel(rgb, gtb, gtid, hist):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    _draw_trails(bgr, gtb, gtid, hist)
    _rects(bgr, gtb, C_GT)
    _id_labels(bgr, gtb, gtid)
    n_id = len(np.unique(gtid)) if len(gtid) else 0
    return _header(bgr, f"Ground Truth     GT={len(gtb)}  ids={n_id}")


def _model_panel(rgb, gtb, tb, tids, title, hist, iou_thr):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    tp, fp, fn = classify_dets(gtb, tb, iou_thr)
    _draw_trails(bgr, tb, tids, hist)
    _rects(bgr, gtb[fn], C_FN)
    _rects(bgr, tb[fp], C_FP)
    _rects(bgr, tb[tp], C_TP)
    _id_labels(bgr, tb, tids)
    txt = (f"{title}   TP={int(tp.sum())} FP={int(fp.sum())} FN={int(fn.sum())}"
           f"  ids={len(np.unique(tids)) if len(tids) else 0}")
    return _header(bgr, txt)


def _legend(W):
    strip = np.zeros((26, W, 3), np.uint8)
    x = 8
    for color, name in [(C_TP, "TP"), (C_FP, "FP"), (C_FN, "FN (missed GT)")]:
        cv2.rectangle(strip, (x, 7), (x + 16, 19), color, -1)
        cv2.putText(strip, name, (x + 20, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (255, 255, 255), 1, cv2.LINE_AA)
        x += 20 + 10 * len(name) + 16
    cv2.putText(strip, "| trail + number = track id (path=motion, same number=kept identity)",
                (x, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
    return strip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="test1024/008")
    ap.add_argument("--dataset", default="rscardata", choices=list(_DATASETS),
                    help="which car MOT dataset the video belongs to")
    ap.add_argument("--mot-file", required=True, help="MOTChallenge txt for this video")
    ap.add_argument("--tracker", default="SORT")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=600, help="0 = all")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--scale", type=float, default=1.0, help="upscale each panel")
    ap.add_argument("--out-dir", default="/work/ziwen/experiments/car_mot_qualitative")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()

    vid = args.video
    cls, root, extra = _DATASETS[args.dataset]
    ds = cls(root=root, split="test", class_map={"car": 0}, **extra)
    video = next(v for v in ds.videos if v.video_id == vid)
    tracks = load_mot_tracks(args.mot_file)

    frame_ids = video.frame_ids[args.start::args.stride]
    if args.max_frames:
        frame_ids = frame_ids[:args.max_frames]

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    safe = vid.replace("/", "_")
    mp4_path = out_dir / f"{safe}_hieum_{args.tracker.lower()}_panel.mp4"

    rgb0 = ds._load_frame(video, frame_ids[0]); H, W = rgb0.shape[:2]
    panel_h = H + HEADER_H
    canvas_w, canvas_h = W * 2, panel_h + 26
    if args.scale != 1.0:
        canvas_w = int(canvas_w * args.scale); canvas_h = int(canvas_h * args.scale)
    vw = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"),
                         args.fps, (canvas_w, canvas_h))
    gif_frames = []

    hist = {"GT": defaultdict(lambda: deque(maxlen=TRAIL_LEN)),
            "M": defaultdict(lambda: deque(maxlen=TRAIL_LEN))}

    for fid in frame_ids:
        rgb = ds._load_frame(video, fid)
        ann = ds._load_annotations(video, fid)
        gtb = np.asarray(ann["boxes"], np.float32).reshape(-1, 4)
        gtid = np.asarray(ann["track_ids"], np.int64).reshape(-1)

        t = tracks.get(fid, {"boxes": np.zeros((0, 4), np.float32),
                             "track_ids": np.zeros(0, np.int64)})
        tb, tid = t["boxes"], t["track_ids"]

        left = _gt_panel(rgb, gtb, gtid, hist["GT"])
        right = _model_panel(rgb, gtb, tb, tid, f"HiEUM + {args.tracker}",
                             hist["M"], args.iou)
        grid = np.vstack([np.hstack([left, right]), _legend(W * 2)])
        if args.scale != 1.0:
            grid = cv2.resize(grid, (canvas_w, canvas_h), interpolation=cv2.INTER_LINEAR)
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
        subprocess.run([ff, "-y", "-i", str(mp4_path), "-c:v", "libx264", "-crf", "23",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(h264)],
                       check=True, capture_output=True)
        mp4_path.unlink(); mp4_path = h264
        print(f"transcoded -> {h264}  ({h264.stat().st_size/1e6:.1f} MB)")
    except Exception as e:
        print(f"H.264 transcode skipped ({e}); kept mp4v")

    if args.gif and gif_frames:
        try:
            import imageio
            gif_path = out_dir / f"{safe}_hieum_{args.tracker.lower()}_panel.gif"
            imageio.mimsave(str(gif_path), gif_frames[::2], fps=max(6, args.fps // 2))
            print(f"wrote {gif_path}")
        except Exception as e:
            print(f"gif skipped: {e}")


if __name__ == "__main__":
    main()
