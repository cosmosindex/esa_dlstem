"""2-panel synchronized qualitative video for any Space-Tracker-MOT sequence.

Same layout as ``viz_car_track_panel.py`` (which stays as the car-only entry
point), generalized on three axes so the airplane / ship / train half of the
benchmark can be shown too:

  * every MOT dataset in the benchmark is selectable, not just the three car
    sets — AIR-MOT-100 / VISO / SAT-MTB non-car included;
  * the right-panel title is a free string, so a one-shot JDT model (FairMOT,
    TGraM) can be named instead of "<detector> + <tracker>";
  * line widths, ring radius and font size scale with the frame, because these
    sequences are 1920x1080 rather than the cars' 1024x1024.

Left panel  = Ground Truth (white boxes + id-coloured motion trails).
Right panel = the model, showing BOTH
  * detection correctness — BORDER colour:
        TP (green)   predicted box matched to a GT box (IoU >= --iou)
        FP (red)     predicted box with no matching GT
        FN (orange)  GT box no track covered, drawn as the GT box
  * tracking / identity — a trailing centre path over the last TRAIL_LEN frames
    coloured by track id, plus the id number.

The id-trail palette deliberately avoids green / red / orange so the two colour
schemes never collide.

Usage::

    python evaluation/viz_mot_track_panel.py --dataset airmot --video 58 \
        --split train --mot-file <run>/mot_format/ship/58.txt \
        --model "FairMOT" --iou 0.3 --fps 12
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
from datasets.airmot import AIRMOTDataset
from datasets.rscardata import RsCarDataset
from datasets.satmtb import SATMTBDataset
from datasets.sdmcar import SDMCarDataset
from datasets.viso import VISODataset
from project_paths import DATA_ROOT

#: Experiment/scratch root. Real paths are machine-specific, so they are
#: never written into the repository — set ``WORK_ROOT`` to point at yours.
WORK = Path(os.environ.get("WORK_ROOT", "/work/anon"))

# dataset key -> (class, root, extra-kwargs, class_map, default split)
_DATASETS = {
    "rscardata": (RsCarDataset, f"{DATA_ROOT}/data/trafic/RsCarData",
                  {}, {"car": 0}, "test"),
    "satmtb": (SATMTBDataset, f"{DATA_ROOT}/data/trafic/SAT-MTB",
               {"task": "mot", "categories": ["car"]}, {"car": 0}, "test"),
    "sdmcar": (SDMCarDataset, f"{DATA_ROOT}/data/trafic/SDM-Car",
               {}, {"car": 0}, "test"),
    "airmot": (AIRMOTDataset, f"{DATA_ROOT}/data/trafic/AIR-MOT-100",
               {}, {"airplane": 0, "ship": 1}, "test"),
    "viso_nocar": (VISODataset, f"{DATA_ROOT}/data/trafic/VISO",
                   {"categories": ["plane", "ship", "train"]},
                   {"plane": 0, "ship": 1, "train": 2}, "test"),
    "satmtb_nocar": (SATMTBDataset, f"{DATA_ROOT}/data/trafic/SAT-MTB",
                     {"task": "mot", "categories": ["airplane", "ship", "train"]},
                     {"airplane": 0, "ship": 1, "train": 2}, "test"),
}

TRAIL_LEN = 60

# detection-quality BGR colours
C_TP = (90, 210, 90)    # green
C_FP = (60, 60, 235)    # red
C_FN = (40, 170, 250)   # orange
C_GT = (235, 235, 235)  # white (GT reference panel)

# Identity palette for trails — blues / cyans / magentas / purples / yellows ONLY,
# so it never clashes with TP-green / FP-red / FN-orange.
_PALETTE = [
    (255, 255, 0), (255, 128, 0), (255, 0, 200), (200, 100, 255), (255, 200, 0),
    (180, 0, 200), (255, 0, 120), (220, 220, 0), (255, 150, 80), (200, 0, 255),
    (255, 100, 160), (150, 80, 255), (255, 220, 120), (200, 0, 140), (120, 200, 255),
]


def _color(tid: int):
    return _PALETTE[int(tid) % len(_PALETTE)]


class Style:
    """Drawing sizes derived from the frame width (tuned at 1024 px)."""

    def __init__(self, width: int):
        k = max(1.0, width / 1024.0)
        self.k = k
        self.lw = max(1, int(round(k)))            # box / trail line width
        self.ring = int(round(7 * k))              # ring marker radius
        self.dot = max(2, int(round(2 * k)))       # trail head dot
        self.font = 0.34 * k                       # id label
        self.header_h = int(round(28 * k))
        self.legend_h = int(round(26 * k))
        self.head_font = 0.6 * k
        self.legend_font = 0.48 * k


def detect_content_box(ds, video, frame_ids, thresh=8, sample=40):
    """AIR-MOT (and some VISO) sequences carry black padding bars on the bottom
    and/or right. Return (x0, y0, x1, y1) of the non-black region, taken over the
    per-pixel max of a frame sample so a dark-but-valid frame can't shrink it."""
    acc = None
    for fid in frame_ids[::max(1, sample)]:
        g = cv2.cvtColor(ds._load_frame(video, fid), cv2.COLOR_RGB2GRAY)
        acc = g if acc is None else np.maximum(acc, g)
    rows = np.where(acc.max(axis=1) > thresh)[0]
    cols = np.where(acc.max(axis=0) > thresh)[0]
    if not len(rows) or not len(cols):
        return 0, 0, acc.shape[1], acc.shape[0]
    return int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1


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


def _draw_trails(bgr, boxes, ids, hist, st):
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
            cv2.line(bgr, pts[i - 1], pts[i], col, st.lw, cv2.LINE_AA)
        cv2.circle(bgr, pts[-1], st.dot, col, -1, cv2.LINE_AA)


# Boxes bigger than this (px, longest side) are legible on their own — a ring
# marker on a 300 px train box is pure clutter.
RING_MAX_SIDE = 64


def _rects(bgr, boxes, color, st, ring=True):
    """Box + (for small objects) a ring marker so they stay visible."""
    for x1, y1, x2, y2 in boxes.astype(int).reshape(-1, 4):
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, st.lw, cv2.LINE_AA)
        if ring and max(x2 - x1, y2 - y1) <= RING_MAX_SIDE:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(bgr, (cx, cy), st.ring, color, st.lw, cv2.LINE_AA)


def _id_labels(bgr, boxes, ids, st):
    off = int(round(8 * st.k))
    for (x1, y1, x2, y2), tid in zip(boxes.astype(int).reshape(-1, 4), ids.astype(int).reshape(-1)):
        cv2.putText(bgr, str(int(tid) % 1_000_000), (int(x1), max(off + 1, int(y1) - off)),
                    cv2.FONT_HERSHEY_SIMPLEX, st.font, _color(int(tid)), st.lw, cv2.LINE_AA)


def _header(bgr, text, st):
    head = np.zeros((st.header_h, bgr.shape[1], 3), np.uint8)
    cv2.putText(head, text, (int(8 * st.k), int(19 * st.k)), cv2.FONT_HERSHEY_SIMPLEX,
                st.head_font, (255, 255, 255), st.lw, cv2.LINE_AA)
    return np.vstack([head, bgr])


def _gt_panel(rgb, gtb, gtid, hist, st):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    _draw_trails(bgr, gtb, gtid, hist, st)
    _rects(bgr, gtb, C_GT, st)
    _id_labels(bgr, gtb, gtid, st)
    n_id = len(np.unique(gtid)) if len(gtid) else 0
    return _header(bgr, f"Ground Truth     GT={len(gtb)}  ids={n_id}", st)


def _model_panel(rgb, gtb, tb, tids, title, hist, iou_thr, st):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    tp, fp, fn = classify_dets(gtb, tb, iou_thr)
    _draw_trails(bgr, tb, tids, hist, st)
    _rects(bgr, gtb[fn], C_FN, st)
    _rects(bgr, tb[fp], C_FP, st)
    _rects(bgr, tb[tp], C_TP, st)
    _id_labels(bgr, tb, tids, st)
    txt = (f"{title}   TP={int(tp.sum())} FP={int(fp.sum())} FN={int(fn.sum())}"
           f"  ids={len(np.unique(tids)) if len(tids) else 0}")
    return _header(bgr, txt, st)


def _legend(W, st):
    strip = np.zeros((st.legend_h, W, 3), np.uint8)
    x, y = int(8 * st.k), int(18 * st.k)
    for color, name in [(C_TP, "TP"), (C_FP, "FP"), (C_FN, "FN (missed GT)")]:
        cv2.rectangle(strip, (x, int(7 * st.k)), (x + int(16 * st.k), int(19 * st.k)),
                      color, -1)
        cv2.putText(strip, name, (x + int(20 * st.k), y), cv2.FONT_HERSHEY_SIMPLEX,
                    st.legend_font, (255, 255, 255), st.lw, cv2.LINE_AA)
        x += int((20 + 10 * len(name) + 16) * st.k)
    cv2.putText(strip, "| trail + number = track id (path=motion, same number=kept identity)",
                (x, y), cv2.FONT_HERSHEY_SIMPLEX, st.legend_font * 0.94,
                (235, 235, 235), st.lw, cv2.LINE_AA)
    return strip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="airmot", choices=list(_DATASETS))
    ap.add_argument("--video", required=True, help="video_id, e.g. 58")
    ap.add_argument("--split", default=None,
                    help="dataset split the video lives in (default: test)")
    ap.add_argument("--mot-file", required=True, help="MOTChallenge txt for this video")
    ap.add_argument("--model", default="Model", help="right-panel title")
    ap.add_argument("--classes", default=None,
                    help="comma-separated GT class names to keep (default: all)")
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--scale", type=float, default=1.0, help="rescale the canvas")
    ap.add_argument("--autocrop", action="store_true",
                    help="crop the black padding bars AIR-MOT / VISO frames carry")
    ap.add_argument("--out-dir", default=str(WORK / "experiments" / "mot_qualitative"))
    ap.add_argument("--out-name", default=None, help="output stem (default: auto)")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()

    cls, root, extra, cmap, def_split = _DATASETS[args.dataset]
    ds = cls(root=root, split=args.split or def_split, class_map=cmap, **extra)
    video = next(v for v in ds.videos if v.video_id == args.video)
    tracks = load_mot_tracks(args.mot_file)

    keep_labels = None
    if args.classes:
        wanted = {c.strip() for c in args.classes.split(",") if c.strip()}
        keep_labels = {cmap[c] for c in wanted}

    frame_ids = video.frame_ids[args.start::args.stride]
    if args.max_frames:
        frame_ids = frame_ids[:args.max_frames]

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_name or (f"{args.dataset}_{args.video.replace('/', '_')}_"
                             f"{args.model.lower().replace(' ', '').replace('+', '_')}_panel")
    mp4_path = out_dir / f"{stem}.mp4"

    cx0, cy0, cx1, cy1 = 0, 0, 0, 0
    if args.autocrop:
        # over the WHOLE sequence, so --start/--max-frames can't shrink the crop
        cx0, cy0, cx1, cy1 = detect_content_box(ds, video, video.frame_ids)
        print(f"autocrop -> x[{cx0}:{cx1}] y[{cy0}:{cy1}]")

    rgb0 = ds._load_frame(video, frame_ids[0])
    if args.autocrop:
        rgb0 = rgb0[cy0:cy1, cx0:cx1]
    H, W = rgb0.shape[:2]
    st = Style(W)
    canvas_w, canvas_h = W * 2, H + st.header_h + st.legend_h
    if args.scale != 1.0:
        canvas_w, canvas_h = int(canvas_w * args.scale), int(canvas_h * args.scale)
    # libx264 + yuv420p needs even dimensions
    canvas_w, canvas_h = canvas_w // 2 * 2, canvas_h // 2 * 2
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
        if keep_labels is not None:
            lab = np.asarray(ann["labels"], np.int64).reshape(-1)
            m = np.isin(lab, list(keep_labels))
            gtb, gtid = gtb[m], gtid[m]

        t = tracks.get(fid, {"boxes": np.zeros((0, 4), np.float32),
                             "track_ids": np.zeros(0, np.int64)})
        tb, tid = t["boxes"], t["track_ids"]

        if args.autocrop:
            rgb = rgb[cy0:cy1, cx0:cx1]
            shift = np.array([cx0, cy0, cx0, cy0], np.float32)
            gtb = gtb - shift
            tb = tb - shift

        left = _gt_panel(rgb, gtb, gtid, hist["GT"], st)
        right = _model_panel(rgb, gtb, tb, tid, args.model, hist["M"], args.iou, st)
        grid = np.vstack([np.hstack([left, right]), _legend(W * 2, st)])
        if (grid.shape[1], grid.shape[0]) != (canvas_w, canvas_h):
            grid = cv2.resize(grid, (canvas_w, canvas_h), interpolation=cv2.INTER_AREA)
        vw.write(grid)
        if args.gif:
            gif_frames.append(cv2.cvtColor(
                cv2.resize(grid, (canvas_w // 2, canvas_h // 2)), cv2.COLOR_BGR2RGB))
    vw.release()
    print(f"wrote raw {mp4_path}  ({len(frame_ids)} frames @ {args.fps}fps, "
          f"{canvas_w}x{canvas_h})")

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
            gif_path = out_dir / f"{stem}.gif"
            imageio.mimsave(str(gif_path), gif_frames[::2], fps=max(6, args.fps // 2))
            print(f"wrote {gif_path}")
        except Exception as e:
            print(f"gif skipped: {e}")


if __name__ == "__main__":
    main()
