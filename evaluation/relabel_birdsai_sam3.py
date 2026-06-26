"""
Re-annotate BIRDSAI bounding boxes with SAM3 (box-prompted refinement).
=====================================================================
Motivation (user): BIRDSAI GT boxes are often loose/too large. SAM3, prompted
with the GT box on the SAME frame (SAM2-style interactive segmentation), returns
a tight mask whose AABB hugs the thermal blob better. We use that tighter box to
RE-ANNOTATE the dataset, keeping every other CSV column (track id, class,
species, occlusion, noise) verbatim.

This is NOT detection/tracking: every box is refined independently on its own
frame using its own GT box as the prompt. No propagation, no cross-frame or
cross-object contamination (each frame is run as a 1-frame "video").

Acceptance policy ("best-box + guards"): adopt SAM3's box unless it is clearly
wrong, in which case we keep the ORIGINAL GT box:
    * SAM3 mask empty / degenerate (w<1 or h<1)        -> keep original
    * IoU(refined, original) < --iou-min               -> keep original
    * area(refined) > --grow-max * area(original)       -> keep original
    * area(refined) < --shrink-min * area(original)     -> keep original

Output: parallel `annotations_sam3/<video>.csv` in each split dir, SAME 10-column
integer schema. The original `annotations/` is never touched.

    # pilot: 2 videos + side-by-side viz, no full write
    CUDA_VISIBLE_DEVICES=0 python evaluation/relabel_birdsai_sam3.py \
        --splits TrainReal --limit-videos 2 --pilot-viz

    # full dataset (resumable: re-run skips finished videos)
    CUDA_VISIBLE_DEVICES=0 python evaluation/relabel_birdsai_sam3.py --resume
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import time
from collections import defaultdict

import cv2
import numpy as np

BIRDSAI_ROOT = "/data/ESA_DLSTEM_2025/data/wild_animal/BIRDSAI"


# ---------------- helpers ----------------------------------------------------
def iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + ab - inter, 1e-9)


def load_frame(img_path: Path) -> np.ndarray | None:
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    img = img[..., ::-1].copy()  # BGR -> RGB
    if img.ndim == 2:
        img = np.stack([img, img, img], -1)
    elif img.shape[2] == 1:
        img = np.repeat(img, 3, 2)
    return img


def parse_csv_lines(path: Path):
    """Yield (raw_line, parts) for every non-empty line, preserving order."""
    out = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            out.append((s, s.split(",")))
    return out


# ---------------- per-video re-annotation ------------------------------------
def relabel_video(sot, csv_path: Path, img_dir: Path, out_csv: Path, args,
                  viz_dir: Path | None):
    video_id = csv_path.stem
    rows = parse_csv_lines(csv_path)

    # group line-indices by frame (only "box" rows: >=6 numeric cols)
    frame_rows: dict[int, list[int]] = defaultdict(list)
    for li, (_, parts) in enumerate(rows):
        if len(parts) < 6:
            continue
        try:
            fid = int(parts[0])
        except ValueError:
            continue
        frame_rows[fid].append(li)

    new_parts = [list(p) for _, p in rows]   # mutable copy we rewrite into
    stats = {"boxes": 0, "refined": 0, "kept_empty": 0, "kept_iou": 0,
             "kept_grow": 0, "kept_shrink": 0, "no_image": 0}
    viz_saved = 0

    for fid in sorted(frame_rows):
        img_path = img_dir / f"{video_id}_{fid:010d}.jpg"
        frame = load_frame(img_path)
        line_idxs = frame_rows[fid]
        if frame is None:
            stats["no_image"] += len(line_idxs)
            continue
        H, W = frame.shape[:2]

        # build box prompts (xyxy, clipped) in file order -> obj_id = local index
        boxes, oids, orig_boxes = [], [], {}
        for k, li in enumerate(line_idxs):
            p = rows[li][1]
            try:
                x, y, w, h = (float(p[2]), float(p[3]), float(p[4]), float(p[5]))
            except (ValueError, IndexError):
                continue
            if w < 1 or h < 1:
                continue
            x1 = float(np.clip(x, 0, W - 1)); y1 = float(np.clip(y, 0, H - 1))
            x2 = float(np.clip(x + w, 1, W)); y2 = float(np.clip(y + h, 1, H))
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            oid = len(boxes)
            boxes.append([x1, y1, x2, y2]); oids.append(oid)
            orig_boxes[oid] = (li, [x1, y1, x2, y2])
        if not boxes:
            continue
        stats["boxes"] += len(boxes)

        # SAM3 box->mask, each frame as an independent 1-frame video
        sot.init_video([frame])
        sot.add_prompts(0, np.asarray(boxes, np.float32),
                        labels=np.zeros(len(boxes), np.int64), obj_ids=oids)
        outs = sot.propagate()
        sot.reset_state()
        o0 = outs[0] if outs else None
        refined_by_oid = {}
        if o0 is not None and len(o0["boxes"]):
            rb = o0["boxes"].numpy(); rid = o0["track_ids"].numpy()
            for b, t in zip(rb, rid):
                refined_by_oid[int(t)] = [float(v) for v in b]

        viz_items = []  # (orig_xyxy, new_xyxy, accepted)
        for oid, (li, ob) in orig_boxes.items():
            rbox = refined_by_oid.get(oid)
            accept, reason = _accept(ob, rbox, args)
            if accept:
                nb = rbox
                stats["refined"] += 1
            else:
                nb = ob
                stats["kept_" + reason] += 1
            # rewrite x,y,w,h as integers (schema is all-int)
            nx = int(round(np.clip(nb[0], 0, W - 1)))
            ny = int(round(np.clip(nb[1], 0, H - 1)))
            nw = max(1, int(round(nb[2] - nb[0])))
            nh = max(1, int(round(nb[3] - nb[1])))
            new_parts[li][2] = str(nx); new_parts[li][3] = str(ny)
            new_parts[li][4] = str(nw); new_parts[li][5] = str(nh)
            viz_items.append((ob, [nx, ny, nx + nw, ny + nh], accept))

        if viz_dir is not None and viz_saved < args.viz_frames and viz_items:
            _save_viz(frame, viz_items, viz_dir / f"{video_id}_{fid:010d}.jpg")
            viz_saved += 1

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w") as f:
        f.write("\n".join(",".join(p) for p in new_parts) + "\n")
    return stats


def _accept(orig, refined, args):
    """Return (accept: bool, reason: str). reason names the keep-original cause."""
    if refined is None:
        return False, "empty"
    rw = refined[2] - refined[0]; rh = refined[3] - refined[1]
    if rw < 1 or rh < 1:
        return False, "empty"
    ao = max((orig[2] - orig[0]) * (orig[3] - orig[1]), 1e-9)
    ar = rw * rh
    if iou_xyxy(orig, refined) < args.iou_min:
        return False, "iou"
    if ar > args.grow_max * ao:
        return False, "grow"
    if ar < args.shrink_min * ao:
        return False, "shrink"
    return True, ""


def _save_viz(frame_rgb, items, out_path: Path):
    img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    scale = 2 if max(img.shape[:2]) < 800 else 1
    if scale > 1:
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_NEAREST)
    for ob, nb, acc in items:
        ob = [int(v * scale) for v in ob]; nb = [int(v * scale) for v in nb]
        cv2.rectangle(img, (ob[0], ob[1]), (ob[2], ob[3]), (0, 0, 255), 1)  # orig=red
        col = (0, 255, 0) if acc else (0, 200, 255)  # green=refined, yellow=rejected
        cv2.rectangle(img, (nb[0], nb[1]), (nb[2], nb[3]), col, 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


# ---------------- main -------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=BIRDSAI_ROOT)
    ap.add_argument("--splits", default="TrainReal,TestReal",
                    help="comma list of split subdirs")
    ap.add_argument("--out-dirname", default="annotations_sam3")
    ap.add_argument("--limit-videos", type=int, default=0, help="0=all")
    ap.add_argument("--resume", action="store_true",
                    help="skip videos whose output csv already exists")
    ap.add_argument("--pilot-viz", action="store_true",
                    help="write side-by-side viz under <out>/_viz/ (implies a look-only check)")
    ap.add_argument("--viz-frames", type=int, default=20,
                    help="max viz frames saved per video")
    ap.add_argument("--iou-min", type=float, default=0.1)
    ap.add_argument("--grow-max", type=float, default=4.0)
    ap.add_argument("--shrink-min", type=float, default=0.05)
    args = ap.parse_args()

    root = Path(args.root)
    from models.sam3 import SAM3Tracker
    sot = SAM3Tracker()

    grand = defaultdict(int)
    t0 = time.perf_counter()
    for split in args.splits.split(","):
        split = split.strip()
        ann_dir = root / split / "annotations"
        img_root = root / split / "images"
        out_dir = root / split / args.out_dirname
        if not ann_dir.exists():
            print(f"[skip] {ann_dir} missing"); continue

        csvs = [c for c in sorted(ann_dir.glob("*.csv"))
                if not c.name.startswith("._")]
        if args.limit_videos:
            csvs = csvs[:args.limit_videos]

        viz_root = (out_dir / "_viz") if args.pilot_viz else None
        for vi, csv_path in enumerate(csvs, 1):
            video_id = csv_path.stem
            img_dir = img_root / video_id
            out_csv = out_dir / csv_path.name
            if not img_dir.exists():
                print(f"  [{split} {vi}/{len(csvs)}] {video_id}: no images, skip")
                continue
            if args.resume and out_csv.exists():
                print(f"  [{split} {vi}/{len(csvs)}] {video_id}: exists, skip")
                continue
            tv = time.perf_counter()
            st = relabel_video(sot, csv_path, img_dir, out_csv, args,
                               (viz_root / video_id) if viz_root else None)
            for k, v in st.items():
                grand[k] += v
            ref_pct = 100 * st["refined"] / max(st["boxes"], 1)
            print(f"  [{split} {vi}/{len(csvs)}] {video_id}: "
                  f"{st['boxes']} boxes, {st['refined']} refined ({ref_pct:.1f}%), "
                  f"kept[empty={st['kept_empty']} iou={st['kept_iou']} "
                  f"grow={st['kept_grow']} shrink={st['kept_shrink']}] "
                  f"{time.perf_counter()-tv:.1f}s", flush=True)

    dt = time.perf_counter() - t0
    print("\n" + "=" * 64)
    print(f"DONE in {dt/60:.1f} min  | totals: " + json.dumps(dict(grand)))
    if grand["boxes"]:
        print(f"refined {grand['refined']}/{grand['boxes']} "
              f"({100*grand['refined']/grand['boxes']:.1f}%)")
    print("=" * 64)


if __name__ == "__main__":
    main()
