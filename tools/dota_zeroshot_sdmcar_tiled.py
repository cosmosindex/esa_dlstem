"""
Tiled zero-shot DOTA inference on one SDM-Car test sequence.

Same as ``dota_zeroshot_sdmcar.py``, but slices each frame into
overlapping ``tile_size`` x ``tile_size`` patches and runs the model
on each tile separately. The model's internal Resize-to-1024² then
upscales the patch ~2× (with tile_size=512), bringing 5-10 px SDM-Car
cars to ~10-20 px in network input — closer to the DOTA training
distribution.

Predictions are mapped back to frame coordinates and de-duplicated
across overlapping tiles via class-aware NMS.

Usage:
  micromamba run -n mmrotate_dota python tools/dota_zeroshot_sdmcar_tiled.py \\
      --video-id test/1-2 --tile-size 512 --tile-overlap 128 \\
      --score-thr 0.05 --iou-thr 0.3 \\
      --out-dir /work/ziwen/experiments/dota_zeroshot_sdmcar_tiled/test_1-2_TS

Run inside the mmrotate_dota env.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from datasets.sdmcar import SDMCarDataset

KEEP_CLASSES = {4: "small-vehicle", 5: "large-vehicle"}


def rbox_to_aabb_batch(rb: np.ndarray) -> np.ndarray:
    """Convert (N, 5) cx,cy,w,h,theta → (N, 4) xyxy AABBs."""
    if len(rb) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    cx, cy, w, h, theta = rb[:, 0], rb[:, 1], rb[:, 2], rb[:, 3], rb[:, 4]
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    dx, dy = w / 2, h / 2
    # 4 corners per box → (N, 4, 2)
    corners = np.stack([
        np.stack([cx + dx * cos_t - dy * sin_t, cy + dx * sin_t + dy * cos_t], axis=-1),
        np.stack([cx - dx * cos_t - dy * sin_t, cy - dx * sin_t + dy * cos_t], axis=-1),
        np.stack([cx - dx * cos_t + dy * sin_t, cy - dx * sin_t - dy * cos_t], axis=-1),
        np.stack([cx + dx * cos_t + dy * sin_t, cy + dx * sin_t - dy * cos_t], axis=-1),
    ], axis=1)
    x1 = corners[..., 0].min(axis=1)
    y1 = corners[..., 1].min(axis=1)
    x2 = corners[..., 0].max(axis=1)
    y2 = corners[..., 1].max(axis=1)
    return np.stack([x1, y1, x2, y2], axis=-1).astype(np.float32)


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)
    inter = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(inter_y2 - inter_y1, 0, None)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return (inter / np.maximum(union, 1e-6)).astype(np.float32)


def greedy_match(pred_xyxy: np.ndarray, pred_scores: np.ndarray,
                 gt_xyxy: np.ndarray, iou_thr: float) -> tuple[int, int, int]:
    if len(gt_xyxy) == 0:
        return 0, len(pred_xyxy), 0
    if len(pred_xyxy) == 0:
        return 0, 0, len(gt_xyxy)
    order = np.argsort(-pred_scores)
    iou = iou_xyxy(pred_xyxy[order], gt_xyxy)
    matched_gt = np.zeros(len(gt_xyxy), dtype=bool)
    tp = 0
    for i in range(len(order)):
        ov = iou[i]
        ov[matched_gt] = -1
        j = int(np.argmax(ov))
        if ov[j] >= iou_thr:
            matched_gt[j] = True
            tp += 1
    fp = len(pred_xyxy) - tp
    fn = int((~matched_gt).sum())
    return tp, fp, fn


def class_aware_nms(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray,
                    iou_thr: float) -> np.ndarray:
    """Returns indices to keep after per-class NMS."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.int64)
    keep_all = []
    for cls in np.unique(labels):
        mask = labels == cls
        idxs = np.where(mask)[0]
        b = torch.from_numpy(boxes[mask]).float()
        s = torch.from_numpy(scores[mask]).float()
        from torchvision.ops import nms
        kept = nms(b, s, iou_thr).numpy()
        keep_all.extend(idxs[kept].tolist())
    keep_all.sort()
    return np.asarray(keep_all, dtype=np.int64)


def gen_tiles(H: int, W: int, tile: int, overlap: int) -> list[tuple[int, int, int, int]]:
    """Generate (x0, y0, x1, y1) tile rectangles covering the full frame
    with stride = tile - overlap. Last tiles in each row/col are right-/
    bottom-aligned so we never crop *more* than the image."""
    stride = tile - overlap
    xs = list(range(0, max(1, W - tile + 1), stride))
    if not xs or xs[-1] + tile < W:
        xs.append(max(0, W - tile))
    ys = list(range(0, max(1, H - tile + 1), stride))
    if not ys or ys[-1] + tile < H:
        ys.append(max(0, H - tile))
    return [
        (x, y, min(W, x + tile), min(H, y + tile))
        for y in ys for x in xs
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-id", default="test/1-2")
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt",
                    default="/work/ziwen/checkpoints/mmrotate/"
                            "rotated_faster_rcnn_r50_fpn_1x_dota_le90.pth")
    ap.add_argument("--dataset-root",
                    default="/data/ESA_DLSTEM_2025/data/trafic/SDM-Car")
    ap.add_argument("--score-thr", type=float, default=0.05)
    ap.add_argument("--iou-thr", type=float, default=0.3)
    ap.add_argument("--nms-iou", type=float, default=0.3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--tile-size", type=int, default=512,
                    help="Crop side length before feeding the model.")
    ap.add_argument("--tile-overlap", type=int, default=128,
                    help="Tile stride = tile_size - tile_overlap.")
    ap.add_argument("--viz-frames",
                    default="0,30,60,90,120,150,180")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "viz").mkdir(exist_ok=True)

    from mmrotate.utils import register_all_modules
    register_all_modules(init_default_scope=True)
    from mmdet.apis import init_detector, inference_detector
    import mmrotate

    config_path = args.config
    if config_path is None:
        cfg_root = Path(mmrotate.__file__).parent / ".mim" / "configs"
        cands = list(cfg_root.glob(
            "rotated_faster_rcnn/rotated-faster-rcnn-le90_r50_fpn_1x_dota.py"))
        if not cands:
            cands = list(cfg_root.rglob("rotated*faster*r50_fpn_1x_dota*.py"))
        config_path = str(cands[0])
    print(f"using config: {config_path}")

    model = init_detector(config_path, args.ckpt, device=args.device)

    ds = SDMCarDataset(
        root=args.dataset_root, split="test", mode="detection",
        class_map={"car": 0},
    )
    video = next((v for v in ds.videos if v.video_id == args.video_id), None)
    if video is None:
        raise SystemExit(f"video_id {args.video_id} not found")

    # Probe frame dims.
    sample = ds._load_frame(video, video.frame_ids[0])
    H, W = sample.shape[:2]
    tiles = gen_tiles(H, W, args.tile_size, args.tile_overlap)
    print(f"frame {H}x{W}; {len(tiles)} tiles per frame "
          f"(tile={args.tile_size}, overlap={args.tile_overlap})")
    print(f"running on {video.video_id}: {video.num_frames} frames "
          f"-> {video.num_frames * len(tiles)} inferences")

    viz_set = set(int(s) for s in args.viz_frames.split(",") if s.strip())
    rows: list[dict] = []
    tot_tp = tot_fp = tot_fn = 0

    for fi, fid in enumerate(video.frame_ids):
        img_rgb = ds._load_frame(video, fid)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        all_boxes = []  # xyxy in frame coords
        all_scores = []
        all_labels = []

        for (x0, y0, x1, y1) in tiles:
            tile = img_bgr[y0:y1, x0:x1]
            result = inference_detector(model, tile)
            pred = result.pred_instances
            rb = pred.bboxes.cpu().numpy()
            sc = pred.scores.cpu().numpy()
            lb = pred.labels.cpu().numpy()

            keep = np.isin(lb, list(KEEP_CLASSES.keys())) & (sc >= args.score_thr)
            rb, sc, lb = rb[keep], sc[keep], lb[keep]
            if len(rb) == 0:
                continue

            aabb = rbox_to_aabb_batch(rb)
            # Map tile-local coords back to frame coords.
            aabb[:, [0, 2]] += x0
            aabb[:, [1, 3]] += y0
            all_boxes.append(aabb)
            all_scores.append(sc)
            all_labels.append(lb)

        if all_boxes:
            boxes = np.concatenate(all_boxes, axis=0)
            scores = np.concatenate(all_scores, axis=0)
            labels = np.concatenate(all_labels, axis=0)
            keep_idx = class_aware_nms(boxes, scores, labels, args.nms_iou)
            boxes, scores, labels = boxes[keep_idx], scores[keep_idx], labels[keep_idx]
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)

        for box, sc, lbl in zip(boxes, scores, labels):
            rows.append({
                "frame_id": int(fid),
                "category": KEEP_CLASSES[int(lbl)],
                "bbox_xyxy": [float(b) for b in box],
                "score": float(sc),
            })

        gt = ds._load_annotations(video, fid)
        tp, fp, fn = greedy_match(boxes, scores, gt["boxes"], args.iou_thr)
        tot_tp += tp; tot_fp += fp; tot_fn += fn

        if int(fid) in viz_set:
            viz = img_bgr.copy()
            for box in gt["boxes"]:
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 0, 255), 1)  # GT red
            for box, sc, lbl in zip(boxes, scores, labels):
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 0), 1)  # pred green
            cv2.imwrite(str(args.out_dir / "viz" / f"frame_{int(fid):06d}.jpg"), viz)

        if (fi + 1) % 20 == 0 or fi + 1 == video.num_frames:
            print(f"  {fi + 1}/{video.num_frames}; "
                  f"running totals: TP={tot_tp} FP={tot_fp} FN={tot_fn}")

    json.dump(rows, open(args.out_dir / "predictions.json", "w"))

    precision = tot_tp / max(tot_tp + tot_fp, 1)
    recall = tot_tp / max(tot_tp + tot_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    summary = {
        "video_id": video.video_id,
        "num_frames": video.num_frames,
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "tiles_per_frame": len(tiles),
        "score_thr": args.score_thr,
        "iou_thr": args.iou_thr,
        "nms_iou": args.nms_iou,
        "tp": tot_tp, "fp": tot_fp, "fn": tot_fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "total_predictions": len(rows),
    }
    json.dump(summary, open(args.out_dir / "summary.json", "w"), indent=2)
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
