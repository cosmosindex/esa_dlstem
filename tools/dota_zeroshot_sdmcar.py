"""
Zero-shot detection on one SDM-Car test sequence using a Faster R-CNN
trained on DOTA-v1.0 (rotated_faster_rcnn_r50_fpn_1x_dota_le90, mmrotate
model zoo). Useful as a quick check for whether external "small-object"
pretrained detectors transfer to satellite vehicle imagery without any
fine-tuning.

DOTA-v1.0 has 15 classes:
  0 plane            5 large-vehicle    10 soccer-ball-field
  1 baseball-diamond 6 ship             11 roundabout
  2 bridge           7 tennis-court     12 harbor
  3 ground-track-field 8 basketball-court 13 swimming-pool
  4 small-vehicle    9 storage-tank     14 helicopter

For SDM-Car (single-class `car`) we keep predictions in classes
{small-vehicle, large-vehicle} and discard the rest.

The model outputs **rotated** boxes (cx, cy, w, h, angle); we collapse
each to its enclosing AABB (xyxy) so that we can compare against the
HBB GT shipped by SDMCarDataset.

Usage:
  micromamba run -n mmrotate_dota python tools/dota_zeroshot_sdmcar.py \\
      --video-id test/1-2 --score-thr 0.05 \\
      --out-dir /work/ziwen/experiments/dota_zeroshot_sdmcar/test_1-2

Run inside the mmrotate_dota env (mmrotate 1.0.0rc1 + mmdet 3.x +
mmcv 2.x). The host esa_dlstem env's PyTorch 2.10/cu128 is incompatible
with mmcv prebuilt wheels.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Make our datasets/ package importable regardless of cwd.
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from datasets.sdmcar import SDMCarDataset

# DOTA-v1.0 class id → name. Keep classes are the only ones we report.
DOTA_CLASSES = {
    0: "plane", 1: "baseball-diamond", 2: "bridge", 3: "ground-track-field",
    4: "small-vehicle", 5: "large-vehicle", 6: "ship", 7: "tennis-court",
    8: "basketball-court", 9: "storage-tank", 10: "soccer-ball-field",
    11: "roundabout", 12: "harbor", 13: "swimming-pool", 14: "helicopter",
}
KEEP_CLASSES = {4: "small-vehicle", 5: "large-vehicle"}


def rbox_to_aabb(cx: float, cy: float, w: float, h: float, theta: float) -> tuple[float, float, float, float]:
    """Convert a rotated box (cx, cy, w, h, angle) to its enclosing AABB."""
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    dx, dy = w / 2, h / 2
    # 4 corners
    corners = np.array([
        [cx + dx * cos_t - dy * sin_t, cy + dx * sin_t + dy * cos_t],
        [cx - dx * cos_t - dy * sin_t, cy - dx * sin_t + dy * cos_t],
        [cx - dx * cos_t + dy * sin_t, cy - dx * sin_t - dy * cos_t],
        [cx + dx * cos_t + dy * sin_t, cy + dx * sin_t - dy * cos_t],
    ])
    return (
        float(corners[:, 0].min()), float(corners[:, 1].min()),
        float(corners[:, 0].max()), float(corners[:, 1].max()),
    )


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between two sets of xyxy boxes — returns (Na, Nb)."""
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
    """Greedy matching for TP/FP/FN counts at one IoU threshold."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-id", default="test/1-2",
                    help="SDM-Car video id (e.g. test/1-2).")
    ap.add_argument("--config",
                    default=None,
                    help="mmrotate config .py path. Auto-resolved from "
                         "the mmrotate package install when None.")
    ap.add_argument("--ckpt",
                    default="/work/ziwen/checkpoints/mmrotate/"
                            "rotated_faster_rcnn_r50_fpn_1x_dota_le90.pth")
    ap.add_argument("--dataset-root",
                    default="/data/ESA_DLSTEM_2025/data/trafic/SDM-Car")
    ap.add_argument("--score-thr", type=float, default=0.05)
    ap.add_argument("--iou-thr", type=float, default=0.3,
                    help="IoU threshold for the precision/recall summary.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--viz-frames",
                    default="0,30,60,90,120,150,180",
                    help="Comma-separated list of frame ids to render.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "viz").mkdir(exist_ok=True)

    # Resolve mmrotate config path lazily — only when mmrotate is available.
    from mmrotate.utils import register_all_modules  # noqa: F401
    register_all_modules(init_default_scope=True)

    config_path = args.config
    if config_path is None:
        import mmrotate
        cfg_root = Path(mmrotate.__file__).parent / ".mim" / "configs"
        cands = list(cfg_root.glob(
            "rotated_faster_rcnn/rotated-faster-rcnn-le90_r50_fpn_1x_dota.py"))
        if not cands:
            cands = list(cfg_root.rglob("rotated*faster*r50_fpn_1x_dota*.py"))
        if not cands:
            raise RuntimeError(f"could not auto-resolve config under {cfg_root}")
        config_path = str(cands[0])
        print(f"using config: {config_path}")

    # Load model.
    from mmdet.apis import init_detector, inference_detector
    model = init_detector(config_path, args.ckpt, device=args.device)

    # Load dataset.
    ds = SDMCarDataset(
        root=args.dataset_root, split="test", mode="detection",
        class_map={"car": 0},
    )
    video = next((v for v in ds.videos if v.video_id == args.video_id), None)
    if video is None:
        raise SystemExit(f"video_id {args.video_id} not found in test split")

    print(f"running on {video.video_id}: {video.num_frames} frames")
    viz_set = set(int(s) for s in args.viz_frames.split(",") if s.strip())

    rows: list[dict] = []
    tot_tp = tot_fp = tot_fn = 0

    for fid in video.frame_ids:
        img_rgb = ds._load_frame(video, fid)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        result = inference_detector(model, img_bgr)
        pred = result.pred_instances
        # mmrotate stores rotated bboxes as (cx, cy, w, h, theta).
        rb = pred.bboxes.cpu().numpy()
        scores = pred.scores.cpu().numpy()
        labels = pred.labels.cpu().numpy()

        # Filter classes + score.
        keep = np.isin(labels, list(KEEP_CLASSES.keys())) & (scores >= args.score_thr)
        rb, scores, labels = rb[keep], scores[keep], labels[keep]

        # Convert OBB → AABB.
        if len(rb) > 0:
            aabbs = np.array([
                rbox_to_aabb(*r[:5].tolist()) for r in rb
            ], dtype=np.float32)
        else:
            aabbs = np.zeros((0, 4), dtype=np.float32)

        for box, sc, lbl in zip(aabbs, scores, labels):
            rows.append({
                "frame_id": int(fid),
                "category": KEEP_CLASSES[int(lbl)],
                "bbox_xyxy": [float(b) for b in box],
                "score": float(sc),
            })

        # Per-frame eval (IoU-based, class-agnostic since SDM-Car is car-only).
        gt = ds._load_annotations(video, fid)
        gt_boxes = gt["boxes"]
        tp, fp, fn = greedy_match(aabbs, scores, gt_boxes, args.iou_thr)
        tot_tp += tp
        tot_fp += fp
        tot_fn += fn

        # Visualize selected frames.
        if int(fid) in viz_set:
            viz = img_bgr.copy()
            for box in gt_boxes:
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 0, 255), 1)  # GT red
            for box, sc, lbl in zip(aabbs, scores, labels):
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 0), 1)  # pred green
                cv2.putText(viz, f"{KEEP_CLASSES[int(lbl)]} {sc:.2f}",
                            (x1, max(8, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.35, (0, 255, 0), 1)
            cv2.imwrite(str(args.out_dir / "viz" / f"frame_{int(fid):06d}.jpg"), viz)

        if (fid + 1) % 50 == 0:
            print(f"  {fid + 1}/{video.num_frames} frames done; "
                  f"running totals: TP={tot_tp} FP={tot_fp} FN={tot_fn}")

    json.dump(rows, open(args.out_dir / "predictions.json", "w"))

    precision = tot_tp / max(tot_tp + tot_fp, 1)
    recall = tot_tp / max(tot_tp + tot_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    summary = {
        "video_id": video.video_id,
        "num_frames": video.num_frames,
        "score_thr": args.score_thr,
        "iou_thr": args.iou_thr,
        "tp": tot_tp, "fp": tot_fp, "fn": tot_fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "total_predictions": len(rows),
    }
    json.dump(summary, open(args.out_dir / "summary.json", "w"), indent=2)
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))
    print(f"viz frames at {args.out_dir}/viz/, "
          f"raw preds at {args.out_dir}/predictions.json")


if __name__ == "__main__":
    main()
