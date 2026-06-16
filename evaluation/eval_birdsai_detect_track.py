"""
BIRDSAI detection + tracking eval for a trained detector (YOLO / FasterRCNN / DINOv3).

The wild-animal-use-case companion to eval_tracker.py. Where eval_tracker.py runs a
tracker over *cached* HiEUM detections, this script generates detections from one of
our three trained detectors, then:

  1. dumps EVERY frame's predictions to a single ``predictions.json`` (original-image
     pixel coords) — the shared artifact for the later unified, all-models-in-one-image
     visualization (no per-run JPEGs are written here; visualization is skipped);
  2. runs a per-class online tracker (OC-SORT) to assign track IDs;
  3. scores detection (Pr/Re/F1 @IoU 0.5, class-aware) and tracking (MOTA / IDF1 /
     IDsw, class-aware) and writes MOTChallenge-format files for optional HOTA.

All three detectors are evaluated identically (same input size, same tracker, same
matching), so the numbers are directly comparable. Labels are stored 0-indexed and
canonical ({0: animal, 1: human}) regardless of each model's internal indexing, so
the JSON dumps align across models for the combined visualization.

Usage::

    CUDA_VISIBLE_DEVICES=0 python eval_birdsai_detect_track.py \\
        --model yolo --config configs/Detection/yolo11_birdsai.yaml \\
        --checkpoint /work/ziwen/experiments/yolo11l_birdsai_manual_.../checkpoints/best.pt

    # FasterRCNN / DINOv3 take the Lightning best-*.ckpt:
    CUDA_VISIBLE_DEVICES=1 python eval_birdsai_detect_track.py \\
        --model fasterrcnn --config configs/Detection/fasterrcnn_birdsai.yaml \\
        --checkpoint .../checkpoints/best-epoch=..-val_mAP=...ckpt
"""
from __future__ import annotations

# --- repo root on path so top-level modules (transforms, obb_utils, ...) import ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import torch
import yaml

from datasets.birdsai_mot import BIRDSAIMOTDataset
from models.trackers.ocsort import OCSortTracker

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BIRDSAI_ROOT = "/data/ESA_DLSTEM_2025/data/wild_animal/BIRDSAI"

# Canonical 0-indexed fine-grained taxonomy used in the JSON dump (aligns all three
# models). Matches the YOLO/DINOv3 config class_map; FasterRCNN is this +1 (background).
CANON_NAMES = {0: "human", 1: "elephant", 2: "giraffe", 3: "lion", 4: "unknown"}


# ----------------------------------------------------------------------
# Model loading — one builder per detector, all returning an eval-mode nn.Module
# whose forward(list[CHW float[0,1]]) -> list[{boxes xyxy, scores, labels}].
# Each builder also returns a label-mapping fn -> canonical 0-indexed class.
# ----------------------------------------------------------------------

def _load_yolo(cfg, ckpt_path):
    from models import YOLODetector
    img = cfg.get("img_size", 640)
    model = YOLODetector(
        model_name=cfg.get("model_name", "yolo11l.pt"),
        num_classes=cfg["num_classes"], enable_tracking=False,
        conf_thresh=cfg.get("conf_thresh", 0.05),
        iou_thresh=cfg.get("iou_thresh", 0.5), img_size=img,
    )
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)
    # YOLO class_map is already 0-indexed canonical {animal:0, human:1}.
    return model, (lambda lab: int(lab))


def _load_lightning(cfg, ckpt_path, build_model, label_offset):
    """Shared loader for the FasterRCNN / DINOv3 Lightning checkpoints."""
    from lightning_modules import ObjectDetectionModule
    model = build_model(cfg)
    module = ObjectDetectionModule.load_from_checkpoint(
        ckpt_path, model=model, has_tracking=False, map_location="cpu",
    )
    return module.model, (lambda lab: int(lab) - label_offset)


def _build_fasterrcnn(cfg):
    from models import FasterRCNNDetector
    anchor_sizes = tuple(tuple(s) for s in cfg["anchor_sizes"])
    anchor_aspect_ratios = tuple(tuple(r) for r in cfg["anchor_aspect_ratios"])
    return FasterRCNNDetector(
        num_classes=cfg["num_classes"], pretrained=cfg.get("pretrained", True),
        use_v2=cfg.get("use_v2", False),
        trainable_backbone_layers=cfg.get("trainable_backbone_layers", 3),
        score_thresh=cfg.get("score_thresh", 0.05), nms_thresh=cfg.get("nms_thresh", 0.5),
        detections_per_img=cfg.get("detections_per_img", 300), enable_tracking=False,
        anchor_sizes=anchor_sizes, anchor_aspect_ratios=anchor_aspect_ratios,
        rpn_fg_iou_thresh=cfg.get("rpn_fg_iou_thresh"), rpn_bg_iou_thresh=cfg.get("rpn_bg_iou_thresh"),
        box_fg_iou_thresh=cfg.get("box_fg_iou_thresh"), box_bg_iou_thresh=cfg.get("box_bg_iou_thresh"),
        rpn_pre_nms_top_n_train=cfg.get("rpn_pre_nms_top_n_train"),
        rpn_post_nms_top_n_train=cfg.get("rpn_post_nms_top_n_train"),
        min_size=cfg.get("min_size"), max_size=cfg.get("max_size"),
    )


def _build_dinov3(cfg):
    from models import DINOv3Detector
    return DINOv3Detector(
        num_classes=cfg["num_classes"],
        hf_model_name=cfg.get("hf_model_name", "facebook/dinov3-vitb16-pretrain-lvd1689m"),
        freeze_backbone=cfg.get("freeze_backbone", True), head_type=cfg.get("head_type", "fcos"),
        fcos_num_convs=cfg.get("fcos_num_convs", 4), fcos_hidden=cfg.get("fcos_hidden", 256),
        fcos_center_radius=cfg.get("fcos_center_radius", 1.5), nms_thresh=cfg.get("nms_thresh", 0.6),
        conf_thresh=cfg.get("conf_thresh", 0.05),
    )


def build_model(model_type, cfg, ckpt_path):
    if model_type == "yolo":
        return _load_yolo(cfg, ckpt_path)
    if model_type == "fasterrcnn":
        # FasterRCNN class_map is 1-indexed (0 = background) → subtract 1 for canonical.
        return _load_lightning(cfg, ckpt_path, _build_fasterrcnn, label_offset=1)
    if model_type == "dinov3":
        return _load_lightning(cfg, ckpt_path, _build_dinov3, label_offset=0)
    raise ValueError(f"unknown model {model_type!r}")


# ----------------------------------------------------------------------
# Inference on one original-resolution RGB frame.
# ----------------------------------------------------------------------

@torch.no_grad()
def detect_frame(model, rgb, img_size, amp_dtype, to_canon, dump_floor):
    """Run the detector on one HxWx3 uint8 RGB frame.

    Returns (boxes_xyxy_orig Nx4, scores N, labels N canonical), filtered to
    score >= dump_floor. Boxes are rescaled from the img_size² input back to the
    original frame resolution.
    """
    H, W = rgb.shape[:2]
    resized = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(resized).permute(2, 0, 1).float().div_(255.0).to(DEVICE)
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=DEVICE == "cuda"):
        out = model([t])[0]
    boxes = out["boxes"].float().cpu().numpy()
    scores = out["scores"].float().cpu().numpy()
    labels = np.array([to_canon(l) for l in out["labels"].cpu().numpy()], dtype=np.int64)
    if len(boxes):
        sx, sy = W / img_size, H / img_size
        boxes[:, [0, 2]] *= sx
        boxes[:, [1, 3]] *= sy
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)
    keep = scores >= dump_floor
    return boxes[keep], scores[keep], labels[keep]


# ----------------------------------------------------------------------
# Matching (class-aware greedy IoU), mirrors eval_tracker.py.
# ----------------------------------------------------------------------

def _iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def _greedy_match(gt_boxes, pred_boxes, iou_thr):
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return []
    iou = _iou_matrix(gt_boxes, pred_boxes)
    rows, cols = np.where(iou >= iou_thr)
    if len(rows) == 0:
        return []
    order = iou[rows, cols].argsort()[::-1]
    rows, cols = rows[order], cols[order]
    mg, mp, matches = set(), set(), []
    for r, c in zip(rows.tolist(), cols.tolist()):
        if r in mg or c in mp:
            continue
        mg.add(r); mp.add(c); matches.append((r, c))
    return matches


def main():
    ap = argparse.ArgumentParser(description="BIRDSAI detection + tracking eval")
    ap.add_argument("--model", required=True, choices=["yolo", "fasterrcnn", "dinov3"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--iou-thresh", type=float, default=0.5, help="det/track match IoU")
    ap.add_argument("--dump-floor", type=float, default=0.05,
                    help="min score to record a detection in predictions.json")
    ap.add_argument("--track-thresh", type=float, default=0.3,
                    help="OC-SORT det_thresh: min score for a det to be tracked")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    img_size = cfg.get("img_size", 640)
    amp_dtype = torch.bfloat16 if "bf16" in str(cfg.get("precision", "bf16-mixed")) else torch.float16

    torch.set_float32_matmul_precision("high")
    model, to_canon = build_model(args.model, cfg, args.checkpoint)
    model.eval().to(DEVICE)

    run_name = f"{args.model}_birdsai_dettrack"
    exp_root = cfg.get("experiment_root", "/work/ziwen/experiments")
    experiment_dir = Path(f"{exp_root}/{run_name}_{datetime.now():%Y%m%d_%H%M%S}")
    (experiment_dir / "mot_format").mkdir(parents=True, exist_ok=True)

    # Fine-grained species GT, with labels mapped to the canonical 0-indexed ids
    # so GT and (to_canon'd) predictions share one taxonomy.
    canon_map = {v: k for k, v in CANON_NAMES.items()}
    dataset = BIRDSAIMOTDataset(
        root=BIRDSAI_ROOT, split=args.split, granularity="fine", class_map=canon_map)

    print("=" * 64)
    print(f"BIRDSAI detect+track: model={args.model}  split={args.split}")
    print(f"  ckpt:   {args.checkpoint}")
    print(f"  output: {experiment_dir}")
    print(f"  imgsz={img_size}  iou_thr={args.iou_thresh}  track_thr={args.track_thresh}")
    print("=" * 64, flush=True)

    # Global accumulators (class-aware): per-class det/track counters.
    classes = sorted(CANON_NAMES)
    det = {c: {"tp": 0, "fp": 0, "fn": 0} for c in classes}
    trk = {c: {"tp": 0, "fp": 0, "fn": 0, "idsw": 0, "ngt": 0} for c in classes}

    predictions = {
        "model": args.model, "dataset": "BIRDSAI", "split": args.split,
        "img_size": img_size, "class_names": CANON_NAMES,
        "checkpoint": str(args.checkpoint), "videos": {},
    }
    per_video = {}
    t0 = time.perf_counter()

    for v_idx, video in enumerate(dataset.videos, 1):
        img_dir = dataset._img_dir_cache[video.video_id]
        # One independent tracker per class (OC-SORT is class-agnostic internally).
        trackers = {c: OCSortTracker(det_thresh=args.track_thresh, iou_threshold=0.3,
                                     min_hits=3, max_age=30) for c in classes}
        last_gt_to_pred = {c: {} for c in classes}
        mot_lines = []
        vdet = {c: {"tp": 0, "fp": 0, "fn": 0} for c in classes}
        vtrk = {c: {"tp": 0, "fp": 0, "fn": 0, "idsw": 0, "ngt": 0} for c in classes}
        frames_out = {}

        for fid in video.frame_ids:
            rgb = dataset._load_frame(video, fid)
            boxes, scores, labels = detect_frame(
                model, rgb, img_size, amp_dtype, to_canon, args.dump_floor)
            ann = dataset._load_annotations(video, fid)
            gt_boxes = np.asarray(ann["boxes"], dtype=np.float32)
            gt_labels = np.asarray(ann["labels"], dtype=np.int64)
            gt_tids = np.asarray(ann["track_ids"], dtype=np.int64)

            # ---- per-class tracking + scoring ----
            frame_track_boxes, frame_track_scores = [], []
            frame_track_labels, frame_track_ids = [], []
            for c in classes:
                cd = labels == c
                dets_c = (np.column_stack([boxes[cd], scores[cd]])
                          if cd.any() else np.zeros((0, 5), dtype=np.float32))
                tracks = trackers[c].update(dets_c, frame_id=fid)
                tb = tracks[:, :4] if len(tracks) else np.zeros((0, 4), np.float32)
                ts = tracks[:, 4] if len(tracks) else np.zeros(0, np.float32)
                # Globally-unique track id: namespace by class.
                tid = (tracks[:, 5].astype(np.int64) + c * 1_000_000
                       if len(tracks) else np.zeros(0, np.int64))

                gm = gt_labels == c
                gtb_c = gt_boxes[gm]; gtid_c = gt_tids[gm]
                matches = _greedy_match(gtb_c, tb, args.iou_thresh)
                tp = len(matches)
                vdet[c]["tp"] += tp
                vdet[c]["fp"] += len(tb) - tp
                vdet[c]["fn"] += len(gtb_c) - tp
                vtrk[c]["ngt"] += len(gtb_c)
                vtrk[c]["tp"] += tp
                vtrk[c]["fp"] += len(tb) - tp
                vtrk[c]["fn"] += len(gtb_c) - tp
                for r, cc in matches:
                    g = int(gtid_c[r]); p = int(tid[cc])
                    prev = last_gt_to_pred[c].get(g)
                    if prev is not None and prev != p:
                        vtrk[c]["idsw"] += 1
                    last_gt_to_pred[c][g] = p

                for j in range(len(tb)):
                    x1, y1, x2, y2 = tb[j]
                    mot_lines.append(
                        f"{int(fid)},{int(tid[j])},{x1:.2f},{y1:.2f},"
                        f"{x2 - x1:.2f},{y2 - y1:.2f},{float(ts[j]):.4f},-1,-1,-1")
                    frame_track_boxes.append([float(x1), float(y1), float(x2), float(y2)])
                    frame_track_scores.append(float(ts[j]))
                    frame_track_labels.append(int(c))
                    frame_track_ids.append(int(tid[j]))

            # ---- JSON dump: raw detections + tracked outputs (original coords) ----
            frames_out[str(int(fid))] = {
                "image_path": str(img_dir / f"{video.video_id}_{fid:010d}.jpg"),
                "detections": {
                    "boxes": boxes.round(2).tolist(),
                    "scores": scores.round(4).tolist(),
                    "labels": labels.tolist(),
                },
                "tracks": {
                    "boxes": frame_track_boxes,
                    "scores": frame_track_scores,
                    "labels": frame_track_labels,
                    "track_ids": frame_track_ids,
                },
            }

        with open(experiment_dir / "mot_format" / f"{video.video_id}.txt", "w") as f:
            f.write("\n".join(mot_lines))
        predictions["videos"][video.video_id] = {
            "image_dir": str(img_dir), "frames": frames_out}

        # per-video pooled metrics (over classes)
        def _pool(d, keys):
            return {k: sum(d[c][k] for c in classes) for k in keys}
        pd_ = _pool(vdet, ["tp", "fp", "fn"]); pt_ = _pool(vtrk, ["tp", "fp", "fn", "idsw", "ngt"])
        prec = pd_["tp"] / max(pd_["tp"] + pd_["fp"], 1)
        rec = pd_["tp"] / max(pd_["tp"] + pd_["fn"], 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        mota = 1.0 - (pt_["fp"] + pt_["fn"] + pt_["idsw"]) / max(pt_["ngt"], 1)
        idp = pt_["tp"] / max(pt_["tp"] + pt_["fp"], 1)
        idr = pt_["tp"] / max(pt_["tp"] + pt_["fn"], 1)
        idf1 = 2 * idp * idr / max(idp + idr, 1e-9)
        per_video[video.video_id] = {
            "Precision": prec, "Recall": rec, "F1": f1,
            "MOTA": mota, "IDF1": idf1, "IDsw": pt_["idsw"], "num_gt": pt_["ngt"]}

        for c in classes:
            for k in ("tp", "fp", "fn"):
                det[c][k] += vdet[c][k]
            for k in ("tp", "fp", "fn", "idsw", "ngt"):
                trk[c][k] += vtrk[c][k]
        print(f"[{v_idx}/{len(dataset.videos)}] {video.video_id}  "
              f"F1={f1:.3f} MOTA={mota:.3f} IDF1={idf1:.3f} IDsw={pt_['idsw']}", flush=True)

    # ---- global summary: per-class + pooled ----
    def metrics_from(d, t):
        prec = d["tp"] / max(d["tp"] + d["fp"], 1)
        rec = d["tp"] / max(d["tp"] + d["fn"], 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        mota = 1.0 - (t["fp"] + t["fn"] + t["idsw"]) / max(t["ngt"], 1)
        idp = t["tp"] / max(t["tp"] + t["fp"], 1)
        idr = t["tp"] / max(t["tp"] + t["fn"], 1)
        idf1 = 2 * idp * idr / max(idp + idr, 1e-9)
        return {"Precision": prec, "Recall": rec, "F1": f1, "MOTA": mota,
                "IDF1": idf1, "IDsw": t["idsw"], "num_gt": t["ngt"]}

    per_class = {CANON_NAMES[c]: metrics_from(det[c], trk[c]) for c in classes}
    pooled_d = {k: sum(det[c][k] for c in classes) for k in ("tp", "fp", "fn")}
    pooled_t = {k: sum(trk[c][k] for c in classes) for k in ("tp", "fp", "fn", "idsw", "ngt")}
    overall = metrics_from(pooled_d, pooled_t)

    summary = {
        "model": args.model, "dataset": "BIRDSAI", "split": args.split,
        "tracker": "ocsort", "iou_thresh": args.iou_thresh,
        "track_thresh": args.track_thresh, "total_videos": len(dataset.videos),
        "total_time_s": time.perf_counter() - t0,
        "overall": overall, "per_class": per_class,
    }
    with open(experiment_dir / "test_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(experiment_dir / "per_video_metrics.json", "w") as f:
        json.dump(per_video, f, indent=2)
    with open(experiment_dir / "predictions.json", "w") as f:
        json.dump(predictions, f)

    print("\n" + "=" * 64)
    print(f"OVERALL  Pr={overall['Precision']:.3f} Re={overall['Recall']:.3f} "
          f"F1={overall['F1']:.3f} | MOTA={overall['MOTA']:.3f} "
          f"IDF1={overall['IDF1']:.3f} IDsw={overall['IDsw']}")
    for name, m in per_class.items():
        print(f"  {name:7s} Pr={m['Precision']:.3f} Re={m['Recall']:.3f} "
              f"F1={m['F1']:.3f} MOTA={m['MOTA']:.3f} IDF1={m['IDF1']:.3f}")
    print(f"predictions.json + test_metrics.json → {experiment_dir}")
    print("=" * 64)


if __name__ == "__main__":
    main()
