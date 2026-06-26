"""
cache_birdsai_feats.py
======================
Build a FastReID appearance-feature cache for the appearance-aware TBD trackers
(BoT-SORT-ReID / TrackTrack) on BIRDSAI.

Reads a detector's ``predictions.json`` (produced by
``eval_birdsai_detect_track.py`` — per-frame raw ``detections``), re-loads each
frame, runs the TrackTrack FastReID SBS-S50 model on every detection's box crop,
and writes one ``.npz`` per video::

    <out_dir>/<safe_video_id>.npz
        flat_frame : int32   (Ndet,)
        boxes      : float32 (Ndet, 4)   # xyxy, same order as predictions.json
        scores     : float32 (Ndet,)
        feats      : float16 (Ndet, D)   # L2-normalized FastReID embedding

Feature rows follow the exact order of ``detections["boxes"]`` per frame, so the
sweep eval can align them back to the cached detections without re-detecting.

NOTE: the FastReID weights are MOT17 RGB-pedestrian — applied to grayscale
thermal animal crops this is a strong domain mismatch; appearance cues are
expected to be near-useless. Run anyway for benchmark completeness.

Usage::

    python evaluation/cache_birdsai_feats.py \\
        --predictions /work/.../yolo_birdsai_dettrack_*/predictions.json \\
        --out-dir /data/.../MOT_birdsai_sweep/feats/yolo
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Upstream TrackTrack FastReID lives in a numbered dir; add to path before import.
_TT_FASTREID_DIR = _PROJECT_ROOT / "TrackTrack" / "2. FastReID"
sys.path.insert(0, str(_TT_FASTREID_DIR))
from fastreid.emb_computer import EmbeddingComputer  # noqa: E402


def _safe_video_id(video_id: str) -> str:
    return video_id.replace("/", "_")


def _process_video(frames: dict, embedder: EmbeddingComputer):
    """frames: {frame_id_str -> {image_path, detections{boxes,scores,labels}}}.

    Returns (flat_frame, flat_box, flat_score, feats_arr)."""
    flat_frame, flat_box, flat_score, flat_feat = [], [], [], []

    for fid_str in sorted(frames, key=lambda s: int(s)):
        fr = frames[fid_str]
        boxes = fr["detections"]["boxes"]
        scores = fr["detections"]["scores"]
        if not boxes:
            continue

        img_bgr = cv2.imread(fr["image_path"])  # BGR; FastReID flips to RGB internally
        if img_bgr is None:
            raise FileNotFoundError(fr["image_path"])

        boxes_arr = np.asarray(boxes, dtype=np.float32)
        feats = embedder.compute_embedding(img_bgr, boxes_arr)  # (N, D) fp16, L2-norm

        fid = int(fid_str)
        flat_frame.extend([fid] * len(boxes_arr))
        flat_box.extend(boxes_arr.tolist())
        flat_score.extend(list(scores))
        flat_feat.append(np.asarray(feats, dtype=np.float16))

    feats_arr = (np.concatenate(flat_feat, axis=0) if flat_feat
                 else np.zeros((0, 2048), dtype=np.float16))
    return flat_frame, flat_box, flat_score, feats_arr


def main():
    ap = argparse.ArgumentParser(description="FastReID feature cache for BIRDSAI TBD ReID trackers")
    ap.add_argument("--predictions", required=True, help="detector predictions.json")
    ap.add_argument("--out-dir", required=True, help="output dir for per-video .npz")
    ap.add_argument("--fastreid-config",
                    default=str(_TT_FASTREID_DIR / "configs" / "MOT17" / "sbs_S50.yml"))
    ap.add_argument("--fastreid-weight",
                    default=str(_TT_FASTREID_DIR / "weights" / "mot17_sbs_S50.pth"))
    args = ap.parse_args()

    with open(args.predictions) as f:
        preds = json.load(f)
    videos = preds["videos"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"FastReID feat cache: {args.predictions}")
    print(f"  model={preds.get('model')}  videos={len(videos)}  → {out_dir}")
    embedder = EmbeddingComputer(config_path=args.fastreid_config,
                                 weight_path=args.fastreid_weight)

    n_dets_total = 0
    for i, (vid, vdata) in enumerate(videos.items(), 1):
        out_path = out_dir / f"{_safe_video_id(vid)}.npz"
        if out_path.exists():
            print(f"  [{i}/{len(videos)}] {vid} — already done, skipping")
            continue
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        flat_frame, flat_box, flat_score, feats_arr = _process_video(vdata["frames"], embedder)
        np.savez_compressed(
            out_path,
            flat_frame=np.asarray(flat_frame, dtype=np.int32),
            boxes=np.asarray(flat_box, dtype=np.float32).reshape(-1, 4),
            scores=np.asarray(flat_score, dtype=np.float32),
            feats=feats_arr,
        )
        n_dets_total += int(feats_arr.shape[0])
        dt = time.perf_counter() - t0
        print(f"  [{i}/{len(videos)}] {vid}  {feats_arr.shape[0]} dets  {dt:.1f}s", flush=True)

    print(f"DONE — {n_dets_total} detection features cached → {out_dir}")


if __name__ == "__main__":
    main()
