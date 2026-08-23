"""FastReID appearance-feature cache on GT boxes for the Exp2 association ORACLE.

The appearance-aware TBD trackers (BoT-SORT-ReID / TrackTrack) need a per-detection
embedding. In the GT-box oracle the "detections" ARE the ground-truth boxes, so this
script re-loads every test frame of a MOT dataset, runs the TrackTrack FastReID
SBS-S50 model on each GT box crop, and writes one ``.npz`` per video::

    <out_dir>/<safe_video_id>.npz
        flat_frame : int32   (Nbox,)
        boxes      : float32 (Nbox, 4)   # xyxy, EXACT order of _load_annotations
        feats      : float16 (Nbox, D)   # L2-normalized FastReID embedding

Rows follow the exact per-frame order of ``_load_annotations(video, fid)["boxes"]``,
so ``eval_tbd_oracle.py`` can align them back without recomputing anything.

Mirrors ``cache_birdsai_feats.py`` (which does the same for the BIRDSAI GT oracle);
the same MOT17-RGB-pedestrian domain-mismatch caveat applies — satellite crops are
5-40 px, so appearance is expected to add little. Run for benchmark completeness.

Usage::

    python evaluation/cache_gt_feats_mot.py --dataset rscardata \\
        --out-dir /data/.../exp2_oracle_20260608/gt_feats/rscardata
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Upstream TrackTrack FastReID lives in a numbered dir; add to path before import.
_TT_FASTREID_DIR = _PROJECT_ROOT / "TrackTrack" / "2. FastReID"
sys.path.insert(0, str(_TT_FASTREID_DIR))
from fastreid.emb_computer import EmbeddingComputer  # noqa: E402

from evaluation.eval_tbd_oracle import _build_dataset, _safe_video_id  # noqa: E402

_FEAT_DIM = 2048


def _sanitize(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    """Clip to the frame and force >=2px extent so cv2.resize never sees an empty crop."""
    b = boxes.astype(np.float32).copy()
    b[:, 0] = np.clip(b[:, 0], 0, w - 2); b[:, 1] = np.clip(b[:, 1], 0, h - 2)
    b[:, 2] = np.clip(b[:, 2], b[:, 0] + 2, w); b[:, 3] = np.clip(b[:, 3], b[:, 1] + 2, h)
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fastreid-config",
                    default=str(_TT_FASTREID_DIR / "configs" / "MOT17" / "sbs_S50.yml"))
    ap.add_argument("--fastreid-weight",
                    default=str(_TT_FASTREID_DIR / "weights" / "mot17_sbs_S50.pth"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    dataset = _build_dataset(args.dataset)
    embedder = EmbeddingComputer(config_path=args.fastreid_config,
                                 weight_path=args.fastreid_weight)

    print(f"GT-box FastReID cache: {args.dataset}  ({len(dataset.videos)} videos) -> {out_dir}")
    n_total = 0
    for v_idx, video in enumerate(dataset.videos, 1):
        out_path = out_dir / f"{_safe_video_id(video.video_id)}.npz"
        if out_path.exists():
            print(f"  [{v_idx}/{len(dataset.videos)}] {video.video_id} — cached, skip", flush=True)
            continue
        t0 = time.perf_counter()
        flat_frame, flat_box, flat_feat = [], [], []
        for fid in video.frame_ids:
            ann = dataset._load_annotations(video, fid)
            boxes = np.asarray(ann["boxes"], dtype=np.float32).reshape(-1, 4)
            if not len(boxes):
                continue
            img = dataset._load_frame(video, fid)          # RGB uint8
            h, w = img.shape[:2]
            img_bgr = np.ascontiguousarray(img[..., ::-1])  # FastReID flips back internally
            feats = embedder.compute_embedding(img_bgr, _sanitize(boxes, w, h))
            flat_frame.extend([int(fid)] * len(boxes))
            flat_box.append(boxes)
            flat_feat.append(np.asarray(feats, dtype=np.float16))

        feats_arr = (np.concatenate(flat_feat, axis=0) if flat_feat
                     else np.zeros((0, _FEAT_DIM), dtype=np.float16))
        boxes_arr = (np.concatenate(flat_box, axis=0) if flat_box
                     else np.zeros((0, 4), dtype=np.float32))
        np.savez(out_path,
                 flat_frame=np.asarray(flat_frame, dtype=np.int32),
                 boxes=boxes_arr, feats=feats_arr)
        n_total += len(feats_arr)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  [{v_idx}/{len(dataset.videos)}] {video.video_id}  "
              f"{len(feats_arr)} boxes  {time.perf_counter()-t0:.1f}s", flush=True)

    print(f"DONE — {n_total} GT-box features -> {out_dir}")


if __name__ == "__main__":
    main()
