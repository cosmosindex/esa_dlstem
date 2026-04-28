"""
Extend the HiEUM detection cache with FastReID appearance features.

Reads an existing per-video JSON produced by ``cache_hieum_dets.py``
(``boxes``, ``scores``, ``frame_ids``), re-loads the corresponding
frames, runs the upstream TrackTrack FastReID SBS-S50 model on each
detection's bounding box crop, and writes a NumPy ``.npz`` file::

    /data/ESA_DLSTEM_2025/experiments/Detection/hieum_dets_with_feats_cache/<dataset>/<safe_vid>.npz

with arrays::

    frame_ids:  int32   (T,)        # native dataset frame ids
    flat_frame: int32   (Ndet,)     # repeated frame id per detection
    boxes:      float32 (Ndet, 4)   # xyxy in original-image coords
    scores:     float32 (Ndet,)
    feats:      float16 (Ndet, D)   # FastReID L2-normalized embedding
    image_size: int32   (2,)        # H, W

Plus per-frame ``boxes_per_frame`` slices reconstructable from
``flat_frame``. The flat layout matches what ``eval_tracktrack.py``
needs to assemble per-frame ``[N, 6+D]`` arrays.

Usage::

    python cache_hieum_dets_with_feats.py --dataset rscardata
    python cache_hieum_dets_with_feats.py --dataset satmtb
    python cache_hieum_dets_with_feats.py --dataset sdmcar

Reuses the existing detection cache by default; pass
``--in-dir`` to point at a different one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


# Add upstream FastReID dir to sys.path before importing it.
_PROJECT_ROOT = Path(__file__).resolve().parent
_TT_FASTREID_DIR = _PROJECT_ROOT / "TrackTrack" / "2. FastReID"
sys.path.insert(0, str(_TT_FASTREID_DIR))

from fastreid.emb_computer import EmbeddingComputer  # noqa: E402

from datasets.rscardata import RsCarDataset  # noqa: E402
from datasets.satmtb import SATMTBDataset  # noqa: E402
from datasets.sdmcar import SDMCarDataset  # noqa: E402


_DATASET_TABLE = {
    "rscardata": (RsCarDataset, "/data/ESA_DLSTEM_2025/data/trafic/RsCarData", {}),
    "satmtb":    (SATMTBDataset, "/data/ESA_DLSTEM_2025/data/trafic/SAT-MTB",
                  {"task": "mot", "categories": ["car"]}),
    "sdmcar":    (SDMCarDataset, "/data/ESA_DLSTEM_2025/data/trafic/SDM-Car", {}),
}


def _safe_video_id(video_id: str) -> str:
    return video_id.replace("/", "_")


def _build_dataset(name: str, split: str = "test"):
    cls, root, extra = _DATASET_TABLE[name]
    return cls(root=root, split=split, mode="detection",
               class_map={"car": 0}, **extra)


def _process_video(
    cache_path: Path,
    out_path: Path,
    dataset,
    video,
    embedder: EmbeddingComputer,
) -> tuple[int, int]:
    """Returns (n_frames, n_dets)."""
    with open(cache_path) as f:
        cache = json.load(f)

    frame_ids: list[int] = cache["frame_ids"]
    boxes_per_frame: list[list[list[float]]] = cache["boxes"]
    scores_per_frame: list[list[float]] = cache["scores"]
    H, W = cache.get("image_size", [None, None])

    flat_frame: list[int] = []
    flat_box: list[list[float]] = []
    flat_score: list[float] = []
    flat_feat: list[np.ndarray] = []

    for fid, fboxes, fscores in zip(frame_ids, boxes_per_frame, scores_per_frame):
        if not fboxes:
            continue

        # _load_frame returns RGB; FastReID's compute_embedding expects BGR
        # (it does cv2.cvtColor(BGR -> RGB) internally), so flip back.
        img_rgb = dataset._load_frame(video, fid)
        img_bgr = img_rgb[..., ::-1]
        if H is None:
            H, W = img_bgr.shape[:2]

        boxes_arr = np.asarray(fboxes, dtype=np.float32)
        feats = embedder.compute_embedding(img_bgr, boxes_arr)  # (N, D) float16, L2-normalized

        flat_frame.extend([fid] * len(boxes_arr))
        flat_box.extend(boxes_arr.tolist())
        flat_score.extend(list(fscores))
        flat_feat.append(feats.astype(np.float16))

    if flat_feat:
        feats_arr = np.concatenate(flat_feat, axis=0)
    else:
        feats_arr = np.zeros((0, embedder.model.cfg.MODEL.HEADS.EMBEDDING_DIM
                              if embedder.model is not None else 2048),
                             dtype=np.float16)

    np.savez_compressed(
        out_path,
        frame_ids=np.asarray(frame_ids, dtype=np.int32),
        flat_frame=np.asarray(flat_frame, dtype=np.int32),
        boxes=np.asarray(flat_box, dtype=np.float32).reshape(-1, 4),
        scores=np.asarray(flat_score, dtype=np.float32),
        feats=feats_arr,
        image_size=np.asarray([H, W], dtype=np.int32),
    )
    return len(frame_ids), int(feats_arr.shape[0])


def main():
    parser = argparse.ArgumentParser(description="Cache FastReID feats per HiEUM detection")
    parser.add_argument("--dataset", required=True,
                        choices=sorted(_DATASET_TABLE.keys()))
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--in-dir", default="/data/ESA_DLSTEM_2025/experiments/Detection/hieum_dets_cache",
        help="Source detection cache root (per-dataset subdir).",
    )
    parser.add_argument(
        "--out-dir", default="/data/ESA_DLSTEM_2025/experiments/Detection/hieum_dets_with_feats_cache",
        help="Output cache root (per-dataset subdir).",
    )
    parser.add_argument(
        "--fastreid-config",
        default=str(_TT_FASTREID_DIR / "configs" / "MOT17" / "sbs_S50.yml"),
    )
    parser.add_argument(
        "--fastreid-weight",
        default=str(_TT_FASTREID_DIR / "weights" / "mot17_sbs_S50.pth"),
    )
    args = parser.parse_args()

    in_dir  = Path(args.in_dir) / args.dataset
    out_dir = Path(args.out_dir) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        raise FileNotFoundError(
            f"Source detection cache not found: {in_dir}\n"
            f"Run cache_hieum_dets.py --dataset {args.dataset} first."
        )

    print(f"Adding FastReID feats to {in_dir} → {out_dir}")
    print(f"  Config: {args.fastreid_config}")
    print(f"  Weight: {args.fastreid_weight}")

    dataset = _build_dataset(args.dataset, split=args.split)

    # FastReID — model is built lazily on first call.
    embedder = EmbeddingComputer(
        config_path=args.fastreid_config,
        weight_path=args.fastreid_weight,
    )

    t_total = 0.0
    n_frames_total = 0
    n_dets_total = 0
    for i, video in enumerate(dataset.videos, 1):
        cache_path = in_dir / f"{_safe_video_id(video.video_id)}.json"
        out_path = out_dir / f"{_safe_video_id(video.video_id)}.npz"

        if not cache_path.exists():
            print(f"  [{i}/{len(dataset.videos)}] {video.video_id} "
                  f"— SKIP, no cache at {cache_path}")
            continue
        if out_path.exists():
            print(f"  [{i}/{len(dataset.videos)}] {video.video_id} "
                  f"— already done, skipping")
            continue

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        n_frames, n_dets = _process_video(cache_path, out_path, dataset, video, embedder)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        t_total += dt
        n_frames_total += n_frames
        n_dets_total += n_dets

        print(f"  [{i}/{len(dataset.videos)}] {video.video_id} "
              f"| {n_frames} frames | {n_dets} dets | {dt:.1f}s "
              f"({n_frames / max(dt, 1e-9):.1f} fps)")

    print(f"\nDone. {n_frames_total} frames, {n_dets_total} dets, "
          f"{t_total:.1f}s "
          f"({n_frames_total / max(t_total, 1e-9):.1f} fps)")


if __name__ == "__main__":
    main()
