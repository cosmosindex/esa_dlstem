"""Add FastReID 2048-D features to the FasterRCNN multi-class detection
cache so BoT-SORT-ReID and TrackTrack can be evaluated on the non-car
half of Space-Tracker-MOT.

Reads:    <fasterrcnn-cache>/<dataset>/<safe_vid>.json   (boxes/scores/labels)
Writes:   <out-dir>/<dataset>/<safe_vid>.npz  with arrays
              frame_ids:   (T,)        int32
              flat_frame:  (Ndet,)     int32
              boxes:       (Ndet, 4)   float32  xyxy
              scores:      (Ndet,)     float32
              labels:      (Ndet,)     int32    1=airplane 2=ship 3=train
              feats:       (Ndet, D)   float16  L2-normalised
              image_size:  (2,)        int32    H, W
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent
_TT_FASTREID_DIR = _PROJECT_ROOT / "TrackTrack" / "2. FastReID"
sys.path.insert(0, str(_TT_FASTREID_DIR))

from fastreid.emb_computer import EmbeddingComputer  # noqa: E402

from datasets.airmot import AIRMOTDataset  # noqa: E402
from datasets.satmtb import SATMTBDataset  # noqa: E402
from datasets.viso import VISODataset  # noqa: E402


_DATASET_TABLE = {
    "satmtb_nocar": (SATMTBDataset, "/data/ESA_DLSTEM_2025/data/trafic/SAT-MTB",
                     {"task": "mot", "categories": ["airplane", "ship", "train"]},
                     "test"),
    "viso_nocar":   (VISODataset, "/data/ESA_DLSTEM_2025/data/trafic/VISO",
                     {"categories": ["plane", "ship", "train"]}, "no_split"),
    "airmot":       (AIRMOTDataset, "/data/ESA_DLSTEM_2025/data/trafic/AIR-MOT-100",
                     {}, "no_split"),
}


def _safe_video_id(video_id: str) -> str:
    return video_id.replace("/", "_")


def _build_dataset(name: str):
    cls, root, extra, split = _DATASET_TABLE[name]
    return cls(root=root, split=split, **extra)


def _process_video(cache_path, out_path, dataset, video, embedder):
    with open(cache_path) as f:
        cache = json.load(f)

    frame_ids = cache["frame_ids"]
    boxes_pf  = cache["boxes"]
    scores_pf = cache["scores"]
    labels_pf = cache["labels"]
    H, W = cache.get("image_size", [None, None])

    flat_frame: list[int] = []
    flat_box: list[list[float]] = []
    flat_score: list[float] = []
    flat_label: list[int] = []
    flat_feat: list[np.ndarray] = []

    for fid, fboxes, fscores, flabels in zip(frame_ids, boxes_pf, scores_pf, labels_pf):
        if not fboxes:
            continue
        img_bgr = dataset._load_frame(video, fid)     # _load_frame returns BGR
        if H is None:
            H, W = img_bgr.shape[:2]

        boxes_arr  = np.asarray(fboxes,  dtype=np.float32)
        scores_arr = np.asarray(fscores, dtype=np.float32)
        labels_arr = np.asarray(flabels, dtype=np.int32)

        # Drop boxes that are degenerate after clipping to image bounds
        # (FastReID's compute_embedding cv2.cvtColor blows up on zero-area
        # crops). Use the same clip the embedder applies.
        cb = np.round(boxes_arr).astype(np.int32)
        cb[:, [0, 2]] = cb[:, [0, 2]].clip(0, W)
        cb[:, [1, 3]] = cb[:, [1, 3]].clip(0, H)
        valid = (cb[:, 2] > cb[:, 0]) & (cb[:, 3] > cb[:, 1])
        if not valid.any():
            continue
        boxes_arr  = boxes_arr[valid]
        scores_arr = scores_arr[valid]
        labels_arr = labels_arr[valid]

        feats = embedder.compute_embedding(img_bgr, boxes_arr)
        flat_frame.extend([fid] * len(boxes_arr))
        flat_box.extend(boxes_arr.tolist())
        flat_score.extend(scores_arr.tolist())
        flat_label.extend(labels_arr.tolist())
        flat_feat.append(feats.astype(np.float16))

    if flat_feat:
        feats_arr = np.concatenate(flat_feat, axis=0)
    else:
        feats_arr = np.zeros((0, 2048), dtype=np.float16)

    np.savez_compressed(
        out_path,
        frame_ids=np.asarray(frame_ids, dtype=np.int32),
        flat_frame=np.asarray(flat_frame, dtype=np.int32),
        boxes=np.asarray(flat_box, dtype=np.float32).reshape(-1, 4),
        scores=np.asarray(flat_score, dtype=np.float32),
        labels=np.asarray(flat_label, dtype=np.int32),
        feats=feats_arr,
        image_size=np.asarray([H, W], dtype=np.int32),
    )
    return len(frame_ids), int(feats_arr.shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["satmtb_nocar", "viso_nocar", "airmot"])
    ap.add_argument("--in-root",
                    default="/data/ESA_DLSTEM_2025/experiments/Detection/fasterrcnn_satmtb_hbb_dets_cache")
    ap.add_argument("--out-root",
                    default="/data/ESA_DLSTEM_2025/experiments/Detection/fasterrcnn_satmtb_hbb_dets_with_feats_cache")
    ap.add_argument("--fastreid-config",
                    default=str(_TT_FASTREID_DIR / "configs" / "MOT17" / "sbs_S50.yml"))
    ap.add_argument("--fastreid-weight",
                    default=str(_TT_FASTREID_DIR / "weights" / "mot17_sbs_S50.pth"))
    args = ap.parse_args()

    embedder = EmbeddingComputer(
        config_path=args.fastreid_config, weight_path=args.fastreid_weight,
    )

    t_total0 = time.time()
    grand_frames = grand_dets = 0
    for ds_name in args.datasets:
        in_dir  = Path(args.in_root)  / ds_name
        out_dir = Path(args.out_root) / ds_name
        out_dir.mkdir(parents=True, exist_ok=True)
        if not in_dir.exists():
            print(f"[skip] {ds_name}: missing in-dir {in_dir}")
            continue
        dataset = _build_dataset(ds_name)
        print(f"\n[{ds_name}] {len(dataset.videos)} videos  {in_dir} -> {out_dir}")
        for i, v in enumerate(dataset.videos, 1):
            cache = in_dir / f"{_safe_video_id(v.video_id)}.json"
            out   = out_dir / f"{_safe_video_id(v.video_id)}.npz"
            if not cache.exists():
                print(f"  [{i:>3d}/{len(dataset.videos)}] {v.video_id}  no JSON, skip")
                continue
            if out.exists():
                print(f"  [{i:>3d}/{len(dataset.videos)}] {v.video_id}  cached, skip")
                continue
            t0 = time.time()
            n_f, n_d = _process_video(cache, out, dataset, v, embedder)
            grand_frames += n_f; grand_dets += n_d
            dt = time.time() - t0
            print(f"  [{i:>3d}/{len(dataset.videos)}] {v.video_id:35s} "
                  f"{n_f:>5d} frames  {n_d:>6d} dets  {dt:5.1f}s "
                  f"({n_d/dt if dt>0 else 0:.0f} dets/s)")
    print(f"\nALL DONE in {(time.time()-t_total0)/60:.1f} min  "
          f"({grand_frames} frames / {grand_dets} dets)")


if __name__ == "__main__":
    main()
