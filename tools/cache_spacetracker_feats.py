"""
FastReID appearance-feature cache for the appearance-aware TBD trackers
(BoT-SORT-ReID / TrackTrack) on the Space-Tracker MOT release.

The plain TBD trackers read the JSON detection cache written by
``tools/cache_fasterrcnn_dets.py``; the two appearance-aware ones additionally
need an embedding per detection. This reads that same JSON cache -- so the
boxes are byte-identical to what every other tracker sees, and the comparison
stays a pure association comparison -- crops each box, and runs the vendored
TrackTrack FastReID SBS-S50 model over it.

Output, one ``.npz`` per video, matching what
``evaluation/eval_tracker_multiclass_reid.py::_load_feat_cache`` expects::

    frame_ids  : int32   (T,)      every frame, including empty ones
    flat_frame : int32   (Ndet,)   frame id of each detection
    boxes      : float32 (Ndet, 4) xyxy
    scores     : float32 (Ndet,)
    labels     : int64   (Ndet,)
    feats      : float16 (Ndet, D) L2-normalised embedding

NOTE on domain mismatch: these weights are MOT17 RGB pedestrians (~128x384
crops). Our targets are 4-40 px satellite ships and aircraft, so appearance
carries very little and matching largely falls back to IoU. That gap is part of
what the benchmark measures -- it is reported as a caveat column, not hidden.

Usage:
    python tools/cache_spacetracker_feats.py \
        --det-cache /path/to/spacetracker_nocar_dets_cache/spacetracker_nocar \
        --out-dir   /path/to/spacetracker_nocar_feats_cache
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

_TT_FASTREID_DIR = _PROJECT_ROOT / "TrackTrack" / "2. FastReID"
sys.path.insert(0, str(_TT_FASTREID_DIR))

from datasets import SpaceTrackerMOTDataset  # noqa: E402
from fastreid.emb_computer import EmbeddingComputer  # noqa: E402


def _safe_video_id(video_id: str) -> str:
    return video_id.replace("/", "_")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--det-cache", required=True,
                    help="Directory of per-video detection JSONs")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--release", default="/data/ESA_DLSTEM_2025/release/space_tracker")
    ap.add_argument("--fastreid-config",
                    default=str(_TT_FASTREID_DIR / "configs" / "MOT17" / "sbs_S50.yml"))
    ap.add_argument("--weights",
                    default=str(_TT_FASTREID_DIR / "weights" / "mot17_sbs_S50.pth"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--categories", nargs="+", default=["airplane", "ship"],
                    help="Track categories to cache. Use --categories car for "
                         "the HiEUM-detected car half.")
    args = ap.parse_args()

    det_dir = Path(args.det_cache)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = SpaceTrackerMOTDataset(
        root=args.release, split=args.split, mode="detection",
        class_map=({"car": 1} if args.categories == ["car"]
                   else {"airplane": 1, "ship": 2}),
        # car GT annotates moving objects only and was never completed with
        # static ones, so the completeness filter (correct for airplane/ship)
        # would leave 2 of 48 car sequences.
        complete_only=(args.categories != ["car"]),
        categories=args.categories,
    )
    if args.categories == ["car"]:
        # categories= filters by TRACK class, so it also admits mixed sequences
        # that merely contain a car. The car benchmark is the 48 sequences whose
        # RELEASE category is 'car' -- the set HiEUM was trained and scored on.
        import json as _json
        ann = _json.loads((Path(args.release) / "mot" / "annotations"
                           / "space_tracker_mot.json").read_text())
        keep = {v["name"] for v in ann["videos"] if v["category"] == "car"}
        before = len(ds.videos)
        ds.videos = [v for v in ds.videos if v.video_id in keep]
        print(f"[setup] car category filter: {before} -> {len(ds.videos)}")
    by_id = {v.video_id: v for v in ds.videos}
    print(f"[setup] {len(ds.videos)} sequences, weights={args.weights}")

    embedder = EmbeddingComputer(config_path=args.fastreid_config,
                                 weight_path=args.weights)

    t_all = time.time()
    for vi, video in enumerate(ds.videos, 1):
        vid = video.video_id
        out_path = out_dir / f"{_safe_video_id(vid)}.npz"
        if out_path.exists():
            print(f"  [{vi:>3d}/{len(ds.videos)}] {vid}  (cached, skip)")
            continue
        det_path = det_dir / f"{_safe_video_id(vid)}.json"
        if not det_path.exists():
            print(f"  [{vi:>3d}/{len(ds.videos)}] {vid}  MISSING detections, skip")
            continue

        d = json.loads(det_path.read_text())
        frame_ids = d["frame_ids"]
        t0 = time.time()

        flat_frame, flat_box, flat_score, flat_label, flat_feat = [], [], [], [], []
        for k, fid in enumerate(frame_ids):
            boxes = np.asarray(d["boxes"][k], dtype=np.float32).reshape(-1, 4)
            if len(boxes) == 0:
                continue
            img_rgb = ds._load_frame(by_id[vid], fid)
            # EmbeddingComputer expects BGR (it converts to RGB internally),
            # while the dataset hands out RGB.
            img_bgr = np.ascontiguousarray(img_rgb[..., ::-1])
            feats = embedder.compute_embedding(img_bgr, boxes)

            flat_frame.extend([int(fid)] * len(boxes))
            flat_box.append(boxes)
            flat_score.extend(list(d["scores"][k]))
            # HiEUM is single-class and writes no label array.
            flat_label.extend(list(d["labels"][k]) if "labels" in d
                              else [1] * len(d["boxes"][k]))
            flat_feat.append(np.asarray(feats, dtype=np.float16))

        if flat_feat:
            boxes_arr = np.concatenate(flat_box, axis=0).astype(np.float32)
            feats_arr = np.concatenate(flat_feat, axis=0)
        else:
            boxes_arr = np.zeros((0, 4), dtype=np.float32)
            feats_arr = np.zeros((0, 2048), dtype=np.float16)

        np.savez_compressed(
            out_path,
            frame_ids=np.asarray(frame_ids, dtype=np.int32),
            flat_frame=np.asarray(flat_frame, dtype=np.int32),
            boxes=boxes_arr,
            scores=np.asarray(flat_score, dtype=np.float32),
            labels=np.asarray(flat_label, dtype=np.int64),
            feats=feats_arr,
        )
        print(f"  [{vi:>3d}/{len(ds.videos)}] {vid}  {len(flat_frame)} dets  "
              f"{time.time() - t0:.1f}s")

    print(f"\ndone in {time.time() - t_all:.1f}s -> {out_dir}")


if __name__ == "__main__":
    main()
