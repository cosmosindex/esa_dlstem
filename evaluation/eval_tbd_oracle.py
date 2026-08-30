"""GT-box association ORACLE for the TBD (detection-by-tracking) trackers —
fair-comparison Experiment 2 (association vs object size).

Instead of the cached HiEUM detections that ``eval_tracker.py`` feeds, this
script feeds the GROUND-TRUTH boxes (score=1) to a motion-based TBD tracker
(sort / bytetrack / ocsort / botsort) and runs its UNCHANGED association. With
perfect, identical detection input for every method, any per-size difference in
the resulting tracks is PURE association ability — the same isolation the JDT
oracle (eval_fairmot.py / eval_tgram.py ``--gt-oracle``) provides for FairMOT /
TGraM. Outputs the standard contract so the size-stratified scorer can read it:

    mot_format/<safe_video_id>.txt   frame,id,x,y,w,h,conf,-1,-1,-1
    test_metrics.json                run config + box/track counts (sanity)

Full size range: GT is surfaced for ALL classes (car→airplane→ship→train) via
the per-dataset class maps, so cars (~5px) through aircraft (~44px) are all
present and binnable. Detection is GT, so it is dataset/class-agnostic — unlike
the HiEUM cache this does NOT depend on any trained detector.

Usage::

    python eval_tbd_oracle.py --tracker sort --dataset satmtb
    python eval_tbd_oracle.py --tracker bytetrack --dataset airmot
"""
from __future__ import annotations

# --- repo root on path so top-level modules (transforms, obb_utils, ...) import ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from datasets.airmot import AIRMOTDataset
from datasets.rscardata import RsCarDataset
from datasets.satmtb import SATMTBDataset
from datasets.sdmcar import SDMCarDataset
from datasets.viso import VISODataset
from models.trackers import build_tracker
from project_paths import DATA_ROOT


_DATASET_TABLE = {
    "rscardata": (RsCarDataset, f"{DATA_ROOT}/data/trafic/RsCarData", {}),
    "satmtb":    (SATMTBDataset, f"{DATA_ROOT}/data/trafic/SAT-MTB",
                  {"task": "mot"}),
    "sdmcar":    (SDMCarDataset, f"{DATA_ROOT}/data/trafic/SDM-Car", {}),
    "airmot":    (AIRMOTDataset, f"{DATA_ROOT}/data/trafic/AIR-MOT-100", {}),
    "viso_no_car": (VISODataset, f"{DATA_ROOT}/data/trafic/VISO",
                    {"categories": ["plane", "ship", "train"]}),
}

# Per-dataset class maps — every category non-negative so _load_annotations
# surfaces ALL GT (mirrors eval_fairmot.py / compute_hota). Integer values are
# irrelevant: the oracle pools all classes into one association problem.
_ALLCLASS_MAPS = {
    "rscardata":   {"car": 0},
    "satmtb":      {"airplane": 0, "car": 1, "ship": 2, "train": 3},
    "sdmcar":      {"car": 0},
    "airmot":      {"airplane": 0, "ship": 1},
    "viso_no_car": {"plane": 0, "ship": 1, "train": 2},
}

_MOTION_TBD = ("sort", "bytetrack", "ocsort", "botsort")
# Appearance-aware TBD: consume a FastReID embedding per GT box, cached by
# ``cache_gt_feats_mot.py`` (their update signature differs from the motion ones).
_REID_TBD = ("botsort_reid", "tracktrack")
_FEAT_DIM = 2048

# Oracle-only override: TrackTrack's ``min_box_area`` is an OUTPUT filter that
# would delete every satellite car (5px -> area 25) before association is even
# scored. The oracle isolates association, so the filter is disabled here; every
# other hyperparameter stays at the wrapper default, exactly like the motion TBDs.
_REID_KWARGS = {
    "botsort_reid": {"feat_dim": _FEAT_DIM},
    "tracktrack":   {"feat_dim": _FEAT_DIM, "min_box_area": 0.0},
}


def _safe_video_id(video_id: str) -> str:
    return video_id.replace("/", "_")


def _load_feat_cache(cache_dir: Path, video_id: str) -> dict[int, np.ndarray]:
    """{frame_id: [N, D] feats}, rows in GT annotation order (see cache_gt_feats_mot)."""
    z = np.load(cache_dir / f"{_safe_video_id(video_id)}.npz")
    flat_frame = z["flat_frame"]
    feats = z["feats"].astype(np.float32)
    out: dict[int, np.ndarray] = {}
    if not len(flat_frame):
        return out
    starts = np.concatenate(([0], np.where(np.diff(flat_frame) != 0)[0] + 1, [len(flat_frame)]))
    for s, e in zip(starts[:-1], starts[1:]):
        out[int(flat_frame[s])] = feats[s:e]
    return out


def _build_dataset(name: str, split: str = "test"):
    cls, root, extra = _DATASET_TABLE[name]
    cmap = _ALLCLASS_MAPS[name]
    if cls in (AIRMOTDataset, VISODataset):          # MOT-only loaders
        return cls(root=root, split=split, class_map=cmap, **extra)
    return cls(root=root, split=split, mode="detection", class_map=cmap, **extra)


def _update_reid(tracker, name: str, gt_boxes: np.ndarray, feats: np.ndarray) -> np.ndarray:
    """One frame for an appearance-aware tracker; GT boxes carry score 1.0."""
    scores = np.ones(len(gt_boxes), dtype=np.float32)
    if name == "botsort_reid":
        return tracker.update_with_feats(gt_boxes, scores, feats)
    # TrackTrack: [N, 6+D] = x1,y1,x2,y2,score,_pad,*feat
    if len(gt_boxes):
        dets = np.concatenate([gt_boxes, scores[:, None],
                               np.zeros((len(gt_boxes), 1), dtype=np.float32),
                               feats], axis=1).astype(np.float32)
    else:
        dets = np.zeros((0, 6 + feats.shape[1] if feats.size else 6 + _FEAT_DIM),
                        dtype=np.float32)
    return tracker.update_with_feats(dets, dets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker", required=True, choices=_MOTION_TBD + _REID_TBD,
                    help="TBD tracker; the appearance-aware ones need --feat-cache")
    ap.add_argument("--dataset", required=True, choices=list(_DATASET_TABLE))
    ap.add_argument("--tracker-kwargs", default="{}",
                    help="JSON dict of tracker constructor kwargs (default {}).")
    ap.add_argument("--feat-cache", default=None,
                    help="dir of per-video FastReID .npz on GT boxes "
                         "(cache_gt_feats_mot.py); required for botsort_reid/tracktrack")
    args = ap.parse_args()

    use_reid = args.tracker in _REID_TBD
    if use_reid and not args.feat_cache:
        ap.error(f"--feat-cache is required for {args.tracker}")
    feat_dir = Path(args.feat_cache) if args.feat_cache else None

    tracker_kwargs = dict(_REID_KWARGS.get(args.tracker, {}))
    tracker_kwargs.update(json.loads(args.tracker_kwargs))
    dataset_key = args.dataset

    exp_root = os.environ.get("EXPERIMENT_ROOT", "/work/anon/experiments")
    run_name = f"{args.tracker}_oracle_{dataset_key}"
    experiment_dir = Path(f"{exp_root}/{run_name}_{datetime.now():%Y%m%d_%H%M%S}")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    mot_dir = experiment_dir / "mot_format"; mot_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print(f"TBD GT-box ORACLE: {args.tracker} on {dataset_key} (Exp2, records-only)")
    print(f"output: {experiment_dir}")
    print("=" * 60)

    dataset = _build_dataset(dataset_key)
    tracker = build_tracker(args.tracker, **tracker_kwargs)

    total_gt = total_out = 0
    t0 = time.perf_counter()
    n_frames_total = 0

    for v_idx, video in enumerate(dataset.videos, 1):
        tracker.reset(vid_name=_safe_video_id(video.video_id)) if use_reid else tracker.reset()
        feats_by_fid = _load_feat_cache(feat_dir, video.video_id) if use_reid else {}
        mot_lines: list[str] = []
        v_gt = v_out = 0

        for fid in video.frame_ids:
            ann = dataset._load_annotations(video, fid)
            gt_boxes = np.asarray(ann["boxes"], dtype=np.float32).reshape(-1, 4)
            n_frames_total += 1
            v_gt += len(gt_boxes)
            if len(gt_boxes):
                dets = np.column_stack([gt_boxes,
                                        np.ones(len(gt_boxes), dtype=np.float32)])
            else:
                dets = np.zeros((0, 5), dtype=np.float32)

            if use_reid:
                feats = feats_by_fid.get(int(fid),
                                         np.zeros((0, _FEAT_DIM), dtype=np.float32))
                if len(feats) != len(gt_boxes):
                    raise RuntimeError(
                        f"feat cache misaligned for {video.video_id} frame {fid}: "
                        f"{len(feats)} feats vs {len(gt_boxes)} GT boxes")
                tracks = _update_reid(tracker, args.tracker, gt_boxes, feats)
            else:
                tracks = tracker.update(dets, frame_id=fid)
            if len(tracks):
                tracks = np.asarray(tracks, dtype=np.float32).reshape(-1, 6)
                v_out += len(tracks)
                for tr in tracks:
                    x1, y1, x2, y2, sc, tid = tr
                    mot_lines.append(
                        f"{int(fid)},{int(tid)},{x1:.2f},{y1:.2f},"
                        f"{x2 - x1:.2f},{y2 - y1:.2f},{float(sc):.4f},-1,-1,-1")

        (mot_dir / f"{_safe_video_id(video.video_id)}.txt").write_text("\n".join(mot_lines))
        total_gt += v_gt; total_out += v_out
        print(f"[{v_idx}/{len(dataset.videos)}] {video.video_id}  "
              f"gt_boxes={v_gt}  out_tracks={v_out}")

    t_total = time.perf_counter() - t0
    summary = {
        "tracker": f"{args.tracker}_oracle", "dataset": dataset_key,
        "mode": "gt_box_oracle", "tracker_kwargs": tracker_kwargs,
        "total_videos": len(dataset.videos), "total_frames": n_frames_total,
        "total_gt_boxes": total_gt, "total_out_tracks": total_out,
        "total_time_s": t_total, "fps": n_frames_total / max(t_total, 1e-9),
    }
    (experiment_dir / "test_metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nframes={n_frames_total} gt_boxes={total_gt} out_tracks={total_out} "
          f"time={t_total:.1f}s  output: {experiment_dir}")


if __name__ == "__main__":
    main()
