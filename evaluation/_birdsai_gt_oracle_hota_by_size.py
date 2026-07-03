#!/usr/bin/env python
"""Size-stratified TrackEval HOTA/AssA for the BIRDSAI GT-box oracle.

Reuses the Exp2 machinery (`compute_hota_by_size._eval_bin` / `_trackeval_one`):
each GT TRACK is binned by its representative size (median √area over its boxes),
GT+preds are filtered to that bin, and real TrackEval HOTA/CLEAR/Identity is run
→ **AssA / IDF1 / IDsw as a function of object size**. AssA is the principled
pure-association metric (replaces the home-made per-frame "ID purity").

Runs consumed: `gtoracle_{tracker}_birdsai_track_*` under --oracle-root
(produced by run_birdsai_gt_oracle.sh). Detection is oracle (GT in) so DetA ~ 1
and per-size differences are pure association.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
from collections import defaultdict

import numpy as np

from datasets.birdsai_mot import BIRDSAIMOTDataset
from evaluation.compute_hota_by_size import _eval_bin, _safe_video_id

BIRDSAI_ROOT = "/data/ESA_DLSTEM_2025/data/wild_animal/BIRDSAI"
CANON = {0: "human", 1: "elephant", 2: "giraffe", 3: "lion", 4: "unknown"}
EDGES = [0.0, 14.0, 20.0, 28.0, 38.0, 50.0, float("inf")]
TRACKERS = ["sort", "ocsort", "bytetrack", "botsort", "botsort_reid", "tracktrack"]


def _load_gt(annotations: str):
    """videos: [(seq, frame_ids, offset, {fid:(boxes,tids)})] + track_sizes{seq:{tid:[√area]}}.

    Track ids are namespaced by class (tid + class*1e6) to match the per-class
    pred ids in mot_format, so GT and preds share one identity space.
    """
    ds = BIRDSAIMOTDataset(root=BIRDSAI_ROOT, split="test", granularity="fine",
                           annotations_dirname=annotations,
                           class_map={v: k for k, v in CANON.items()})
    videos = []
    track_sizes: dict[str, dict[int, list]] = {}
    for v in ds.videos:
        seq = _safe_video_id(v.video_id)
        first = min(v.frame_ids)
        offset = 1 - int(first) if first < 1 else 0
        per_frame = {}
        sizes: dict[int, list] = defaultdict(list)
        for fid in v.frame_ids:
            a = ds._load_annotations(v, fid)
            boxes = np.asarray(a["boxes"], np.float32).reshape(-1, 4)
            tids = np.asarray(a["track_ids"], np.int64).reshape(-1)
            labels = np.asarray(a["labels"], np.int64).reshape(-1)
            gtid = tids + labels * 1_000_000          # class-namespaced, matches preds
            per_frame[fid] = (boxes, gtid)
            for j in range(len(boxes)):
                w = boxes[j, 2] - boxes[j, 0]; h = boxes[j, 3] - boxes[j, 1]
                sizes[int(gtid[j])].append(float(np.sqrt(max(w * h, 0.0))))
        videos.append((seq, list(v.frame_ids), offset, per_frame))
        track_sizes[seq] = sizes
    return videos, track_sizes


def _collect(root: Path):
    runs = {}
    for trk in TRACKERS:
        hits = sorted(root.glob(f"gtoracle_{trk}_birdsai_track_*"))
        if hits:
            runs[trk] = hits[-1]
        else:
            print(f"!! missing gtoracle_{trk}")
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle-root", default="/data/ESA_DLSTEM_2025/experiments/MOT_birdsai_gtoracle")
    ap.add_argument("--annotations", default="annotations_sam3")
    ap.add_argument("--workspace", default="/tmp/birdsai_hota_size_ws")
    ap.add_argument("--output", default="docs/use_case_results/birdsai_gt_oracle_assa_vs_size.csv")
    args = ap.parse_args()

    import shutil
    ws = Path(args.workspace)
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)

    runs = _collect(Path(args.oracle_root))
    methods = [t for t in TRACKERS if t in runs]
    videos, track_sizes = _load_gt(args.annotations)
    print(f"videos={len(videos)} methods={methods}\n")

    all_rows = []
    # unbinned "all" row
    for r in _eval_bin("birdsai", methods, runs, videos, track_sizes,
                       [0.0, float("inf")], 0, ws):
        r["size_bin"] = "all"; r["bin_idx"] = -1
        all_rows.append(r)
    for bi in range(len(EDGES) - 1):
        all_rows.extend(_eval_bin("birdsai", methods, runs, videos, track_sizes,
                                  EDGES, bi, ws))

    cols = ["dataset", "method", "size_bin", "bin_idx", "n_gt_tracks",
            "n_gt_boxes", "HOTA", "DetA", "AssA", "IDF1", "IDsw", "MOTA"]
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"\nwrote {out} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
