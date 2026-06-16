"""Size-stratified HOTA for the GT-box association ORACLE (Experiment 2).

Every oracle run (``{method}_oracle_{dataset}_{TS}/mot_format/``) was produced
by feeding the SAME GT boxes to a method's association. Because detection is
perfect and identical, per-size differences in the resulting tracks are PURE
association ability. This script bins GT tracks by object size and runs
TrackEval per bin, so AssA / IDF1 / IDsw are reported **as a function of object
pixel size** — exposing the small-object association cliff (and, for the JDT
ReID methods, the stride-4 embedding floor).

Mechanics per (dataset, method, size-bin):
  1. Each GT TRACK is assigned to one bin by its representative size
     (median sqrt_area over its boxes); whole tracks stay together.
  2. GT for the bin = only the in-bin tracks' boxes.
  3. Preds for the bin = only pred boxes that IoU-match (>=0.5) an in-bin GT box
     that frame. Detection is oracle (pred boxes are GT-derived) so this keeps
     DetA ~ 1 and strips other-bin objects without spurious FPs.
  4. TrackEval HOTA/CLEAR/Identity on the filtered GT+preds.

Output CSV rows: dataset, method, size_bin, n_gt_tracks, n_gt_boxes,
HOTA, DetA, AssA, IDF1, IDsw, plus an "all" (unbinned) row per (dataset,method).

Usage::

    python compute_hota_by_size.py \
        --oracle-root /data/.../experiments/MOT/exp2_oracle_20260608 \
        --workspace /tmp/hota_size_ws \
        --output /data/.../experiments/MOT/exp2_oracle_20260608/assa_vs_size.csv
"""
from __future__ import annotations

# --- repo root on path so top-level modules (transforms, obb_utils, ...) import ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from datasets.airmot import AIRMOTDataset
from datasets.rscardata import RsCarDataset
from datasets.satmtb import SATMTBDataset
from datasets.sdmcar import SDMCarDataset
from datasets.viso import VISODataset


# Per-dataset class maps — surface ALL GT classes (mirror eval_*_oracle).
_CLASS_MAPS = {
    "rscardata":   {"car": 0},
    "satmtb":      {"airplane": 0, "car": 1, "ship": 2, "train": 3},
    "sdmcar":      {"car": 0},
    "airmot":      {"airplane": 0, "ship": 1},
    "viso_no_car": {"plane": 0, "ship": 1, "train": 2},
}
_DATASET_TABLE = {
    "rscardata":   (RsCarDataset, "/data/ESA_DLSTEM_2025/data/trafic/RsCarData", {}),
    "satmtb":      (SATMTBDataset, "/data/ESA_DLSTEM_2025/data/trafic/SAT-MTB", {"task": "mot"}),
    "sdmcar":      (SDMCarDataset, "/data/ESA_DLSTEM_2025/data/trafic/SDM-Car", {}),
    "airmot":      (AIRMOTDataset, "/data/ESA_DLSTEM_2025/data/trafic/AIR-MOT-100", {}),
    "viso_no_car": (VISODataset, "/data/ESA_DLSTEM_2025/data/trafic/VISO",
                    {"categories": ["plane", "ship", "train"]}),
}

# sqrt_area (px) bin edges — resolve the small-object end finely (cars ~5px,
# ships ~10-20px) and span to aircraft/train (>=40px). Override with --bins.
_DEFAULT_EDGES = [0.0, 5.0, 8.0, 12.0, 20.0, 40.0, float("inf")]

_RUN_RE = re.compile(
    r"^(?P<method>[a-zA-Z][a-zA-Z0-9_]*)_oracle_"
    r"(?P<dataset>rscardata|satmtb|sdmcar|viso_no_car|airmot)_"
    r"(?P<ts>\d{8}_\d{6})$"
)


def _safe_video_id(video_id: str) -> str:
    return video_id.replace("/", "_")


def _bin_label(lo: float, hi: float) -> str:
    if hi == float("inf"):
        return f">={lo:g}"
    if lo == 0.0:
        return f"<{hi:g}"
    return f"{lo:g}-{hi:g}"


def _build_dataset(name: str):
    cls, root, extra = _DATASET_TABLE[name]
    cmap = _CLASS_MAPS[name]
    if cls in (AIRMOTDataset, VISODataset):
        return cls(root=root, split="test", class_map=cmap, **extra)
    return cls(root=root, split="test", mode="detection", class_map=cmap, **extra)


def _iou_vec(box, boxes):
    """IoU of one xyxy box against an [N,4] array."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(a + b - inter, 1e-9)


def _collect_runs(root: Path) -> dict[str, dict[str, Path]]:
    """{dataset: {method: most_recent_run_dir}} for *_oracle_* dirs."""
    by_pair: dict[tuple[str, str], Path] = {}
    for d in root.iterdir():
        if not d.is_dir():
            continue
        m = _RUN_RE.match(d.name)
        if not m:
            continue
        key = (m["dataset"], m["method"])
        prev = by_pair.get(key)
        if prev is None or d.name > prev.name:
            by_pair[key] = d
    grouped: dict[str, dict[str, Path]] = defaultdict(dict)
    for (ds, meth), path in by_pair.items():
        grouped[ds][meth] = path
    return grouped


def _load_gt(dataset_name: str):
    """Return per-video GT plus per-track size bin.

    videos: [(seq, frame_ids, offset, {fid: (boxes[N,4], tids[N])})]
    track_bin: {seq: {tid: bin_index}}  (-1 → unbinned/sentinel, excluded)
    """
    ds = _build_dataset(dataset_name)
    videos = []
    track_sizes: dict[str, dict[int, list]] = {}
    for v in ds.videos:
        seq = _safe_video_id(v.video_id)
        first_fid = min(v.frame_ids)
        offset = 1 - int(first_fid) if first_fid < 1 else 0
        per_frame = {}
        sizes: dict[int, list] = defaultdict(list)
        for fid in v.frame_ids:
            ann = ds._load_annotations(v, fid)
            boxes = np.asarray(ann["boxes"], dtype=np.float32).reshape(-1, 4)
            tids = np.asarray(ann["track_ids"], dtype=np.int64).reshape(-1)
            per_frame[fid] = (boxes, tids)
            for j in range(len(boxes)):
                if int(tids[j]) < 0:
                    continue
                w = boxes[j, 2] - boxes[j, 0]; h = boxes[j, 3] - boxes[j, 1]
                sizes[int(tids[j])].append(float(np.sqrt(max(w * h, 0.0))))
        videos.append((seq, list(v.frame_ids), offset, per_frame))
        track_sizes[seq] = sizes
    return videos, track_sizes


def _trackeval_one(workspace: Path, benchmark: str, method: str,
                   seqmap_file: Path):
    import trackeval
    eval_cfg = trackeval.Evaluator.get_default_eval_config()
    eval_cfg.update({"USE_PARALLEL": False, "PRINT_RESULTS": False,
                     "PRINT_CONFIG": False, "TIME_PROGRESS": False,
                     "BREAK_ON_ERROR": False, "RETURN_ON_ERROR": True})
    ds_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    ds_cfg.update({
        "GT_FOLDER": str(workspace / "gt"),
        "TRACKERS_FOLDER": str(workspace / "trackers"),
        "OUTPUT_FOLDER": str(workspace / "output"),
        "TRACKERS_TO_EVAL": [method], "CLASSES_TO_EVAL": ["pedestrian"],
        "BENCHMARK": benchmark, "SPLIT_TO_EVAL": "test",
        "PRINT_CONFIG": False, "DO_PREPROC": False,
        "SEQMAP_FOLDER": str(workspace / "seqmaps"),
        "SEQMAP_FILE": str(seqmap_file), "SKIP_SPLIT_FOL": False,
    })
    metrics = [trackeval.metrics.HOTA({"PRINT_CONFIG": False}),
               trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
               trackeval.metrics.Identity({"PRINT_CONFIG": False})]
    evaluator = trackeval.Evaluator(eval_cfg)
    try:
        raw, _ = evaluator.evaluate([trackeval.datasets.MotChallenge2DBox(ds_cfg)], metrics)
    except Exception as exc:
        print(f"    TrackEval raised {exc!r}")
        return None
    bench_res = raw.get("MotChallenge2DBox", {}).get(method)
    if not bench_res or "COMBINED_SEQ" not in bench_res:
        return None
    cs = bench_res["COMBINED_SEQ"].get("pedestrian")
    if cs is None:
        return None
    hota, clear, ident = cs["HOTA"], cs["CLEAR"], cs["Identity"]
    return {
        "HOTA": float(np.mean(hota["HOTA"])), "DetA": float(np.mean(hota["DetA"])),
        "AssA": float(np.mean(hota["AssA"])), "IDF1": float(ident["IDF1"]),
        "IDsw": int(clear["IDSW"]), "MOTA": float(clear["MOTA"]),
    }


def _eval_bin(dataset_name, methods, runs, videos, track_sizes,
              edges, bin_idx, workspace):
    """Write size-filtered GT + preds for one bin, eval each method."""
    lo, hi = edges[bin_idx], edges[bin_idx + 1]
    benchmark = f"{dataset_name}_b{bin_idx}"
    gt_root = workspace / "gt" / f"{benchmark}-test"
    tr_root = workspace / "trackers" / f"{benchmark}-test"
    for p in (gt_root, tr_root):
        if p.exists():
            shutil.rmtree(p)
    gt_root.mkdir(parents=True); tr_root.mkdir(parents=True)

    # in-bin track ids per seq + in-bin GT boxes per (seq, fid)
    inbin_tids: dict[str, set] = {}
    inbin_gt_boxes: dict[str, dict] = {}     # seq -> fid -> [K,4]
    n_gt_tracks = n_gt_boxes = 0
    seq_names = []
    for seq, frame_ids, offset, per_frame in videos:
        sizes = track_sizes[seq]
        tids_in = set()
        for tid, slist in sizes.items():
            med = float(np.median(slist))
            if lo <= med < hi:
                tids_in.add(tid)
        inbin_tids[seq] = tids_in
        n_gt_tracks += len(tids_in)

        (gt_root / seq / "gt").mkdir(parents=True, exist_ok=True)
        lines = []
        box_by_fid = {}
        for fid in frame_ids:
            boxes, tids = per_frame[fid]
            keep_boxes = []
            seen = set()
            for j in range(len(boxes)):
                tid = int(tids[j])
                if tid not in tids_in or tid in seen:
                    continue
                seen.add(tid)
                x1, y1, x2, y2 = boxes[j]
                w, h = float(x2 - x1), float(y2 - y1)
                lines.append(f"{int(fid)+offset},{tid},{float(x1):.2f},"
                             f"{float(y1):.2f},{w:.2f},{h:.2f},1,1,1.0")
                keep_boxes.append([float(x1), float(y1), float(x2), float(y2)])
                n_gt_boxes += 1
            box_by_fid[fid] = np.asarray(keep_boxes, dtype=np.float32).reshape(-1, 4)
        inbin_gt_boxes[seq] = box_by_fid
        (gt_root / seq / "gt" / "gt.txt").write_text("\n".join(lines))
        (gt_root / seq / "seqinfo.ini").write_text(
            f"[Sequence]\nname={seq}\nseqLength={len(frame_ids)}\n"
            "imWidth=1024\nimHeight=1024\nimExt=.jpg\n")
        seq_names.append(seq)

    seqmap_dir = workspace / "seqmaps"; seqmap_dir.mkdir(parents=True, exist_ok=True)
    seqmap_file = seqmap_dir / f"{benchmark}-test.txt"
    seqmap_file.write_text("name\n" + "\n".join(seq_names) + "\n")

    # filter each method's preds to boxes matching an in-bin GT box that frame
    offset_by_seq = {seq: off for seq, _, off, _ in videos}
    rows = []
    for method in methods:
        run_dir = runs[method]
        src = run_dir / "mot_format"
        dst = tr_root / method / "data"
        dst.mkdir(parents=True, exist_ok=True)
        for seq in seq_names:
            off = offset_by_seq[seq]
            gt_boxes_fid = inbin_gt_boxes[seq]
            f = src / f"{seq}.txt"
            out_lines = []
            if f.is_file():
                for ln in f.read_text().splitlines():
                    if not ln:
                        continue
                    p = ln.split(",")
                    fid = int(p[0])
                    pb = np.array([float(p[2]), float(p[3]),
                                   float(p[2]) + float(p[4]),
                                   float(p[3]) + float(p[5])], dtype=np.float32)
                    gtb = gt_boxes_fid.get(fid, np.zeros((0, 4), dtype=np.float32))
                    if len(gtb) and _iou_vec(pb, gtb).max() >= 0.5:
                        p[0] = str(fid + off)
                        out_lines.append(",".join(p))
            (dst / f"{seq}.txt").write_text("\n".join(out_lines))

        m = _trackeval_one(workspace, benchmark, method, seqmap_file)
        row = {"dataset": dataset_name, "method": method,
               "size_bin": _bin_label(lo, hi), "bin_idx": bin_idx,
               "n_gt_tracks": n_gt_tracks, "n_gt_boxes": n_gt_boxes}
        if m:
            row.update({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in m.items()})
        rows.append(row)
        if m:
            print(f"    {method:16s} {_bin_label(lo,hi):>7s}  "
                  f"AssA={m['AssA']*100:5.2f} IDF1={m['IDF1']*100:5.2f} "
                  f"IDsw={m['IDsw']:6d} DetA={m['DetA']*100:5.2f} "
                  f"(tracks={n_gt_tracks})")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle-root", required=True)
    ap.add_argument("--workspace", default="/tmp/hota_size_ws")
    ap.add_argument("--output", required=True)
    ap.add_argument("--bins", default=None,
                    help="comma list of sqrt_area edges, e.g. 0,5,8,12,20,40,inf")
    args = ap.parse_args()

    edges = _DEFAULT_EDGES
    if args.bins:
        edges = [float("inf") if x.strip() in ("inf", "Inf") else float(x)
                 for x in args.bins.split(",")]

    root = Path(args.oracle_root)
    workspace = Path(args.workspace)
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)

    runs = _collect_runs(root)
    if not runs:
        print(f"no *_oracle_* runs under {root}", file=sys.stderr); sys.exit(1)
    print("Discovered oracle runs:")
    for ds in sorted(runs):
        print(f"  {ds}: {sorted(runs[ds])}")

    all_rows = []
    for ds in sorted(runs):
        print(f"\n=== {ds} ===")
        methods = sorted(runs[ds])
        videos, track_sizes = _load_gt(ds)
        # "all" (unbinned) row uses a single [0, inf) bin
        for rows in [_eval_bin(ds, methods, runs[ds], videos, track_sizes,
                               [0.0, float("inf")], 0, workspace)]:
            for r in rows:
                r["size_bin"] = "all"; r["bin_idx"] = -1
            all_rows.extend(rows)
        for bi in range(len(edges) - 1):
            all_rows.extend(_eval_bin(ds, methods, runs[ds], videos, track_sizes,
                                      edges, bi, workspace))

    if all_rows:
        cols = ["dataset", "method", "size_bin", "bin_idx", "n_gt_tracks",
                "n_gt_boxes", "HOTA", "DetA", "AssA", "IDF1", "IDsw", "MOTA"]
        out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in all_rows:
                w.writerow(r)
        print(f"\nwrote {out}  ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
