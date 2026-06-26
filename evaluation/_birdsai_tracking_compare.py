#!/usr/bin/env python
"""Offline TRACKING-task comparison on BIRDSAI annotations_sam3.

Re-scores the cached MOT track outputs (mot_format/*.txt) of the tracker sweep
(6 TBD trackers x 3 detectors = 18 runs) against the SAM3-refined GT
(annotations_sam3) — the SAME GT as the detection table — so the tracking and
detection numbers are directly comparable.

The sweep ran per-class trackers and encoded the class in the track id as
  class = track_id // 1_000_000
(verified: elephant-GT videos -> id-prefix 1, giraffe -> 2, human -> 0, ...).

Pure offline — parses the .txt tracks, no GPU / no re-tracking.

CAVEAT: the sweep covers 11 of the 16 test videos (the 5 it skipped are not
re-tracked here); all 18 runs share this same 11-video set so they are mutually
comparable. Metrics mirror the project MOT eval: P/R/F1 (detection level),
MOTA, IDsw (association), at IoU 0.5. IDF1 in this pipeline == det-F1, so it is
omitted to avoid implying a true identity-F1.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from datasets.birdsai_mot import BIRDSAIMOTDataset

BIRDSAI_ROOT = "/data/ESA_DLSTEM_2025/data/wild_animal/BIRDSAI"
ANN = "annotations_sam3"
IOU_THR = 0.5
SWEEP_ROOT = Path("/data/ESA_DLSTEM_2025/experiments/MOT_birdsai_sweep")
CANON = {0: "human", 1: "elephant", 2: "giraffe", 3: "lion", 4: "unknown"}
CLASSES = sorted(CANON)
DETECTORS = ["fasterrcnn", "yolo", "dinov3"]
DET_LABEL = {"fasterrcnn": "FasterRCNN", "yolo": "YOLO11l", "dinov3": "DINOv3"}
TRACKERS = ["sort", "ocsort", "bytetrack", "botsort", "botsort_reid", "tracktrack"]
TRK_LABEL = {"sort": "SORT", "ocsort": "OC-SORT", "bytetrack": "ByteTrack",
             "botsort": "BoT-SORT", "botsort_reid": "BoT-SORT+ReID", "tracktrack": "TrackTrack"}
OUT_MD = Path("docs/use_case_results/birdsai_tracking_sam3gt_compare.md")


# ---------------- IoU + greedy match ------------------------------------------
def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    A = a[:, None, :]; B = b[None, :, :]
    x1 = np.maximum(A[..., 0], B[..., 0]); y1 = np.maximum(A[..., 1], B[..., 1])
    x2 = np.minimum(A[..., 2], B[..., 2]); y2 = np.minimum(A[..., 3], B[..., 3])
    iw = np.clip(x2 - x1, 0, None); ih = np.clip(y2 - y1, 0, None)
    inter = iw * ih
    ar = (A[..., 2] - A[..., 0]) * (A[..., 3] - A[..., 1])
    br = (B[..., 2] - B[..., 0]) * (B[..., 3] - B[..., 1])
    return inter / np.clip(ar + br - inter, 1e-9, None)


def greedy_match(g, p, thr):
    if len(g) == 0 or len(p) == 0:
        return []
    iou = iou_matrix(g, p); rs, cs = np.where(iou >= thr)
    order = iou[rs, cs].argsort()[::-1]
    mg, mp, out = set(), set(), []
    for k in order:
        r, c = rs[k], cs[k]
        if r in mg or c in mp:
            continue
        mg.add(r); mp.add(c); out.append((r, c))
    return out


# ---------------- parse mot_format -------------------------------------------
def parse_tracks(txt):
    """frame -> dict(boxes[n,4] xyxy, labels[n], track_ids[n])."""
    by_frame = {}
    if not txt.exists():
        return by_frame
    for line in open(txt):
        p = line.strip().split(",")
        if len(p) < 6:
            continue
        f = int(float(p[0])); tid = int(float(p[1]))
        x, y, w, h = (float(p[2]), float(p[3]), float(p[4]), float(p[5]))
        cls = tid // 1_000_000
        by_frame.setdefault(f, {"boxes": [], "labels": [], "track_ids": []})
        by_frame[f]["boxes"].append([x, y, x + w, y + h])
        by_frame[f]["labels"].append(cls)
        by_frame[f]["track_ids"].append(tid)
    return by_frame


def build_gt(ds, video_ids):
    gt = {}
    vmap = {v.video_id: v for v in ds.videos}
    for vid in video_ids:
        v = vmap[vid]
        fr = {}
        for fid in v.frame_ids:
            a = ds._load_annotations(v, fid)
            fr[int(fid)] = (a["boxes"].reshape(-1, 4).astype(np.float32),
                            a["labels"].reshape(-1).astype(np.int64),
                            a["track_ids"].reshape(-1).astype(np.int64))
        gt[vid] = fr
    return gt


def score_run(run_dir, gt):
    det = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CLASSES}
    trk = {c: {"idsw": 0, "ngt": 0} for c in CLASSES}
    for vid, frames in gt.items():
        tracks = parse_tracks(run_dir / "mot_format" / f"{vid}.txt")
        last = {c: {} for c in CLASSES}
        for fid in sorted(frames):
            gb, gl, gid = frames[fid]
            t = tracks.get(fid, {"boxes": [], "labels": [], "track_ids": []})
            pb = np.asarray(t["boxes"], np.float32).reshape(-1, 4)
            pl = np.asarray(t["labels"], np.int64).reshape(-1)
            pid = np.asarray(t["track_ids"], np.int64).reshape(-1)
            for c in CLASSES:
                gm = gl == c; pm = pl == c
                gbc = gb[gm]; gic = gid[gm]; cbc = pb[pm]; cic = pid[pm]
                ms = greedy_match(gbc, cbc, IOU_THR)
                tp = len(ms)
                det[c]["tp"] += tp
                det[c]["fp"] += len(cbc) - tp
                det[c]["fn"] += len(gbc) - tp
                trk[c]["ngt"] += len(gbc)
                for r, cc in ms:
                    g = int(gic[r]); pp = int(cic[cc])
                    prev = last[c].get(g)
                    if prev is not None and prev != pp:
                        trk[c]["idsw"] += 1
                    last[c][g] = pp
    return det, trk


def metrics(det, trk, classes):
    tp = sum(det[c]["tp"] for c in classes)
    fp = sum(det[c]["fp"] for c in classes)
    fn = sum(det[c]["fn"] for c in classes)
    idsw = sum(trk[c]["idsw"] for c in classes)
    ngt = sum(trk[c]["ngt"] for c in classes)
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    F1 = 2 * P * R / max(P + R, 1e-9)
    MOTA = 1.0 - (fp + fn + idsw) / max(ngt, 1)
    return {"P": P, "R": R, "F1": F1, "MOTA": MOTA, "IDsw": idsw, "nGT": ngt}


def find_run(det_name, trk_name):
    hits = sorted(SWEEP_ROOT.glob(f"{det_name}_{trk_name}_birdsai_track_*"))
    return hits[-1] if hits else None


def main():
    ds = BIRDSAIMOTDataset(root=BIRDSAI_ROOT, split="test", granularity="fine",
                           annotations_dirname=ANN,
                           class_map={v: k for k, v in CANON.items()})
    # videos actually covered by the sweep
    any_run = find_run("dinov3", "botsort")
    sweep_videos = sorted(p.stem for p in (any_run / "mot_format").glob("*.txt"))
    gt = build_gt(ds, sweep_videos)
    nfr = sum(len(f) for f in gt.values())
    print(f"sweep videos: {len(sweep_videos)}/16, frames={nfr}\n")

    res = {}      # (det,trk) -> overall metrics
    perclass = {} # (det,trk) -> {class: metrics}
    for d in DETECTORS:
        for t in TRACKERS:
            run = find_run(d, t)
            if run is None:
                print(f"!! missing {d}_{t}"); continue
            de, tr = score_run(run, gt)
            res[(d, t)] = metrics(de, tr, CLASSES)
            perclass[(d, t)] = {c: metrics({c: de[c]}, {c: tr[c]}, [c]) for c in CLASSES}
            m = res[(d, t)]
            print(f"{DET_LABEL[d]:11s} {TRK_LABEL[t]:14s} "
                  f"MOTA={m['MOTA']:+.3f} F1={m['F1']:.3f} IDsw={m['IDsw']:5d} "
                  f"P={m['P']:.3f} R={m['R']:.3f}")

    # ---------------- markdown ----------------
    L = []
    L.append("# BIRDSAI Tracking Comparison on SAM3-refined GT (`annotations_sam3`)\n")
    L.append(f"6 TBD trackers × 3 detectors (cached detections → online tracking), "
             f"re-scored on the **same GT as the detection table**.")
    L.append(f"Videos: {len(sweep_videos)}/16 (sweep subset, all 18 runs identical set) · "
             f"{nfr} frames · IoU {IOU_THR} · fine 5-class.")
    L.append("MOTA = 1−(FP+FN+IDsw)/GT · IDsw = identity switches · F1/P/R = detection level.\n")

    for metric, lo_better, fmt in [("MOTA", False, "{:+.3f}"),
                                   ("F1", False, "{:.3f}"),
                                   ("IDsw", True, "{:d}")]:
        L.append(f"## {metric} (rows = tracker, cols = detector)\n")
        L.append("| Tracker | " + " | ".join(DET_LABEL[d] for d in DETECTORS) + " |")
        L.append("|---|" + "|".join(["---:"] * len(DETECTORS)) + "|")
        for t in TRACKERS:
            cells = []
            vals = {d: res.get((d, t), {}).get(metric) for d in DETECTORS}
            best = None
            present = [v for v in vals.values() if v is not None]
            if present:
                best = min(present) if lo_better else max(present)
            for d in DETECTORS:
                v = vals[d]
                s = fmt.format(v) if v is not None else "—"
                if v is not None and v == best:
                    s = f"**{s}**"
                cells.append(s)
            L.append(f"| {TRK_LABEL[t]} | " + " | ".join(cells) + " |")
        L.append("")

    # per-class F1 for the best detector backbone (FasterRCNN), all trackers
    L.append("## Per-class F1 — FasterRCNN backbone (IoU 0.5)\n")
    L.append("| Tracker | " + " | ".join(CANON[c] for c in CLASSES) + " |")
    L.append("|---|" + "|".join(["---:"] * len(CLASSES)) + "|")
    for t in TRACKERS:
        pc = perclass.get(("fasterrcnn", t), {})
        cells = [f"{pc[c]['F1']:.3f}" if c in pc else "—" for c in CLASSES]
        L.append(f"| {TRK_LABEL[t]} | " + " | ".join(cells) + " |")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L) + "\n")
    print(f"\nwrote {OUT_MD}")

    jpath = OUT_MD.with_suffix(".json")
    jpath.write_text(json.dumps(
        {f"{d}+{t}": {"overall": res[(d, t)],
                      "per_class": {CANON[c]: perclass[(d, t)][c] for c in CLASSES}}
         for (d, t) in res}, indent=2))
    print(f"wrote {jpath}")


if __name__ == "__main__":
    main()
