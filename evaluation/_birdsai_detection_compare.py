#!/usr/bin/env python
"""Offline DETECTION-task comparison on BIRDSAI annotations_sam3.

Re-scores the cached per-frame `detections` of the 3 trained detectors
(FasterRCNN / YOLO11 / DINOv3) and SAM3-train-exemplar against the SAM3-refined
GT (annotations_sam3), holding the frame set identical (all 4 cover the same
15,494 test frames). No GPU / no re-inference — pure offline box matching.

Detection metric (per fine class, IoU 0.5):
  * P / R / F1 at a single operating point (score >= thr; detectors thr=0.5,
    SAM3-xexemplar boxes carry score 1.0 so thr is inert).
  * mAP@0.5 (VOC all-point, score-ranked) for the 3 detectors only — SAM3-x has
    no score ranking (all 1.0) so AP degenerates to the operating point and is
    omitted.

Writes a markdown table to docs/birdsai_detection_sam3gt_compare.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from datasets.birdsai_mot import BIRDSAIMOTDataset

BIRDSAI_ROOT = "/data/ESA_DLSTEM_2025/data/wild_animal/BIRDSAI"
ANN = "annotations_sam3"
IOU_THR = 0.5
CANON = {0: "human", 1: "elephant", 2: "giraffe", 3: "lion", 4: "unknown"}
CLASSES = sorted(CANON)

METHODS = [
    ("FasterRCNN", "/work/ziwen/experiments/fasterrcnn_birdsai_dettrack_20260617_185512/predictions.json", 0.5, True),
    ("YOLO11l",    "/work/ziwen/experiments/yolo_birdsai_dettrack_20260617_214738/predictions.json",        0.5, True),
    ("DINOv3",     "/work/ziwen/experiments/dinov3_birdsai_dettrack_20260617_170549/predictions.json",      0.5, True),
    ("SAM3-xexemplar", "/work/ziwen/experiments/sam3_birdsai_xexemplar_full_20260622_142509/predictions.json", 0.0, False),
]
OUT_MD = Path("docs/use_case_results/birdsai_detection_sam3gt_compare.md")


# ---------------- IoU + greedy match (mirror eval_birdsai_sam3_xexemplar) -----
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
    """Highest-IoU-first greedy match (count metric, score-agnostic)."""
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


def match_score_order(g, p, scores, thr):
    """Score-descending greedy match for AP. Returns tp flag per det (in input
    order) and #matched GT. Each GT used once; det matched to highest-IoU
    unmatched GT if IoU>=thr."""
    n = len(p)
    tp = np.zeros(n, bool)
    if n == 0:
        return tp
    iou = iou_matrix(p, g)  # [n_det, n_gt]
    used = set()
    for di in np.argsort(-scores):
        if len(g) == 0:
            break
        gi = np.argsort(-iou[di])
        for gj in gi:
            if iou[di, gj] < thr:
                break
            if gj in used:
                continue
            used.add(gj); tp[di] = True
            break
    return tp


def voc_ap(scores, tp, npos):
    """All-point (VOC2010+) AP from per-det scores+tp flags and total #GT."""
    if npos == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores)
    tp = tp[order].astype(np.float64)
    fp = 1.0 - tp
    ctp = np.cumsum(tp); cfp = np.cumsum(fp)
    rec = ctp / npos
    prec = ctp / np.clip(ctp + cfp, 1e-9, None)
    mrec = np.concatenate([[0.0], rec, [rec[-1]]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


# ---------------- GT ----------------------------------------------------------
def build_gt(ds):
    gt = {}  # video_id -> fid -> (boxes[n,4], labels[n])
    for v in ds.videos:
        fr = {}
        for fid in v.frame_ids:
            a = ds._load_annotations(v, fid)
            fr[int(fid)] = (a["boxes"].reshape(-1, 4).astype(np.float32),
                            a["labels"].reshape(-1).astype(np.int64))
        gt[v.video_id] = fr
    return gt


def score_method(pred_path, thr, do_map, gt):
    d = json.load(open(pred_path))
    vids = d["videos"]
    # operating-point counters
    cnt = {c: {"tp": 0, "fp": 0, "fn": 0, "ngt": 0} for c in CLASSES}
    # AP accumulators
    ap_sc = {c: [] for c in CLASSES}
    ap_tp = {c: [] for c in CLASSES}
    ap_npos = {c: 0 for c in CLASSES}

    for vid, frames in gt.items():
        pv = vids.get(vid, {}).get("frames", {})
        for fid, (gb, gl) in frames.items():
            det = pv.get(str(fid), {}).get("detections",
                                           {"boxes": [], "scores": [], "labels": []})
            pb = np.asarray(det["boxes"], np.float32).reshape(-1, 4)
            ps = np.asarray(det["scores"], np.float32).reshape(-1)
            pl = np.asarray(det["labels"], np.int64).reshape(-1)
            for c in CLASSES:
                gm = gl == c
                gbc = gb[gm]
                ngt = len(gbc)
                cnt[c]["ngt"] += ngt
                ap_npos[c] += ngt
                # --- operating point (apply thr) ---
                pm = (pl == c) & (ps >= thr)
                pbc = pb[pm]
                ms = greedy_match(gbc, pbc, IOU_THR)
                tp = len(ms)
                cnt[c]["tp"] += tp
                cnt[c]["fp"] += len(pbc) - tp
                cnt[c]["fn"] += ngt - tp
                # --- AP (all dets of this class, score-ranked) ---
                if do_map:
                    am = pl == c
                    pbc_a = pb[am]; psc_a = ps[am]
                    tpf = match_score_order(gbc, pbc_a, psc_a, IOU_THR)
                    ap_sc[c].extend(psc_a.tolist())
                    ap_tp[c].extend(tpf.tolist())

    rows = {}
    for c in CLASSES:
        k = cnt[c]
        prec = k["tp"] / max(k["tp"] + k["fp"], 1)
        rec = k["tp"] / max(k["tp"] + k["fn"], 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        ap = (voc_ap(np.asarray(ap_sc[c]), np.asarray(ap_tp[c], bool), ap_npos[c])
              if do_map else float("nan"))
        rows[c] = {"P": prec, "R": rec, "F1": f1, "AP": ap, "ngt": k["ngt"]}
    # overall (micro for P/R/F1, macro-mean for mAP over classes w/ GT)
    tot = {kk: sum(cnt[c][kk] for c in CLASSES) for kk in ("tp", "fp", "fn", "ngt")}
    P = tot["tp"] / max(tot["tp"] + tot["fp"], 1)
    R = tot["tp"] / max(tot["tp"] + tot["fn"], 1)
    F1 = 2 * P * R / max(P + R, 1e-9)
    if do_map:
        aps = [rows[c]["AP"] for c in CLASSES if rows[c]["ngt"] > 0]
        mAP = float(np.mean(aps)) if aps else float("nan")
    else:
        mAP = float("nan")
    rows["__overall__"] = {"P": P, "R": R, "F1": F1, "AP": mAP, "ngt": tot["ngt"]}
    return rows


def fmt(x):
    return "  —  " if (x != x) else f"{x:.3f}"


def main():
    ds = BIRDSAIMOTDataset(root=BIRDSAI_ROOT, split="test", granularity="fine",
                           annotations_dirname=ANN, class_map={v: k for k, v in CANON.items()})
    gt = build_gt(ds)
    print(f"GT loaded: {len(gt)} videos, {sum(len(f) for f in gt.values())} frames\n")

    results = {}
    for name, path, thr, do_map in METHODS:
        results[name] = score_method(path, thr, do_map, gt)
        o = results[name]["__overall__"]
        print(f"{name:16s} P={o['P']:.3f} R={o['R']:.3f} F1={o['F1']:.3f} mAP={fmt(o['AP'])}")

    # ---- markdown ----
    names = [m[0] for m in METHODS]
    L = []
    L.append("# BIRDSAI Detection Comparison on SAM3-refined GT (`annotations_sam3`)\n")
    L.append(f"Frame set: identical 15,494 test frames · IoU = {IOU_THR} · fine 5-class.")
    L.append("Detector operating point = score ≥ 0.5; SAM3-xexemplar boxes carry score 1.0.")
    L.append("mAP@0.5 = VOC all-point, score-ranked (only meaningful for the score-producing detectors).\n")

    # per-class F1 table
    L.append("## Per-class F1 (IoU 0.5, single operating point)\n")
    L.append("| Class | nGT | " + " | ".join(names) + " |")
    L.append("|---|---:|" + "|".join(["---:"] * len(names)) + "|")
    for c in CLASSES:
        ng = results[names[0]][c]["ngt"]
        cells = [fmt(results[n][c]["F1"]) for n in names]
        L.append(f"| {CANON[c]} | {ng} | " + " | ".join(cells) + " |")
    ovng = results[names[0]]["__overall__"]["ngt"]
    L.append(f"| **OVERALL** | {ovng} | " +
             " | ".join(f"**{fmt(results[n]['__overall__']['F1'])}**" for n in names) + " |")

    # overall P/R/F1/mAP table
    L.append("\n## Overall detection metrics\n")
    L.append("| Method | Precision | Recall | F1 | mAP@0.5 |")
    L.append("|---|---:|---:|---:|---:|")
    for n in names:
        o = results[n]["__overall__"]
        L.append(f"| {n} | {fmt(o['P'])} | {fmt(o['R'])} | {fmt(o['F1'])} | {fmt(o['AP'])} |")

    # per-class mAP for detectors
    L.append("\n## Per-class mAP@0.5 (detectors only)\n")
    det_names = [m[0] for m in METHODS if m[3]]
    L.append("| Class | nGT | " + " | ".join(det_names) + " |")
    L.append("|---|---:|" + "|".join(["---:"] * len(det_names)) + "|")
    for c in CLASSES:
        ng = results[det_names[0]][c]["ngt"]
        cells = [fmt(results[n][c]["AP"]) for n in det_names]
        L.append(f"| {CANON[c]} | {ng} | " + " | ".join(cells) + " |")
    L.append("| **mean** | — | " +
             " | ".join(f"**{fmt(results[n]['__overall__']['AP'])}**" for n in det_names) + " |")

    # per-class P/R appendix
    L.append("\n## Appendix — per-class Precision / Recall (IoU 0.5)\n")
    for n in names:
        L.append(f"\n### {n}\n")
        L.append("| Class | P | R | F1 |")
        L.append("|---|---:|---:|---:|")
        for c in CLASSES:
            r = results[n][c]
            L.append(f"| {CANON[c]} | {fmt(r['P'])} | {fmt(r['R'])} | {fmt(r['F1'])} |")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L) + "\n")
    print(f"\nwrote {OUT_MD}")

    # also dump json
    jpath = OUT_MD.with_suffix(".json")
    jpath.write_text(json.dumps(
        {n: {("overall" if c == "__overall__" else CANON.get(c, c)): v
             for c, v in rows.items()} for n, rows in results.items()},
        indent=2))
    print(f"wrote {jpath}")


if __name__ == "__main__":
    main()
