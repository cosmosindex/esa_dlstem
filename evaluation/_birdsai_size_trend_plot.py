#!/usr/bin/env python
"""Detection performance vs object size — FasterRCNN / YOLO11l / DINOv3 on BIRDSAI.

Size-stratified (sqrt-area px) per-bin Precision / Recall / F1 at IoU 0.5 on the
SAM3-refined GT (annotations_sam3). One panel per detector; coloured lines =
metrics. Shows the sharp collapse on small thermal objects.

Pure offline — reuses cached predictions.json, no GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets.birdsai_mot import BIRDSAIMOTDataset
from tools.plot_style import apply_neurips_style

BIRDSAI_ROOT = "/data/ESA_DLSTEM_2025/data/wild_animal/BIRDSAI"
ANN = "annotations_sam3"
IOU_THR = 0.5
SCORE_THR = 0.5
CANON = {0: "human", 1: "elephant", 2: "giraffe", 3: "lion", 4: "unknown"}
CLASSES = sorted(CANON)

# sqrt-area (px) bin edges — data-driven (test GT spans ~5..94px, median ~36)
EDGES = [0, 14, 20, 28, 38, 50, 95]
BIN_LABELS = ["<14", "14–20", "20–28", "28–38", "38–50", ">50"]
BIN_CENTERS = [10, 17, 24, 33, 44, 60]   # x positions (px), last is open-ended

DETECTORS = [
    ("FasterRCNN", "/work/ziwen/experiments/fasterrcnn_birdsai_dettrack_20260617_185512/predictions.json"),
    ("YOLO11l",    "/work/ziwen/experiments/yolo_birdsai_dettrack_20260617_214738/predictions.json"),
    ("DINOv3",     "/work/ziwen/experiments/dinov3_birdsai_dettrack_20260617_170549/predictions.json"),
]
DISPLAY = {"FasterRCNN": "Faster R-CNN", "YOLO11l": "YOLO11l", "DINOv3": "DINOv3+FCOS"}
METRIC_COLORS = {"Precision": "#1f77b4", "Recall": "#d62728", "F1": "#2ca02c"}
OUT_PNG = Path("docs/use_case_results/figures/birdsai_size_trend.png")
OUT_PDF = Path("docs/use_case_results/figures/birdsai_size_trend.pdf")


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


def sqrt_area(b):
    return np.sqrt(np.clip((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), 0, None))


def size_bin(s):
    return np.clip(np.digitize(s, EDGES) - 1, 0, len(BIN_LABELS) - 1)


def build_gt(ds):
    gt = {}
    for v in ds.videos:
        fr = {}
        for fid in v.frame_ids:
            a = ds._load_annotations(v, fid)
            fr[int(fid)] = (a["boxes"].reshape(-1, 4).astype(np.float32),
                            a["labels"].reshape(-1).astype(np.int64))
        gt[v.video_id] = fr
    return gt


def score(pred_path, gt):
    nb = len(BIN_LABELS)
    # recall: GT binned by GT size; precision: pred binned by pred size
    gt_tot = np.zeros(nb); gt_tp = np.zeros(nb)
    pr_tot = np.zeros(nb); pr_tp = np.zeros(nb)
    vids = json.load(open(pred_path))["videos"]
    for vid, frames in gt.items():
        pv = vids.get(vid, {}).get("frames", {})
        for fid, (gb, gl) in frames.items():
            det = pv.get(str(fid), {}).get("detections",
                                           {"boxes": [], "scores": [], "labels": []})
            pb = np.asarray(det["boxes"], np.float32).reshape(-1, 4)
            ps = np.asarray(det["scores"], np.float32).reshape(-1)
            pl = np.asarray(det["labels"], np.int64).reshape(-1)
            keep = ps >= SCORE_THR
            pb, pl = pb[keep], pl[keep]
            for c in CLASSES:
                gm = gl == c; pm = pl == c
                gbc = gb[gm]; pbc = pb[pm]
                gbin = size_bin(sqrt_area(gbc)) if len(gbc) else np.zeros(0, int)
                pbin = size_bin(sqrt_area(pbc)) if len(pbc) else np.zeros(0, int)
                for b in gbin:
                    gt_tot[b] += 1
                for b in pbin:
                    pr_tot[b] += 1
                for r, cc in greedy_match(gbc, pbc, IOU_THR):
                    gt_tp[gbin[r]] += 1     # TP credited to GT size bin (recall)
                    pr_tp[pbin[cc]] += 1    # TP credited to pred size bin (precision)
    rec = gt_tp / np.clip(gt_tot, 1, None)
    prec = pr_tp / np.clip(pr_tot, 1, None)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    return {"Precision": prec, "Recall": rec, "F1": f1, "n_gt": gt_tot}


def main():
    ds = BIRDSAIMOTDataset(root=BIRDSAI_ROOT, split="test", granularity="fine",
                           annotations_dirname=ANN,
                           class_map={v: k for k, v in CANON.items()})
    gt = build_gt(ds)
    res = {name: score(path, gt) for name, path in DETECTORS}

    apply_neurips_style(base_size=10)
    x = np.array(BIN_CENTERS, float)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=True)
    for ax, (name, _) in zip(axes, DETECTORS):
        r = res[name]
        for metric, color in METRIC_COLORS.items():
            ax.plot(x, r[metric], "-o", color=color, lw=1.8, ms=5,
                    markerfacecolor="white", markeredgewidth=1.4, label=metric)
        ax.set_title(DISPLAY.get(name, name))
        ax.set_xlabel(r"object size  $\sqrt{\mathrm{area}}$  (px)")
        ax.set_xticks(x)
        ax.set_xticklabels(BIN_LABELS, rotation=0)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, axis="y", alpha=0.25, lw=0.4)
        ax.axvspan(x[0] - 4, EDGES[1], color="0.85", alpha=0.35, zorder=0)  # tiny region
    axes[0].set_ylabel("score @ IoU 0.5")
    axes[0].legend(loc="upper left", frameon=False, handlelength=1.6)
    fig.suptitle("BIRDSAI detection vs object size — small objects collapse", y=1.02)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG)
    fig.savefig(OUT_PDF)
    print(f"wrote {OUT_PNG}\nwrote {OUT_PDF}")

    # console dump
    for name, _ in DETECTORS:
        r = res[name]
        print(f"\n{name}  (n_gt per bin: {r['n_gt'].astype(int).tolist()})")
        for i, lab in enumerate(BIN_LABELS):
            print(f"  {lab:>6}  P={r['Precision'][i]:.3f}  R={r['Recall'][i]:.3f}  F1={r['F1'][i]:.3f}")


if __name__ == "__main__":
    main()
