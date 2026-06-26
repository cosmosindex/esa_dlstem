#!/usr/bin/env python
"""Fire detection performance vs object size — FasterRCNN / YOLO11l / DINOv3.

Mirrors evaluation/_birdsai_size_trend_plot.py but for the FireRGBT (RGBT-3M)
wildfire use case. Reads the per-frame dump produced by
``evaluation/eval_fire_detect_dump.py`` and, for each detector, computes
size-stratified Precision / Recall / F1 at IoU 0.5 (operating point score >= 0.5)
in the canonical 0-indexed class space {smoke, fire, person}.

Size = sqrt(area) in px, in the 640² eval space (the space mAP was reported in).
Bins are quantile-derived from the GT size distribution so each bin holds a
comparable number of boxes; tick labels show the px range. One panel per
detector, coloured lines = metrics. Shows whether performance collapses on the
small fire/person objects vs the large smoke.

Pure offline — no GPU.

Run:
    micromamba run -n esa_dlstem python tools/plot_fire_size_trend.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import apply_neurips_style  # noqa: E402

DUMP = Path("/work/ziwen/experiments/fire_detect_dump/fire_detect_predictions.json")
OUT_PNG = Path("docs/use_case_results/figures/fire_size_trend.png")
OUT_PDF = Path("docs/use_case_results/figures/fire_size_trend.pdf")
OUT_CSV = Path("docs/use_case_results/figures/fire_size_trend.csv")

IOU_THR = 0.5
SCORE_THR = 0.5
N_BINS = 6
CLASSES = [0, 1, 2]  # smoke / fire / person
DETECTORS = ["FasterRCNN", "YOLO11l", "DINOv3"]  # keys into the dump
DISPLAY = {"FasterRCNN": "Faster R-CNN", "YOLO11l": "YOLO11l", "DINOv3": "DINOv3+FCOS"}
METRIC_COLORS = {"Precision": "#1f77b4", "Recall": "#d62728", "F1": "#2ca02c"}


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
    if len(b) == 0:
        return np.zeros(0, np.float32)
    return np.sqrt(np.clip((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), 0, None))


def make_edges(gt):
    """Quantile bin edges from the GT sqrt-area distribution."""
    areas = np.concatenate([sqrt_area(np.asarray(f["boxes"], np.float32).reshape(-1, 4))
                            for f in gt])
    qs = np.linspace(0, 1, N_BINS + 1)
    edges = np.quantile(areas, qs)
    edges[0] = 0.0
    edges[-1] = max(edges[-1], areas.max()) + 1
    # de-duplicate (in case of ties) while keeping N_BINS+1 monotone edges
    edges = np.maximum.accumulate(edges)
    return edges, areas


def size_bin(s, edges):
    return np.clip(np.digitize(s, edges) - 1, 0, len(edges) - 2)


def score_model(preds, gt, edges):
    nb = len(edges) - 1
    gt_tot = np.zeros(nb); gt_tp = np.zeros(nb)
    pr_tot = np.zeros(nb); pr_tp = np.zeros(nb)
    for det, g in zip(preds, gt):
        gb = np.asarray(g["boxes"], np.float32).reshape(-1, 4)
        gl = np.asarray(g["labels"], np.int64).reshape(-1)
        pb = np.asarray(det["boxes"], np.float32).reshape(-1, 4)
        ps = np.asarray(det["scores"], np.float32).reshape(-1)
        pl = np.asarray(det["labels"], np.int64).reshape(-1)
        keep = ps >= SCORE_THR
        pb, pl = pb[keep], pl[keep]
        for c in CLASSES:
            gbc = gb[gl == c]; pbc = pb[pl == c]
            gbin = size_bin(sqrt_area(gbc), edges) if len(gbc) else np.zeros(0, int)
            pbin = size_bin(sqrt_area(pbc), edges) if len(pbc) else np.zeros(0, int)
            for b in gbin:
                gt_tot[b] += 1
            for b in pbin:
                pr_tot[b] += 1
            for r, cc in greedy_match(gbc, pbc, IOU_THR):
                gt_tp[gbin[r]] += 1
                pr_tp[pbin[cc]] += 1
    rec = gt_tp / np.clip(gt_tot, 1, None)
    prec = pr_tp / np.clip(pr_tot, 1, None)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    return {"Precision": prec, "Recall": rec, "F1": f1, "n_gt": gt_tot}


def main():
    data = json.load(open(DUMP))
    gt = data["gt"]
    edges, areas = make_edges(gt)
    bin_labels = [f"{int(round(edges[i]))}–{int(round(edges[i+1]))}"
                  for i in range(len(edges) - 1)]
    bin_labels[-1] = f">{int(round(edges[-2]))}"
    res = {name: score_model(data["models"][name], gt, edges) for name in DETECTORS}

    print(f"GT sqrt-area px (640² space): median={np.median(areas):.1f} "
          f"p5={np.percentile(areas,5):.1f} p95={np.percentile(areas,95):.1f}")
    print(f"bin edges: {np.round(edges,1).tolist()}")

    apply_neurips_style(base_size=10)
    x = np.arange(len(bin_labels))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=True)
    for ax, name in zip(axes, DETECTORS):
        r = res[name]
        for metric, color in METRIC_COLORS.items():
            ax.plot(x, r[metric], "-o", color=color, lw=1.8, ms=5,
                    markerfacecolor="white", markeredgewidth=1.4, label=metric)
        ax.set_title(DISPLAY[name])
        ax.set_xlabel(r"object size  $\sqrt{\mathrm{area}}$  (px)")
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, rotation=30, ha="right")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, axis="y", alpha=0.25, lw=0.4)
    axes[0].set_ylabel("score @ IoU 0.5")
    axes[0].legend(loc="lower right", frameon=False, handlelength=1.6)
    fig.suptitle("Fire detection vs object size (RGBT-3M) — small fire/person vs large smoke", y=1.02)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG)
    fig.savefig(OUT_PDF)
    print(f"wrote {OUT_PNG}\nwrote {OUT_PDF}")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["detector", "bin", "px_range", "n_gt", "Precision", "Recall", "F1"])
        for name in DETECTORS:
            r = res[name]
            for i, lab in enumerate(bin_labels):
                w.writerow([name, i, lab, int(r["n_gt"][i]),
                            f"{r['Precision'][i]:.4f}", f"{r['Recall'][i]:.4f}",
                            f"{r['F1'][i]:.4f}"])
    print(f"wrote {OUT_CSV}")

    for name in DETECTORS:
        r = res[name]
        print(f"\n{name}  (n_gt per bin: {r['n_gt'].astype(int).tolist()})")
        for i, lab in enumerate(bin_labels):
            print(f"  {lab:>10}  P={r['Precision'][i]:.3f}  R={r['Recall'][i]:.3f}  F1={r['F1'][i]:.3f}")


if __name__ == "__main__":
    main()
