"""
Per-epoch HiEUM union-VAL curve.

Runs every per-epoch checkpoint of the space-tracker car-union retrain over the
union VAL set (``test1024_mot.json``, 31 videos / 6,876 frames) and reports a
single comparable detection score per epoch — so we can SEE whether epoch 15 had
converged or whether continuing training is worthwhile. This is the validation
signal that the in-training val slot could not produce (HiEUM's built-in val
crashes on full-res frames, so it was disabled with --val_intervals 99999).

Protocol mirrors the project's HiEUM eval configs:
  - HiEUMDetector wrapper at hieum_image_size 1024x1024, thresh 3.0, max_dets 128,
    nms_iou 0.1, layers 3, seq_len 20 (non-overlapping clips = HiEUM's own inference).
  - centroid matching, dist <= 5 px in original-image pixels (HiEUM paper protocol).
  - score sweep -> best-F1 operating point per epoch (how HiEUM reports F1).

Results are written incrementally to OUT/val_curve_results.json so the run is
resumable and plottable at any time.

Usage:
    python evaluation/_hieum_union_val_curve.py            # all 15 epochs, full VAL
    python evaluation/_hieum_union_val_curve.py --smoke    # 1 epoch, 2 videos
    python evaluation/_hieum_union_val_curve.py --plot-only # just (re)draw from json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import os

import cv2
import numpy as np
import torch

from models import HiEUMDetector

# ----------------------------------------------------------------------------
DATA_ROOT = Path("/work/ziwen/data/hieum_car_union")
VAL_JSON = DATA_ROOT / "annotations" / "test1024_mot.json"
CKPT_DIR = Path(
    "/work/ziwen/experiments/hieum_car_union/rs_car_multi/sp_centerDet_minus/"
    "spacetracker_car_supMode_0_seglen20_weights2026_06_29_17_52_37"
)
OUT_DIR = Path("/work/ziwen/experiments/hieum_car_union/val_curve")
RESULTS_JSON = OUT_DIR / "val_curve_results.json"   # default (single-process)


def merged_results():
    """Merge every val_curve_results*.json shard in OUT_DIR (dual-GPU shards)."""
    merged = {}
    for fp in sorted(OUT_DIR.glob("val_curve_results*.json")):
        for ep, val in json.load(open(fp)).items():
            merged[ep] = val
    return merged

FIG_PNG = Path("docs/use_case_results/figures/hieum_union_val_curve.png")
FIG_PDF = Path("docs/use_case_results/figures/hieum_union_val_curve.pdf")

# HiEUM inference params (match configs/MOT/hieum_*.yaml)
IMAGE_SIZE = (1024, 1024)
SEQ_LEN = 20
LAYERS = 3
THRESH = 3.0
MAX_DETS = 128
NMS_IOU = 0.1
SCORE_FLOOR = 0.05            # wrapper floor; sweep tests cutoffs above this
CENTROID_DIST = 5.0          # px, original-image coords
SCORE_SWEEP = [round(x, 2) for x in np.arange(0.10, 0.55, 0.05)]
FRAME_WINDOW = 100           # feed videos to the detector in 100-frame windows
                             # (multiple of SEQ_LEN -> identical to whole-video)


def load_val():
    d = json.load(open(VAL_JSON))
    imgs = {im["id"]: im for im in d["images"]}
    # group images per video, sorted by frame_id
    vids = {}
    for im in d["images"]:
        vids.setdefault(im["video_id"], []).append(im)
    for v in vids:
        vids[v].sort(key=lambda im: im["frame_id"])
    # gt centroids per image_id
    gt = {im_id: [] for im_id in imgs}
    for a in d["annotations"]:
        x, y, w, h = a["bbox"]
        gt[a["image_id"]].append((x + w / 2.0, y + h / 2.0))
    names = {v: ims[0]["file_name"].split("/")[1] for v, ims in vids.items()}
    return vids, gt, names


def run_video(det, images):
    """Return list of per-frame pred lists [(cx, cy, score), ...] for one video."""
    preds = []
    n = len(images)
    for start in range(0, n, FRAME_WINDOW):
        window = images[start:start + FRAME_WINDOW]
        frames = []
        for im in window:
            bgr = cv2.imread(str(DATA_ROOT / im["file_name"]))
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        det.reset_state()
        det.init_video(frames)
        outs = det.propagate()
        for o in outs:
            boxes = o["boxes"].numpy()
            scores = o["scores"].numpy()
            if len(boxes) == 0:
                preds.append([])
                continue
            cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
            cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
            preds.append(list(zip(cx.tolist(), cy.tolist(), scores.tolist())))
    return preds


def match_frame(preds, gts, thr):
    """Greedy centroid match at score >= thr. Returns (tp, fp, n_pred, n_gt)."""
    p = sorted([q for q in preds if q[2] >= thr], key=lambda q: -q[2])
    n_pred = len(p)
    n_gt = len(gts)
    if n_pred == 0 or n_gt == 0:
        return 0, n_pred, n_pred, n_gt
    used = [False] * n_gt
    tp = 0
    gtc = np.array(gts, dtype=np.float32)
    for (cx, cy, _s) in p:
        d = np.hypot(gtc[:, 0] - cx, gtc[:, 1] - cy)
        d[used] = np.inf
        j = int(d.argmin())
        if d[j] <= CENTROID_DIST:
            used[j] = True
            tp += 1
    return tp, n_pred - tp, n_pred, n_gt


def score_epoch(preds_per_video, gt, vids):
    """Sweep thresholds; return best-F1 operating point + per-threshold table."""
    table = []
    for thr in SCORE_SWEEP:
        TP = FP = FN = 0
        for v, ims in vids.items():
            vpreds = preds_per_video[v]
            for im, fp_preds in zip(ims, vpreds):
                gts = gt[im["id"]]
                tp, fp, _np, ngt = match_frame(fp_preds, gts, thr)
                TP += tp
                FP += fp
                FN += ngt - tp
        prec = TP / (TP + FP) if (TP + FP) else 0.0
        rec = TP / (TP + FN) if (TP + FN) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        table.append({"thr": thr, "precision": prec, "recall": rec, "f1": f1,
                      "tp": TP, "fp": FP, "fn": FN})
    best = max(table, key=lambda r: r["f1"])
    return best, table


def epoch_ckpt(ep):
    return CKPT_DIR / f"model_ep{ep}.pth"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, nargs="*", default=list(range(1, 16)))
    ap.add_argument("--smoke", action="store_true",
                    help="1 epoch (15), 2 videos — quick wiring check")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="",
                    help="results-file shard tag for dual-GPU runs, e.g. g0/g1")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_json = (OUT_DIR / f"val_curve_results_{args.tag}.json"
                    if args.tag else RESULTS_JSON)
    results = {}
    if results_json.exists():
        results = json.load(open(results_json))

    if not args.plot_only:
        vids, gt, names = load_val()
        epochs = args.epochs
        if args.smoke:
            epochs = [15]
            keep = sorted(vids)[:2]
            vids = {v: vids[v] for v in keep}

        for ep in epochs:
            ck = epoch_ckpt(ep)
            if not ck.exists():
                print(f"[ep{ep}] missing {ck}, skip")
                continue
            if str(ep) in results and not args.smoke:
                print(f"[ep{ep}] cached -> best F1 {results[str(ep)]['best']['f1']:.4f}")
                continue
            print(f"[ep{ep}] building detector from {ck.name} ...", flush=True)
            det = HiEUMDetector(
                checkpoint_path=str(ck), seq_len=SEQ_LEN, image_size=IMAGE_SIZE,
                layers=LAYERS, thresh=THRESH, car_label=0,
                score_thresh=SCORE_FLOOR, nms_iou=NMS_IOU, max_dets=MAX_DETS,
                device=args.device,
            )
            preds_per_video = {}
            with torch.no_grad():
                for i, v in enumerate(sorted(vids)):
                    preds_per_video[v] = run_video(det, vids[v])
                    print(f"  [ep{ep}] video {i+1}/{len(vids)} {names[v]} "
                          f"({len(vids[v])} frames)", flush=True)
            best, table = score_epoch(preds_per_video, gt, vids)
            print(f"[ep{ep}] best F1 {best['f1']:.4f} "
                  f"(P {best['precision']:.3f} R {best['recall']:.3f} @thr {best['thr']})",
                  flush=True)
            del det
            torch.cuda.empty_cache()
            if not args.smoke:
                results[str(ep)] = {"best": best, "table": table}
                json.dump(results, open(results_json, "w"), indent=2)

    allres = merged_results()
    if allres:
        plot(allres)


def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    from plot_style import apply_neurips_style
    apply_neurips_style(base_size=10)

    eps = sorted(int(e) for e in results)
    f1 = [results[str(e)]["best"]["f1"] for e in eps]
    prec = [results[str(e)]["best"]["precision"] for e in eps]
    rec = [results[str(e)]["best"]["recall"] for e in eps]

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.plot(eps, f1, "-o", label="best F1", color="#1f77b4", lw=1.8, ms=4)
    ax.plot(eps, prec, "--s", label="precision", color="#2ca02c", lw=1.2, ms=3)
    ax.plot(eps, rec, "--^", label="recall", color="#d62728", lw=1.2, ms=3)
    best_ep = eps[int(np.argmax(f1))]
    ax.axvline(best_ep, color="grey", ls=":", lw=1.0)
    ax.annotate(f"best ep{best_ep}\nF1={max(f1):.3f}",
                xy=(best_ep, max(f1)), xytext=(4, -28),
                textcoords="offset points", fontsize=8, color="grey")
    ax.set_xlabel("epoch")
    ax.set_ylabel("union VAL score")
    ax.set_xticks(eps)
    ax.set_title("HiEUM car-union retrain — per-epoch VAL")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    FIG_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PNG, dpi=200)
    fig.savefig(FIG_PDF)
    print(f"saved {FIG_PNG}")
    print(f"saved {FIG_PDF}")


if __name__ == "__main__":
    main()
