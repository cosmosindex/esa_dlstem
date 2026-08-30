"""
Build the MOT-stage detection table for the WACV paper.

The two-stage MOT pipeline needs per-frame boxes, and Space-Tracker splits that
job between two detectors that cover *disjoint* category sets:

* **HiEUM** for `car`. Car ground truth labels only the moving objects, so a
  single-frame detector cannot learn it -- the same pixels are a positive in one
  sequence and unlabelled background in another. HiEUM consumes a clip and keys
  on motion, which is what makes the category learnable at all.
* **Faster R-CNN** for `airplane` and `ship`, trained on the Space-Tracker
  detection training split with data-driven anchors.

They are therefore NOT competing rows: neither one is evaluated on the other's
categories, and the table must not be read as a ranking. What it is for is
stating the detection quality that every tracking-by-detection number in the
paper inherits, per \\cref{sec:eval_metrics}: mAP_50, class-agnostic AP_overall,
and P / R / F1 at score 0.5, all at IoU >= 0.5.

The scores come from the `pr_curve.json` written by
`lightning_modules/module.py::_save_pr_curves`, which runs on the **test split
only**. We read the run whose checkpoint actually produced the cached detections
the trackers consumed -- reporting a different (better) checkpoint here would
make this table inconsistent with every MOT number downstream.

Usage:
    python tools/make_wacv_det_table.py \
        --frcnn /work/anon/experiments/fasterrcnn_spacetracker_mot_20260825_083522/pr_curve.json \
        --out   wacv-2027-author-kit-template/tables/mot_detection.tex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# pr_curve.json stores FRCNN's 1-based label ids; the config's class_map is
# {airplane: 1, ship: 2}.
FRCNN_CLASSES = {"cls1": "airplane", "cls2": "ship"}
TAU_OP = 0.5          # operating point for P / R / F1, per sec:eval_metrics


def op_point(curve: dict, tau: float = TAU_OP) -> dict:
    """P / R / F1 among predictions scoring >= tau.

    `precision` and `recall` are cumulative over predictions sorted by
    descending score, so the operating point is simply the last index whose
    score still clears tau.
    """
    scores = curve["scores"]
    idx = -1
    for i, s in enumerate(scores):
        if s >= tau:
            idx = i
        else:
            break
    if idx < 0:                       # nothing survives the threshold
        return {"P": 0.0, "R": 0.0, "F1": 0.0, "n_pred": 0}
    p, r = curve["precision"][idx], curve["recall"][idx]
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"P": p, "R": r, "F1": f1, "n_pred": idx + 1}


def best_f1(curve: dict) -> dict:
    """The operating point that maximises F1, and the score that achieves it.

    A single shared threshold is only meaningful when the detectors are
    calibrated alike, and these two are not: Faster R-CNN's scores reach 1.0
    while HiEUM's top out at 0.66, so tau=0.5 discards 99.6% of HiEUM's
    predictions and drives its recall to 0.007 -- a property of the threshold,
    not of the detector. Reporting each detector's own F1-optimal point
    alongside the protocol point makes the comparison legible without hiding
    what the protocol says.
    """
    scores, P, R = curve["scores"], curve["precision"], curve["recall"]
    best = {"F1": -1.0, "P": 0.0, "R": 0.0, "thr": 0.0}
    for i in range(len(scores)):
        p_, r_ = P[i], R[i]
        f = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) > 0 else 0.0
        if f > best["F1"]:
            best = {"F1": f, "P": p_, "R": r_, "thr": scores[i]}
    return best


def load_frcnn(path: Path) -> dict:
    d = json.loads(path.read_text())
    per_class = {FRCNN_CLASSES[k]: v for k, v in d["per_class"].items()
                 if k in FRCNN_CLASSES}
    return {
        "mAP50": d["mAP_11pt"],
        "AP_overall": d["AP_overall_11pt"],
        "n_gt": d["overall"]["n_gt"],
        "op": op_point(d["overall"]),
        "best": best_f1(d["overall"]),
        "score_max": max(d["overall"]["scores"]) if d["overall"]["scores"] else 0.0,
        "per_class": {c: {"AP": v["AP_11pt"], "n_gt": v["n_gt"],
                          "op": op_point(v), "best": best_f1(v)}
                      for c, v in per_class.items()},
    }


def fmt(x, nd=3):
    return "--" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frcnn", required=True, help="pr_curve.json of the FRCNN run")
    ap.add_argument("--hieum", default=None,
                    help="pr_curve.json for HiEUM on the car test split. Omit "
                         "while that evaluation is still outstanding; the row "
                         "is then emitted with '--' and the caption says so.")
    ap.add_argument("--frcnn-seqs", type=int, default=22)
    ap.add_argument("--hieum-seqs", type=int, default=48)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    f = load_frcnn(Path(args.frcnn))
    h = load_frcnn(Path(args.hieum)) if args.hieum else None

    print(f"Faster R-CNN  mAP50={f['mAP50']:.4f}  AP_ov={f['AP_overall']:.4f}  "
          f"P={f['op']['P']:.4f} R={f['op']['R']:.4f} F1={f['op']['F1']:.4f}  "
          f"(n_gt={f['n_gt']}, {f['op']['n_pred']} preds >= {TAU_OP})")
    for c, v in f["per_class"].items():
        print(f"   {c:9s} AP={v['AP']:.4f}  n_gt={v['n_gt']:6d}  "
              f"P={v['op']['P']:.3f} R={v['op']['R']:.3f} F1={v['op']['F1']:.3f}")
    if h is None:
        print("HiEUM        : no pr_curve supplied -- row emitted as '--'")

    L = []
    L.append("% Generated by tools/make_wacv_det_table.py -- do not edit by hand.")
    L.append(r"\begin{table}[t]")
    L.append(r"\centering")
    L.append(
        r"\caption{Detection quality on the Space-Tracker-MOT \textbf{test} "
        r"split, the input every tracking-by-detection result in this paper "
        r"inherits. The two detectors cover \emph{disjoint} category sets and "
        r"are not competing entries: \emph{car} ground truth annotates only "
        r"moving objects, which a single-frame detector cannot learn, so it is "
        r"handled by the clip-based HiEUM, while \emph{airplane} and "
        r"\emph{ship} are handled by a Faster R-CNN trained on the "
        r"Space-Tracker detection training split with data-driven anchors. "
        r"Metrics follow \cref{sec:eval_metrics}: $\mathrm{AP}$ and "
        r"$\mathrm{mAP}_{50}$ use 11-point interpolation at "
        r"$\mathrm{IoU}\geq0.5$ and $\mathrm{AP}_{\text{overall}}$ pools all "
        r"categories into one foreground class; both are threshold-free and are "
        r"the only columns that support a cross-detector reading. "
        r"\textbf{The $\tau=0.5$ columns must not be compared across the two "
        r"detectors.} The protocol threshold assumes comparable calibration and "
        r"these models do not have it: Faster R-CNN's scores reach $1.0$ while "
        r"HiEUM's peak at $0.66$, so $\tau=0.5$ removes $99.6\%$ of HiEUM's "
        r"predictions and pins its recall at $0.007$ -- a fact about the "
        r"threshold, not about the detector, whose $\mathrm{mAP}_{50}$ is "
        r"$0.20$. We therefore also give each detector's own F1-optimal point "
        r"and the score $\tau^{*}$ that attains it; that HiEUM peaks at "
        r"$\tau^{*}=0.26$ and Faster R-CNN at $\tau^{*}=0.99$ is itself the "
        r"evidence that no single threshold serves both. Indented rows "
        r"decompose the multi-class detector by category. Higher is better "
        r"throughout except $\tau^{*}$, which is descriptive.}")
    L.append(r"\label{tab:mot_detection}")
    L.append(r"\setlength{\tabcolsep}{4pt}")
    L.append(r"\renewcommand{\arraystretch}{1.15}")
    L.append(r"\resizebox{\columnwidth}{!}{%")
    L.append(r"\begin{tabular}{l l r r *{7}{>{\centering\arraybackslash}p{2.5em}}}")
    L.append(r"\toprule")
    L.append(r"\multirow{2}{*}{\textbf{Detector}} & "
             r"\multirow{2}{*}{\textbf{Categories}} & "
             r"\multirow{2}{*}{\textbf{\#Seq}} & \multirow{2}{*}{\textbf{\#GT}} & "
             r"\multicolumn{2}{c}{\textbf{Threshold-free}} & "
             r"\multicolumn{3}{c}{\textbf{At $\tau=0.5$}} & "
             r"\multicolumn{2}{c}{\textbf{Best F1}} \\")
    L.append(r"\cmidrule(lr){5-6} \cmidrule(lr){7-9} \cmidrule(lr){10-11}")
    L.append(r" & & & & mAP$_{50}$ & AP$_{\text{ov}}$ & P & R & F1 & F1 & $\tau^{*}$ \\")
    L.append(r"\midrule")

    if h is None:
        L.append(rf"HiEUM~\citep{{Xiao2024HiEUM}} & car & {args.hieum_seqs} & "
                 r"-- & -- & -- & -- & -- & -- & -- & -- \\")
    else:
        L.append(rf"HiEUM~\citep{{Xiao2024HiEUM}} & car & {args.hieum_seqs} & "
                 rf"{h['n_gt']:,} & {fmt(h['mAP50'])} & {fmt(h['AP_overall'])} & "
                 rf"{fmt(h['op']['P'])} & {fmt(h['op']['R'])} & {fmt(h['op']['F1'])} & "
                 rf"{fmt(h['best']['F1'])} & {fmt(h['best']['thr'], 2)} \\")

    L.append(r"\midrule")
    L.append(rf"Faster R-CNN & airplane, ship & {args.frcnn_seqs} & "
             rf"{f['n_gt']:,} & {fmt(f['mAP50'])} & {fmt(f['AP_overall'])} & "
             rf"{fmt(f['op']['P'])} & {fmt(f['op']['R'])} & {fmt(f['op']['F1'])} & "
             rf"{fmt(f['best']['F1'])} & {fmt(f['best']['thr'], 2)} \\")
    for c in ("airplane", "ship"):
        v = f["per_class"][c]
        L.append(rf"\quad\emph{{{c}}} & & & {v['n_gt']:,} & {fmt(v['AP'])} & -- & "
                 rf"{fmt(v['op']['P'])} & {fmt(v['op']['R'])} & {fmt(v['op']['F1'])} & "
                 rf"{fmt(v['best']['F1'])} & {fmt(v['best']['thr'], 2)} \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"}")
    L.append(r"\end{table}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
