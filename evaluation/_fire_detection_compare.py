#!/usr/bin/env python
"""Fire detection comparison at a UNIFIED score>=0.5 operating point.

Mirrors ``evaluation/_birdsai_detection_compare.py``: re-scores the cached
per-frame predictions of the three trained detectors (FasterRCNN / YOLO11l /
DINOv3) from the single unified dump so that *every* reported number shares one
inference pass and one matcher.

Detection metric (per fine class, IoU 0.5):
  * P / R / F1 at a single operating point (score >= 0.5), class-aware greedy
    (highest-IoU-first) matching.
  * OVERALL = micro-average (sum per-class tp/fp/fn), so it always lies within
    the per-class range — no class-agnostic / threshold mismatch.
  * small/large split by COCO area (< 32x32 = 1024 px^2), recall binned by GT
    box size, precision binned by predicted box size.

Why this file exists: the training-time ``test/Precision`` for FasterRCNN and
DINOv3 was a *class-agnostic* count over ALL raw detections down to torchvision's
default score_thresh=0.05, so it fell below every per-class precision. This
recompute puts all three detectors on the same 0.5 footing as the BIRDSAI table.

mAP@0.5 (VOC all-point, score-ranked, threshold-independent) is reported from
each model's official test run, not recomputed here.

Source dump: ``eval_fire_detect_dump.py`` -> fire_detect_predictions.json
"""
import json

import numpy as np

DUMP = "/work/ziwen/experiments/fire_detect_dump/fire_detect_predictions.json"
IOU_THR = 0.5
SCORE_THR = 0.5
SMALL_AREA = 1024.0  # COCO small: area < 32x32
CLASSES = {"smoke": 0, "fire": 1, "person": 2}
MODELS = ("FasterRCNN", "YOLO11l", "DINOv3")


def iou_matrix(a, b):
    a = np.asarray(a, np.float32).reshape(-1, 4)
    b = np.asarray(b, np.float32).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(aa[:, None] + ab[None, :] - inter, 1e-9, None)


def greedy_pairs(g, p, thr):
    """Highest-IoU-first greedy 1-to-1 match; returns list of (gi, pi)."""
    iou = iou_matrix(g, p)
    rs, cs = np.where(iou >= thr)
    pairs, ug, up = [], set(), set()
    for k in np.argsort(-iou[rs, cs]) if len(rs) else []:
        r, c = int(rs[k]), int(cs[k])
        if r in ug or c in up:
            continue
        ug.add(r); up.add(c); pairs.append((r, c))
    return pairs


def area(b):
    b = np.asarray(b, np.float32).reshape(-1, 4)
    return (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])


def prf(tp, fp, fn):
    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    return P, R, 2 * P * R / max(P + R, 1e-9)


def score_model(gt, preds):
    cnt = {c: {"tp": 0, "fp": 0, "fn": 0, "ngt": 0} for c in CLASSES}
    size = {s: {"tp": 0, "fp": 0, "fn": 0, "ngt": 0} for s in ("small", "large")}

    for gframe, pframe in zip(gt, preds):
        gb = np.asarray(gframe["boxes"], np.float32).reshape(-1, 4)
        gl = np.asarray(gframe["labels"], int).reshape(-1)
        pb = np.asarray(pframe["boxes"], np.float32).reshape(-1, 4)
        pl = np.asarray(pframe["labels"], int).reshape(-1)
        ps = np.asarray(pframe["scores"], np.float32).reshape(-1)

        for name, c in CLASSES.items():
            gbc = gb[gl == c]
            pbc = pb[(pl == c) & (ps >= SCORE_THR)]
            ngt = len(gbc)
            pairs = greedy_pairs(gbc, pbc, IOU_THR)
            tp = len(pairs)
            cnt[name]["ngt"] += ngt
            cnt[name]["tp"] += tp
            cnt[name]["fp"] += len(pbc) - tp
            cnt[name]["fn"] += ngt - tp
            # per-size: every GT starts as FN in its own size bin, matched -> TP;
            # unmatched predictions are FP in the predicted box's size bin.
            ga, pa = area(gbc), area(pbc)
            for i in range(ngt):
                s = "small" if ga[i] < SMALL_AREA else "large"
                size[s]["ngt"] += 1; size[s]["fn"] += 1
            matched_p = {pi for _, pi in pairs}
            for gi, _ in pairs:
                s = "small" if ga[gi] < SMALL_AREA else "large"
                size[s]["tp"] += 1; size[s]["fn"] -= 1
            for j in range(len(pbc)):
                if j not in matched_p:
                    s = "small" if pa[j] < SMALL_AREA else "large"
                    size[s]["fp"] += 1

    rows = {}
    for name in CLASSES:
        k = cnt[name]
        P, R, F1 = prf(k["tp"], k["fp"], k["fn"])
        rows[name] = dict(P=P, R=R, F1=F1, **k)
    tot = {kk: sum(cnt[c][kk] for c in CLASSES) for kk in ("tp", "fp", "fn", "ngt")}
    P, R, F1 = prf(tot["tp"], tot["fp"], tot["fn"])
    rows["__overall__"] = dict(P=P, R=R, F1=F1, **tot)
    for s in size:
        k = size[s]
        k["P"], k["R"], k["F1"] = prf(k["tp"], k["fp"], k["fn"])
    rows["__size__"] = size
    return rows


def main():
    d = json.load(open(DUMP))
    gt = d["gt"]
    print(f"frames={len(gt)}  operating point score>={SCORE_THR}  IoU={IOU_THR}\n")
    for m in MODELS:
        r = score_model(gt, d["models"][m])
        o = r["__overall__"]
        print(f"== {m} ==")
        print(f"  OVERALL  P={o['P']:.3f} R={o['R']:.3f} F1={o['F1']:.3f}"
              f"  (tp={o['tp']} fp={o['fp']} fn={o['fn']} ngt={o['ngt']})")
        for c in CLASSES:
            x = r[c]
            print(f"    {c:7s} nGT={x['ngt']:5d}  P={x['P']:.3f} R={x['R']:.3f} F1={x['F1']:.3f}")
        for sz in ("small", "large"):
            x = r["__size__"][sz]
            print(f"    {sz:5s} nGT={x['ngt']:5d}  P={x['P']:.3f} R={x['R']:.3f} F1={x['F1']:.3f}")
        print()


if __name__ == "__main__":
    main()
