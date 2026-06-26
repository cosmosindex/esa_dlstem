"""Step-A offline mAP: recompute per-class AP for the 3 detectors on different
TestReal subsets, purely from cached raw detections (no model re-run).

Sets compared:
  old_test  : 11 videos (original reported `test` split)
  full16    : all 16 TestReal videos (most representative; includes the only lion)
  strat_test: a class+size stratified test split proposed for option B
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from datasets.birdsai_mot import BIRDSAIMOTDataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision

ROOT = "/data/ESA_DLSTEM_2025/data/wild_animal/BIRDSAI"
CANON = {0: "human", 1: "elephant", 2: "giraffe", 3: "lion", 4: "unknown"}
A_DIR = Path("/work/ziwen/experiments/birdsai_resplit_A")

CACHED = {
    "fasterrcnn": "/work/ziwen/experiments/fasterrcnn_birdsai_dettrack_20260615_182903/predictions.json",
    "yolo": "/work/ziwen/experiments/yolo_birdsai_dettrack_20260615_140408/predictions.json",
    "dinov3": "/work/ziwen/experiments/dinov3_birdsai_dettrack_20260615_182902/predictions.json",
}


def load_preds(model):
    """video_id -> frame_id(str) -> {boxes,scores,labels} merged over 11+5 videos."""
    out = {}
    for path in (CACHED[model], str(A_DIR / f"missing_preds_{model}.json")):
        d = json.load(open(path))
        for vid, ventry in d["videos"].items():
            out.setdefault(vid, {})
            for fid, fentry in ventry["frames"].items():
                out[vid][fid] = fentry["detections"]
    return out


def load_gt():
    """video_id -> frame_id(int) -> (boxes Nx4, labels N) canonical, + per-video stats."""
    canon_map = {v: k for k, v in CANON.items()}
    ds = BIRDSAIMOTDataset(root=ROOT, split="no_split", granularity="fine", class_map=canon_map)
    # restrict to TestReal = old val (5) ∪ old test (11)
    val_ids = {v.video_id for v in BIRDSAIMOTDataset(root=ROOT, split="val", granularity="fine").videos}
    test_ids = {v.video_id for v in BIRDSAIMOTDataset(root=ROOT, split="test", granularity="fine").videos}
    testreal = val_ids | test_ids

    gt = {}
    vstats = {}  # vid -> {cls_count, size_count, n}
    for v in ds.videos:
        if v.video_id not in testreal:
            continue
        gt[v.video_id] = {}
        cc = defaultdict(int); sc = {"s": 0, "m": 0, "l": 0}; n = 0
        for fid in v.frame_ids:
            ann = ds._load_annotations(v, fid)
            b = np.asarray(ann["boxes"], np.float32).reshape(-1, 4)
            l = np.asarray(ann["labels"], np.int64).reshape(-1)
            gt[v.video_id][fid] = (b, l)
            for bb, ll in zip(b, l):
                a = max(0., bb[2]-bb[0]) * max(0., bb[3]-bb[1])
                cc[int(ll)] += 1
                sc["s" if a < 32*32 else ("m" if a < 96*96 else "l")] += 1
                n += 1
        vstats[v.video_id] = {"cls": dict(cc), "size": sc, "n": n}
    return gt, vstats, sorted(val_ids), sorted(test_ids)


def ap_for(model_preds, gt, vids):
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True,
                                  iou_thresholds=[0.5])
    for vid in vids:
        for fid_int, (gb, gl) in gt[vid].items():
            p = model_preds.get(vid, {}).get(str(fid_int),
                                             {"boxes": [], "scores": [], "labels": []})
            pb = torch.tensor(p["boxes"], dtype=torch.float32).reshape(-1, 4)
            ps = torch.tensor(p["scores"], dtype=torch.float32).reshape(-1)
            pl = torch.tensor(p["labels"], dtype=torch.int64).reshape(-1)
            metric.update(
                [{"boxes": pb, "scores": ps, "labels": pl}],
                [{"boxes": torch.tensor(gb, dtype=torch.float32).reshape(-1, 4),
                  "labels": torch.tensor(gl, dtype=torch.int64).reshape(-1)}],
            )
    res = metric.compute()
    per = {}
    classes = res.get("classes")
    map_pc = res.get("map_per_class")
    if map_pc is not None and map_pc.ndim > 0:
        for c, ap in zip(classes.tolist(), map_pc.tolist()):
            per[CANON.get(int(c), c)] = ap
    return float(res["map_50"]), per


def main():
    gt, vstats, val_ids, test_ids = load_gt()
    full16 = sorted(gt.keys())

    print("\n##### per-video composition (TestReal) #####")
    print(f"{'video':24s} {'n':>6s} | classes (count) | small/med/large")
    for vid in full16:
        s = vstats[vid]
        cls = ", ".join(f"{CANON[c]}={n}" for c, n in sorted(s["cls"].items()))
        tag = "VAL" if vid in val_ids else "test"
        print(f"{vid:24s} {s['n']:>6d} [{tag}] | {cls:50s} | "
              f"{s['size']['s']}/{s['size']['m']}/{s['size']['l']}")

    # ---- propose a class+size stratified split (manual, data-driven) ----
    # Goal: every class present in BOTH val & test; size mix closer to overall.
    # Filled after inspecting composition (see STRAT_VAL below).
    STRAT_VAL = STRAT.get("val") if (STRAT := globals().get("STRAT_SPLIT", {})) else None

    sets = {"old_test(11)": test_ids, "full16": full16}
    if STRAT_VAL is not None:
        strat_test = [v for v in full16 if v not in set(STRAT_VAL)]
        sets["strat_test"] = strat_test
        sets["strat_val"] = STRAT_VAL

    preds = {m: load_preds(m) for m in CACHED}

    for setname, vids in sets.items():
        print("\n" + "=" * 70)
        print(f"SET = {setname}  ({len(vids)} videos)")
        # GT class totals for this set
        tot = defaultdict(int)
        for vid in vids:
            for c, n in vstats[vid]["cls"].items():
                tot[c] += n
        print("  GT class boxes:", {CANON[c]: n for c, n in sorted(tot.items())})
        print(f"  {'model':12s} {'mAP@.5':>8s} | per-class AP@.5")
        for m in CACHED:
            map50, per = ap_for(preds[m], gt, vids)
            ps = "  ".join(f"{k}={v:.3f}" for k, v in sorted(per.items()))
            print(f"  {m:12s} {map50:8.3f} | {ps}")


if __name__ == "__main__":
    main()
