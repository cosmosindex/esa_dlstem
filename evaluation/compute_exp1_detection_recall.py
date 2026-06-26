"""Exp1 — detection ability vs. object size, per class, on SAT-MTB.

Complement of Exp2 (association axis). Here each method uses ITS OWN detections
and we measure **Recall vs object size**, *per class* — the cleanest cut, since
the detector identity is tied to class, not to a pixel threshold:

    car            -> HiEUM           (car specialist)
    airplane/ship/train -> Faster R-CNN (SAT-MTB det_hbb, in-domain, 3-class)
    all classes    -> FairMOT, TGraM  (4-class union, their own decoded boxes)

Recall only needs GT class+size (pred class is irrelevant to "was this GT box
found"), so it works uniformly for every detector — including the JDT
``mot_format`` outputs, which carry no class column.

GT class is taken from the SAT-MTB video_id prefix (``airplane/07`` -> airplane);
SAT-MTB stores one category per sequence folder, so this is exact.

Output: ``docs/figures/exp1_detection_recall_by_size.csv`` (detector,class,bin,
recall,n_gt,n_hit). Plot is drawn by ``tools/plot_exp1_detection_vs_size.py``.

Run:
    python compute_exp1_detection_recall.py
"""
from __future__ import annotations

# --- repo root on path so top-level modules (transforms, obb_utils, ...) import ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from compute_hota_by_size import _build_dataset

# sqrt_area (px) bin edges — same as compute_hota_by_size (Exp2), so the two
# experiments share an x-axis.
BIN_EDGES = [0, 5, 8, 12, 20, 40, np.inf]
BIN_LABELS = ["<5", "5-8", "8-12", "12-20", "20-40", ">=40"]
IOU_THRESH = 0.5  # a GT box counts as detected if some pred has IoU >= this

CLASSES = ["car", "airplane", "ship", "train"]
# SAT-MTB `_load_annotations` returns labels re-indexed to COARSE_CATEGORIES
# (alphabetical), NOT the raw MOT cls. Verified empirically (airplane->0 @50px,
# car->1 @4.6px, ship->2 @17px, train->3 @128px). See datasets/satmtb.py.
COARSE = ("airplane", "car", "ship", "train")   # label index -> class name

# specialist detector per class (HiEUM=car, Faster R-CNN=the rest)
SPECIALIST = {"car": "HiEUM", "airplane": "FasterRCNN",
              "ship": "FasterRCNN", "train": "FasterRCNN"}

HIEUM_DIR = "/data/ESA_DLSTEM_2025/experiments/Detection/hieum_dets_cache/satmtb"
FRCNN_DIR = "/data/ESA_DLSTEM_2025/experiments/Detection/fasterrcnn_satmtb_hbb_dets_cache/satmtb_nocar"
FAIRMOT_GLOB = "/data/ESA_DLSTEM_2025/experiments/MOT/allclass_20260608/fairmot_all_satmtb_*/mot_format"
TGRAM_GLOB = "/data/ESA_DLSTEM_2025/experiments/MOT/allclass_20260608/tgram_all_satmtb_*/mot_format"

OUT_CSV = Path("/home/anon/code/esa_dlstem/docs/figures/exp1_detection_recall_by_size.csv")

# detector -> which classes it is scored on
DETECTOR_CLASSES = {
    "HiEUM":   ["car"],
    "FasterRCNN": ["airplane", "ship", "train"],
    "FairMOT": CLASSES,
    "TGraM":   CLASSES,
}


def _bin_idx(sqrt_area: float) -> int:
    for i in range(len(BIN_EDGES) - 1):
        if BIN_EDGES[i] <= sqrt_area < BIN_EDGES[i + 1]:
            return i
    return len(BIN_LABELS) - 1


def _safe(video_id: str) -> str:
    return video_id.replace("/", "_")


def _iou_matrix(gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
    """gt [N,4] xywh, pr [M,4] xywh -> IoU [N,M]."""
    if len(gt) == 0 or len(pr) == 0:
        return np.zeros((len(gt), len(pr)), dtype=np.float32)
    g = gt.copy(); p = pr.copy()
    gx1, gy1, gx2, gy2 = g[:, 0], g[:, 1], g[:, 0] + g[:, 2], g[:, 1] + g[:, 3]
    px1, py1, px2, py2 = p[:, 0], p[:, 1], p[:, 0] + p[:, 2], p[:, 1] + p[:, 3]
    ix1 = np.maximum(gx1[:, None], px1[None, :])
    iy1 = np.maximum(gy1[:, None], py1[None, :])
    ix2 = np.minimum(gx2[:, None], px2[None, :])
    iy2 = np.minimum(gy2[:, None], py2[None, :])
    iw = np.clip(ix2 - ix1, 0, None); ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    ga = (g[:, 2] * g[:, 3])[:, None]; pa = (p[:, 2] * p[:, 3])[None, :]
    return inter / np.maximum(ga + pa - inter, 1e-9)


# ---- prediction loaders: video_id -> {frame_id: boxes[M,4] xywh} ----

def _load_cache_dir(cache_dir: str) -> dict[str, dict[int, np.ndarray]]:
    out: dict[str, dict[int, np.ndarray]] = {}
    for f in glob.glob(os.path.join(cache_dir, "*.json")):
        d = json.load(open(f))
        vid = d["video_id"]
        per = {}
        for fid, boxes in zip(d["frame_ids"], d["boxes"]):
            arr = np.asarray(boxes, dtype=np.float32).reshape(-1, 4) if boxes else np.zeros((0, 4), np.float32)
            per[int(fid)] = arr  # cache boxes are already xywh? -> verified below
        out[vid] = per
    return out


def _load_mot_format(mot_dir: str) -> dict[str, dict[int, np.ndarray]]:
    out: dict[str, dict[int, np.ndarray]] = defaultdict(lambda: defaultdict(list))
    for f in glob.glob(os.path.join(mot_dir, "*.txt")):
        vid = Path(f).stem.replace("_", "/", 1)  # airplane_07 -> airplane/07
        for line in open(f):
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            fid = int(p[0]); x, y, w, h = map(float, p[2:6])
            out[vid][fid].append([x, y, w, h])
    return {v: {fid: np.asarray(b, np.float32).reshape(-1, 4) for fid, b in fr.items()}
            for v, fr in out.items()}


def main():
    ds = _build_dataset("satmtb")

    preds = {
        "HiEUM": _load_cache_dir(HIEUM_DIR),
        "FasterRCNN": _load_cache_dir(FRCNN_DIR),
        "FairMOT": _load_mot_format(sorted(glob.glob(FAIRMOT_GLOB))[-1]),
        "TGraM": _load_mot_format(sorted(glob.glob(TGRAM_GLOB))[-1]),
    }
    # cache box format check: HiEUM/FRCNN cache stores xyxy; convert to xywh once.
    for det in ("HiEUM", "FasterRCNN"):
        for vid, fr in preds[det].items():
            for fid, arr in fr.items():
                if len(arr):
                    arr[:, 2] = arr[:, 2] - arr[:, 0]
                    arr[:, 3] = arr[:, 3] - arr[:, 1]

    # Each class is scored only on the videos where its specialist detector
    # actually ran (HiEUM -> car-folder vids; Faster R-CNN -> the rest). This
    # keeps specialist vs JDT on the *same* GT boxes, and excludes e.g. cars
    # that appear inside train scenes (HiEUM never saw those frames).
    class_video_set = {
        "car": set(preds["HiEUM"].keys()),
        "airplane": set(preds["FasterRCNN"].keys()),
        "ship": set(preds["FasterRCNN"].keys()),
        "train": set(preds["FasterRCNN"].keys()),
    }

    # accumulate hits/totals per (detector, class, bin)
    n_gt = defaultdict(int)
    n_hit = defaultdict(int)

    # cache best-IoU per (detector) for each frame so we score every detector once
    for video in ds.videos:
        vid = video.video_id
        for fid in video.frame_ids:
            ann = ds._load_annotations(video, fid)
            gt = np.asarray(ann["boxes"], dtype=np.float32).reshape(-1, 4)
            if not len(gt):
                continue
            gt[:, 2] = gt[:, 2] - gt[:, 0]   # xyxy -> xywh (GT is xyxy)
            gt[:, 3] = gt[:, 3] - gt[:, 1]
            labels = np.asarray(ann["labels"]).reshape(-1)
            gt_cls = [COARSE[int(l)] for l in labels]
            gt_bin = [_bin_idx(float(np.sqrt(max(w * h, 0.0)))) for *_, w, h in gt]

            # which detectors are relevant for the classes present in this frame
            specialists = {SPECIALIST[c] for c in gt_cls}
            best = {}
            for det in specialists | {"FairMOT", "TGraM"}:
                pr = preds[det].get(vid, {}).get(int(fid), np.zeros((0, 4), np.float32))
                iou = _iou_matrix(gt, pr)
                best[det] = iou.max(axis=1) if iou.shape[1] else np.zeros(len(gt))

            for i, (c, b) in enumerate(zip(gt_cls, gt_bin)):
                if vid not in class_video_set[c]:
                    continue  # this class's specialist didn't run on this video
                for det in (SPECIALIST[c], "FairMOT", "TGraM"):
                    n_gt[(det, c, b)] += 1
                    if best[det][i] >= IOU_THRESH:
                        n_hit[(det, c, b)] += 1

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w") as f:
        f.write("detector,class,bin_idx,size_bin,n_gt,n_hit,recall\n")
        for (det, cls, b), tot in sorted(n_gt.items()):
            hit = n_hit[(det, cls, b)]
            f.write(f"{det},{cls},{b},{BIN_LABELS[b]},{tot},{hit},{hit/max(tot,1):.4f}\n")
    print(f"wrote {OUT_CSV}  (IoU>={IOU_THRESH})")
    # quick console summary
    for cls in CLASSES:
        print(f"\n[{cls}]")
        for det in DETECTOR_CLASSES:
            if cls not in DETECTOR_CLASSES[det]:
                continue
            row = [f"{BIN_LABELS[b]}={n_hit[(det,cls,b)]/max(n_gt[(det,cls,b)],1):.2f}({n_gt[(det,cls,b)]})"
                   for b in range(len(BIN_LABELS)) if n_gt[(det, cls, b)] > 0]
            print(f"  {det:11s} " + "  ".join(row))


if __name__ == "__main__":
    main()
