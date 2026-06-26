"""
BIRDSAI SAM3 ORACLE eval (two modes). Both use test GT -> oracle / upper-bound,
NOT fair MOT rows. Scored at FINE 5-class, same matching as
eval_birdsai_detect_track.py, so directly comparable to the 3 detectors.

  mode=gt_init        (Exp1): each GT track is initialised with its GT box at its
                      first-appearance frame (within a clip), then SAM3 propagates
                      it (SAM2-style mask memory). Tests the TRACKER ceiling.

  mode=exemplar_detect(Exp2): at each clip's first frame, every GT box of a class
                      is looped as a single visual exemplar through SAM3's
                      find-grounding head; the union of detected boxes (dedup by
                      IoU) is then used as the SOT init that SAM3 propagates.
                      Tests exemplar-seeded detection + tracking.

Videos are processed in non-overlapping clips (SAM3 loads a whole clip); track
ids are offset to stay globally unique. Output mirrors eval_birdsai_detect_track:
predictions.json (detections==tracks) + test_metrics.json.

    CUDA_VISIBLE_DEVICES=0 python evaluation/eval_birdsai_sam3_oracle.py --mode gt_init
    CUDA_VISIBLE_DEVICES=0 python evaluation/eval_birdsai_sam3_oracle.py --mode exemplar_detect
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import time
from datetime import datetime

import numpy as np
import torch

from datasets.birdsai_mot import BIRDSAIMOTDataset
from models.sam3 import SAM3Tracker, SAM3TextTracker

BIRDSAI_ROOT = "/data/ESA_DLSTEM_2025/data/wild_animal/BIRDSAI"
OUT_ROOT = "/work/ziwen/experiments"
CANON = {0: "human", 1: "elephant", 2: "giraffe", 3: "lion", 4: "unknown"}
PROMPT = {0: "person", 1: "elephant", 2: "giraffe", 3: "lion", 4: "animal"}
CLIP_LEN = 32
EXEMPLAR_CAP = 20          # max exemplar boxes looped per class per keyframe
DEDUP_IOU = 0.5


# ---------------- IoU helpers (class-aware greedy, mirrors eval_tracker) -------
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


# ---------------- per-clip producers -----------------------------------------
def clip_gt_init(sot: SAM3Tracker, frames, clip_anns, gid_offset):
    """Exp1: init each GT track at its first-appearance frame within the clip."""
    sot.init_video(frames)
    # group (box,label,global_id) by first-appearance frame
    seen = {}
    by_frame: dict[int, list] = {}
    for j, ann in enumerate(clip_anns):
        for b, l, tid in zip(ann["boxes"], ann["labels"], ann["track_ids"]):
            key = int(tid)
            if key in seen:
                continue
            seen[key] = gid_offset + len(seen)
            by_frame.setdefault(j, []).append(
                (np.asarray(b, np.float32), int(l), seen[key]))
    if not seen:
        sot.reset_state(); return [{} for _ in frames], gid_offset
    for j, items in by_frame.items():
        boxes = np.stack([it[0] for it in items])
        labels = np.array([it[1] for it in items])
        oids = [it[2] for it in items]
        sot.add_prompts(j, boxes, labels, obj_ids=oids)
    outs = sot.propagate()
    per = _outs_to_per(outs, len(frames))
    sot.reset_state()
    return per, gid_offset + len(seen)


def clip_exemplar_detect(finder: SAM3TextTracker, sot: SAM3Tracker, frames,
                         clip_anns, gid_offset):
    """Exp2: exemplar-seeded detection at frame 0 -> SOT init -> propagate."""
    H, W = frames[0].shape[:2]
    pred = finder.predictor
    finder.init_video(frames)
    st = finder._inference_state
    a0 = clip_anns[0]
    # detected (box_xyxy, label, score) merged across exemplars
    det = []
    for c in sorted(CANON):
        cbs = [np.asarray(b, np.float32) for b, l in zip(a0["boxes"], a0["labels"])
               if int(l) == c]
        if not cbs:
            continue
        text = PROMPT[c]
        cand = []
        for b in cbs[:EXEMPLAR_CAP]:
            xywh = np.clip([b[0] / W, b[1] / H, (b[2] - b[0]) / W, (b[3] - b[1]) / H], 0, 1)
            pred.reset_state(st)
            _, out = pred.add_prompt(
                inference_state=st, frame_idx=0, text_str=text,
                boxes_xywh=torch.tensor(xywh, dtype=torch.float32).view(1, 4),
                box_labels=torch.ones(1, dtype=torch.long))
            for i in range(len(out["out_probs"])):
                if not out["out_binary_masks"][i].any():
                    continue
                x, y, w, h = [float(v) for v in out["out_boxes_xywh"][i]]
                cand.append(([x * W, y * H, (x + w) * W, (y + h) * H],
                             float(out["out_probs"][i])))
        # dedup within class by IoU, keep highest score
        cand.sort(key=lambda t: -t[1])
        kept = []
        for box, sc in cand:
            bb = np.array(box, np.float32)
            if any(iou_matrix(bb[None], np.array(k[0], np.float32)[None])[0, 0] > DEDUP_IOU
                   for k in kept):
                continue
            kept.append((box, sc))
            det.append((bb, c, sc))
    finder.reset_state()

    if not det:
        return [{} for _ in frames], gid_offset
    # SOT init with detected boxes at frame 0
    sot.init_video(frames)
    boxes = np.stack([d[0] for d in det])
    labels = np.array([d[1] for d in det])
    oids = [gid_offset + i for i in range(len(det))]
    sot.add_prompts(0, boxes, labels, obj_ids=oids)
    outs = sot.propagate()
    per = _outs_to_per(outs, len(frames))
    sot.reset_state()
    return per, gid_offset + len(det)


def _outs_to_per(outs, nframes):
    """SAM3Tracker.propagate() list -> {j: {boxes,labels,scores,track_ids}}."""
    per = {}
    for j in range(nframes):
        o = outs[j] if j < len(outs) else None
        if o is None or len(o["boxes"]) == 0:
            per[j] = {"boxes": np.zeros((0, 4), np.float32), "labels": np.zeros(0, int),
                      "scores": np.zeros(0, np.float32), "track_ids": np.zeros(0, int)}
        else:
            per[j] = {"boxes": o["boxes"].numpy(), "labels": o["labels"].numpy(),
                      "scores": o["scores"].numpy(), "track_ids": o["track_ids"].numpy()}
    return per


# ---------------- main --------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gt_init", "exemplar_detect"], required=True)
    ap.add_argument("--iou-thresh", type=float, default=0.5)
    ap.add_argument("--limit-videos", type=int, default=0, help="0=all (debug)")
    args = ap.parse_args()

    ds = BIRDSAIMOTDataset(root=BIRDSAI_ROOT, split="test", granularity="fine",
                           class_map={v: k for k, v in CANON.items()})
    sot = SAM3Tracker()
    finder = SAM3TextTracker(class_names=["animal"], label_to_id={"animal": 0}) \
        if args.mode == "exemplar_detect" else None

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(OUT_ROOT) / f"sam3_birdsai_oracle_{args.mode}_{ts}"
    (exp_dir / "mot_format").mkdir(parents=True, exist_ok=True)

    classes = sorted(CANON)
    det = {c: {"tp": 0, "fp": 0, "fn": 0} for c in classes}
    trk = {c: {"tp": 0, "fp": 0, "fn": 0, "idsw": 0, "ngt": 0} for c in classes}
    predictions = {"model": f"sam3_oracle_{args.mode}", "dataset": "BIRDSAI",
                   "split": "test", "class_names": CANON, "videos": {}}
    t0 = time.perf_counter()

    videos = ds.videos[:args.limit_videos] if args.limit_videos else ds.videos
    for v_idx, video in enumerate(videos, 1):
        fids = video.frame_ids
        gid = 1
        frames_out = {}
        last_gt_to_pred = {c: {} for c in classes}
        vtp = vfp = vfn = vng = vidsw = 0
        mot_lines = []
        for s in range(0, len(fids), CLIP_LEN):
            cf = fids[s:s + CLIP_LEN]
            frames = [ds._load_frame(video, f) for f in cf]
            clip_anns = [ds._load_annotations(video, f) for f in cf]
            if args.mode == "gt_init":
                per, gid = clip_gt_init(sot, frames, clip_anns, gid)
            else:
                per, gid = clip_exemplar_detect(finder, sot, frames, clip_anns, gid)

            for j, fid in enumerate(cf):
                ann = clip_anns[j]
                gtb = np.asarray(ann["boxes"], np.float32).reshape(-1, 4)
                gtl = np.asarray(ann["labels"], np.int64).reshape(-1)
                gtid = np.asarray(ann["track_ids"], np.int64).reshape(-1)
                p = per[j]
                pb, pl, ps, pid = p["boxes"], p["labels"], p["scores"], p["track_ids"]
                fb, fs, flab, fids_ = [], [], [], []
                for c in classes:
                    gm = gtl == c; pm = pl == c
                    gb = gtb[gm]; gi = gtid[gm]
                    cb = pb[pm]; ci = pid[pm]
                    ms = greedy_match(gb, cb, args.iou_thresh)
                    tp = len(ms)
                    det[c]["tp"] += tp; det[c]["fp"] += len(cb) - tp; det[c]["fn"] += len(gb) - tp
                    trk[c]["tp"] += tp; trk[c]["fp"] += len(cb) - tp; trk[c]["fn"] += len(gb) - tp
                    trk[c]["ngt"] += len(gb)
                    vtp += tp; vfp += len(cb) - tp; vfn += len(gb) - tp; vng += len(gb)
                    for r, cc in ms:
                        g = int(gi[r]); pp = int(ci[cc])
                        prev = last_gt_to_pred[c].get(g)
                        if prev is not None and prev != pp:
                            trk[c]["idsw"] += 1; vidsw += 1
                        last_gt_to_pred[c][g] = pp
                for k in range(len(pb)):
                    x1, y1, x2, y2 = [float(v) for v in pb[k]]
                    fb.append([x1, y1, x2, y2]); fs.append(float(ps[k]))
                    flab.append(int(pl[k])); fids_.append(int(pid[k]))
                    mot_lines.append(f"{int(fid)},{int(pid[k])},{x1:.2f},{y1:.2f},"
                                     f"{x2-x1:.2f},{y2-y1:.2f},{float(ps[k]):.4f},-1,-1,-1")
                frames_out[str(int(fid))] = {
                    "image_path": str(ds._img_dir_cache[video.video_id] /
                                      f"{video.video_id}_{fid:010d}.jpg"),
                    "detections": {"boxes": fb, "scores": fs, "labels": flab},
                    "tracks": {"boxes": fb, "scores": fs, "labels": flab, "track_ids": fids_},
                }
        with open(exp_dir / "mot_format" / f"{video.video_id}.txt", "w") as f:
            f.write("\n".join(mot_lines))
        predictions["videos"][video.video_id] = {
            "image_dir": str(ds._img_dir_cache[video.video_id]), "frames": frames_out}
        prec = vtp / max(vtp + vfp, 1); rec = vtp / max(vtp + vfn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        mota = 1.0 - (vfp + vfn + vidsw) / max(vng, 1)
        print(f"[{v_idx}/{len(videos)}] {video.video_id}  "
              f"P={prec:.3f} R={rec:.3f} F1={f1:.3f} MOTA={mota:.3f} IDsw={vidsw}", flush=True)

    def metrics_from(d, t):
        prec = d["tp"] / max(d["tp"] + d["fp"], 1)
        rec = d["tp"] / max(d["tp"] + d["fn"], 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        mota = 1.0 - (t["fp"] + t["fn"] + t["idsw"]) / max(t["ngt"], 1)
        idp = t["tp"] / max(t["tp"] + t["fp"], 1); idr = t["tp"] / max(t["tp"] + t["fn"], 1)
        idf1 = 2 * idp * idr / max(idp + idr, 1e-9)
        return {"Precision": prec, "Recall": rec, "F1": f1, "MOTA": mota,
                "IDF1": idf1, "IDsw": t["idsw"], "num_gt": t["ngt"]}

    per_class = {CANON[c]: metrics_from(det[c], trk[c]) for c in classes}
    pd_ = {k: sum(det[c][k] for c in classes) for k in ("tp", "fp", "fn")}
    pt_ = {k: sum(trk[c][k] for c in classes) for k in ("tp", "fp", "fn", "idsw", "ngt")}
    overall = metrics_from(pd_, pt_)
    summary = {"model": f"sam3_oracle_{args.mode}", "dataset": "BIRDSAI", "split": "test",
               "oracle": True, "iou_thresh": args.iou_thresh, "clip_len": CLIP_LEN,
               "total_videos": len(videos), "total_time_s": time.perf_counter() - t0,
               "overall": overall, "per_class": per_class}
    with open(exp_dir / "test_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(exp_dir / "predictions.json", "w") as f:
        json.dump(predictions, f)

    print("\n" + "=" * 64)
    print(f"OVERALL [{args.mode}]  P={overall['Precision']:.3f} R={overall['Recall']:.3f} "
          f"F1={overall['F1']:.3f} MOTA={overall['MOTA']:.3f} IDF1={overall['IDF1']:.3f}")
    for c in classes:
        m = per_class[CANON[c]]
        print(f"  {CANON[c]:8s} P={m['Precision']:.3f} R={m['Recall']:.3f} "
              f"F1={m['F1']:.3f} MOTA={m['MOTA']:.3f} (nGT={m['num_gt']})")
    print(f"→ {exp_dir}")
    print("=" * 64)


if __name__ == "__main__":
    main()
