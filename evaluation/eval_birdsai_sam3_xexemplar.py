"""
BIRDSAI SAM3 cross-image (train-set) exemplar MOT eval — the FAIR few-shot row.

Unlike eval_birdsai_sam3_oracle.py (which seeds from *test* GT → oracle), this
script seeds detection from *training* exemplars only, so it is a legitimately
fair open-vocabulary detector+tracker. Design decisions (user-confirmed
2026-06-22):

  * PRIOR = fine / known-species. We are told which species a test video
    contains (true for 15/16 BIRDSAI videos — each is single-species). Only that
    species's exemplar bank(s) run on the video, so fine labels stay clean and we
    isolate the pure recall gain. The mixed video runs both its species.

  * EXEMPLAR SELECTION = transductive. For each (video, class) we gather N_CAND
    candidate exemplars from the TRAIN set, score each by the detection
    confidence it produces on the *test video's own frame 0* (image only, no GT),
    and keep the top-K. This matches the user's "pick the exemplar whose
    prediction has the highest confidence" idea. It peeks at the unlabelled test
    frame 0 → TRANSDUCTIVE (disclose as such; still never uses test GT). The
    pooled bank is K positives (+ M negatives from train background), exactly the
    K-pos/M-neg prototype that the xexemplar probe found best.

  * DETECTION = SAM3 *image* grounding head with a cross-image exemplar bank
    (build_sam3_image_model). The geometry encoder RoI-pools each exemplar's
    appearance from ITS OWN train frame; the encoder/decoder ground those tokens
    on the (different) test frame. Output box coords come from the test image's
    own anchors — the ref box location never leaks into coords.

  * TRACKING = SAM3 *video* SOT propagation (SAM3Tracker, SAM2 mask memory) — the
    path proven to hold tiny thermal objects (giraffe R≈0.93). New objects that
    first appear AFTER frame 0 are caught by PERIODIC RE-DETECTION: every R frames
    we re-run the exemplar detector and spawn new tracks for boxes that match no
    active track (bounded fixed-point loop, so re-found objects don't duplicate).

  * GT = annotations_sam3 (SAM3 box-refined, tighter thermal boxes) for BOTH the
    exemplar boxes AND the eval GT — per the user's "use the new SAM3 labels from
    the start". NOTE: the 3 detector rows + the oracle rows were scored on the
    ORIGINAL annotations, so to drop this row into that table you must re-score
    them on annotations_sam3 (cheap — read their cached predictions.json).

Output mirrors eval_birdsai_detect_track.py / eval_birdsai_sam3_oracle.py
(predictions.json with detections==tracks + test_metrics.json), fine 5-class,
greedy IoU matching, so it is directly comparable once GT is held equal.

    CUDA_VISIBLE_DEVICES=0 python evaluation/eval_birdsai_sam3_xexemplar.py
    CUDA_VISIBLE_DEVICES=0 python evaluation/eval_birdsai_sam3_xexemplar.py \
        --limit-videos 1 --max-frames 64        # quick smoke
"""
from __future__ import annotations

import os
# Reduce CUDA fragmentation (two SAM3 models + many short-lived SOT states across
# 16 videos otherwise fragments VRAM → OOM mid-run). Set before any CUDA init.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import time
from datetime import datetime

import numpy as np
import torch
from PIL import Image

from datasets.birdsai_mot import BIRDSAIMOTDataset
from models.sam3 import SAM3Tracker

BIRDSAI_ROOT = "/data/ESA_DLSTEM_2025/data/wild_animal/BIRDSAI"
OUT_ROOT = "/work/ziwen/experiments"
CANON = {0: "human", 1: "elephant", 2: "giraffe", 3: "lion", 4: "unknown"}

CLIP_LEN = 32          # frames per SAM3 video pass (matches oracle for parity)
REDETECT_R = 8         # re-run exemplar detector every R frames within a clip
MATCH_TOL_PX = 20.0    # base center-dist tol for "this re-det is an existing track"
MOTION_TOL_PX = 50.0   # max center travel of a real object between two keyframes
SEED_CAP = 24          # max frame-0 seeds per clip. SAM3 memory-attention VRAM
                       # scales with (#objects × #memory frames); the human FP
                       # flood otherwise reaches ~88 objects/clip → 28 GB → OOM.
                       # 24 covers every real density (max GT ~7/frame) and trims FP.
PROMOTE_CAP = 8        # max confirmed new objects added per clip via re-detect
N_CAND = 24            # train exemplar candidates scored per (video, class)
K_POS = 4              # positive exemplars pooled into the bank
M_NEG = 2              # negative (train background) exemplars pooled in
DET_SCORE_THR = 0.2    # image-detector score floor (exemplar dets are low-conf)
NMS_IOU = 0.5          # per-image class-agnostic NMS on the exemplar dets
SEED = 0


# ---------------- IoU helpers (mirror eval_birdsai_sam3_oracle) ---------------
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


def _outs_to_per(outs, nframes):
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


def _norm_cxcywh(box_xyxy, W, H):
    x1, y1, x2, y2 = box_xyxy
    cx = (x1 + x2) / 2 / W; cy = (y1 + y2) / 2 / H
    w = (x2 - x1) / W; h = (y2 - y1) / H
    return np.clip([cx, cy, w, h], 0, 1)


# ---------------- SAM3 image grounding head w/ cross-image exemplar bank ------
class XExemplarDetector:
    """Detect a concept on a query frame using exemplar boxes pooled from OTHER
    (training) frames. Reuses 100% of SAM3's image scoring tail; only the
    geometry-encoder output is overridden with a precomputed bank."""

    def __init__(self, score_thr=DET_SCORE_THR, nms_iou=NMS_IOU):
        SAM3Tracker._ensure_pkg_resources_shim()
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        import torchvision
        self._nms = torchvision.ops.nms
        self.nms_iou = nms_iou
        self.model = build_sam3_image_model(load_from_HF=True)
        self.proc = Sam3Processor(self.model, confidence_threshold=score_thr)
        self.dev = self.proc.device

    @torch.inference_mode()
    def token_for(self, ref_pil, box_xyxy, positive):
        """One exemplar box on its own frame → (geo_feats[1,1,C], geo_masks[1,1])."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            W, H = ref_pil.size
            st = self.proc.set_image(ref_pil)
            self.proc.reset_all_prompts(st)
            geo = self.model._get_dummy_prompt()
            cxcywh = _norm_cxcywh(box_xyxy, W, H)
            boxes = torch.tensor(cxcywh, dtype=torch.float32, device=self.dev).view(1, 1, 4)
            labels = torch.tensor([bool(positive)], device=self.dev).view(1, 1)
            geo.append_boxes(boxes, labels)
            _, img_feats, img_pos, sizes = self.model._get_img_feats(
                st["backbone_out"], self.proc.find_stage.img_ids)
            feats, masks = self.model.geometry_encoder(geo, img_feats, sizes, img_pos)
            # forward output order is [box_token, cls_token]; keep box token only
            return feats[:1].clone(), masks[:, :1].clone()

    def build_bank(self, exemplars):
        """exemplars: list of (ref_pil, box_xyxy, positive_bool).
        Returns concatenated (geo_feats[S,1,C], geo_masks[1,S])."""
        feats, masks = [], []
        for ref_pil, box, pos in exemplars:
            f, m = self.token_for(ref_pil, box, pos)
            feats.append(f); masks.append(m)
        return torch.cat(feats, dim=0), torch.cat(masks, dim=1)

    @torch.inference_mode()
    def detect(self, query_pil, bank):
        """Ground the bank on query_pil. Returns (boxes_xyxy[np], scores[np])."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            st = self.proc.set_image(query_pil)
            self.proc.reset_all_prompts(st)
            st = self.proc.set_text_prompt("visual", st)
            geo_feats, geo_masks = bank
            orig = self.model.geometry_encoder.forward
            self.model.geometry_encoder.forward = lambda *a, **k: (geo_feats, geo_masks)
            try:
                st["geometric_prompt"] = self.model._get_dummy_prompt()
                st = self.proc._forward_grounding(st)
            finally:
                self.model.geometry_encoder.forward = orig
            boxes = st["boxes"].float(); scores = st["scores"].float()
        if len(boxes) == 0:
            return boxes.cpu().numpy(), scores.cpu().numpy()
        keep = self._nms(boxes, scores, self.nms_iou)
        return boxes[keep].cpu().numpy(), scores[keep].cpu().numpy()


# ---------------- train exemplar gathering ------------------------------------
def gather_pos(ds_tr, cls, n_cand):
    """Diverse-size candidate train boxes for `cls` (xyxy pixel)."""
    cands = []  # (diag, video, fid, box)
    for v in ds_tr.videos:
        for fid in v.frame_ids[::12]:
            ann = ds_tr._load_annotations(v, fid)
            for b, l in zip(ann["boxes"], ann["labels"]):
                if int(l) != cls:
                    continue
                b = np.asarray(b, np.float32)
                cands.append((float(np.hypot(b[2] - b[0], b[3] - b[1])), v, fid, b))
        if len(cands) > 600:
            break
    if not cands:
        return []
    cands.sort(key=lambda t: -t[0])
    idx = np.linspace(0, min(len(cands), 200) - 1, min(n_cand, len(cands))).astype(int)
    return [cands[i] for i in sorted(set(idx))]


def gather_neg(ds_tr, m, rng):
    """M background boxes from random train frames (no GT overlap)."""
    out = []
    vids = list(ds_tr.videos)
    tries = 0
    while len(out) < m and tries < 4000:
        tries += 1
        v = vids[rng.integers(len(vids))]
        fid = int(v.frame_ids[rng.integers(len(v.frame_ids))])
        ann = ds_tr._load_annotations(v, fid)
        gt = np.asarray(ann["boxes"], np.float32).reshape(-1, 4)
        H, W = ds_tr._load_frame(v, fid).shape[:2]
        s = int(rng.integers(14, 30))
        x = int(rng.integers(0, max(1, W - s))); y = int(rng.integers(0, max(1, H - s)))
        box = np.array([x, y, x + s, y + s], np.float32)
        if len(gt) and (iou_matrix(box[None], gt) > 0.05).any():
            continue
        out.append((Image.fromarray(ds_tr._load_frame(v, fid)), box, False))
    return out


def select_banks(detector, ds_tr, present_classes, frame0_pil, rng, verbose=True):
    """Transductive bank selection: rank train candidates by detection
    confidence on the test video's frame 0, pool top-K (+M neg) per class."""
    banks = {}
    for c in sorted(present_classes):
        cand = gather_pos(ds_tr, c, N_CAND)
        if not cand:
            if verbose:
                print(f"    [select] {CANON[c]}: no train exemplar, skip", flush=True)
            continue
        scored = []
        for diag, vv, ff, bx in cand:
            ref_pil = Image.fromarray(ds_tr._load_frame(vv, ff))
            tok = detector.build_bank([(ref_pil, bx, True)])
            _, scs = detector.detect(frame0_pil, tok)
            scored.append((float(scs.max()) if len(scs) else 0.0, diag, ref_pil, bx))
        scored.sort(key=lambda t: -t[0])
        top = scored[:K_POS]
        pos_ex = [(rp, bx, True) for _, _, rp, bx in top]
        neg_ex = gather_neg(ds_tr, M_NEG, rng)
        banks[c] = detector.build_bank(pos_ex + neg_ex)
        if verbose:
            confs = ",".join(f"{s:.2f}" for s, *_ in top)
            print(f"    [select] {CANON[c]}: {len(cand)} cand, top{K_POS} f0-conf=[{confs}]",
                  flush=True)
    return banks


# ---------------- per-clip producer: detect@frame0 + SOT + periodic re-detect -
def clip_xexemplar_track(detector, sot, frames, banks, gid_offset,
                         redetect_r=REDETECT_R, det_thr=DET_SCORE_THR,
                         seed_cap=SEED_CAP, promote_cap=PROMOTE_CAP):
    """Returns (per[j]=dict, next_gid). Detects with the prebuilt per-class
    exemplar banks, propagates with SAM3 SOT, and discovers later-appearing
    objects via a bounded periodic-re-detect fixed point. redetect_r<=0 disables
    periodic re-detection (frame-0 SOT only)."""
    nf = len(frames)
    pils = {}                      # frame_idx -> PIL (lazy, only keyframes)
    det_cache = {}                 # frame_idx -> list[(box_xyxy, label, score)]

    def detect_at(k):
        if k in det_cache:
            return det_cache[k]
        if k not in pils:
            pils[k] = Image.fromarray(frames[k])
        out = []
        for c, bank in banks.items():
            bx, sc = detector.detect(pils[k], bank)
            for b, s in zip(bx, sc):
                if float(s) >= det_thr:
                    out.append((b.astype(np.float32), int(c), float(s)))
        det_cache[k] = out
        return out

    det0 = detect_at(0)
    if not det0:
        empty = {"boxes": np.zeros((0, 4), np.float32), "labels": np.zeros(0, int),
                 "scores": np.zeros(0, np.float32), "track_ids": np.zeros(0, int)}
        return {j: dict(empty) for j in range(nf)}, gid_offset

    keyframes = list(range(redetect_r, nf, redetect_r)) if redetect_r > 0 else []

    def _centers(boxes):
        if len(boxes) == 0:
            return np.zeros((0, 2), np.float32)
        return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2,
                         (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)

    def _propagate(seeds):
        """seeds: list[(frame_idx, box, label)] -> {j: per-frame dict}."""
        sot.init_video(frames)
        by_frame: dict[int, list] = {}
        for i, (fi, box, lab) in enumerate(seeds):
            by_frame.setdefault(fi, []).append((box, lab, gid_offset + i))
        for fi, items in by_frame.items():
            sot.add_prompts(fi, np.stack([it[0] for it in items]),
                            np.array([it[1] for it in items]),
                            obj_ids=[it[2] for it in items])
        out = _outs_to_per(sot.propagate(), nf)
        sot.reset_state()
        torch.cuda.empty_cache()
        return out

    # --- pass 1: frame-0 detections seed the tracks (capped by score) ---
    det0 = sorted(det0, key=lambda t: -t[2])[:seed_cap]
    accepted = [(0, b, lab) for b, lab, _ in det0]   # (frame_idx, box, label)
    per = _propagate(accepted)
    if not keyframes:
        return per, gid_offset + len(accepted)

    # --- periodic re-detect: collect detections NOT already tracked ---
    # Generous center-distance gate (tolerates SOT mask drift) so an existing
    # track / FP hot-spot is NOT re-spawned as a duplicate — that duplication,
    # not genuinely-new objects, was what flooded FPs in the naive version.
    def unmatched_at(k):
        tcent = _centers(per[k]["boxes"])
        out = []
        for box, lab, _ in detect_at(k):
            c = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], np.float32)
            tol = max(float(np.hypot(box[2] - box[0], box[3] - box[1])), MATCH_TOL_PX)
            if len(tcent) and (np.linalg.norm(tcent - c, axis=1) <= tol).any():
                continue                          # already tracked
            out.append((k, box, lab, c))
        return out

    cand = [unmatched_at(k) for k in keyframes]

    # --- persistence confirmation: a new object must survive to the NEXT
    # keyframe (within motion tolerance) to be promoted — transient one-frame
    # FP blobs are dropped. Greedy dedup keeps ONE seed (earliest) per object. ---
    promoted = []   # (frame_idx, box, label, center)
    for i in range(len(keyframes) - 1):
        nxt = cand[i + 1]
        ncent = np.stack([c for *_, c in nxt]) if nxt else np.zeros((0, 2), np.float32)
        if not len(ncent):
            continue
        pcent = np.stack([c for *_, c in promoted]) if promoted else np.zeros((0, 2), np.float32)
        for (k, box, lab, c) in cand[i]:
            if (np.linalg.norm(ncent - c, axis=1) > MOTION_TOL_PX).all():
                continue                          # not corroborated next keyframe
            if len(pcent) and (np.linalg.norm(pcent - c, axis=1) <= MOTION_TOL_PX).any():
                continue                          # already promoted nearby (earlier kf)
            promoted.append((k, box, lab, c))
            pcent = np.stack([c for *_, c in promoted])

    if not promoted:
        return per, gid_offset + len(accepted)

    # --- pass 2: re-propagate with frame-0 seeds + confirmed new objects ---
    accepted += [(k, box, lab) for (k, box, lab, _) in promoted[:promote_cap]]
    per = _propagate(accepted)
    return per, gid_offset + len(accepted)


# ---------------- scoring (shared by fresh + resumed videos) ------------------
def score_video(det, trk, frames_out, video, ds, classes, iou_thresh):
    """Accumulate per-class det/trk counters from a video's frames_out (fid ->
    {tracks: {boxes,labels,track_ids}}). Identical to the original inline scoring
    (clips are consecutive fid ranges → sorted-fid order == clip order, so IDsw
    matches). Returns this video's (tp, fp, fn, ngt, idsw) for the progress line."""
    last = {c: {} for c in classes}
    vtp = vfp = vfn = vng = vidsw = 0
    for fid in sorted(int(f) for f in frames_out):
        ann = ds._load_annotations(video, fid)
        gtb = np.asarray(ann["boxes"], np.float32).reshape(-1, 4)
        gtl = np.asarray(ann["labels"], np.int64).reshape(-1)
        gtid = np.asarray(ann["track_ids"], np.int64).reshape(-1)
        tr = frames_out[str(fid)]["tracks"]
        pb = np.asarray(tr["boxes"], np.float32).reshape(-1, 4)
        pl = np.asarray(tr["labels"], np.int64).reshape(-1)
        pid = np.asarray(tr["track_ids"], np.int64).reshape(-1)
        for c in classes:
            gm = gtl == c; pm = pl == c
            gb = gtb[gm]; gi = gtid[gm]; cb = pb[pm]; ci = pid[pm]
            ms = greedy_match(gb, cb, iou_thresh)
            tp = len(ms)
            det[c]["tp"] += tp; det[c]["fp"] += len(cb) - tp; det[c]["fn"] += len(gb) - tp
            trk[c]["tp"] += tp; trk[c]["fp"] += len(cb) - tp; trk[c]["fn"] += len(gb) - tp
            trk[c]["ngt"] += len(gb)
            vtp += tp; vfp += len(cb) - tp; vfn += len(gb) - tp; vng += len(gb)
            for r, cc in ms:
                g = int(gi[r]); pp = int(ci[cc])
                prev = last[c].get(g)
                if prev is not None and prev != pp:
                    trk[c]["idsw"] += 1; vidsw += 1
                last[c][g] = pp
    return vtp, vfp, vfn, vng, vidsw


# ---------------- main --------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iou-thresh", type=float, default=0.5)
    ap.add_argument("--annotations", default="annotations_sam3",
                    help="annotation subdir for BOTH exemplars and eval GT")
    ap.add_argument("--limit-videos", type=int, default=0, help="0=all (debug)")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="cap frames per video (debug; 0=all)")
    ap.add_argument("--video-ids", default="",
                    help="comma-separated video_ids to restrict to (debug)")
    ap.add_argument("--redetect-r", type=int, default=REDETECT_R,
                    help="periodic re-detect interval; <=0 disables (frame-0 SOT only)")
    ap.add_argument("--det-thr", type=float, default=DET_SCORE_THR,
                    help="seed/detection score floor")
    ap.add_argument("--seed-cap", type=int, default=SEED_CAP,
                    help="max frame-0 seeds per clip (VRAM bound)")
    ap.add_argument("--promote-cap", type=int, default=PROMOTE_CAP,
                    help="max confirmed new objects per clip")
    ap.add_argument("--resume-dir", default="",
                    help="existing exp dir: videos with pred_<id>.json are reused, "
                         "rest are (re)computed → crash/OOM-resumable")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cmap = {v: k for k, v in CANON.items()}
    ds = BIRDSAIMOTDataset(root=BIRDSAI_ROOT, split="test", granularity="fine",
                           annotations_dirname=args.annotations, class_map=cmap)
    ds_tr = BIRDSAIMOTDataset(root=BIRDSAI_ROOT, split="train", granularity="fine",
                              annotations_dirname=args.annotations, class_map=cmap)

    if args.resume_dir:
        exp_dir = Path(args.resume_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = (args.tag + "_") if args.tag else ""
        exp_dir = Path(OUT_ROOT) / f"sam3_birdsai_xexemplar_{tag}{ts}"
    (exp_dir / "mot_format").mkdir(parents=True, exist_ok=True)

    classes = sorted(CANON)
    det = {c: {"tp": 0, "fp": 0, "fn": 0} for c in classes}
    trk = {c: {"tp": 0, "fp": 0, "fn": 0, "idsw": 0, "ngt": 0} for c in classes}
    predictions = {"model": "sam3_xexemplar", "dataset": "BIRDSAI", "split": "test",
                   "annotations": args.annotations, "class_names": CANON, "videos": {}}
    t0 = time.perf_counter()

    def metrics_from(d, t):
        prec = d["tp"] / max(d["tp"] + d["fp"], 1)
        rec = d["tp"] / max(d["tp"] + d["fn"], 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        mota = 1.0 - (t["fp"] + t["fn"] + t["idsw"]) / max(t["ngt"], 1)
        idp = t["tp"] / max(t["tp"] + t["fp"], 1); idr = t["tp"] / max(t["tp"] + t["fn"], 1)
        idf1 = 2 * idp * idr / max(idp + idr, 1e-9)
        return {"Precision": prec, "Recall": rec, "F1": f1, "MOTA": mota,
                "IDF1": idf1, "IDsw": t["idsw"], "num_gt": t["ngt"]}

    def build_summary(n_done):
        per_class = {CANON[c]: metrics_from(det[c], trk[c]) for c in classes}
        pd_ = {k: sum(det[c][k] for c in classes) for k in ("tp", "fp", "fn")}
        pt_ = {k: sum(trk[c][k] for c in classes) for k in ("tp", "fp", "fn", "idsw", "ngt")}
        return {"model": "sam3_xexemplar", "dataset": "BIRDSAI", "split": "test",
                "oracle": False, "transductive": True, "prior": "fine_species",
                "annotations": args.annotations, "iou_thresh": args.iou_thresh,
                "clip_len": CLIP_LEN, "redetect_r": args.redetect_r,
                "k_pos": K_POS, "m_neg": M_NEG, "n_cand": N_CAND,
                "det_score_thr": args.det_thr, "seed_cap": args.seed_cap,
                "promote_cap": args.promote_cap, "videos_done": n_done,
                "total_videos": None, "total_time_s": time.perf_counter() - t0,
                "overall": metrics_from(pd_, pt_), "per_class": per_class}

    videos = ds.videos
    if args.video_ids:
        want = set(args.video_ids.split(","))
        videos = [v for v in videos if v.video_id in want]
    if args.limit_videos:
        videos = videos[:args.limit_videos]

    # lazily build the two SAM3 models only if some video actually needs computing
    models = {}

    def ensure_models():
        if not models:
            print("Building SAM3 image detector (exemplar grounding)...", flush=True)
            models["det"] = XExemplarDetector()
            print("Building SAM3 video tracker (SOT propagation)...", flush=True)
            models["sot"] = SAM3Tracker()
        return models["det"], models["sot"]

    for v_idx, video in enumerate(videos, 1):
        predf = exp_dir / f"pred_{video.video_id}.json"
        fids = video.frame_ids[:args.max_frames] if args.max_frames else video.frame_ids

        # --- resume: reuse a previously-computed video ---
        if predf.exists():
            data = json.load(open(predf))
            frames_out = data["frames"]
            predictions["videos"][video.video_id] = {
                "image_dir": data["image_dir"], "frames": frames_out}
            vtp, vfp, vfn, vng, vidsw = score_video(
                det, trk, frames_out, video, ds, classes, args.iou_thresh)
            prec = vtp / max(vtp + vfp, 1); rec = vtp / max(vtp + vfn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            print(f"[{v_idx}/{len(videos)}] {video.video_id} CACHED  "
                  f"P={prec:.3f} R={rec:.3f} F1={f1:.3f} IDsw={vidsw}", flush=True)
            continue

        # --- fine prior: which species are present in this video (from GT) ---
        present = set()
        for fid in fids:
            for l in ds._load_annotations(video, fid)["labels"]:
                present.add(int(l))
        print(f"[{v_idx}/{len(videos)}] {video.video_id} nf={len(fids)} "
              f"present={[CANON[c] for c in sorted(present)]}", flush=True)

        detector, sot = ensure_models()

        # --- transductive bank selection on the video's frame 0 ---
        # per-video rng → negatives are deterministic regardless of video order
        vrng = np.random.default_rng(SEED + (int(video.video_id.split("_")[0]) % 100000))
        frame0 = Image.fromarray(ds._load_frame(video, fids[0]))
        banks = select_banks(detector, ds_tr, present, frame0, vrng)
        if not banks:
            print("    no banks → empty predictions for this video", flush=True)

        gid = 1
        frames_out = {}
        mot_lines = []
        for s in range(0, len(fids), CLIP_LEN):
            cf = fids[s:s + CLIP_LEN]
            frames = [ds._load_frame(video, f) for f in cf]
            if banks:
                per, gid = clip_xexemplar_track(detector, sot, frames, banks, gid,
                                                redetect_r=args.redetect_r,
                                                det_thr=args.det_thr,
                                                seed_cap=args.seed_cap,
                                                promote_cap=args.promote_cap)
            else:
                per = {j: {"boxes": np.zeros((0, 4), np.float32), "labels": np.zeros(0, int),
                           "scores": np.zeros(0, np.float32), "track_ids": np.zeros(0, int)}
                       for j in range(len(frames))}

            for j, fid in enumerate(cf):
                p = per[j]
                pb, pl, ps, pid = p["boxes"], p["labels"], p["scores"], p["track_ids"]
                fb, fs, flab, fids_ = [], [], [], []
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

        # write per-video artifacts FIRST (crash-safety / resume) then score
        with open(exp_dir / "mot_format" / f"{video.video_id}.txt", "w") as f:
            f.write("\n".join(mot_lines))
        img_dir = str(ds._img_dir_cache[video.video_id])
        with open(predf, "w") as f:
            json.dump({"image_dir": img_dir, "frames": frames_out}, f)
        predictions["videos"][video.video_id] = {"image_dir": img_dir, "frames": frames_out}

        vtp, vfp, vfn, vng, vidsw = score_video(
            det, trk, frames_out, video, ds, classes, args.iou_thresh)
        prec = vtp / max(vtp + vfp, 1); rec = vtp / max(vtp + vfn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        mota = 1.0 - (vfp + vfn + vidsw) / max(vng, 1)
        print(f"    P={prec:.3f} R={rec:.3f} F1={f1:.3f} MOTA={mota:.3f} IDsw={vidsw}",
              flush=True)
        torch.cuda.empty_cache()
        with open(exp_dir / "test_metrics.json", "w") as f:
            json.dump(build_summary(v_idx), f, indent=2)

    summary = build_summary(len(videos))
    summary["total_videos"] = len(videos)
    overall = summary["overall"]; per_class = summary["per_class"]
    with open(exp_dir / "test_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(exp_dir / "predictions.json", "w") as f:
        json.dump(predictions, f)

    print("\n" + "=" * 64)
    print(f"OVERALL [xexemplar/fine/transductive]  P={overall['Precision']:.3f} "
          f"R={overall['Recall']:.3f} F1={overall['F1']:.3f} "
          f"MOTA={overall['MOTA']:.3f} IDF1={overall['IDF1']:.3f}")
    for c in classes:
        m = per_class[CANON[c]]
        print(f"  {CANON[c]:8s} P={m['Precision']:.3f} R={m['Recall']:.3f} "
              f"F1={m['F1']:.3f} MOTA={m['MOTA']:.3f} (nGT={m['num_gt']})")
    print(f"→ {exp_dir}")
    print("=" * 64)


if __name__ == "__main__":
    main()
