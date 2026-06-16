"""Evaluate a trained TGraM (canonical `tgrammbseg`: MobileNetV3-Small + graph
spatiotemporal reasoning, one-shot JDT) model on a car MOT test split.

TGraM is the remote-sensing-specialized JDT row; like FairMOT it does its OWN
detection + ReID every frame and so must be run frame-by-frame and scored
against GT. This script is the TGraM sibling of ``eval_fairmot.py`` and writes
the EXACT same output contract so the downstream HOTA / master-CSV merge is
unchanged:

    test_metrics.json          micro + macro Pr/Re/F1 (det), MOTA/IDF1/IDsw
    per_video_metrics.json     same metrics per video
    mot_format/<video>.txt     MOTChallenge rows: frame,id,x,y,w,h,conf,-1,-1,-1

Two TGraM specifics vs FairMOT:
  * The model is TEMPORAL — it consumes a clip ``[1, num_frames, 3, H, W]``
    (current frame + ``num_frames-1`` previous frames). We buffer previous
    frames per video and pad with the current frame at the sequence start
    (mirrors upstream ``datasets.LoadImages``).
  * arch ``tgrammbseg`` needs DCNv2 — provided by the torchvision
    ``deform_conv2d`` shim at ``TGraM/src/lib/models/networks/dcn_v2.py``.

Predictions are mapped back to ORIGINAL pixel coords by the tracker's
ctdet_post_process (CenterNet inverse affine), so they score against unmodified
GT exactly like the HiEUM / TBD / FairMOT rows. The pedestrian min_box_area /
vertical (w/h>1.6) filters are NOT applied (they delete tiny / wide cars).

Usage::

    python eval_tgram.py --config configs/MOT/tgram_rscardata.yaml
    python eval_tgram.py --config configs/MOT/tgram_satmtb.yaml --checkpoint <ckpt>
"""
from __future__ import annotations

# --- repo root on path so top-level modules (transforms, obb_utils, ...) import ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import os
import sys
import time
import types
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace


def _install_numba_stub():
    """TGraM's tracker does `from numba import jit`, but every `@jit` is
    commented out — numba is imported and never used. Avoid adding the dep
    (and its numpy-version constraints) by injecting a passthrough stub."""
    if "numba" in sys.modules:
        return
    m = types.ModuleType("numba")

    def jit(*a, **k):
        if len(a) == 1 and callable(a[0]) and not k:
            return a[0]                      # bare @jit
        return lambda f: f                   # @jit(...) factory
    m.jit = jit
    sys.modules["numba"] = m


_install_numba_stub()

import cv2
import numpy as np
import torch
import yaml

# Parent-repo dataset classes FIRST, so all of our dataset code is imported and
# resolved before we juggle sys.path for TGraM's (colliding) top-level names.
from datasets.airmot import AIRMOTDataset
from datasets.rscardata import RsCarDataset
from datasets.satmtb import SATMTBDataset
from datasets.sdmcar import SDMCarDataset
from datasets.viso import VISODataset

_REPO = Path(__file__).resolve().parent
_TGRAM_LIB = str(_REPO / "TGraM" / "src" / "lib")
_TGRAM_NETWORKS = str(_REPO / "TGraM" / "src" / "lib" / "models" / "networks")

# TGraM's src/lib exposes top-level packages (models, tracker, tracking_utils,
# utils, opts, lib) that collide with this repo's. We import our datasets above,
# then purge those names and prepend TGraM's lib so JDETracker's internal
# absolute imports (`from models.model import ...`) bind to TGraM's copies.
_TGRAM_COLLIDING = {"models", "tracker", "tracking_utils", "utils", "opts", "lib"}


def _activate_tgram_lib():
    for key in [k for k in list(sys.modules)
                if k.split(".")[0] in _TGRAM_COLLIDING]:
        del sys.modules[key]
    # Drop the repo root: our `models/`/`datasets/` packages would shadow
    # TGraM/src/lib/models even when the latter is prepended. Our dataset
    # classes are already imported (held by reference) so we no longer need it.
    _repo = str(_REPO)
    sys.path[:] = [p for p in sys.path if p not in ("", ".", _repo)]
    for p in (_TGRAM_LIB, _TGRAM_NETWORKS):   # networks dir -> `from dcn_v2 import DCN`
        if p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, _TGRAM_NETWORKS)
    sys.path.insert(0, _TGRAM_LIB)


_DATASET_TABLE = {
    "rscardata": (RsCarDataset, "/data/ESA_DLSTEM_2025/data/trafic/RsCarData", {}),
    "satmtb":    (SATMTBDataset, "/data/ESA_DLSTEM_2025/data/trafic/SAT-MTB",
                  {"task": "mot"}),
    "sdmcar":    (SDMCarDataset, "/data/ESA_DLSTEM_2025/data/trafic/SDM-Car", {}),
    "airmot":    (AIRMOTDataset, "/data/ESA_DLSTEM_2025/data/trafic/AIR-MOT-100", {}),
    "viso_no_car": (VISODataset, "/data/ESA_DLSTEM_2025/data/trafic/VISO",
                    {"categories": ["plane", "ship", "train"]}),
}

# 4-class union model: per-dataset class maps (every category non-negative so
# _load_annotations surfaces ALL GT — HOTA is class-agnostic, see compute_hota
# `_SAM3_CLASS_MAPS`). Integer values are irrelevant downstream (TrackEval pools
# every track as one foreground class); only "present vs dropped" matters.
_ALLCLASS_MAPS = {
    "rscardata":   {"car": 0},
    "satmtb":      {"airplane": 0, "car": 1, "ship": 2, "train": 3},
    "sdmcar":      {"car": 0},
    "airmot":      {"airplane": 0, "ship": 1},
    "viso_no_car": {"plane": 0, "ship": 1, "train": 2},
}

# Per-dataset eval input size (the native-res training bucket from train_union).
_ALLCLASS_INPUT = {
    "rscardata": (1024, 1024), "satmtb": (1024, 1024), "sdmcar": (1920, 1088),
    "airmot": (1920, 1088), "viso_no_car": (1472, 768),
}


def _build_dataset(name: str, split: str = "test", all_class: bool = False):
    cls, root, extra = _DATASET_TABLE[name]
    kwargs = dict(extra)
    if all_class:
        cmap = _ALLCLASS_MAPS[name]
    else:
        cmap = {"car": 0}
        if name == "satmtb":
            kwargs["categories"] = ["car"]     # car-only protocol (legacy rows)
    # AIRMOT / VISO are MOT-only loaders — they don't accept mode="detection".
    if cls in (AIRMOTDataset, VISODataset):
        return cls(root=root, split=split, class_map=cmap, **kwargs)
    return cls(root=root, split=split, mode="detection", class_map=cmap, **kwargs)


def _safe_video_id(video_id: str) -> str:
    return video_id.replace("/", "_")


# ---------------------------------------------------------------- preprocessing

def _letterbox(img, height, width, color=(127.5, 127.5, 127.5)):
    """TGraM's letterbox (datasets/dataset/tgram.py), inlined. Channel-order
    agnostic (symmetric pad colour)."""
    shape = img.shape[:2]                                       # (h, w)
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # (w, h)
    dw = (width - new_shape[0]) / 2
    dh = (height - new_shape[1]) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img


def _to_chw(frame_rgb, input_h, input_w):
    """letterbox -> RGB CHW -> /255 (matches TGraM training: RGB ToTensor)."""
    lb = _letterbox(frame_rgb, input_h, input_w)
    return np.ascontiguousarray(lb.transpose(2, 0, 1), dtype=np.float32) / 255.0


def _build_clip(cur_chw, prev_chw, num_frames, device):
    """Stack the current frame + (num_frames-1) previous frames into a
    [1, num_frames, 3, H, W] blob. `prev_chw` is a rolling list ending at t-1
    (most recent last). Missing history (sequence start) pads with the current
    frame, mirroring upstream LoadImages."""
    clip = [cur_chw]
    for i in range(1, num_frames):
        clip.append(prev_chw[-i] if len(prev_chw) >= i else cur_chw)
    blob = np.stack(clip, axis=0)                               # [nf, 3, H, W]
    return torch.from_numpy(blob).unsqueeze(0).to(device)       # [1, nf, 3, H, W]


# ----------------------------------------------------------------- GT + matching
# (mirror eval_fairmot.py / eval_tracktrack.py exactly so all rows score
#  identically; only the prediction source differs.)

def _gt_per_frame(dataset, video):
    out = {}
    for fid in video.frame_ids:
        ann = dataset._load_annotations(video, fid)
        out[fid] = {
            "boxes":     np.asarray(ann["boxes"], dtype=np.float32),
            "track_ids": np.asarray(ann["track_ids"], dtype=np.int64),
        }
    return out


def _iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def _centroid_dist(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    cx_a = (a[:, 0] + a[:, 2]) * 0.5; cy_a = (a[:, 1] + a[:, 3]) * 0.5
    cx_b = (b[:, 0] + b[:, 2]) * 0.5; cy_b = (b[:, 1] + b[:, 3]) * 0.5
    dx = cx_a[:, None] - cx_b[None, :]; dy = cy_a[:, None] - cy_b[None, :]
    return np.sqrt(dx * dx + dy * dy).astype(np.float32)


def _greedy_match(gt_boxes, pred_boxes, metric, iou_thr, dist_thr):
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return []
    if metric == "centroid":
        score = _centroid_dist(gt_boxes, pred_boxes)
        accept = score <= dist_thr
        descending = False
    else:
        score = _iou_matrix(gt_boxes, pred_boxes)
        accept = score >= iou_thr
        descending = True
    rows, cols = np.where(accept)
    if len(rows) == 0:
        return []
    order = score[rows, cols].argsort()
    if descending:
        order = order[::-1]
    rows, cols = rows[order], cols[order]
    mg, mp, matches = set(), set(), []
    for r, c in zip(rows.tolist(), cols.tolist()):
        if r in mg or c in mp:
            continue
        mg.add(r); mp.add(c); matches.append((r, c))
    return matches


# ----------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", default=None,
                    help="override cfg['dataset'] (union model -> run per split)")
    ap.add_argument("--checkpoint", default=None, help="override cfg['checkpoint']")
    ap.add_argument("--all-class", action="store_true",
                    help="4-class union model: surface all GT classes + pool "
                         "every predicted class as one foreground track "
                         "(class-agnostic HOTA, matches compute_hota).")
    ap.add_argument("--gt-oracle", action="store_true",
                    help="Experiment 2 (association vs size): feed GT boxes as "
                         "detections, sample the model's ReID at GT centres, run "
                         "unchanged association. Isolates association from "
                         "detection. Implies --all-class (uses the 4-class union "
                         "model + all GT classes for full size range).")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    gt_oracle   = args.gt_oracle
    all_class   = args.all_class or gt_oracle or int(cfg.get("num_classes", 1)) == 4
    dataset_key = args.dataset or cfg["dataset"]
    checkpoint  = args.checkpoint or cfg["checkpoint"]
    arch        = cfg.get("arch", "tgrammbseg")
    if all_class:                                   # input size is per-dataset bucket
        input_w, input_h = _ALLCLASS_INPUT[dataset_key]
    else:
        input_w = int(cfg.get("input_w", 1024))
        input_h = int(cfg.get("input_h", 1024))
    num_frames  = int(cfg.get("num_frames", 3))
    conf_thres  = float(cfg.get("conf_thres", 0.4))
    track_buffer = int(cfg.get("track_buffer", 30))
    frame_rate  = int(cfg.get("frame_rate", 30))
    K           = int(cfg.get("K", 500))
    metric      = cfg.get("match_metric", "centroid")
    iou_thr     = float(cfg.get("iou_thresh", 0.5))
    dist_thr    = float(cfg.get("centroid_dist_thresh", 5.0))
    min_box_area = float(cfg.get("min_box_area", 0.0))   # 0 => keep all (tiny cars)

    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    exp_root = os.environ.get("EXPERIMENT_ROOT", "/work/ziwen/experiments")
    _prefix = "tgram_oracle" if gt_oracle else ("tgram_all" if all_class else "tgram")
    run_name = f"{_prefix}_{dataset_key}"
    experiment_dir = Path(f"{exp_root}/{run_name}_{datetime.now():%Y%m%d_%H%M%S}")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    mot_dir = experiment_dir / "mot_format"; mot_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print(f"TGraM eval on {dataset_key}  (test split, records-only)")
    print(f"ckpt:   {checkpoint}")
    print(f"input:  {input_w}x{input_h}  num_frames={num_frames}  "
          f"conf_thres={conf_thres}  K={K}")
    print(f"match:  {metric} (iou>={iou_thr} / dist<={dist_thr}px)")
    print(f"output: {experiment_dir}")
    print("=" * 60)

    dataset = _build_dataset(dataset_key, all_class=all_class)

    # ---- build the TGraM tracker (after our datasets are imported) ----
    _activate_tgram_lib()
    from tracker.multitracker import JDETracker  # noqa: E402

    device_ok = torch.cuda.is_available()
    opt = SimpleNamespace(
        gpus=[0] if device_ok else [-1],
        arch=arch,
        # hm head must match the checkpoint (4 channels for the 4-class union
        # model); num_classes stays 1 so merge_outputs reads only the pooled
        # key-1 bucket the JDETrackerPooled.post_process produces below.
        heads={"hm": 4 if all_class else 1, "wh": 4, "id": 128, "reg": 2},
        head_conv=256,
        load_model=checkpoint,
        conf_thres=conf_thres,
        track_buffer=track_buffer,
        K=K,
        mean=[0.408, 0.447, 0.470], std=[0.289, 0.274, 0.278],  # vestigial (unused)
        num_classes=1,
        down_ratio=4,
        reg_offset=True,
        ltrb=True,
    )
    device = torch.device("cuda" if device_ok else "cpu")

    if all_class and not gt_oracle:
        # Class-agnostic pooled tracking — see eval_fairmot.py for the rationale.
        # Override post_process to inverse-affine ALL K decoded boxes (order
        # preserved → id_feature alignment intact) and pool them under key 1, so
        # car/plane/ship/train all flow into one association pool (the single-
        # foreground protocol compute_hota scores against).
        from utils.image import transform_preds  # noqa: E402  (TGraM's copy)

        class JDETrackerPooled(JDETracker):
            def post_process(self, dets, meta):
                d = dets.detach().cpu().numpy()
                d = d.reshape(-1, d.shape[2])               # [K, 6] bbox4,score,cls
                d[:, 0:2] = transform_preds(
                    d[:, 0:2], meta["c"], meta["s"],
                    (meta["out_width"], meta["out_height"]))
                d[:, 2:4] = transform_preds(
                    d[:, 2:4], meta["c"], meta["s"],
                    (meta["out_width"], meta["out_height"]))
                return {1: d[:, :5].astype(np.float32)}

        TrackerCls = JDETrackerPooled
    else:
        TrackerCls = JDETracker

    det_tp = det_fp = det_fn = 0
    tr_tp = tr_fp = tr_fn = id_switches = total_gt = 0
    per_video = {}
    model_mb = os.path.getsize(checkpoint) / 1e6
    t0 = time.perf_counter()
    n_frames_total = 0

    for v_idx, video in enumerate(dataset.videos, 1):
        gt = _gt_per_frame(dataset, video)
        # one tracker per sequence -> per-sequence track ids (reloads model each
        # video; cheap relative to inference and keeps state fully isolated).
        tracker = TrackerCls(opt, frame_rate=frame_rate)
        last_gt_to_pred = {}
        prev_chw = []                                  # rolling previous-frame buffer

        vd_tp = vd_fp = vd_fn = vt_tp = vt_fp = vt_fn = v_idsw = v_gt = 0
        mot_lines = []

        for fid in video.frame_ids:
            frame_rgb = dataset._load_frame(video, fid)
            cur_chw = _to_chw(frame_rgb, input_h, input_w)
            im_blob = _build_clip(cur_chw, prev_chw, num_frames, device)
            if gt_oracle:
                online = tracker.update_oracle(im_blob, frame_rgb, gt[fid]["boxes"])
            else:
                online = tracker.update(im_blob, frame_rgb)
            n_frames_total += 1
            prev_chw.append(cur_chw)
            if len(prev_chw) > num_frames:             # only need last num_frames-1
                prev_chw.pop(0)

            pb, pid, psc = [], [], []
            for t in online:
                w, h = float(t.tlwh[2]), float(t.tlwh[3])
                if w * h < min_box_area:
                    continue
                x1, y1 = float(t.tlwh[0]), float(t.tlwh[1])
                pb.append([x1, y1, x1 + w, y1 + h]); pid.append(int(t.track_id))
                psc.append(float(t.score))
            pred_boxes = np.asarray(pb, dtype=np.float32).reshape(-1, 4)
            pred_ids = np.asarray(pid, dtype=np.int64)
            pred_scores = np.asarray(psc, dtype=np.float32)

            gt_boxes, gt_tids = gt[fid]["boxes"], gt[fid]["track_ids"]
            matches = _greedy_match(gt_boxes, pred_boxes, metric, iou_thr, dist_thr)
            tp_d = len(matches)
            vd_tp += tp_d; vd_fp += len(pred_boxes) - tp_d; vd_fn += len(gt_boxes) - tp_d
            v_gt += len(gt_boxes)
            for r, c in matches:
                gid, prid = int(gt_tids[r]), int(pred_ids[c])
                prev = last_gt_to_pred.get(gid)
                if prev is not None and prev != prid:
                    v_idsw += 1
                last_gt_to_pred[gid] = prid; vt_tp += 1
            vt_fn += len(gt_boxes) - tp_d; vt_fp += len(pred_boxes) - tp_d

            for j in range(len(pred_boxes)):
                x1, y1, x2, y2 = pred_boxes[j]
                mot_lines.append(
                    f"{int(fid)},{int(pred_ids[j])},{x1:.2f},{y1:.2f},"
                    f"{x2 - x1:.2f},{y2 - y1:.2f},{float(pred_scores[j]):.4f},-1,-1,-1")

        (mot_dir / f"{_safe_video_id(video.video_id)}.txt").write_text("\n".join(mot_lines))

        v_prec = vd_tp / max(vd_tp + vd_fp, 1)
        v_rec = vd_tp / max(vd_tp + vd_fn, 1)
        v_f1 = 2 * v_prec * v_rec / max(v_prec + v_rec, 1e-9)
        v_mota = 1.0 - (vt_fp + vt_fn + v_idsw) / max(v_gt, 1)
        v_idp = vt_tp / max(vt_tp + vt_fp, 1); v_idr = vt_tp / max(vt_tp + vt_fn, 1)
        v_idf1 = 2 * v_idp * v_idr / max(v_idp + v_idr, 1e-9)
        per_video[video.video_id] = {
            "det": {"tp": vd_tp, "fp": vd_fp, "fn": vd_fn,
                    "Precision": v_prec, "Recall": v_rec, "F1": v_f1},
            "track": {"tp": vt_tp, "fp": vt_fp, "fn": vt_fn, "id_switches": v_idsw,
                      "num_gt": v_gt, "MOTA": v_mota, "IDF1": v_idf1},
        }
        det_tp += vd_tp; det_fp += vd_fp; det_fn += vd_fn
        tr_tp += vt_tp; tr_fp += vt_fp; tr_fn += vt_fn
        id_switches += v_idsw; total_gt += v_gt
        print(f"[{v_idx}/{len(dataset.videos)}] {video.video_id}  "
              f"F1={v_f1:.3f} MOTA={v_mota:.3f} IDF1={v_idf1:.3f} IDsw={v_idsw}")

    t_total = time.perf_counter() - t0
    prec = det_tp / max(det_tp + det_fp, 1)
    rec = det_tp / max(det_tp + det_fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    mota = 1.0 - (tr_fp + tr_fn + id_switches) / max(total_gt, 1)
    idp = tr_tp / max(tr_tp + tr_fp, 1); idr = tr_tp / max(tr_tp + tr_fn, 1)
    idf1 = 2 * idp * idr / max(idp + idr, 1e-9)
    mac = lambda key, sub: (float(np.mean([m[sub][key] for m in per_video.values()]))
                            if per_video else 0.0)

    summary = {
        "tracker": "tgram_oracle" if gt_oracle else ("tgram_all" if all_class else "tgram"),
        "dataset": dataset_key, "checkpoint": checkpoint,
        "match_metric": metric, "iou_thresh": iou_thr, "centroid_dist_thresh": dist_thr,
        "conf_thres": conf_thres, "input_w": input_w, "input_h": input_h,
        "num_frames": num_frames,
        "total_videos": len(dataset.videos), "total_frames": n_frames_total,
        "total_time_s": t_total, "fps": n_frames_total / max(t_total, 1e-9),
        "model_size_MB": model_mb,
        "Precision": prec, "Recall": rec, "F1": f1,
        "MOTA": mota, "IDF1": idf1, "ID_switches": id_switches, "num_gt": total_gt,
        "macro_Precision": mac("Precision", "det"), "macro_Recall": mac("Recall", "det"),
        "macro_F1": mac("F1", "det"), "macro_MOTA": mac("MOTA", "track"),
        "macro_IDF1": mac("IDF1", "track"),
    }
    (experiment_dir / "test_metrics.json").write_text(json.dumps(summary, indent=2))
    (experiment_dir / "per_video_metrics.json").write_text(json.dumps(per_video, indent=2))

    print("\n" + "=" * 60)
    print(f"micro:  Pr={prec:.3f} Re={rec:.3f} F1={f1:.3f} "
          f"MOTA={mota:.3f} IDF1={idf1:.3f} IDsw={id_switches}")
    print(f"macro:  Pr={summary['macro_Precision']:.3f} "
          f"Re={summary['macro_Recall']:.3f} F1={summary['macro_F1']:.3f}")
    print(f"frames={n_frames_total}  time={t_total:.1f}s  fps={summary['fps']:.1f}")
    print(f"output: {experiment_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
