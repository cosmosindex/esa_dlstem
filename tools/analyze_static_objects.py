"""
Does the Space-tracker MOT ground truth annotate *static* objects, or only
moving ones?

This matters for benchmark fairness: if a parked aircraft or a moored ship is
visible but unlabelled, every detector that finds it is charged a false
positive it cannot avoid.

Two independent lines of evidence:

**1. Track displacement** (all 5 MOT datasets). For each GT track, net
displacement of the box centre from first to last frame, normalised by the
object's own size. A GT that only covers movers has no tracks near zero; a GT
that also covers static objects has a spike there.

**2. SAT-MTB det-vs-MOT delta** (airplane / ship / train only). SAT-MTB ships
per-frame detection XML *and* MOT CSV for these three categories. Objects
present in the detection XML but absent from the MOT CSV are exactly the
annotations a MOT-trained model is penalised for finding. (``car`` has no
detection XML — its MOT CSV is the only source, so it cannot be cross-checked.)

Usage::

    python tools/analyze_static_objects.py --out-dir docs/size_split
"""

from __future__ import annotations

import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from space_tracker.data_mot import _parse_gt
from space_tracker.manifest_mot import MOTManifest
from tools.analyze_size_split import MOT_ROOTS, REPO

SATMTB_DET_ROOT = MOT_ROOTS["satmtb"] / "SAT-MTB_Dataset"

# A track counts as static when its centre never wanders further than this
# many multiples of its own sqrt(area) from where it started.
STATIC_NET_DISP_RATIO = 0.5


# ---------------------------------------------------------------------------
# 1. Track displacement
# ---------------------------------------------------------------------------

def track_displacements(seq, root: Path) -> list[dict]:
    """One record per GT track in the sequence."""
    ann = _parse_gt(seq, root)
    tracks: dict[tuple[int, str], list[tuple[int, np.ndarray]]] = defaultdict(list)
    for fid, objs in ann.items():
        for o in objs:
            tracks[(o.track_id, o.category)].append((fid, o.bbox_xyxy))

    out = []
    for (tid, cat), obs in tracks.items():
        obs.sort(key=lambda t: t[0])
        boxes = np.stack([b for _, b in obs]).astype(np.float64)
        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        size = np.sqrt(np.maximum(boxes[:, 2] - boxes[:, 0], 0) *
                       np.maximum(boxes[:, 3] - boxes[:, 1], 0))
        med_size = float(np.median(size)) or 1.0

        net = float(np.hypot(cx[-1] - cx[0], cy[-1] - cy[0]))
        step = np.hypot(np.diff(cx), np.diff(cy)) if len(cx) > 1 else np.array([0.0])
        # Path length bounds how far it *could* have gone; net alone would call
        # a car that drives out and back "static".
        path = float(step.sum())

        out.append({
            "seq_id": seq.id, "dataset": seq.dataset, "category": cat,
            "track_id": tid, "n_obs": len(obs),
            "median_sqrt_area": round(med_size, 2),
            "net_disp_px": round(net, 2),
            "path_len_px": round(path, 2),
            "net_disp_ratio": round(net / med_size, 3),
            "path_len_ratio": round(path / med_size, 3),
            "mean_step_px": round(float(step.mean()), 3),
        })
    return out


# ---------------------------------------------------------------------------
# 2. SAT-MTB det XML vs MOT CSV
# ---------------------------------------------------------------------------

def _parse_hbb_xml(path: Path) -> list[tuple[str, str, np.ndarray]]:
    """-> [(category, objectID, xyxy), ...]"""
    out = []
    for obj in ET.parse(path).findall("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue
        out.append((
            (obj.findtext("name") or "").strip(),
            (obj.findtext("objectID") or "").strip(),
            np.array([
                float(bb.findtext("xmin", "0")), float(bb.findtext("ymin", "0")),
                float(bb.findtext("xmax", "0")), float(bb.findtext("ymax", "0")),
            ]),
        ))
    return out


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.prod(np.clip(a[:, 2:] - a[:, :2], 0, None), axis=1)
    area_b = np.prod(np.clip(b[:, 2:] - b[:, :2], 0, None), axis=1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def satmtb_det_vs_mot(seq, iou_thresh: float = 0.5) -> list[dict]:
    """One record per *detection-XML track*: is it also in the MOT GT, and does
    it move?

    Matching is done per category — the detection XML for an ``airplane``
    sequence still labels the ships in frame, while the MOT CSV of a mixed
    sequence additionally labels cars the XML ignores. Comparing across
    categories would attribute those bookkeeping differences to static objects.
    """
    cat_dir, num = seq.video_id.split("/", 1)
    hbb_dir = SATMTB_DET_ROOT / cat_dir / num / "det" / "HBB"
    if not hbb_dir.is_dir():
        return []

    ann = _parse_gt(seq, MOT_ROOTS["satmtb"])
    # det track -> observations, and how many of them the MOT GT also has
    obs: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    hits: dict[tuple[str, str], int] = defaultdict(int)

    for xml_path in sorted(hbb_dir.glob("*.xml")):
        try:
            fid = int(xml_path.stem)
        except ValueError:
            continue
        det = _parse_hbb_xml(xml_path)
        if not det:
            continue
        mot = ann.get(fid, [])
        for category in {c for c, _, _ in det}:
            d_idx = [i for i, (c, _, _) in enumerate(det) if c == category]
            d = np.stack([det[i][2] for i in d_idx])
            m = np.stack([o.bbox_xyxy.astype(np.float64) for o in mot
                          if o.category == category]) if any(
                              o.category == category for o in mot) else np.zeros((0, 4))
            best = _iou_matrix(d, m).max(axis=1) if m.shape[0] else np.zeros(len(d))
            for j, i in enumerate(d_idx):
                key = (det[i][0], det[i][1])
                obs[key].append(det[i][2])
                if best[j] >= iou_thresh:
                    hits[key] += 1

    out = []
    for (category, oid), boxes in obs.items():
        b = np.stack(boxes)
        cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
        size = np.sqrt(np.maximum(b[:, 2] - b[:, 0], 0) * np.maximum(b[:, 3] - b[:, 1], 0))
        med = float(np.median(size)) or 1.0
        net = float(np.hypot(cx[-1] - cx[0], cy[-1] - cy[0]))
        frac_in_mot = hits[(category, oid)] / len(boxes)
        out.append({
            "seq_id": seq.id, "seq_category": cat_dir, "category": category,
            "det_object_id": oid, "n_obs": len(boxes),
            "median_sqrt_area": round(med, 2),
            "net_disp_px": round(net, 2),
            "net_disp_ratio": round(net / med, 3),
            "frac_frames_in_mot": round(frac_in_mot, 3),
            # "in MOT" = the MOT GT covers most of this det track
            "in_mot": bool(frac_in_mot >= 0.5),
        })
    return out


# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}  ({len(rows)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    man = MOTManifest.load(REPO / "space_tracker" / "space_tracker_mot.json")

    print("== 1. track displacement ==")
    tracks: list[dict] = []
    for i, seq in enumerate(man.sequences, 1):
        try:
            tracks += track_displacements(seq, MOT_ROOTS[seq.dataset])
        except Exception as exc:                                # noqa: BLE001
            print(f"  [warn] {seq.id}: {exc}")
        if i % 100 == 0:
            print(f"  {i}/{len(man.sequences)}")
    _write_csv(args.out_dir / "track_displacement.csv", tracks)

    print(f"\n{'dataset':<12}{'category':<10}{'tracks':>8}{'static%':>9}"
          f"{'net_p50':>9}{'net_p90':>9}{'path_p50':>10}")
    print("-" * 67)
    by = defaultdict(list)
    for t in tracks:
        by[(t["dataset"], t["category"])].append(t)
    for k, ts in sorted(by.items()):
        net = np.array([t["net_disp_ratio"] for t in ts])
        path = np.array([t["path_len_ratio"] for t in ts])
        static = float((net < STATIC_NET_DISP_RATIO).mean()) * 100
        print(f"{k[0]:<12}{k[1]:<10}{len(ts):>8}{static:>8.1f}%"
              f"{np.percentile(net, 50):>9.2f}{np.percentile(net, 90):>9.2f}"
              f"{np.percentile(path, 50):>10.2f}")

    print("\n== 2. SAT-MTB detection XML vs MOT CSV ==")
    deltas: list[dict] = []
    satmtb = [s for s in man.sequences if s.dataset == "satmtb"]
    for i, seq in enumerate(satmtb, 1):
        deltas += satmtb_det_vs_mot(seq)
        if i % 50 == 0:
            print(f"  {i}/{len(satmtb)}")
    _write_csv(args.out_dir / "satmtb_det_vs_mot.csv", deltas)

    if deltas:
        print("\nDetection-XML tracks, split by whether the MOT GT also has them.")
        print("If MOT only covers movers, the 'missing' column is static "
              "(net displacement ~ 0) and the 'in MOT' column is not.\n")
        print(f"{'category':<12}{'det_tracks':>11}{'in MOT':>9}{'missing':>9}"
              f"{'miss%':>8}   {'net_disp_ratio (median)':>24}   {'size (median)':>15}")
        print(f"{'':<12}{'':>11}{'':>9}{'':>9}{'':>8}   "
              f"{'in MOT':>11}{'missing':>13}   {'in MOT':>7}{'missing':>8}")
        print("-" * 96)
        bycat = defaultdict(list)
        for d in deltas:
            bycat[d["category"]].append(d)
        for cat, ds in sorted(bycat.items()):
            inm = [d for d in ds if d["in_mot"]]
            mis = [d for d in ds if not d["in_mot"]]
            f = lambda xs, k: f"{np.median([x[k] for x in xs]):.2f}" if xs else "-"
            print(f"{cat:<12}{len(ds):>11}{len(inm):>9}{len(mis):>9}"
                  f"{len(mis) / len(ds):>8.1%}   "
                  f"{f(inm, 'net_disp_ratio'):>11}{f(mis, 'net_disp_ratio'):>13}   "
                  f"{f(inm, 'median_sqrt_area'):>7}{f(mis, 'median_sqrt_area'):>8}")


if __name__ == "__main__":
    main()
