#!/usr/bin/env python
"""Give the shared SOT attributes one operational definition instead of a union.

The problem
-----------
SatSOT, SV248S and OOTB each publish a sequence-attribute vocabulary, and the
names overlap.  Merging by name is not the same as merging by meaning: SV248S
defines in-plane rotation as ``an angle greater than or equal to 30 degrees``
and derives it automatically, SatSOT and OOTB label it by eye and state no
threshold at all.  A reviewer of the earlier version put it exactly right --- we
had taken the union of the vocabularies, not unified the definitions.

The approach
------------
SV248S already shows the way.  Its attribute table has a ``DEPENDENCE`` and an
``AUTO`` column: a handful of measured primitives (centre point, movement
velocity, object size) and per-frame state flags, and every sequence attribute
is a stated predicate over them.  We generalise that architecture to all three
datasets: measure the primitive identically everywhere, write one predicate, and
demote each source's native label from ground truth to a **validation signal**.

What this buys is more than tidiness. Because the primitive is continuous, each
source's native labels can be projected onto it, and the threshold they behave
like can be read off. Where those thresholds differ, that is the definitional
heterogeneity, measured rather than asserted.

What came out
-------------
Only one of the four shared attributes reachable without new annotation turns
out to be unifiable this way, and saying so is the point of the exercise.

``ROT`` --- in-plane rotation.  Primitive: the orientation of the minimum-area
rotated rectangle of the target's own geometry, unwrapped modulo 180 degrees,
smoothed over 9 frames, reduced to its range over the sequence.  It reproduces
the native labels of both annotating datasets well (Youden's J of 0.84 and 0.64)
**at different operating points** --- OOTB's labels behave like a 15-degree
threshold and SV248S's like 35 degrees, against the 30 degrees SV248S states.
Same name, same measurable, one source twice as permissive as the other.  That
is the definitional heterogeneity, measured.

``IV``, ``BC``, ``SOB`` --- illumination variation, background clutter, similar
object.  Every primitive tried reproduces the native labels only weakly
(J between 0.24 and 0.52, against 0.84 for ROT) and, worse, inconsistently:
``texture_ratio`` reaches J = 0.52 on SatSOT and 0.26 on OOTB for the same
attribute.  Two readings are possible --- our primitives are inadequate, or the
native labels are subjective judgements no pixel statistic reproduces --- and
the ROT result argues for the second, since there a good primitive showed
itself immediately.  These three are therefore **not pooled**.  The continuous
primitive is published for each so a user can pick an operating point, but the
native labels are reported per dataset and never merged.

Requires oriented geometry, which after ``import_sv248s_polygons.py`` exists for
OOTB (4-point boxes) and SV248S (tight contours) but not for SatSOT, which ships
horizontal boxes only.  SatSOT is therefore reported as *not supporting* this
attribute rather than being given a substitute: the obvious substitute, the
direction of travel, was tested and rejected --- against OOTB's own oriented
boxes it correlates at r = 0.31, because an aircraft can taxi in a straight line
without turning and can turn without its centre moving.

Usage
-----
    python tools/unify_sot_attributes.py --out docs/space_tracker/sot_attribute_unification.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

RELEASE = Path(os.environ.get("SPACE_TRACKER_RELEASE",
                              "/data/anon/release/space_tracker"))

#: Frames averaged before the orientation range is taken. A 5 px car's
#: minimum-area rectangle jitters by tens of degrees frame to frame; a rotation
#: that matters to a tracker persists for far longer than nine frames.
SMOOTH_FRAMES = 9

#: The unified threshold. 30 degrees is not our invention: it is the only
#: threshold any of the three sources states, and it comes from the only source
#: that derives the attribute automatically rather than by eye.
ROT_THRESHOLD_DEG = 30.0

#: Which native labels of each dataset are claimed to mean this attribute.
NATIVE = {
    "ROT": {"satsot": ["ROT"], "sv248s": ["IPR"], "ootb": ["IPR"]},
    "IV":  {"satsot": ["IV"],  "sv248s": ["IV"],  "ootb": ["IV"]},
    "BC":  {"satsot": ["BC"],  "sv248s": [],      "ootb": ["BC"]},
    "SOB": {"satsot": ["SOB"], "sv248s": ["DS"],  "ootb": ["SA"]},
}

#: Attributes whose native labels no primitive reproduced well enough to pool.
#: The best primitive found for each is kept and published as a continuous
#: value; the binary label stays per-dataset.
NOT_POOLED = {
    "IV":  ("iv_global",     True,  "range of the frame-level median intensity "
                                    "over the sequence, relative to its median"),
    "BC":  ("texture_ratio", False, "target contrast relative to its "
                                    "neighbourhood's contrast"),
    "SOB": ("peak",          True,  "strongest normalised cross-correlation of "
                                    "the target template elsewhere within "
                                    "2.5x the object size"),
}


#: Frames sampled per sequence for the imagery primitives. The statistics taken
#: from them are ranges and medians over the sequence, which 80 samples estimate
#: as well as every frame would at a fraction of the reading cost.
IMAGE_SAMPLES = 80


def image_primitives(job):
    """Per-sequence appearance statistics, measured identically on every source.

    ``iv_global`` is deliberately a *frame-level* quantity. Measuring intensity
    in a window that follows the target instead measures the background the
    target is passing over, which is a different attribute (SV248S calls it
    background change) and was what a first attempt accidentally quantified.
    """
    name, dataset, rows = job
    cv2.setNumThreads(1)
    rows.sort()
    idx = np.unique(np.linspace(0, len(rows) - 1,
                                min(IMAGE_SAMPLES, len(rows))).astype(int))
    frame_median, peak, texture = [], [], []
    for i in idx:
        _, file_name, bbox = rows[i]
        im = cv2.imread(str(RELEASE / file_name), cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        height, width = im.shape
        frame_median.append(float(np.median(im)))
        x, y, w, h = bbox
        x0, y0 = max(int(round(x)), 0), max(int(round(y)), 0)
        w0, h0 = max(int(round(w)), 2), max(int(round(h)), 2)
        x1, y1 = min(x0 + w0, width), min(y0 + h0, height)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        target = im[y0:y1, x0:x1].astype(np.float32)
        # 2.5x the object size is SV248S's own radius for "a similar object
        # exists nearby", so its definition is reused rather than reinvented.
        r = int(round(2.5 * np.sqrt(w0 * h0)))
        sx0, sy0 = max(x0 - r, 0), max(y0 - r, 0)
        sx1, sy1 = min(x1 + r, width), min(y1 + r, height)
        search = im[sy0:sy1, sx0:sx1].astype(np.float32)
        if search.shape[0] <= target.shape[0] or search.shape[1] <= target.shape[1]:
            continue
        res = cv2.matchTemplate(search, target, cv2.TM_CCOEFF_NORMED)
        tx, ty = x0 - sx0, y0 - sy0
        res[max(ty - h0, 0):ty + h0, max(tx - w0, 0):tx + w0] = -1.0
        peak.append(float(res.max()))
        texture.append(float(np.std(target)) / (float(np.std(search)) + 1e-3))
    if len(frame_median) < 5:
        return None
    fm = np.asarray(frame_median)
    return {
        "sequence": f"sot/{name}", "source_dataset": dataset,
        "iv_global": round(float((np.percentile(fm, 90) - np.percentile(fm, 10))
                                 / (np.median(fm) + 1e-3)), 4),
        "peak": round(float(np.median(peak)), 4) if peak else None,
        "texture_ratio": round(float(np.median(texture)), 4) if texture else None,
    }


def orientation_range(polygons: list[list[float]]) -> float | None:
    """Range of the target's own orientation over the sequence, in degrees."""
    angles = []
    for poly in polygons:
        pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        (_, _), (w, h), theta = cv2.minAreaRect(pts)
        if w < h:                     # name the long side consistently
            theta += 90.0
        angles.append(np.deg2rad(theta))
    if len(angles) < SMOOTH_FRAMES + 1:
        return None
    # Orientation is periodic with 180 degrees, so unwrap the doubled angle.
    a = np.degrees(np.unwrap(2.0 * np.asarray(angles)) / 2.0)
    k = SMOOTH_FRAMES
    sm = np.convolve(a, np.ones(k) / k, mode="valid")
    return float(sm.max() - sm.min())


def sweep(values: np.ndarray, native: np.ndarray) -> dict:
    """The threshold on `values` whose labels best reproduce `native`."""
    best = {"threshold_deg": None, "youden_j": -1.0}
    for t in np.arange(5.0, 121.0, 1.0):
        pred = values >= t
        tpr = float(pred[native].mean()) if native.any() else 0.0
        fpr = float(pred[~native].mean()) if (~native).any() else 0.0
        if tpr - fpr > best["youden_j"]:
            best = {"threshold_deg": float(t), "youden_j": round(tpr - fpr, 3),
                    "recall": round(tpr, 3), "false_alarm": round(fpr, 3),
                    "agreement": round(float((pred == native).mean()), 3)}
    return best


def at(values: np.ndarray, native: np.ndarray, t: float) -> dict:
    pred = values >= t
    return {
        "threshold_deg": t,
        "n": int(len(values)),
        "n_native_positive": int(native.sum()),
        "n_unified_positive": int(pred.sum()),
        "agreement": round(float((pred == native).mean()), 3),
        "recall": round(float(pred[native].mean()), 3) if native.any() else None,
        "false_alarm": round(float(pred[~native].mean()), 3) if (~native).any() else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=Path("docs/space_tracker/sot_attribute_unification.json"))
    ap.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 8))
    ap.add_argument("--skip-imagery", action="store_true",
                    help="ROT only; skips the pass over the frames")
    args = ap.parse_args()

    data = json.loads((RELEASE / "sot" / "annotations"
                       / "space_tracker_sot.json").read_text())
    videos = {v["id"]: v for v in data["videos"]}
    videos_by_name = {f"sot/{v['name']}": v for v in data["videos"]}
    images = {i["id"]: i for i in data["images"]}

    polys: dict[int, list] = collections.defaultdict(list)
    for a in data["annotations"]:
        seg = a.get("segmentation")
        if seg:
            im = images[a["image_id"]]
            polys[im["video_id"]].append((im.get("frame_id", 0), seg[0]))

    per_seq = []
    for v in data["videos"]:
        rows = sorted(polys.get(v["id"], []))
        rng = orientation_range([r[1] for r in rows]) if rows else None
        native = bool(set(NATIVE["ROT"].get(v["source_dataset"], []))
                      & set(v.get("native_attributes") or []))
        per_seq.append({
            "sequence": f"sot/{v['name']}",
            "source_dataset": v["source_dataset"],
            "rotation_range_deg": None if rng is None else round(rng, 2),
            "rot_unified": None if rng is None else bool(rng >= ROT_THRESHOLD_DEG),
            "rot_native": native,
            "supported": rng is not None,
        })

    report = {"attribute": "ROT", "unified_threshold_deg": ROT_THRESHOLD_DEG,
              "primitive": "range of the smoothed minimum-area-rectangle "
                           "orientation over the sequence, degrees",
              "per_dataset": {}}
    print(f"ROT: unified predicate = orientation range >= "
          f"{ROT_THRESHOLD_DEG:.0f} deg\n")
    print(f"{'dataset':8s} {'n':>4s} {'sup':>4s} {'native+':>8s} "
          f"{'unified+':>9s} {'agree':>6s} {'recall':>7s} {'FA':>5s}   "
          f"native labels behave like")
    for ds in ("ootb", "satsot", "sv248s"):
        rows = [r for r in per_seq if r["source_dataset"] == ds]
        ok = [r for r in rows if r["supported"]]
        if not ok:
            n_native = sum(r["rot_native"] for r in rows)
            report["per_dataset"][ds] = {
                "supported": False, "n": len(rows), "n_native_positive": n_native,
                "reason": "no oriented geometry in the source (horizontal boxes "
                          "only), so the primitive cannot be measured",
            }
            print(f"{ds:8s} {len(rows):4d} {0:4d} {n_native:8d} "
                  f"{'---':>9s} {'---':>6s} {'---':>7s} {'---':>5s}   "
                  f"not supported: source has no oriented geometry")
            continue
        vals = np.array([r["rotation_range_deg"] for r in ok])
        nat = np.array([r["rot_native"] for r in ok])
        fixed, best = at(vals, nat, ROT_THRESHOLD_DEG), sweep(vals, nat)
        report["per_dataset"][ds] = {"supported": True, "at_unified": fixed,
                                     "best_matching": best}
        print(f"{ds:8s} {len(rows):4d} {len(ok):4d} {fixed['n_native_positive']:8d} "
              f"{fixed['n_unified_positive']:9d} {fixed['agreement']:6.2f} "
              f"{fixed['recall']:7.2f} {fixed['false_alarm']:5.2f}   "
              f"{best['threshold_deg']:.0f} deg "
              f"(agreement {best['agreement']:.2f})")

    # ---- the three attributes that turned out not to be poolable -----------
    not_pooled = {}
    if not args.skip_imagery:
        boxes: dict[int, list] = collections.defaultdict(list)
        for a in data["annotations"]:
            im = images[a["image_id"]]
            boxes[im["video_id"]].append(
                (im.get("frame_id", 0), im["file_name"], a["bbox"]))
        jobs = [(videos[k]["name"], videos[k]["source_dataset"], v)
                for k, v in boxes.items()]
        print(f"\nmeasuring appearance primitives over {len(jobs)} sequences "
              f"on {args.workers} workers...", flush=True)
        with Pool(args.workers) as pool:
            prim = [r for r in pool.map(image_primitives, jobs) if r]
        by_name = {r["sequence"]: r for r in prim}
        for rec in per_seq:
            rec.update({k: v for k, v in by_name.get(rec["sequence"], {}).items()
                        if k not in ("sequence", "source_dataset")})

        print(f"\n{'attr':5s} {'primitive':15s} {'dataset':8s} {'n':>4s} "
              f"{'native+':>8s} {'best J':>7s}  verdict")
        for attr, (field, higher, description) in NOT_POOLED.items():
            entry = {"primitive": field, "primitive_description": description,
                     "higher_is_positive": higher, "pooled": False,
                     "per_dataset": {}}
            for ds in ("ootb", "satsot", "sv248s"):
                labels = NATIVE[attr].get(ds) or []
                rows = [r for r in per_seq
                        if r["source_dataset"] == ds and r.get(field) is not None]
                nat = np.array([bool(set(labels) & set(
                    (videos_by_name[r["sequence"]].get("native_attributes") or [])))
                    for r in rows])
                if not labels or nat.sum() < 3 or (~nat).sum() < 3:
                    entry["per_dataset"][ds] = {
                        "annotated": bool(labels), "n_native_positive": int(nat.sum()),
                        "usable": False}
                    print(f"{attr:5s} {field:15s} {ds:8s} {len(rows):4d} "
                          f"{int(nat.sum()):8d} {'---':>7s}  "
                          f"{'not annotated' if not labels else 'too few to judge'}")
                    continue
                x = np.asarray([r[field] for r in rows], dtype=float)
                best_j, best_t = -1.0, None
                for t in np.percentile(x, np.arange(2, 99, 1)):
                    pred = x >= t if higher else x <= t
                    j = float(pred[nat].mean()) - float(pred[~nat].mean())
                    if j > best_j:
                        best_j, best_t = j, float(t)
                entry["per_dataset"][ds] = {
                    "annotated": True, "usable": True, "n": len(rows),
                    "n_native_positive": int(nat.sum()),
                    "best_youden_j": round(best_j, 3),
                    "best_threshold": round(best_t, 4)}
                print(f"{attr:5s} {field:15s} {ds:8s} {len(rows):4d} "
                      f"{int(nat.sum()):8d} {best_j:7.2f}  "
                      f"{'weak - not pooled' if best_j < 0.6 else 'strong'}")
            not_pooled[attr] = entry

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"pooled": {"ROT": report}, "not_pooled": not_pooled,
         "sequences": per_seq}, indent=2) + "\n")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
