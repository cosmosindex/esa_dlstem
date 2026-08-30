#!/usr/bin/env python
"""Find sequences that are genuinely redundant, as opposed to merely related.

Sharing a parent scene is not a defect.  Two crops of one satellite scene that
follow different targets are two different tracking problems, and a sequence in
the SOT half has nothing to do with one in the MOT half — they benchmark
different algorithms.  Only one thing is actually redundant: **the same video,
inside the same task half, carrying the same annotation.**

So this narrows the parent-scene graph down twice.

1. Same half, and the same crop window — the matched pair has scale 1 and no
   translation.  Then verify pixel identity on several frames, not just frame 0.
2. Same annotation.  For SOT that means the tracked target's boxes agree frame
   by frame; for MOT it means the whole box set per frame agrees.  A pair that
   passes (1) but fails (2) is the same footage annotated for a different
   target, and both sequences are kept.

Only a pair that passes both is a duplicate.

Usage
-----
    python tools/find_space_tracker_duplicates.py \
        --scene-groups docs/space_tracker/scene_groups.json \
        --out docs/space_tracker/duplicates.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

import cv2
import numpy as np

RELEASE = Path(os.environ.get("SPACE_TRACKER_RELEASE",
                              "/data/anon/release/space_tracker"))

#: A candidate must be the same crop of the same scene: unit scale, no shift.
MAX_SHIFT_PX = 3.0
SCALE_TOL = 0.01
#: Frames sampled across the sequence for the pixel-identity check.
N_PROBE_FRAMES = 5
#: Boxes are compared at this resolution; the two sources round differently
#: (SatSOT ships integer HBBs, OOTB float OBB-derived ones), and half a pixel of
#: rounding on a 6 px car is not a different annotation.
BOX_ROUND = 0
#: Fraction of frames that must agree before two annotations count as the same.
AGREE_FRAC = 0.95
IOU_SAME = 0.9


def load_half(half: str, fname: str) -> dict:
    data = json.loads((RELEASE / half / "annotations" / fname).read_text())
    cats = {c["id"]: c["name"] for c in data["categories"]}
    img = {i["id"]: i for i in data["images"]}
    frames: dict[int, dict[int, str]] = collections.defaultdict(dict)
    for i in data["images"].__iter__():
        frames[i["video_id"]][i.get("frame_id", 0)] = i["file_name"]
    boxes: dict[int, dict[int, list]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for a in data["annotations"]:
        im = img[a["image_id"]]
        boxes[im["video_id"]][im.get("frame_id", 0)].append(
            (cats[a["category_id"]], tuple(round(x, BOX_ROUND) for x in a["bbox"])))
    return {"videos": {f"{half}/{v['name']}": v for v in data["videos"]},
            "frames": frames, "boxes": boxes}


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / (aw * ah + bw * bh - inter)


def pixels_identical(idx: dict, va: dict, vb: dict) -> tuple[bool, int]:
    """Compare N frames spread over the pair. Returns (identical, n_compared)."""
    fa, fb = idx["frames"][va["id"]], idx["frames"][vb["id"]]
    shared = sorted(set(fa) & set(fb))
    if not shared:
        return False, 0
    probe = [shared[i] for i in
             np.unique(np.linspace(0, len(shared) - 1, N_PROBE_FRAMES).astype(int))]
    for f in probe:
        ia = cv2.imread(str(RELEASE / fa[f]), cv2.IMREAD_GRAYSCALE)
        ib = cv2.imread(str(RELEASE / fb[f]), cv2.IMREAD_GRAYSCALE)
        if ia is None or ib is None or ia.shape != ib.shape:
            return False, len(probe)
        if not np.array_equal(ia, ib):
            return False, len(probe)
    return True, len(probe)


def annotations_agree(idx: dict, va: dict, vb: dict) -> tuple[bool, float, str]:
    """Do the two sequences annotate the same thing on the frames they share?"""
    ba, bb = idx["boxes"][va["id"]], idx["boxes"][vb["id"]]
    shared = sorted(set(ba) & set(bb))
    if not shared:
        return False, 0.0, "no shared frames"
    agree = 0
    for f in shared:
        A, B = ba[f], bb[f]
        if len(A) != len(B):
            continue
        # Greedy: every box on one side must find a match on the other.
        unused = list(B)
        ok = True
        for ca, box_a in A:
            hit = None
            for k, (cb, box_b) in enumerate(unused):
                if ca == cb and iou(box_a, box_b) >= IOU_SAME:
                    hit = k
                    break
            if hit is None:
                ok = False
                break
            unused.pop(hit)
        agree += bool(ok)
    frac = agree / len(shared)
    return frac >= AGREE_FRAC, frac, f"{agree}/{len(shared)} frames"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-groups", type=Path,
                    default=Path("docs/space_tracker/scene_groups.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("docs/space_tracker/duplicates.json"))
    args = ap.parse_args()

    idx = {}
    for half, fname in (("sot", "space_tracker_sot.json"),
                        ("mot", "space_tracker_mot.json")):
        idx[half] = load_half(half, fname)

    edges = json.loads(args.scene_groups.read_text())["edges"]
    cand = [e for e in edges
            if e["a"].split("/")[0] == e["b"].split("/")[0]
            and abs(e["tx"]) < MAX_SHIFT_PX and abs(e["ty"]) < MAX_SHIFT_PX
            and abs(e["scale"] - 1.0) <= SCALE_TOL]
    print(f"{len(cand)} same-half, same-crop-window candidates "
          f"out of {len(edges)} edges", flush=True)

    results = []
    for e in cand:
        half = e["a"].split("/")[0]
        I = idx[half]
        va, vb = I["videos"].get(e["a"]), I["videos"].get(e["b"])
        if va is None or vb is None:
            continue
        same_px, n_probe = pixels_identical(I, va, vb)
        if not same_px:
            verdict, agree, detail = "same_scene_only", None, "frames differ"
        else:
            same_ann, agree, detail = annotations_agree(I, va, vb)
            verdict = "duplicate" if same_ann else "same_video_different_annotation"
        results.append({
            "half": half, "a": e["a"], "b": e["b"], "verdict": verdict,
            "inliers": e["inliers"], "frames_probed": n_probe,
            "annotation_agreement": None if agree is None else round(agree, 3),
            "detail": detail,
            "a_frames": va.get("n_frames"), "b_frames": vb.get("n_frames"),
            "a_tracks": va.get("n_tracks", 1), "b_tracks": vb.get("n_tracks", 1),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"n_candidates": len(cand), "pairs": results}, indent=2) + "\n")

    c = collections.Counter((r["half"], r["verdict"]) for r in results)
    print()
    for (half, verdict), n in sorted(c.items()):
        print(f"  {half}  {verdict:34s} {n}")
    dups = [r for r in results if r["verdict"] == "duplicate"]
    print(f"\n{len(dups)} true duplicate pairs (same video, same annotation)")
    for r in dups:
        print(f"    {r['a']:32s} == {r['b']:32s}  {r['detail']}")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
