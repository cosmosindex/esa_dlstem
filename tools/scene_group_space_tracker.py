#!/usr/bin/env python
"""Group every Space-Tracker sequence by the parent satellite scene it was cut from.

Why this exists
---------------
The source datasets overlap.  OOTB says so itself (its Table 3 lists
``OD = SatSOT, VISO, AIR-MOT``), and SV248S ships six parent videos that 248
sequences were cropped out of.  Two consequences:

* a train/val/test split made per *sequence* leaks pixels across the split;
* frame rate and ground sample distance are properties of the **parent video**,
  not of the sequence, so once sequences are grouped, those two numbers only
  have to be settled once per group instead of once per sequence.

Method
------
Read frame 0 of every sequence, equalise it (satellite scenes are low contrast),
detect ORB keypoints, and match every pair.  A pair that survives a RANSAC
partial-affine fit with enough inliers came from the same parent scene; the
fitted scale is the ratio of their ground sample distances, and the translation
is the offset between their crop windows.  Union-find turns the surviving pairs
into groups.

Partial affine (scale + rotation + translation) is the right model: two crops of
one satellite scene differ by a translation, plus a uniform scale when the
sources resampled to different GSDs.  It is not a homography — there is no
perspective change between two crops of the same frame.

Usage
-----
    python tools/scene_group_space_tracker.py --out space_tracker/data/scene_groups.json
"""

from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import cv2
import numpy as np

import os as _bootstrap_os, sys as _bootstrap_sys
_bootstrap_sys.path.insert(0, _bootstrap_os.path.dirname(
    _bootstrap_os.path.dirname(_bootstrap_os.path.abspath(__file__))))
from project_paths import DATA_ROOT

#: Tracked copies of the pipeline intermediates, shipped in the package so a
#: clean checkout can re-run the build without a scratch directory.
_PKG = Path(__file__).resolve().parents[1] / "space_tracker" / "data"

#: The released package. ``SPACE_TRACKER_ROOT`` is the name the README and the
#: loading code use; ``SPACE_TRACKER_RELEASE`` is accepted as the older spelling.
RELEASE = Path(os.environ.get("SPACE_TRACKER_ROOT")
               or os.environ.get("SPACE_TRACKER_RELEASE")
               or f"{DATA_ROOT}/release/space_tracker")

#: Frame 0 is downscaled so its longest side is at most this before ORB runs.
#: Keeps the cost of a 4096-wide parent scene comparable to a 200-wide crop.
MAX_SIDE = 1024
N_FEATURES = 1500

#: A pair needs this many RANSAC inliers to count as the same parent scene.
#: 12 is well above what unrelated satellite scenes produce (they share roads,
#: rooftops and field boundaries, so a handful of chance matches is normal) and
#: well below what a true overlap produces (hundreds).
MIN_INLIERS = 12

#: Ratio of ground sample distances we are willing to believe. 0.75 m vs 1.1 m
#: is 1.47x; anything beyond 3x is a mis-fit, not a resampled scene.
SCALE_RANGE = (0.33, 3.0)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_sequences() -> list[dict]:
    """One record per sequence across both halves of the release."""
    out = []
    for half, fname in (("sot", "space_tracker_sot.json"),
                        ("mot", "space_tracker_mot.json")):
        path = RELEASE / half / "annotations" / fname
        data = json.loads(path.read_text())
        first: dict[int, dict] = {}
        for im in data["images"]:
            v = im["video_id"]
            if v not in first or im.get("frame_id", 0) < first[v].get("frame_id", 0):
                first[v] = im
        for v in data["videos"]:
            im = first.get(v["id"])
            if im is None:
                continue
            out.append({
                "key": f"{half}/{v['name']}",
                "half": half,
                "name": v["name"],
                "dataset": v.get("source_dataset"),
                "source_sequence_id": v.get("source_sequence_id"),
                "category": v.get("category") or v.get("categories_in_sequence"),
                # file_name is relative to the release root, not to the half.
                "frame0": str(RELEASE / im["file_name"]),
                "width": im["width"],
                "height": im["height"],
            })
    return out


def describe(rec: dict) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    """Keypoints and ORB descriptors for frame 0, plus the resize factor used."""
    img = cv2.imread(rec["frame0"], cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None, 1.0
    f = min(1.0, MAX_SIDE / max(img.shape))
    if f < 1.0:
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    img = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(img)
    orb = cv2.ORB_create(nfeatures=N_FEATURES, fastThreshold=7)
    kp, des = orb.detectAndCompute(img, None)
    if des is None or len(kp) < MIN_INLIERS:
        return None, None, f
    return np.float32([k.pt for k in kp]), des, f


# ---------------------------------------------------------------------------
# pairwise matching
# ---------------------------------------------------------------------------

PTS: list = []
DES: list = []
FAC: list = []


def _init(pts, des, fac):
    global PTS, DES, FAC
    # One OpenCV thread per worker. Left at the default, each of N workers spawns
    # a full thread pool and they spend their time fighting over the cores.
    cv2.setNumThreads(1)
    PTS, DES, FAC = pts, des, fac


def _match_chunk(pairs: list[tuple[int, int]]) -> list[tuple]:
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    hits = []
    for i, j in pairs:
        da, db = DES[i], DES[j]
        if da is None or db is None:
            continue
        if len(da) < 2 or len(db) < 2:
            continue
        good_a, good_b = [], []
        for m, n in bf.knnMatch(da, db, k=2):
            if m.distance < 0.75 * n.distance:
                good_a.append(PTS[i][m.queryIdx])
                good_b.append(PTS[j][m.trainIdx])
        if len(good_a) < MIN_INLIERS:
            continue
        M, inl = cv2.estimateAffinePartial2D(
            np.float32(good_a), np.float32(good_b),
            method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=5000)
        if M is None or inl is None:
            continue
        n_inl = int(inl.sum())
        if n_inl < MIN_INLIERS:
            continue
        # estimateAffinePartial2D returns [s*cos, -s*sin; s*sin, s*cos | tx, ty]
        scale = float(np.hypot(M[0, 0], M[1, 0]))
        # Undo the per-image downscaling to get the true GSD ratio.
        scale *= FAC[i] / FAC[j] if FAC[j] else 1.0
        if not (SCALE_RANGE[0] <= scale <= SCALE_RANGE[1]):
            continue
        rot = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
        hits.append((i, j, n_inl, len(good_a), round(scale, 4), round(rot, 2),
                     round(float(M[0, 2]), 1), round(float(M[1, 2]), 1)))
    return hits


# ---------------------------------------------------------------------------
# parents the datasets already tell us about
# ---------------------------------------------------------------------------

def metadata_parent(rec: dict) -> str | None:
    """Parent video id when the source dataset ships one, else ``None``.

    Two datasets state their parent outright, and both need it: pixel matching
    cannot recover either.

    * SV248S ``sv248s/<video>/<seq>`` — 248 sequences cropped from six parent
      videos (paper Table 4).  Two crops of one parent that do not spatially
      overlap share no keypoints, so matching alone splits a parent into pieces.
    * SDM-Car ``sdmcar/<split>/<video>-<patch>`` — the paper segments each
      original video "into multiple patches **without overlap**", so by
      construction no two patches of one video can ever be matched to each other.
    """
    sid = rec.get("source_sequence_id") or ""
    parts = sid.split("/")
    if rec["dataset"] == "sv248s" and len(parts) >= 3:
        return f"sv248s/{parts[1]}"
    if rec["dataset"] == "sdmcar" and len(parts) >= 3:
        return f"sdmcar/{parts[2].split('-')[0]}"
    return None


# ---------------------------------------------------------------------------
# union-find
# ---------------------------------------------------------------------------

def group(n: int, edges: list[tuple[int, int]]) -> list[int]:
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return [find(i) for i in range(n)]


# ---------------------------------------------------------------------------

def finish(recs: list[dict], hits: list[tuple], out: Path) -> None:
    """Seed union-find with the parents the datasets state, add the matched
    edges on top, and write the groups out."""
    seed_edges: list[tuple[int, int]] = []
    by_parent: dict[str, int] = {}
    for i, r in enumerate(recs):
        p = metadata_parent(r)
        if p is None:
            continue
        if p in by_parent:
            seed_edges.append((by_parent[p], i))
        else:
            by_parent[p] = i
    print(f"{len(seed_edges)} edges seeded from dataset-declared parents", flush=True)
    roots = group(len(recs), seed_edges + [(h[0], h[1]) for h in hits])
    members = collections.defaultdict(list)
    for i, r in enumerate(roots):
        members[r].append(i)

    groups = []
    for gid, (root, idx) in enumerate(sorted(members.items(),
                                             key=lambda kv: -len(kv[1])), start=1):
        groups.append({
            "group_id": gid,
            "n_sequences": len(idx),
            "datasets": sorted({recs[i]["dataset"] for i in idx}),
            "halves": sorted({recs[i]["half"] for i in idx}),
            "sequences": [recs[i]["key"] for i in sorted(idx)],
        })

    edges = [{
        "a": recs[h[0]]["key"], "b": recs[h[1]]["key"],
        "inliers": h[2], "matches": h[3], "scale": h[4], "rotation_deg": h[5],
        "tx": h[6], "ty": h[7],
    } for h in sorted(hits, key=lambda h: -h[2])]

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_sequences": len(recs),
        "n_edges": len(edges),
        "n_groups": len(groups),
        "min_inliers": MIN_INLIERS,
        "groups": groups,
        "edges": edges,
    }, indent=2) + "\n")

    multi = [g for g in groups if g["n_sequences"] > 1]
    cross_ds = [g for g in multi if len(g["datasets"]) > 1]
    cross_half = [g for g in multi if len(g["halves"]) > 1]
    print(f"\n{len(groups)} parent-scene groups "
          f"({len(multi)} with more than one sequence)")
    print(f"  groups spanning more than one source dataset: {len(cross_ds)}")
    print(f"  groups spanning both the SOT and MOT halves : {len(cross_half)}")
    print(f"  sequences in a multi-sequence group        : "
          f"{sum(g['n_sequences'] for g in multi)}/{len(recs)}")
    print(f"\nwritten to {out}")




def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=_PKG / "scene_groups.json")
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8))
    ap.add_argument("--limit", type=int, default=0, help="debug: first N sequences")
    ap.add_argument("--reuse-edges", type=Path, default=None,
                    help="skip matching and take the edges from a previous run")
    args = ap.parse_args()

    recs = load_sequences()
    if args.limit:
        recs = recs[:args.limit]
    print(f"{len(recs)} sequences", flush=True)

    if args.reuse_edges:
        prev = json.loads(args.reuse_edges.read_text())
        pos = {r["key"]: i for i, r in enumerate(recs)}
        hits = [(pos[e["a"]], pos[e["b"]], e["inliers"], e["matches"],
                 e["scale"], e["rotation_deg"], e["tx"], e["ty"])
                for e in prev["edges"] if e["a"] in pos and e["b"] in pos]
        print(f"reused {len(hits)} edges from {args.reuse_edges}", flush=True)
        return finish(recs, hits, args.out)

    pts, des, fac = [], [], []
    for k, r in enumerate(recs):
        p, d, f = describe(r)
        pts.append(p); des.append(d); fac.append(f)
        if (k + 1) % 100 == 0:
            print(f"  described {k + 1}/{len(recs)}", flush=True)
    n_desc = sum(d is not None for d in des)
    print(f"usable descriptors: {n_desc}/{len(recs)}", flush=True)

    pairs = [(i, j) for i in range(len(recs)) for j in range(i + 1, len(recs))
             if des[i] is not None and des[j] is not None]
    print(f"{len(pairs)} pairs to match on {args.workers} workers", flush=True)

    chunks = [pairs[i::args.workers * 4] for i in range(args.workers * 4)]
    ctx = mp.get_context("fork")
    hits: list[tuple] = []
    with ctx.Pool(args.workers, initializer=_init, initargs=(pts, des, fac)) as pool:
        for k, part in enumerate(pool.imap_unordered(_match_chunk, chunks)):
            hits.extend(part)
            print(f"  chunk {k + 1}/{len(chunks)}  hits so far {len(hits)}", flush=True)

    return finish(recs, hits, args.out)


if __name__ == "__main__":
    sys.exit(main())
