#!/usr/bin/env python
"""Per-sequence metadata for Space-Tracker, in three layers.

Layer A -- measured from our own ground truth.  Exact, reproducible from the
released COCO files alone, no citation needed: frame size, length, track and box
counts, object size, per-frame displacement, density, and the motion-derived
flags (static, fast, sudden entry, gap/occlusion).

Layer B -- per-dataset constants from the source papers (``related_work/``).
Platform, ground sample distance, frame rate.  Every field carries a
``*_source`` tag: ``paper`` (quoted from the source publication), ``derived``
(computed from data, e.g. SDM-Car's fps read out of the AVI containers),
``inherited`` (taken from the parent-scene group), or ``unverified``.

Layer C -- what only a human eye can settle: weather, illumination change, low
contrast.  Emitted as empty fields for a later annotation pass; never guessed.

Nothing here invents a number.  A cell we cannot establish stays ``null`` with
``"unverified"`` next to it.

Usage
-----
    python tools/build_space_tracker_metadata.py \
        --scene-groups space_tracker/data/scene_groups.json \
        --out space_tracker/data/sequence_metadata.json
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
from pathlib import Path

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
TRAFIC = Path(os.environ.get("SPACE_TRACKER_DATA_ROOT",
                             "/data/anon/trafic"))

#: Layer B.  Quotes are from the papers in ``related_work/``; see
#: ``source_paper_specs`` for the exact page references.
#: OOTB is deliberately absent: it mixes JL, SkySat and ISS, so it has no
#: dataset-level GSD or frame rate and must be resolved per parent scene.
#: SatSOT is present but with fps ``None``: the paper gives "10 or 25 fps" for
#: the raw videos and no per-sequence table.
DATASET_CONSTANTS = {
    "satmtb":   dict(platform="Jilin-1 03", gsd_m=0.92, fps=10.0, src="paper"),
    "viso":     dict(platform="Jilin-1", gsd_m=0.92, fps=10.0, src="paper"),
    # RsCarData is VISO's car subset re-annotated under the HiEUM protocol, so
    # it is the same imagery and carries VISO's constants. Tagged "paper" for
    # that reason -- "inherited" is reserved for a value propagated through the
    # match graph, and letting the two share a tag makes the summary unreadable.
    "rscardata": dict(platform="Jilin-1", gsd_m=0.92, fps=10.0, src="paper",
                      note="VISO car subset; constants are VISO's"),
    "sv248s":   dict(platform="Jilin-1", gsd_m=0.92, fps=25.0, src="paper",
                     note="25 fps is VFI-upsampled from a native 10 fps"),
    "sdmcar":   dict(platform="Luojia 3-01", gsd_m=0.75, fps=None, src="paper",
                     note="fps is per video, read from the AVI containers"),
    "satsot":   dict(platform="Jilin-1 / Skybox / Carbonite-2", gsd_m=None,
                     fps=None, src="unverified",
                     note="paper gives 10 or 25 fps and no per-sequence platform"),
    "ootb":     dict(platform="Jilin-1 / SkySat / ISS", gsd_m=None, fps=None,
                     src="unverified",
                     note="multi-platform by design; no dataset-level constant"),
}

#: GSD inherited from a matched neighbour is only trusted above this many
#: RANSAC inliers. Below it the affine fit is close enough to the acceptance
#: threshold that its *scale* is not reliable even when the match itself is —
#: the seven sequences that came out at 0.34 / 0.65 / 1.01 m all sat at 12-20
#: inliers, against a median of 172 for a solid match.
GSD_INHERIT_MIN_INLIERS = 50

#: A track whose centre never moves this far in total is stationary.
STATIC_TOTAL_PX = 2.0
#: Chord length used for the displacement estimate. Long enough that integer
#: box rounding (SatSOT ships integer HBBs) stops dominating the difference.
CHORD = 10


def sdmcar_fps() -> dict[str, float]:
    """``{source sequence stem: fps}`` straight out of the AVI headers."""
    import cv2
    out = {}
    for p in glob.glob(str(TRAFIC / "SDM-Car" / "**" / "*.avi"), recursive=True):
        if os.path.basename(p).startswith("._"):
            continue
        cap = cv2.VideoCapture(p)
        fps = round(cap.get(cv2.CAP_PROP_FPS), 3)
        cap.release()
        out[Path(p).stem] = fps
    return out


def inherited_gsd(scene_groups: Path, dataset_of: dict[str, str]) -> dict[str, float]:
    """GSD for sequences whose own dataset has none, taken from what they match.

    The fitted scale of a pair is the ratio of their ground sample distances, so
    a sequence that overlaps a Jilin-1 crop at scale 1.0 is itself at 0.92 m.
    Propagated best-first through the match graph so a strong chain always wins
    over a weak one, and only over edges that clear
    ``GSD_INHERIT_MIN_INLIERS``.
    """
    import heapq
    if not scene_groups.exists():
        return {}
    edges = json.loads(scene_groups.read_text())["edges"]
    adj: dict[str, list] = collections.defaultdict(list)
    for e in edges:
        if e["inliers"] < GSD_INHERIT_MIN_INLIERS:
            continue
        # scale maps a onto b, so gsd_a = scale * gsd_b.
        adj[e["a"]].append((e["b"], 1.0 / e["scale"], e["inliers"]))
        adj[e["b"]].append((e["a"], e["scale"], e["inliers"]))

    known = {k: DATASET_CONSTANTS[d]["gsd_m"] for k, d in dataset_of.items()
             if DATASET_CONSTANTS.get(d, {}).get("gsd_m") is not None}
    gsd = dict(known)
    conf = {k: float("inf") for k in known}
    pq = [(-float("inf"), k) for k in known]
    heapq.heapify(pq)
    while pq:
        negc, u = heapq.heappop(pq)
        if -negc < conf.get(u, -1):
            continue
        for v, mult, inl in adj.get(u, ()):
            c = min(-negc, inl)
            if c > conf.get(v, 0):
                conf[v], gsd[v] = c, gsd[u] * mult
                heapq.heappush(pq, (-c, v))
    return {k: round(v, 3) for k, v in gsd.items() if k not in known}


def load_half(half: str, fname: str) -> tuple[list[dict], dict]:
    data = json.loads((RELEASE / half / "annotations" / fname).read_text())
    cats = {c["id"]: c["name"] for c in data["categories"]}
    img = {i["id"]: i for i in data["images"]}
    per_video: dict[int, list] = collections.defaultdict(list)
    frames: dict[int, set] = collections.defaultdict(set)
    for a in data["annotations"]:
        im = img[a["image_id"]]
        per_video[im["video_id"]].append(
            (im.get("frame_id", 0), a.get("track_id", a["id"]),
             a["bbox"], cats[a["category_id"]]))
        frames[im["video_id"]].add(im.get("frame_id", 0))
    return data["videos"], dict(per_video=per_video, frames=frames, images=img)


def measure(rows: list) -> dict:
    """Layer A statistics for one sequence."""
    by_track: dict = collections.defaultdict(list)
    per_frame: collections.Counter = collections.Counter()
    for f, t, b, c in rows:
        by_track[t].append((f, b))
        per_frame[f] += 1

    sizes, steps, statics, gapped, late = [], [], 0, 0, 0
    first_frame = min(f for f, _, _, _ in rows)
    for t, seq in by_track.items():
        seq.sort()
        f = np.array([s[0] for s in seq], float)
        b = np.array([s[1] for s in seq], float)
        sizes.extend(np.sqrt(b[:, 2] * b[:, 3]).tolist())
        c = np.stack([b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], axis=1)
        total = float(np.linalg.norm(c[-1] - c[0])) if len(c) > 1 else 0.0
        path = float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum()) if len(c) > 1 else 0.0
        if path < STATIC_TOTAL_PX:
            statics += 1
        if len(c) > CHORD:
            steps.append(float(np.median(
                np.linalg.norm(c[CHORD:] - c[:-CHORD], axis=1) / CHORD)))
        elif len(c) > 1:
            steps.append(path / (len(c) - 1))
        # A hole in the frame index of a track is where the annotator lost the
        # object -- occlusion, or it left the frame and came back.
        if len(f) > 1 and (np.diff(f) > 1).any():
            gapped += 1
        if f[0] > first_frame:
            late += 1

    sizes_a = np.array(sizes) if sizes else np.array([0.0])
    steps_a = np.array(steps) if steps else np.array([0.0])
    return {
        "n_tracks": len(by_track),
        "n_boxes": len(rows),
        "objects_per_frame_mean": round(float(np.mean(list(per_frame.values()))), 2),
        "objects_per_frame_max": int(max(per_frame.values())),
        "sqrt_area_px_median": round(float(np.median(sizes_a)), 2),
        "sqrt_area_px_p10": round(float(np.percentile(sizes_a, 10)), 2),
        "sqrt_area_px_p90": round(float(np.percentile(sizes_a, 90)), 2),
        "px_per_frame_median": round(float(np.median(steps_a)), 3),
        "px_per_frame_p90": round(float(np.percentile(steps_a, 90)), 3),
        # Max over tracks, not a percentile over them. "Something in this video
        # moves fast" is a statement about the fastest object, and in a MOT
        # sequence with 200 slow cars a single fast aircraft does not shift any
        # percentile. Checked against the sequences a human tagged
        # ``fast_motion``: on the p90 those 19 spread across the whole
        # distribution, on the max they all sit above its 51st percentile.
        "px_per_frame_max": round(float(np.max(steps_a)), 3),
        "frac_tracks_static": round(statics / max(len(by_track), 1), 3),
        "frac_tracks_with_gap": round(gapped / max(len(by_track), 1), 3),
        "frac_tracks_late_entry": round(late / max(len(by_track), 1), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-groups", type=Path,
                    default=_PKG / "scene_groups.json")
    ap.add_argument("--out", type=Path,
                    default=_PKG / "sequence_metadata.json")
    args = ap.parse_args()

    group_of: dict[str, int] = {}
    if args.scene_groups.exists():
        for g in json.loads(args.scene_groups.read_text())["groups"]:
            for k in g["sequences"]:
                group_of[k] = g["group_id"]

    avi_fps = sdmcar_fps()
    dataset_of: dict[str, str] = {}
    for half, fname in (("sot", "space_tracker_sot.json"),
                        ("mot", "space_tracker_mot.json")):
        for v in json.loads((RELEASE / half / "annotations" / fname).read_text())["videos"]:
            dataset_of[f"{half}/{v['name']}"] = v.get("source_dataset")
    gsd_inherited = inherited_gsd(args.scene_groups, dataset_of)

    records = []
    for half, fname in (("sot", "space_tracker_sot.json"),
                        ("mot", "space_tracker_mot.json")):
        videos, idx = load_half(half, fname)
        for v in videos:
            rows = idx["per_video"].get(v["id"], [])
            if not rows:
                continue
            ds = v.get("source_dataset")
            const = dict(DATASET_CONSTANTS.get(ds, {}))
            note = const.pop("note", None)
            src = const.pop("src", "unverified")
            key = f"{half}/{v['name']}"
            gsd, gsd_src = const.get("gsd_m"), src
            if gsd is None and key in gsd_inherited:
                gsd, gsd_src = gsd_inherited[key], "inherited"
            fps, fps_src = const.get("fps"), src
            if ds == "sdmcar":
                stem = Path(v.get("source_sequence_id", "")).name
                if stem in avi_fps:
                    fps, fps_src = avi_fps[stem], "derived"

            rec = {
                "key": key,
                "half": half,
                "name": v["name"],
                "source_dataset": ds,
                "source_sequence_id": v.get("source_sequence_id"),
                "category": v.get("category"),
                "categories_in_sequence": v.get("categories_in_sequence"),
                "parent_scene_group": group_of.get(key),
                # Layer A
                "width": v.get("width"), "height": v.get("height"),
                "n_frames": len(idx["frames"][v["id"]]),
                **measure(rows),
                # Layer B
                "platform": const.get("platform"),
                "gsd_m": gsd,
                "gsd_source": gsd_src if gsd is not None else "unverified",
                "fps": fps,
                "fps_source": fps_src if fps is not None else "unverified",
                "layer_b_note": note,
                # Layer C -- human only, left empty on purpose
                "weather": None,
                "illumination_change": None,
                "low_contrast": None,
            }
            records.append(rec)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"n_sequences": len(records), "sequences": records}, indent=2) + "\n")

    print(f"{len(records)} sequences -> {args.out}")
    for field in ("gsd_source", "fps_source"):
        c = collections.Counter(r[field] for r in records)
        print(f"  {field:12s} " + "  ".join(f"{k}={n}" for k, n in c.most_common()))
    grouped = sum(r["parent_scene_group"] is not None for r in records)
    print(f"  parent_scene_group set for {grouped}/{len(records)}")


if __name__ == "__main__":
    main()
