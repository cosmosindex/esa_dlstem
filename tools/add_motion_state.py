#!/usr/bin/env python
"""Label every Space-Tracker-MOT track ``moving`` / ``static`` / ``intermittent``.

Why
---
Space-Tracker-MOT annotates stationary aircraft, ships and trains that the
source datasets omit, because a single-frame appearance-based detector cannot be
supervised on a label that flips with motion state.  But feeding those tracks
into MOT metrics unchanged inflates them: a stationary object is a trivial
association problem, so MOTA and IDF1 rise for reasons that have nothing to do
with a tracker's ability to associate.  Numbers computed on the enriched
annotation would not be comparable with numbers published on the original
moving-only ones.

Labelling each track lets one annotation serve both protocols --- ``full``, and
``moving-only`` which drops the static tracks and reproduces the older
convention.

How
---
Speed is measured in **metres per second**, not pixels per frame.  The subsets
differ in ground sample distance (0.92 m vs 0.75 m) and in frame rate (6, 8 and
10 fps), so a pixel threshold would mean a different physical speed in each and
would silently classify the same vehicle differently depending on which
satellite filmed it.  Every MOT sequence has both constants from
``sequence_metadata.json``.

For each track we take the displacement of the box centre over a sliding
``WINDOW``-frame chord, which averages out the sub-pixel jitter that per-frame
differencing amplifies, and convert it with ``gsd * fps``.  A window counts as
moving above ``MOVING_MPS``; the track is then classified by the fraction of its
windows that move.

Outputs (idempotent, additive --- existing fields are overwritten, nothing else
is touched):

* ``motion_state``, ``moving_fraction`` and ``median_speed_mps`` on every entry
  of the COCO-VID ``tracks`` table;
* ``n_moving_tracks`` / ``n_static_tracks`` / ``n_intermittent_tracks`` on every
  video;
* ``<seq>/gt/motion_state.txt`` beside each ``gt.txt``.  A sidecar rather than a
  tenth column, because adding a column to ``gt.txt`` would break the
  MOTChallenge parsers the layout exists to satisfy.

Usage
-----
    python tools/add_motion_state.py
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

import numpy as np

RELEASE = Path(os.environ.get("SPACE_TRACKER_RELEASE",
                              "/data/anon/release/space_tracker"))

#: Chord length, in frames, used to measure speed. Per-frame differencing on a
#: 6 px car is dominated by annotation rounding; over 10 frames it is not.
WINDOW = 10

#: A window moves above this ground speed. 1 m/s is 3.6 km/h -- slower than
#: anything that drives, sails or taxis, and about twice the measured noise
#: floor: the p10 of both the aircraft and the ship distributions sits at
#: 0.46 m/s, which is exactly half a pixel over a 10-frame chord at 0.92 m and
#: 10 fps, i.e. annotation jitter rather than motion.
MOVING_MPS = 1.0

#: A track is static when its box centre never leaves the object's own
#: footprint: the diagonal of the centre cloud is at most this multiple of the
#: object's median sqrt(area). Speed alone is not enough -- a parked aircraft's
#: box wobbles, and over hundreds of windows a few of those wobbles cross any
#: sensible speed threshold, which is why a speed-only rule called 80% of the
#: parked aircraft "intermittent". Displacement relative to the object's own
#: size is the physical statement that matters, and it separates cleanly:
#: aircraft sit at a median ratio of 0.19 and cars at 24.9, with the aircraft
#: p75 (0.58) still below the car p5 (2.28).
STATIC_MAX_SPAN_RATIO = 0.5

#: Fraction of windows that must move before a track counts as moving
#: throughout. Below it, and not static, the track stops and starts -- a vehicle
#: at a light, a vessel berthing -- which is a third case, not a noisy version
#: of the other two.
MOVING_MIN_FRAC = 0.95

#: ``unknown`` is a single-box track: no displacement to measure.
STATES = ("moving", "intermittent", "static", "unknown")


def classify(centres: np.ndarray, size_px: float,
             mps_per_px_per_frame: float | None) -> tuple[str, float, float, float]:
    """(state, fraction of moving windows, median speed m/s, span ratio).

    A single box carries no motion evidence at all and is left ``unknown``;
    two boxes give one displacement sample, which is thin but is still an
    observation, so it is classified rather than discarded.
    """
    if len(centres) < 2 or not mps_per_px_per_frame:
        return "unknown", -1.0, -1.0, -1.0
    w = min(WINDOW, len(centres) - 1)
    speed = (np.linalg.norm(centres[w:] - centres[:-w], axis=1) / w
             * mps_per_px_per_frame)
    frac = float((speed >= MOVING_MPS).mean())
    median_mps = float(np.median(speed))
    span = float(np.linalg.norm(centres.max(axis=0) - centres.min(axis=0)))
    ratio = span / max(size_px, 1e-6)

    # Both conditions, because either alone misfires. Span alone would call a
    # moving train static -- a train is long and thin, so travelling its own
    # length barely moves the ratio. Speed alone would call a parked aircraft
    # intermittent.
    if ratio <= STATIC_MAX_SPAN_RATIO and median_mps < MOVING_MPS:
        state = "static"
    elif frac >= MOVING_MIN_FRAC:
        state = "moving"
    else:
        state = "intermittent"
    return state, round(frac, 3), round(median_mps, 3), round(ratio, 3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", type=Path,
                    default=Path("docs/space_tracker/sequence_metadata.json"))
    ap.add_argument("--dry-run", action="store_true",
                    help="report the distribution without writing anything")
    args = ap.parse_args()

    coco_path = RELEASE / "mot" / "annotations" / "space_tracker_mot.json"
    data = json.loads(coco_path.read_text())
    meta = {r["key"]: r for r in json.loads(args.metadata.read_text())["sequences"]}

    images = {i["id"]: i for i in data["images"]}
    videos = {v["id"]: v for v in data["videos"]}
    cats = {c["id"]: c["name"] for c in data["categories"]}

    boxes: dict[int, dict[int, list]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for a in data["annotations"]:
        im = images[a["image_id"]]
        boxes[im["video_id"]][a["track_id"]].append((im["frame_id"], a["bbox"]))

    verdict: dict[int, tuple[str, float, float]] = {}
    missing_constants = 0
    for vid, tracks in boxes.items():
        m = meta.get(f"mot/{videos[vid]['name']}", {})
        gsd, fps = m.get("gsd_m"), m.get("fps")
        scale = gsd * fps if gsd and fps else None
        if scale is None:
            missing_constants += 1
        for tid, rows in tracks.items():
            rows.sort()
            b = np.array([r[1] for r in rows], float)
            centres = np.stack([b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], axis=1)
            size_px = float(np.median(np.sqrt(b[:, 2] * b[:, 3])))
            verdict[tid] = classify(centres, size_px, scale)

    per_state = collections.Counter(v[0] for v in verdict.values())
    per_class: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter)
    for t in data["tracks"]:
        per_class[cats[t["category_id"]]][verdict[t["id"]][0]] += 1

    print(f"{len(verdict)} tracks classified "
          f"({missing_constants} sequences without gsd/fps)")
    print(f"  window {WINDOW} frames; moving above {MOVING_MPS} m/s; static if "
          f"centre span <= {STATIC_MAX_SPAN_RATIO} x own size and median speed "
          f"below threshold; moving if >= {MOVING_MIN_FRAC:.0%} of windows move")
    print("\n  " + "  ".join(f"{s}={per_state[s]} "
                             f"({100*per_state[s]/len(verdict):.1f}%)"
                             for s in STATES))
    print("\n  by class")
    for cl in sorted(per_class):
        c = per_class[cl]
        n = sum(c.values())
        print(f"    {cl:9s} n={n:6d}  " +
              "  ".join(f"{s}={c[s]:6d} ({100*c[s]/n:5.1f}%)" for s in STATES))

    if args.dry_run:
        print("\ndry run, nothing written")
        return

    for t in data["tracks"]:
        state, frac, speed, ratio = verdict[t["id"]]
        t["motion_state"] = state
        t["moving_fraction"] = frac
        t["median_speed_mps"] = speed
        t["centre_span_over_size"] = ratio

    by_video: dict[int, collections.Counter] = collections.defaultdict(
        collections.Counter)
    for t in data["tracks"]:
        by_video[t["video_id"]][t["motion_state"]] += 1
    for v in data["videos"]:
        c = by_video[v["id"]]
        for s in STATES:
            v[f"n_{s}_tracks"] = c[s]

    data["info"]["motion_state"] = (
        f"Per-track motion label. Speed is the box-centre displacement over a "
        f"{WINDOW}-frame chord, converted to m/s with the sequence's ground "
        f"sample distance and frame rate; a window moves above {MOVING_MPS} m/s. "
        f"static: the box centre never leaves the object's own footprint (span "
        f"at most {STATIC_MAX_SPAN_RATIO} x its median sqrt(area)) and its "
        f"median speed is below threshold. moving: at least "
        f"{MOVING_MIN_FRAC:.0%} of windows move. intermittent: in between -- a vehicle "
        f"that stops, or a vessel that berths. The 'moving-only' evaluation "
        f"protocol drops static tracks and keeps intermittent ones whole; the "
        f"'full' protocol uses every track."
    )
    coco_path.write_text(json.dumps(data, indent=2) + "\n")

    # Sidecar per sequence, so gt.txt stays a valid MOTChallenge file.
    local = collections.defaultdict(dict)
    for t in data["tracks"]:
        local[t["video_id"]][t["local_track_id"]] = t
    written = 0
    for vid, tracks in local.items():
        v = videos[vid]
        seq_dir = RELEASE / "mot" / v["category"] / v["name"] / "gt"
        if not seq_dir.is_dir():
            continue
        lines = ["# track_id,motion_state,moving_fraction,median_speed_mps,centre_span_over_size"]
        for ltid in sorted(tracks):
            t = tracks[ltid]
            lines.append(f"{ltid},{t['motion_state']},{t['moving_fraction']},"
                         f"{t['median_speed_mps']},{t['centre_span_over_size']}")
        (seq_dir / "motion_state.txt").write_text("\n".join(lines) + "\n")
        written += 1

    print(f"\nwritten: {coco_path}")
    print(f"         {written} x gt/motion_state.txt")


if __name__ == "__main__":
    main()
