#!/usr/bin/env python
"""Check that Space-Tracker-SOT reads the same boxes as its three sources.

The published SOT numbers were produced by running each tracker on OOTB, SatSOT
and SV248S in their original layouts and aggregating over the released
sequences (``tools/make_wacv_sot_table.py``). ``datasets/space_tracker_sot.py``
reads the release instead. The two paths have to describe the same ground
truth, or a result obtained through one is not comparable with a result
obtained through the other -- so this compares them box by box.

For every released sequence, both paths are asked for every frame and the
comparison covers: the number of frames, which frames carry the target at all,
and the axis-aligned box on each of those, to within ``--tol`` px.

The sources are 0-indexed where the release is 1-indexed, and SatSOT writes the
literal ``none`` where the release writes ``visible = 0``; both are differences
in encoding and the loaders already resolve them, which is exactly what this
checks.

The default tolerance is two hundredths of a pixel and not zero, because the
release writes its geometry at two decimal places (``f"{v:.2f}"``). A corner
coordinate therefore moves by at most 0.005, and ``x2 = x + w`` accumulates two
of those, so 0.02 is the widest disagreement rounding alone can produce. On a
benchmark whose median target is a few pixels across, anything larger would be
a real difference and is reported as one.

Usage::

    python tools/check_sot_release_equivalence.py
    python tools/check_sot_release_equivalence.py --datasets ootb --limit 20
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.ootb import OOTBDataset  # noqa: E402
from datasets.satsot import SatSOTDataset  # noqa: E402
from datasets.space_tracker_sot import SpaceTrackerSOTDataset  # noqa: E402
from datasets.sv248s import SV248SDataset  # noqa: E402
from project_paths import DATA_ROOT  # noqa: E402

SOURCES = {
    "ootb":   (OOTBDataset,   f"{DATA_ROOT}/data/trafic/OOTB"),
    "satsot": (SatSOTDataset, f"{DATA_ROOT}/data/trafic/SatSOT"),
    "sv248s": (SV248SDataset, f"{DATA_ROOT}/data/trafic/SV248S"),
}
CLASS_MAP = {"car": 0, "airplane": 1, "plane": 1, "ship": 2, "train": 3,
             "car-large": 0}


def boxes_of(ds, video) -> dict[int, np.ndarray]:
    """``{frame index: xyxy}`` for the frames whose target is present."""
    out = {}
    for fid in video.frame_ids:
        ann = ds._load_annotations(video, fid)
        if len(ann["boxes"]):
            out[fid] = np.asarray(ann["boxes"][0], dtype=np.float64)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", default=None,
                    help="release root; default $SPACE_TRACKER_ROOT")
    ap.add_argument("--datasets", nargs="*", default=list(SOURCES))
    ap.add_argument("--limit", type=int, default=None,
                    help="check at most this many sequences per source")
    ap.add_argument("--tol", type=float, default=0.02,
                    help="per-coordinate tolerance in px; the default is the "
                         "most that the release's 2-decimal geometry can "
                         "differ by from the source's full precision")
    args = ap.parse_args()

    rel = SpaceTrackerSOTDataset(args.release, split="no_split",
                                 class_map=CLASS_MAP)
    # Released name -> its source id, from the manifest the loader just read.
    import json
    manifest = json.loads(
        (rel.root / "sot" / "space_tracker_sot.json").read_text())
    src_of = {r["name"]: (r["dataset"], r["source_sequence_id"])
              for r in manifest["sequences"]}
    rel_by_name = {v.video_id: v for v in rel.videos}

    totals, problems = Counter(), []
    for ds_key in args.datasets:
        cls, root = SOURCES[ds_key]
        src = cls(root, split="no_split", class_map=CLASS_MAP)
        src_by_id = {v.video_id: v for v in src.videos}

        names = [n for n, (d, _) in src_of.items() if d == ds_key]
        names.sort()
        if args.limit:
            names = names[:args.limit]

        for name in names:
            totals["checked"] += 1
            _, sid = src_of[name]
            vid = sid.split("/", 1)[1] if sid.startswith(ds_key + "/") else sid
            sv = src_by_id.get(vid)
            if sv is None:
                problems.append((name, f"no source sequence {vid!r}")); continue

            rv = rel_by_name[name]
            a, b = boxes_of(rel, rv), boxes_of(src, sv)
            if rv.num_frames != sv.num_frames:
                problems.append(
                    (name, f"frames {rv.num_frames} vs {sv.num_frames}")); continue
            if a.keys() != b.keys():
                only_rel, only_src = a.keys() - b.keys(), b.keys() - a.keys()
                problems.append(
                    (name, f"visible frames differ: +{len(only_rel)} / -{len(only_src)}"))
                continue
            worst = max((np.abs(a[k] - b[k]).max() for k in a), default=0.0)
            if worst > args.tol:
                problems.append((name, f"box differs by {worst:.3f} px")); continue
            totals["identical"] += 1
            totals["boxes"] += len(a)
        print(f"  {ds_key}: {len(names)} sequences checked", flush=True)

    print(f"\n{totals['identical']}/{totals['checked']} sequences identical "
          f"({totals['boxes']:,} boxes compared, tolerance {args.tol} px)")
    if problems:
        print(f"\n{len(problems)} sequence(s) differ:")
        for name, why in problems[:20]:
            print(f"  {name}: {why}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
