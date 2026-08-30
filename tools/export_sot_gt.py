"""Normalise SOT ground truth to one format, the way the MOT side already is.

The three SOT sources disagree on more than syntax. OOTB annotates an oriented
box and no absence flag; SatSOT writes the literal string ``none`` on frames
where the target is gone; SV248S keeps geometry in ``<seq>.rect`` and a separate
``<seq>.state`` file whose 0/1/2 distinguishes visible from invisible from
occluded. A user who downloads space-tracker should not have to learn all three
to read a box.

So every sequence is rewritten as one comma-separated file, one row per frame::

    frame_id,x,y,w,h,visible,state,px1,py1,px2,py2,px3,py3,px4,py4

``x,y,w,h``
    Axis-aligned box in absolute pixels, top-left origin. ``-1`` on every column
    when the target is absent — the row is still written, so row *i* is always
    frame *i* and no reader has to reconstruct the frame index from line count.
``visible``
    1 / 0. Unifies SatSOT's ``none`` and SV248S's state into one column, which
    is the only absence signal a tracker evaluation actually needs.
``state``
    The native SV248S flag (0 visible, 1 invisible, 2 occluded), ``-1``
    elsewhere. Kept because "occluded but present" is a distinction the unified
    ``visible`` column deliberately collapses, and throwing it away would make
    the export lossy against the source.
``px1..py4``
    OBB corners, ``-1`` when the source annotates no orientation (everything
    except OOTB). OOTB rows carry *both*: the polygon and the axis-aligned box
    derived from it, so ``polygon`` and ``ootb_aabb`` evaluation modes both read
    the same file.

Nothing is re-annotated here — this is a format change, and it is lossless:
``--verify`` re-reads every exported file and compares it to the native loader
box for box.

Usage::

    python tools/export_sot_gt.py --out /work/<user>/space_tracker_sot --verify
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from space_tracker.data import iter_frames
from space_tracker.manifest import Manifest
from project_paths import DATA_ROOT

SOT_MANIFEST = Path(__file__).resolve().parents[1] / "space_tracker" / "space_tracker.json"

DEFAULT_ROOTS = {
    "ootb": f"{DATA_ROOT}/data/trafic/OOTB",
    "satsot": f"{DATA_ROOT}/data/trafic/SatSOT",
    "sv248s": f"{DATA_ROOT}/data/trafic/SV248S",
}

UNIFIED_FORMAT = "sot_csv_unified"
UNIFIED_DESCRIPTION = (
    "Comma-separated, one row per frame, 15 columns: frame_id, x, y, w, h, "
    "visible, state, px1, py1, px2, py2, px3, py3, px4, py4. Coordinates are "
    "absolute pixels with a top-left origin; xywh is axis-aligned. A row is "
    "written for every frame — all-'-1' geometry means the target is absent "
    "(visible=0). 'state' is SV248S's native flag (0 visible, 1 invisible, "
    "2 occluded) and -1 for the other datasets. The eight 'p' columns are OBB "
    "corners, -1 where the source annotates no orientation (OOTB only "
    "annotates it); OOTB rows carry both the polygon and its axis-aligned box."
)


def rows_for(seq, roots: dict[str, str]) -> tuple[list[str], Counter]:
    """Unified rows for one sequence, plus a small tally."""
    counts: Counter = Counter()
    lines = []
    for frame in iter_frames(seq, roots):
        counts["frames"] += 1
        box = frame.gt_box_xyxy
        if box is None or not frame.visible:
            xywh = (-1.0, -1.0, -1.0, -1.0)
            counts["absent"] += 1
        else:
            x1, y1, x2, y2 = (float(v) for v in box)
            xywh = (x1, y1, x2 - x1, y2 - y1)
        obb = frame.gt_obb_8pt
        if obb is None:
            poly = (-1.0,) * 8
        else:
            poly = tuple(float(v) for v in obb)
            counts["with_obb"] += 1
        state = -1 if frame.state is None else int(frame.state)
        lines.append(",".join(
            [str(frame.frame_id)]
            + [f"{v:.2f}" for v in xywh]
            + [str(int(bool(frame.visible))), str(state)]
            + [f"{v:.2f}" for v in poly]))
    return lines, counts


def verify(seq, roots: dict[str, str], path: Path) -> list[str]:
    """Re-read an exported file and diff it against the native loader."""
    problems = []
    rows = [l.split(",") for l in path.read_text().splitlines() if l.strip()]
    native = list(iter_frames(seq, roots))
    if len(rows) != len(native):
        return [f"{seq.id}: {len(rows)} rows vs {len(native)} native frames"]
    for row, frame in zip(rows, native):
        if int(row[0]) != frame.frame_id:
            problems.append(f"{seq.id} frame {frame.frame_id}: id mismatch {row[0]}")
            continue
        vis = bool(int(row[5]))
        if vis != bool(frame.visible):
            problems.append(f"{seq.id} frame {frame.frame_id}: visible "
                            f"{vis} vs {frame.visible}")
        if not vis or frame.gt_box_xyxy is None:
            continue
        x, y, w, h = (float(v) for v in row[1:5])
        got = np.array([x, y, x + w, y + h])
        if not np.allclose(got, np.asarray(frame.gt_box_xyxy, float), atol=0.01):
            problems.append(f"{seq.id} frame {frame.frame_id}: box {got} vs "
                            f"{frame.gt_box_xyxy}")
        if frame.gt_obb_8pt is not None:
            poly = np.array([float(v) for v in row[7:15]])
            if not np.allclose(poly, np.asarray(frame.gt_obb_8pt, float), atol=0.01):
                problems.append(f"{seq.id} frame {frame.frame_id}: obb mismatch")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--roots", type=json.loads, default=None,
                    help='JSON, e.g. \'{"ootb": "/data/OOTB"}\'')
    ap.add_argument("--verify", action="store_true",
                    help="re-read every exported file and diff it against the "
                         "native loader")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    roots = {**DEFAULT_ROOTS, **(args.roots or {})}
    src = json.loads(SOT_MANIFEST.read_text())
    mani = Manifest.load(SOT_MANIFEST)
    by_id = {s.id: s for s in mani.sequences}

    gt_dir = args.out / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)

    kept: list[dict] = []
    totals: Counter = Counter()
    problems: list[str] = []
    for record in src["sequences"]:
        if args.only and record["id"] not in args.only:
            continue
        seq = by_id[record["id"]]
        lines, counts = rows_for(seq, roots)
        rel = Path("gt") / (seq.id.replace("/", "_") + ".txt")
        (args.out / rel).write_text("\n".join(lines) + "\n")
        totals.update(counts)
        totals["sequences"] += 1
        if args.verify:
            problems += verify(seq, roots, args.out / rel)

        out_record = dict(record)
        out_record.update({"gt_path": str(rel), "gt_format": UNIFIED_FORMAT})
        kept.append(out_record)
        if totals["sequences"] % 50 == 0:
            print(f"  {totals['sequences']} sequences...", flush=True)

    out_manifest = dict(src)
    out_manifest["sequences"] = kept
    out_manifest["n_sequences"] = len(kept)
    out_manifest["name"] = src.get("name", "") + " (unified GT)"
    for d in out_manifest.get("datasets", {}).values():
        d["gt_format"] = UNIFIED_FORMAT
        d["gt_format_description"] = UNIFIED_DESCRIPTION
    (args.out / "space_tracker_sot_unified.json").write_text(
        json.dumps(out_manifest, indent=1))

    print(f"\n{totals['sequences']} sequences, {totals['frames']} frames "
          f"({totals['absent']} with the target absent, "
          f"{totals['with_obb']} with an oriented box)")
    if args.verify:
        print(f"verify: {len(problems)} mismatches"
              + ("" if not problems else " — first few:"))
        for p in problems[:10]:
            print("   ", p)
    print(f"-> {args.out}/space_tracker_sot_unified.json")


if __name__ == "__main__":
    main()
