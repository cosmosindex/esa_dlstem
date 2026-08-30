"""Audit the assembled space-tracker release.

The package states the same boxes twice — once per sequence in the CVPR
benchmark layout, once in the COCO-VID JSON — and both have to agree with the
sources they were derived from.  Three independent checks, all of which have to
pass before anything is uploaded:

1. **Round-trip against the source.**  SOT rows are diffed against
   ``space_tracker.data.iter_frames`` reading the native OOTB / SatSOT / SV248S
   files; MOT rows are diffed against the internal 11-column export, allowing
   for the deliberate 1-indexing of SDM-Car's frames and the renumbering of its
   track ids.
2. **JSON vs per-sequence files.**  Every box in the COCO-VID document must
   appear in the sequence's own ``gt.txt`` / ``groundtruth.txt`` and vice versa.
3. **Structural sanity.**  Unique ids, categories in range, boxes inside the
   frame, no zero-area boxes, no track carrying two categories, manifest counts
   matching what is on disk.

Usage::

    python tools/verify_space_tracker_release.py --out /data/<root>/release/space_tracker
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from space_tracker.data import iter_frames
from space_tracker.manifest import SequenceRecord
from tools.build_space_tracker_release import sot_source_records

REPO = Path(__file__).resolve().parents[1]

# Same environment contract as tools/build_space_tracker_release.py.
TRAFIC = Path(os.environ.get("SPACE_TRACKER_DATA_ROOT", "/data/anon/trafic"))
MOT_RELEASE = Path(os.environ.get("SPACE_TRACKER_MOT_RELEASE",
                                  "/work/anon/space_tracker_mot_release"))
MOT_SRC_MANIFEST = MOT_RELEASE / "space_tracker_mot_reviewed.json"
SOT_ROOTS = {"ootb": str(TRAFIC / "OOTB"), "satsot": str(TRAFIC / "SatSOT"),
             "sv248s": str(TRAFIC / "SV248S")}


def check_sot_roundtrip(out: Path, mani: dict, problems: list[str]) -> None:
    by_id = {r["id"]: SequenceRecord.from_dict(r) for r in sot_source_records()}
    for rec in mani["sequences"]:
        rows = [l.split(",") for l in
                (out / rec["gt_path"]).read_text().splitlines() if l.strip()]
        native = list(iter_frames(by_id[rec["source_sequence_id"]], SOT_ROOTS))
        if len(rows) != len(native):
            problems.append(f"sot {rec['name']}: {len(rows)} rows vs "
                            f"{len(native)} native frames")
            continue
        for i, (row, frame) in enumerate(zip(rows, native), start=1):
            if int(row[0]) != i:
                problems.append(f"sot {rec['name']} row {i}: frame_id {row[0]}")
                break
            vis = bool(int(row[5]))
            if vis != bool(frame.visible and frame.gt_box_xyxy is not None):
                problems.append(f"sot {rec['name']} frame {i}: visible {vis}")
                break
            if not vis:
                continue
            x, y, w, h = (float(v) for v in row[1:5])
            if not np.allclose([x, y, x + w, y + h],
                               np.asarray(frame.gt_box_xyxy, float), atol=0.01):
                problems.append(f"sot {rec['name']} frame {i}: box mismatch")
                break
            if frame.gt_obb_8pt is not None and not np.allclose(
                    [float(v) for v in row[7:15]],
                    np.asarray(frame.gt_obb_8pt, float), atol=0.01):
                problems.append(f"sot {rec['name']} frame {i}: obb mismatch")
                break


def check_mot_roundtrip(out: Path, mani: dict, problems: list[str]) -> None:
    """Diff against the source, replaying the documented boundary cleanup.

    The release clips boxes to the image and drops the ones that lie wholly
    outside it, so the source is not expected to match byte for byte — it is
    expected to match *after the same clip*. Applying the rule here rather than
    exempting the difference keeps the check honest: an accidental change to any
    other box still shows up.
    """
    src = {s["id"]: s for s in json.loads(MOT_SRC_MANIFEST.read_text())["sequences"]}
    for rec in mani["sequences"]:
        s = src[rec["source_sequence_id"]]
        base = s["frame_index_base"]
        w_img, h_img = float(s["img_width"]), float(s["img_height"])
        want: Counter = Counter()
        for line in (MOT_RELEASE / s["gt_path"]).read_text().splitlines():
            if not line.strip():
                continue
            p = line.split(",")
            x, y = float(p[2]), float(p[3])
            x1, y1 = max(x, 0.0), max(y, 0.0)
            x2 = min(x + float(p[4]), w_img)
            y2 = min(y + float(p[5]), h_img)
            if x2 - x1 < 1.0 or y2 - y1 < 1.0:
                continue
            want[(int(float(p[0])) - base + 1, round(x1, 2), round(y1, 2),
                  round(x2 - x1, 2), round(y2 - y1, 2),
                  int(float(p[7])) + 1)] += 1
        got: Counter = Counter()
        for line in (out / rec["gt_path"]).read_text().splitlines():
            if not line.strip():
                continue
            p = line.split(",")
            got[(int(p[0]), round(float(p[2]), 2), round(float(p[3]), 2),
                 round(float(p[4]), 2), round(float(p[5]), 2), int(p[7]))] += 1
        if want != got:
            miss, extra = want - got, got - want
            problems.append(f"mot {rec['name']}: {sum(miss.values())} boxes lost, "
                            f"{sum(extra.values())} invented")


def check_json_vs_files(out: Path, task: str, mani: dict,
                        problems: list[str]) -> dict:
    doc = json.loads((out / mani["coco_annotations"]).read_text())
    stats = {"images": len(doc["images"]), "annotations": len(doc["annotations"]),
             "videos": len(doc["videos"]), "tracks": len(doc["tracks"])}

    for key in ("images", "annotations", "videos", "tracks"):
        ids = [e["id"] for e in doc[key]]
        if len(set(ids)) != len(ids):
            problems.append(f"{task} json: duplicate {key} ids")

    n_cats = len(mani["categories"])
    img_by_id = {im["id"]: im for im in doc["images"]}
    track_cat: dict[int, set] = defaultdict(set)
    per_video: dict[int, Counter] = defaultdict(Counter)
    for ann in doc["annotations"]:
        if not 1 <= ann["category_id"] <= n_cats:
            problems.append(f"{task} json ann {ann['id']}: category out of range")
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            problems.append(f"{task} json ann {ann['id']}: non-positive box")
        im = img_by_id.get(ann["image_id"])
        if im is None:
            problems.append(f"{task} json ann {ann['id']}: dangling image_id")
            continue
        if im["video_id"] != ann["video_id"]:
            problems.append(f"{task} json ann {ann['id']}: video_id disagrees "
                            "with its image")
        if (task == "mot"
                and (x < 0 or y < 0
                     or x + w > im["width"] + 0.01
                     or y + h > im["height"] + 0.01)):
            problems.append(f"{task} json ann {ann['id']}: box outside the frame")
        track_cat[ann["track_id"]].add(ann["category_id"])
        per_video[ann["video_id"]][(im["frame_id"], round(x, 2), round(y, 2),
                                    round(w, 2), round(h, 2),
                                    ann["category_id"])] += 1
    for tid, cats in track_cat.items():
        if len(cats) > 1:
            problems.append(f"{task} json track {tid}: carries {len(cats)} categories")

    video_name = {v["id"]: v["name"] for v in doc["videos"]}
    rec_by_name = {r["name"]: r for r in mani["sequences"]}
    for vid, boxes in per_video.items():
        rec = rec_by_name[video_name[vid]]
        on_disk: Counter = Counter()
        for line in (out / rec["gt_path"]).read_text().splitlines():
            if not line.strip():
                continue
            p = line.split(",")
            if task == "sot":
                if int(p[5]) == 0:
                    continue
                on_disk[(int(p[0]), round(float(p[1]), 2), round(float(p[2]), 2),
                         round(float(p[3]), 2), round(float(p[4]), 2),
                         mani["categories"][rec["category"]])] += 1
            else:
                on_disk[(int(p[0]), round(float(p[2]), 2), round(float(p[3]), 2),
                         round(float(p[4]), 2), round(float(p[5]), 2),
                         int(p[7]))] += 1
        if on_disk != boxes:
            problems.append(f"{task} {rec['name']}: json has {sum(boxes.values())} "
                            f"boxes, gt file has {sum(on_disk.values())}")
    return stats


def check_manifest_vs_disk(out: Path, task: str, mani: dict,
                           problems: list[str]) -> None:
    seen = set()
    for rec in mani["sequences"]:
        d = out / rec["path"]
        if not d.is_dir():
            problems.append(f"{task} {rec['name']}: directory missing")
            continue
        if rec["name"] in seen:
            problems.append(f"{task} {rec['name']}: duplicate sequence name")
        seen.add(rec["name"])
        for f in (("groundtruth.txt", "seqinfo.ini") if task == "sot"
                  else ("gt/gt.txt", "seqinfo.ini")):
            if not (d / f).exists():
                problems.append(f"{task} {rec['name']}: {f} missing")
    if len(mani["sequences"]) != mani["n_sequences"]:
        problems.append(f"{task}: n_sequences disagrees with the sequence list")
    on_disk = {p.name for p in (out / task).iterdir()
               if p.is_dir() and p.name != "annotations"}
    declared = {r["category"] for r in mani["sequences"]}
    if on_disk != declared:
        problems.append(f"{task}: folders {sorted(on_disk)} vs categories "
                        f"{sorted(declared)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skip-roundtrip", action="store_true",
                    help="skip the (slow) diff against the source datasets")
    args = ap.parse_args()

    problems: list[str] = []
    summary = {}
    for task, name in (("sot", "space_tracker_sot.json"),
                       ("mot", "space_tracker_mot.json")):
        mani = json.loads((args.out / task / name).read_text())
        print(f"[{task}] {mani['n_sequences']} sequences")
        check_manifest_vs_disk(args.out, task, mani, problems)
        summary[task] = check_json_vs_files(args.out, task, mani, problems)
        print(f"  json/file cross-check done ({summary[task]})")
        if not args.skip_roundtrip:
            (check_sot_roundtrip if task == "sot" else check_mot_roundtrip)(
                args.out, mani, problems)
            print("  round-trip against source done")

    print(f"\n{len(problems)} problems")
    for p in problems[:40]:
        print("   ", p)
    if len(problems) > 40:
        print(f"    ... and {len(problems) - 40} more")
    print(json.dumps(summary, indent=1))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
