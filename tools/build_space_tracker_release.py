"""Assemble the public space-tracker release: one folder, two tasks, one format each.

The three SOT sources and the four MOT sources disagree about everything a user
would have to learn before reading a single box: OOTB annotates an oriented
polygon, SatSOT writes the literal string ``none`` for an absent target, SV248S
splits geometry and visibility across two sibling files, SAT-MTB writes 11-column
CSV, SDM-Car 10-column 0-indexed CSV behind an ``.avi``, VISO uses two different
layouts inside one dataset, RsCarData a COCO JSON holding every sequence at once.

This script rewrites all of it into a single tree::

    space_tracker/
      sot/<category>/<seq>/{img/, groundtruth.txt, seqinfo.ini}
      sot/annotations/space_tracker_sot.json          COCO-VID, every sequence
      sot/annotations/per_class/<category>.json
      mot/<group>/<seq>/{img1/, gt/gt.txt, seqinfo.ini}
      mot/annotations/space_tracker_mot.json          COCO-VID, every sequence
      mot/annotations/per_class/<group>.json

Two conventions, both deliberate:

*Boxes are COCO.*  Every box in every JSON is ``[x, y, width, height]`` in
absolute pixels with a top-left origin, carrying ``area`` and ``iscrowd`` — the
COCO detection contract, unchanged.  What COCO has no field for (a frame index,
an identity that persists across frames) follows the video extension that TAO,
BDD100K and YouTube-VIS all use: ``videos`` at the top level, ``video_id`` +
``frame_id`` on each image, ``track_id`` on each annotation, and a ``tracks``
table.  So a COCO reader works unmodified and a tracking reader gains identity.

*Per-sequence files are the CVPR benchmark convention.*  MOT ships
``<seq>/gt/gt.txt`` + ``seqinfo.ini`` + ``img1/`` exactly as MOTChallenge
defines it, so TrackEval runs on the release with no conversion.  SOT ships
``<seq>/groundtruth.txt`` + ``img/`` the way OTB and LaSOT do.  The JSON and the
per-sequence files describe the same boxes; neither is derived at read time.

Everything is 1-indexed — frame ids, image basenames, track ids, category ids.
The internal exports were 0-indexed on the SOT side; the release is not, because
a package that indexes one half from 0 and the other from 1 is a bug generator.

Scope: an object counts as small when the median of ``sqrt(area)`` over its own
track is <= 32 px.  SOT keeps a sequence when its single target is small; MOT
keeps a sequence when *any* track in it is small, and then keeps every track in
that sequence annotated — a large ship left unlabelled in a kept frame would
cost every detector a false positive it does not deserve.

Usage::

    python tools/build_space_tracker_release.py --stage annotations \
        --out /data/<root>/release/space_tracker
"""

from __future__ import annotations

import argparse
import configparser
import csv
import functools
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from space_tracker.data import iter_frames

REPO = Path(__file__).resolve().parents[1]

# Machine-specific roots come from the environment so nothing identifying is
# committed: SPACE_TRACKER_DATA_ROOT is the directory holding the eight source
# datasets, SPACE_TRACKER_MOT_RELEASE the reviewed internal MOT export.
TRAFIC = Path(os.environ.get("SPACE_TRACKER_DATA_ROOT", "/data/anon/trafic"))
MOT_RELEASE = Path(os.environ.get("SPACE_TRACKER_MOT_RELEASE",
                                  "/work/anon/space_tracker_mot_release"))
MOT_MANIFEST = MOT_RELEASE / "space_tracker_mot_reviewed.json"
SOT_ROOTS = {
    "ootb": str(TRAFIC / "OOTB"),
    "satsot": str(TRAFIC / "SatSOT"),
    "sv248s": str(TRAFIC / "SV248S"),
}
MOT_ROOTS = {
    "satmtb": TRAFIC / "SAT-MTB",
    "viso": TRAFIC / "VISO",
    "sdmcar": TRAFIC / "SDM-Car",
    "rscardata": TRAFIC / "RsCarData",
}

SMALL_MAX_SQRT_AREA_PX = 32.0

# SOT keeps SV248S's large-vehicle class distinct; MOT's four sources never make
# that distinction, so the two task halves have their own category tables. Only
# the spelling is harmonised ('plane' -> 'airplane').
SOT_CATEGORIES = ["car", "car-large", "airplane", "ship", "train"]
MOT_CATEGORIES = ["car", "airplane", "ship", "train"]
SOT_CATEGORY_RENAME = {"plane": "airplane"}
MOT_CLS_ID_TO_NAME = {0: "car", 1: "airplane", 2: "ship", 3: "train"}


def cat_id(names: list[str], name: str) -> int:
    return names.index(name) + 1


# --------------------------------------------------------------------------
# shared writers


#: Per-sequence acquisition constants, keyed ``<half>/<sequence name>``.
#: Ships with the loading code so the release carries the ground sample
#: distance and frame rate every physical-units computation needs -- the
#: motion-state classifier among them.
ACQUISITION_CONSTANTS = Path(__file__).resolve().parents[1] / \
    "space_tracker" / "data" / "sequence_constants.json"


@functools.lru_cache(maxsize=1)
def _acquisition_table() -> dict[str, dict]:
    return json.loads(ACQUISITION_CONSTANTS.read_text())["sequences"]


def acquisition(half: str, seq_name: str) -> dict:
    """``platform`` / ``gsd_m`` / ``fps`` and their provenance, for one sequence.

    Every field may be ``None``: OOTB and SatSOT mix imaging platforms and
    publish no per-sequence table, so parts of the SOT half are genuinely
    unknown and are recorded as such rather than guessed.
    """
    r = _acquisition_table().get(f"{half}/{seq_name}", {})
    return {k: r.get(k) for k in
            ("platform", "gsd_m", "gsd_source", "fps", "fps_source")}


def write_seqinfo(path: Path, *, name: str, im_dir: str, seq_length: int,
                  im_width: int, im_height: int, im_ext: str,
                  frame_rate: float | None = None, gsd_m: float | None = None,
                  platform: str | None = None,
                  extra: dict[str, str] | None = None) -> None:
    """MOTChallenge seqinfo.ini, plus the two constants MOTChallenge has no field for.

    ``frameRate`` carries the source dataset's acquisition rate where that
    dataset publishes one, and ``-1`` -- the MOTChallenge convention for
    absent -- where it does not; ``gsd`` is the ground sample distance in
    metres, written only when known. Together they are what converts a
    displacement in pixels into one in metres per second, which is how the
    released motion states were computed.
    """
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg["Sequence"] = {
        "name": name,
        "imDir": im_dir,
        "frameRate": f"{frame_rate:g}" if frame_rate else "-1",
        "seqLength": str(seq_length),
        "imWidth": str(im_width),
        "imHeight": str(im_height),
        "imExt": im_ext,
        **({"gsd": f"{gsd_m:g}"} if gsd_m else {}),
        **({"platform": platform} if platform else {}),
        **(extra or {}),
    }
    with open(path, "w") as f:
        cfg.write(f)


def coco_skeleton(task: str, categories: list[str], description: str) -> dict:
    return {
        "info": {
            "description": description,
            "task": task,
            "version": "1.0",
            "bbox_format": "coco_xywh_absolute_pixels_topleft_origin",
            "frame_index_base": 1,
            "small_object_criterion":
                f"median sqrt(area) over a track <= {SMALL_MAX_SQRT_AREA_PX:g} px",
        },
        "categories": [{"id": cat_id(categories, n), "name": n,
                        "supercategory": "vehicle"} for n in categories],
        "videos": [],
        "images": [],
        "annotations": [],
        "tracks": [],
    }


def dump_coco(doc: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f)


def split_per_class(doc: dict, out_dir: Path, group_of_video: dict[int, str]) -> None:
    """Write one COCO-VID file per class folder, sharing the global ids."""
    groups: dict[str, dict] = {}
    for video in doc["videos"]:
        g = group_of_video[video["id"]]
        sub = groups.setdefault(g, {**{k: doc[k] for k in ("info", "categories")},
                                    "videos": [], "images": [],
                                    "annotations": [], "tracks": []})
        sub["videos"].append(video)
    vid_group = group_of_video
    for image in doc["images"]:
        groups[vid_group[image["video_id"]]]["images"].append(image)
    for ann in doc["annotations"]:
        groups[vid_group[ann["video_id"]]]["annotations"].append(ann)
    for tr in doc["tracks"]:
        groups[vid_group[tr["video_id"]]]["tracks"].append(tr)
    for g, sub in groups.items():
        dump_coco(sub, out_dir / f"{g}.json")


SOT_SOURCE_INFO = {
    "ootb": {"name": "OOTB",
             "paper": "Chen et al., ISPRS J. Photogramm. Remote Sens. 2024"},
    "satsot": {"name": "SatSOT", "paper": "Zhao et al., IEEE TGRS 2022"},
    "sv248s": {"name": "SV248S", "paper": "Li et al., IEEE TGRS 2022"},
}


def _sot_taxonomy() -> dict:
    """The sequence-attribute taxonomy, imported from its single definition."""
    from tools.build_space_tracker_manifest import (
        UNIFIED_ATTR_SPEC, TAXONOMY_GROUPS, TAXONOMY_SPEC)
    return {
        "unified_attributes": UNIFIED_ATTR_SPEC,
        "attribute_taxonomy": {
            "description":
                "Sequence-attribute taxonomy. Three groups: pooled (rows "
                "more than one source annotates among the released "
                "sequences, scored over all of them), single_source "
                "(reported on the one annotating source, never pooled), and "
                "occlusion_subtypes, which drills into the pooled OCC row "
                "and is not counted among the 18. A sequence's "
                "'taxonomy_attributes' is the flat list of names it carries, "
                "derived from its native attributes via each attribute's "
                "'datasets' mapping.",
            "groups": TAXONOMY_GROUPS,
            "attributes": TAXONOMY_SPEC,
        },
    }


def release_name(category: str, dataset: str, number: int) -> str:
    """The public identifier of a sequence: what it holds, where it came from.

    The source ids cannot be reused. SAT-MTB files a sequence under ``car/42``
    because cars are what *it* annotated, but the review added the ships and
    aircraft in the same video, so the folder name actively misleads about the
    content. The category here is the one derived from the released boxes, and
    the number is assigned per source dataset in sorted source-id order, so it
    is stable across rebuilds. ``source_sequence_id`` in the manifest keeps the
    original name for anyone who needs to go back to it.
    """
    return f"{category.replace('-', '')}_{dataset}_{number:04d}"


def assign_numbers(source_ids: list[str]) -> dict[str, int]:
    """Per-dataset 1..N in sorted source-id order."""
    out: dict[str, int] = {}
    per: Counter = Counter()
    for sid in sorted(source_ids):
        ds = sid.split("/", 1)[0]
        per[ds] += 1
        out[sid] = per[ds]
    return out


def polygon_area(poly) -> float:
    x, y = np.asarray(poly[0::2], float), np.asarray(poly[1::2], float)
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def sot_source_records() -> list[dict]:
    """Sequence records read fresh from the three SOT datasets.

    Deliberately not read from ``space_tracker/space_tracker.json``: that
    manifest was built for an earlier, incremental version of this benchmark and
    its derived fields no longer describe what we ship. Categories and
    attributes come from the dataset classes, geometry from the files.
    """
    from datasets.ootb import OOTBDataset
    from datasets.satsot import SatSOTDataset
    from datasets.sv248s import SV248SDataset
    from tools.build_space_tracker_manifest import (
        _native_to_unified, _native_to_taxonomy)

    ds_objs = {
        "ootb": OOTBDataset(root=SOT_ROOTS["ootb"], split="no_split", mode="detection"),
        "satsot": SatSOTDataset(root=SOT_ROOTS["satsot"], split="no_split", mode="detection"),
        "sv248s": SV248SDataset(root=SOT_ROOTS["sv248s"], split="no_split", mode="detection"),
    }
    records = []
    for name, ds in ds_objs.items():
        attrs = ds.sequence_attributes()
        for v in ds.videos:
            vid = v.video_id
            if name == "sv248s":
                head, seq = vid.split("/")
                image_dir, gt_path = f"{head}/sequences/{seq}", f"{head}/annotations/{seq}.rect"
                gt_format = "xywh_with_state"
            else:
                image_dir, gt_path = f"{vid}/img", f"{vid}/groundtruth.txt"
                gt_format = "obb_8pt" if name == "ootb" else "xywh_with_none"
            native = sorted(attrs.get(vid, []))
            records.append({
                "id": f"{name}/{vid}", "dataset": name, "video_id": vid,
                "category": SOT_CATEGORY_RENAME.get(v.category, v.category),
                "n_frames": int(v.num_frames),
                "image_dir": image_dir, "gt_path": gt_path, "gt_format": gt_format,
                "native_attrs": native,
                "unified_attrs": _native_to_unified(name, native),
                "taxonomy_attrs": _native_to_taxonomy(name, native),
                "median_sqrt_area_px": None, "tiny": False,
            })
    return records


def image_size(path: Path) -> tuple[int, int]:
    from PIL import Image
    with Image.open(path) as im:
        return im.width, im.height


# --------------------------------------------------------------------------
# SOT


SOT_GT_HEADER = ("frame_id,x,y,w,h,visible,state,"
                 "px1,py1,px2,py2,px3,py3,px4,py4")
SOT_GT_DESCRIPTION = (
    "Comma-separated, one row per frame, 15 columns: " + SOT_GT_HEADER + ". "
    "frame_id is 1-indexed and matches the image basename. x,y,w,h is the "
    "axis-aligned box in absolute pixels, top-left origin (COCO convention). A "
    "row exists for every frame; all-'-1' geometry with visible=0 marks a frame "
    "where the target is absent, unifying SatSOT's literal 'none' with SV248S's "
    "state file. 'state' keeps SV248S's native flag (0 visible, 1 invisible, "
    "2 occluded) and is -1 elsewhere, because 'occluded but present' is a "
    "distinction 'visible' deliberately collapses. px1..py4 are oriented-box "
    "corners, -1 where the source annotates no orientation; OOTB rows carry the "
    "polygon and its axis-aligned box together."
)


def build_sot(out: Path, plan_dir: Path, limit: int | None,
              size_metric: str) -> dict:
    from space_tracker.manifest import SequenceRecord

    records = sot_source_records()
    by_id = {r["id"]: SequenceRecord.from_dict(r) for r in records}

    # Target size is measured on the boxes we ship, not taken on trust. The
    # earlier manifest measured OOTB on its oriented polygon and the other two
    # on their axis-aligned box, which judged the same object by two different
    # numbers -- an OOTB plane at 32 px of footprint reads as 22 px, a SatSOT
    # one as 32. 'aabb' is the uniform choice and is exactly the `area` the
    # released COCO file carries, so a reader can reproduce this selection from
    # the package alone. '--size-metric obb' restores the old behaviour.
    print(f"  measuring target size ({size_metric}) from the source ground truth...")
    for i, r in enumerate(records, start=1):
        areas = []
        for frame in iter_frames(by_id[r["id"]], SOT_ROOTS):
            if not frame.visible or frame.gt_box_xyxy is None:
                continue
            if size_metric == "obb" and frame.gt_obb_8pt is not None:
                areas.append(polygon_area(frame.gt_obb_8pt))
            else:
                x1, y1, x2, y2 = frame.gt_box_xyxy
                areas.append(max(x2 - x1, 0.0) * max(y2 - y1, 0.0))
        r["median_sqrt_area_px"] = (round(float(np.sqrt(np.median(areas))), 3)
                                    if areas else None)
        if i % 100 == 0:
            print(f"    {i}/{len(records)}", flush=True)

    kept_records = [r for r in records
                    if r["median_sqrt_area_px"] is not None
                    and r["median_sqrt_area_px"] <= SMALL_MAX_SQRT_AREA_PX]
    dropped = [r for r in records if r not in kept_records]
    numbers = assign_numbers([r["id"] for r in kept_records])
    if limit:
        kept_records = kept_records[:limit]

    doc = coco_skeleton(
        "sot", SOT_CATEGORIES,
        "space-tracker SOT: OOTB, SatSOT and SV248S unified to one annotation "
        "format, restricted to targets whose median sqrt(area) is at most "
        f"{SMALL_MAX_SQRT_AREA_PX:g} px.")
    group_of_video: dict[int, str] = {}
    seq_records: list[dict] = []
    plan_rows: list[tuple[str, str, str]] = []
    totals: Counter = Counter()

    img_id = ann_id = 0
    for vid, record in enumerate(kept_records, start=1):
        seq = by_id[record["id"]]
        category = record["category"]
        seq_name = release_name(category, record["dataset"], numbers[record["id"]])
        seq_dir = out / "sot" / category / seq_name
        (seq_dir).mkdir(parents=True, exist_ok=True)

        frames = list(iter_frames(seq, SOT_ROOTS))
        if not frames:
            raise RuntimeError(f"{record['id']}: no frames resolved")
        ext = frames[0].image_path.suffix.lower()
        width, height = image_size(frames[0].image_path)

        rows = []
        n_visible = 0
        for i, frame in enumerate(frames, start=1):
            dest = f"{i:06d}{ext}"
            plan_rows.append((f"sot/{category}/{seq_name}/img/{dest}",
                              str(frame.image_path), record["id"]))
            box = frame.gt_box_xyxy
            visible = bool(frame.visible) and box is not None
            if visible:
                x1, y1, x2, y2 = (float(v) for v in box)
                xywh = (x1, y1, x2 - x1, y2 - y1)
                n_visible += 1
            else:
                xywh = (-1.0, -1.0, -1.0, -1.0)
            obb = frame.gt_obb_8pt
            poly = tuple(float(v) for v in obb) if obb is not None else (-1.0,) * 8
            state = -1 if frame.state is None else int(frame.state)
            rows.append(",".join(
                [str(i)] + [f"{v:.2f}" for v in xywh]
                + [str(int(visible)), str(state)]
                + [f"{v:.2f}" for v in poly]))

            img_id += 1
            doc["images"].append({
                "id": img_id, "video_id": vid, "frame_id": i,
                "file_name": f"sot/{category}/{seq_name}/img/{dest}",
                "width": width, "height": height,
                "has_target": visible,
            })
            if not visible:
                continue
            ann_id += 1
            ann = {
                "id": ann_id, "image_id": img_id, "video_id": vid,
                "track_id": vid,                       # SOT: one target per sequence
                "category_id": cat_id(SOT_CATEGORIES, category),
                "bbox": [round(v, 2) for v in xywh],
                "area": round(xywh[2] * xywh[3], 2),
                "iscrowd": 0,
            }
            if obb is not None:
                ann["segmentation"] = [[round(float(v), 2) for v in obb]]
            if state >= 0:
                ann["state"] = state
            doc["annotations"].append(ann)

        (seq_dir / "groundtruth.txt").write_text("\n".join(rows) + "\n")
        acq = acquisition("sot", seq_name)
        write_seqinfo(seq_dir / "seqinfo.ini", name=seq_name, im_dir="img",
                      seq_length=len(frames), im_width=width, im_height=height,
                      im_ext=ext, frame_rate=acq["fps"], gsd_m=acq["gsd_m"],
                      platform=acq["platform"],
                      extra={"category": category, "sourceDataset": seq.dataset})

        doc["videos"].append({
            "id": vid, "name": seq_name,
            "source_dataset": record["dataset"],
            "source_sequence_id": record["id"],
            "category": category,
            "category_id": cat_id(SOT_CATEGORIES, category),
            "n_frames": len(frames), "n_visible_frames": n_visible,
            "width": width, "height": height,
            "median_sqrt_area_px": record["median_sqrt_area_px"],
            "native_attributes": record["native_attrs"],
            "unified_attributes": record["unified_attrs"],
            "taxonomy_attributes": record["taxonomy_attrs"],
            **acq,
        })
        doc["tracks"].append({
            "id": vid, "video_id": vid,
            "category_id": cat_id(SOT_CATEGORIES, category),
            "n_boxes": n_visible, "is_small": True,
            "median_sqrt_area_px": record["median_sqrt_area_px"],
        })
        group_of_video[vid] = category

        seq_records.append({
            "id": seq_name, "source_sequence_id": record["id"],
            "name": seq_name,
            "dataset": record["dataset"], "category": category,
            "n_frames": len(frames), "n_visible_frames": n_visible,
            "img_width": width, "img_height": height, "img_ext": ext,
            "path": f"sot/{category}/{seq_name}",
            "gt_path": f"sot/{category}/{seq_name}/groundtruth.txt",
            "image_path_pattern":
                f"sot/{category}/{seq_name}/img/{{frame_id:06d}}{ext}",
            "frame_index_base": 1,
            "median_sqrt_area_px": record["median_sqrt_area_px"],
            "native_attrs": record["native_attrs"],
            "unified_attrs": record["unified_attrs"],
            "taxonomy_attrs": record["taxonomy_attrs"],
            **acq,
        })
        totals["sequences"] += 1
        totals["frames"] += len(frames)
        totals["boxes"] += n_visible
        if totals["sequences"] % 50 == 0:
            print(f"  sot {totals['sequences']}/{len(kept_records)}", flush=True)

    ann_dir = out / "sot" / "annotations"
    dump_coco(doc, ann_dir / "space_tracker_sot.json")
    split_per_class(doc, ann_dir / "per_class", group_of_video)

    manifest = {
        "version": "2.0", "name": "space-tracker-sot", "task": "sot",
        "description": doc["info"]["description"],
        "frame_index_base": 1,
        "small_object_criterion": doc["info"]["small_object_criterion"],
        "gt_format": "sot_csv_unified_15col",
        "gt_format_description": SOT_GT_DESCRIPTION,
        "size_metric": size_metric,
        "coco_annotations": "sot/annotations/space_tracker_sot.json",
        "categories": {n: cat_id(SOT_CATEGORIES, n) for n in SOT_CATEGORIES},
        "source_datasets": SOT_SOURCE_INFO,
        "unified_attributes": _sot_taxonomy()["unified_attributes"],
        "attribute_taxonomy": _sot_taxonomy()["attribute_taxonomy"],
        "n_sequences": len(seq_records),
        "excluded_by_size": [
            {"source_sequence_id": r["id"], "category": r["category"],
             "median_sqrt_area_px": r["median_sqrt_area_px"]} for r in dropped],
        "sequences": seq_records,
    }
    (out / "sot" / "space_tracker_sot.json").write_text(json.dumps(manifest, indent=1))

    plan_dir.mkdir(parents=True, exist_ok=True)
    with open(plan_dir / "sot_images.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerows(plan_rows)

    print(f"SOT: {totals['sequences']} sequences, {totals['frames']} frames, "
          f"{totals['boxes']} boxes, {len(dropped)} sequences dropped as too large")
    return {"kept": totals["sequences"], "frames": totals["frames"],
            "boxes": totals["boxes"], "dropped": len(dropped),
            "images_planned": len(plan_rows)}


# --------------------------------------------------------------------------
# MOT


MOT_GT_DESCRIPTION = (
    "MOTChallenge ground truth: comma-separated, one row per box, 9 columns: "
    "frame_id, track_id, x, y, w, h, conf, class_id, visibility. frame_id and "
    "track_id are 1-indexed; frame_id matches the image basename in img1/. "
    "x,y,w,h is the box in absolute pixels, top-left origin (COCO convention). "
    "conf is 1 and visibility is 1 on every row — the sources annotate neither, "
    "and the columns are kept so MOTChallenge parsers read the file unmodified. "
    "class_id: 1=car, 2=airplane, 3=ship, 4=train."
)


def read_mot_gt(path: Path) -> list[tuple[int, int, float, float, float, float, int]]:
    """(frame, track, x, y, w, h, cls) from the 11-column internal export."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split(",")
            rows.append((int(float(p[0])), int(float(p[1])),
                         float(p[2]), float(p[3]), float(p[4]), float(p[5]),
                         int(float(p[7]))))
    return rows


def build_mot(out: Path, plan_dir: Path, limit: int | None) -> dict:
    src = json.loads(MOT_MANIFEST.read_text())
    records = src["sequences"]
    if limit:
        records = records[:limit]

    doc = coco_skeleton(
        "mot", MOT_CATEGORIES,
        "space-tracker MOT: SAT-MTB, VISO (non-car), SDM-Car and RsCarData "
        "unified to one annotation format, restricted to sequences containing at "
        f"least one object whose median sqrt(area) is at most "
        f"{SMALL_MAX_SQRT_AREA_PX:g} px. AIR-MOT is excluded: no redistribution "
        "licence.")
    group_of_video: dict[int, str] = {}
    seq_records: list[dict] = []
    plan_frames: list[tuple[str, str, str]] = []
    plan_videos: list[dict] = []
    totals: Counter = Counter()
    warnings: list[str] = []
    regrouped: list[str] = []

    numbers = assign_numbers([r["id"] for r in records])
    img_id = ann_id = track_uid = 0
    for vid, record in enumerate(records, start=1):
        base = record["frame_index_base"]
        rows = read_mot_gt(MOT_RELEASE / record["gt_path"])
        n_frames = record["n_frames"]
        width, height = record["img_width"], record["img_height"]

        # Boxes are clipped to the image. SDM-Car and SAT-MTB keep writing a
        # track after its object has driven off the edge, so 0.09% of boxes
        # extend past the border and two thirds of those lie wholly outside it.
        # A box over pixels that do not exist is not ground truth: no method can
        # ever hit it, and it drags recall down for everyone equally. Partly
        # visible objects are clipped to what is actually in frame; wholly
        # absent ones are dropped, as are the few tracks left with no box.
        clipped: list[tuple[int, int, float, float, float, float, int]] = []
        for frame, tid, x, y, w, h, cls in rows:
            x1, y1 = max(x, 0.0), max(y, 0.0)
            x2, y2 = min(x + w, float(width)), min(y + h, float(height))
            cw, ch = x2 - x1, y2 - y1
            if cw < 1.0 or ch < 1.0:
                totals["boxes_dropped_offscreen"] += 1
                continue
            if abs(cw - w) > 0.01 or abs(ch - h) > 0.01:
                totals["boxes_clipped"] += 1
            clipped.append((frame - base + 1, tid, x1, y1, cw, ch, cls))

        # Track ids are renumbered 1..N in order of first appearance among the
        # surviving boxes: SDM-Car starts its ids at 0, MOTChallenge readers
        # assume >= 1, and a track dropped entirely must not leave a hole.
        local_id: dict[int, int] = {}
        by_track: dict[int, list] = defaultdict(list)
        for frame, tid, x, y, w, h, cls in clipped:
            if tid not in local_id:
                local_id[tid] = len(local_id) + 1
            by_track[local_id[tid]].append((frame, x, y, w, h, cls))
        n_src_tracks = len({r[1] for r in rows})
        totals["tracks_dropped_offscreen"] += n_src_tracks - len(by_track)
        if not by_track:
            raise RuntimeError(f"{record['id']}: every box was off-screen")

        # Small-object scope is decided per track, on the median of sqrt(area)
        # over that track's own boxes — one frame's box is too noisy to judge on.
        small_flags: dict[int, tuple[bool, float]] = {}
        for tid, boxes in by_track.items():
            med = float(np.median([np.sqrt(max(b[3], 0.0) * max(b[4], 0.0))
                                   for b in boxes]))
            small_flags[tid] = (med <= SMALL_MAX_SQRT_AREA_PX, round(med, 3))
        if not any(s for s, _ in small_flags.values()):
            warnings.append(f"{record['id']}: no small track — out of scope")

        # The class folder names what the sequence actually contains after the
        # off-screen cleanup, not what the source manifest said before it.
        surviving = {MOT_CLS_ID_TO_NAME[b[5]] for bs in by_track.values() for b in bs}
        group = sorted(surviving)[0] if len(surviving) == 1 else "mixed"
        if group != record["category"]:
            regrouped.append(f"{record['id']}: {record['category']} -> {group}")
        seq_name = release_name(group, record["dataset"], numbers[record["id"]])
        seq_dir = out / "mot" / group / seq_name
        (seq_dir / "gt").mkdir(parents=True, exist_ok=True)

        ext = (".jpg" if record["image_format"] == "video"
               else "." + record["image_path_pattern"].rsplit(".", 1)[-1])

        if record["image_format"] == "video":
            plan_videos.append({
                "seq": record["id"],
                "video_path": str(MOT_ROOTS[record["dataset"]] / record["video_path"]),
                "dest_dir": f"mot/{group}/{seq_name}/img1",
                "n_frames": n_frames, "frame_index_base": base, "ext": ext,
            })
        else:
            root = MOT_ROOTS[record["dataset"]]
            for i in range(1, n_frames + 1):
                native = record["image_path_pattern"].format(frame_id=i - 1 + base)
                plan_frames.append((f"mot/{group}/{seq_name}/img1/{i:06d}{ext}",
                                    str(root / native), record["id"]))

        # per-frame COCO image records, including frames with no annotation
        frame_img_id: dict[int, int] = {}
        for i in range(1, n_frames + 1):
            img_id += 1
            frame_img_id[i] = img_id
            doc["images"].append({
                "id": img_id, "video_id": vid, "frame_id": i,
                "file_name": f"mot/{group}/{seq_name}/img1/{i:06d}{ext}",
                "width": width, "height": height,
            })

        gt_lines: list[tuple] = []
        cats_in_seq: set[str] = set()
        for tid in sorted(by_track):
            boxes = by_track[tid]
            track_uid += 1
            is_small, med = small_flags[tid]
            cls_name = MOT_CLS_ID_TO_NAME[boxes[0][5]]
            cats_in_seq.add(cls_name)
            doc["tracks"].append({
                "id": track_uid, "video_id": vid,
                "local_track_id": tid,
                "category_id": cat_id(MOT_CATEGORIES, cls_name),
                "n_boxes": len(boxes), "is_small": is_small,
                "median_sqrt_area_px": med,
            })
            for frame, x, y, w, h, cls in sorted(boxes):
                cname = MOT_CLS_ID_TO_NAME[cls]
                cid = cat_id(MOT_CATEGORIES, cname)
                gt_lines.append((frame, tid, x, y, w, h, 1, cid, 1))
                ann_id += 1
                doc["annotations"].append({
                    "id": ann_id, "image_id": frame_img_id[frame],
                    "video_id": vid, "track_id": track_uid,
                    "category_id": cid,
                    "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                    "area": round(w * h, 2), "iscrowd": 0,
                })
                totals["boxes"] += 1

        gt_lines.sort(key=lambda r: (r[0], r[1]))
        (seq_dir / "gt" / "gt.txt").write_text("".join(
            f"{f},{t},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{c},{k},{v}\n"
            for f, t, x, y, w, h, c, k, v in gt_lines))
        acq = acquisition("mot", seq_name)
        write_seqinfo(seq_dir / "seqinfo.ini", name=seq_name, im_dir="img1",
                      seq_length=n_frames, im_width=width, im_height=height,
                      im_ext=ext, frame_rate=acq["fps"], gsd_m=acq["gsd_m"],
                      platform=acq["platform"],
                      extra={"category": group,
                             "sourceDataset": record["dataset"]})

        doc["videos"].append({
            "id": vid, "name": seq_name,
            "source_dataset": record["dataset"],
            "source_sequence_id": record["id"],
            "category": group,
            "categories_in_sequence": sorted(cats_in_seq),
            "n_frames": n_frames, "n_tracks": len(by_track),
            "n_small_tracks": sum(1 for s, _ in small_flags.values() if s),
            "width": width, "height": height,
            "tags": record.get("tags", []),
            **acq,
        })
        group_of_video[vid] = group

        seq_records.append({
            "id": seq_name, "source_sequence_id": record["id"],
            "name": seq_name,
            "dataset": record["dataset"], "category": group,
            "categories_in_sequence": sorted(cats_in_seq),
            "n_frames": n_frames, "n_tracks": len(by_track),
            "n_small_tracks": sum(1 for s, _ in small_flags.values() if s),
            "n_boxes": len(gt_lines),
            "img_width": width, "img_height": height, "img_ext": ext,
            "path": f"mot/{group}/{seq_name}",
            "gt_path": f"mot/{group}/{seq_name}/gt/gt.txt",
            "image_path_pattern":
                f"mot/{group}/{seq_name}/img1/{{frame_id:06d}}{ext}",
            "frame_index_base": 1,
            "source_image_format": record["image_format"],
            "review": record.get("review", {}),
            "tags": record.get("tags", []),
            **acq,
        })
        totals["sequences"] += 1
        totals["frames"] += n_frames
        totals["tracks"] += len(by_track)
        if totals["sequences"] % 50 == 0:
            print(f"  mot {totals['sequences']}/{len(records)}", flush=True)

    ann_dir = out / "mot" / "annotations"
    dump_coco(doc, ann_dir / "space_tracker_mot.json")
    split_per_class(doc, ann_dir / "per_class", group_of_video)

    manifest = {
        "version": "2.0", "name": "space-tracker-mot", "task": "mot",
        "description": doc["info"]["description"],
        "frame_index_base": 1,
        "small_object_criterion": doc["info"]["small_object_criterion"],
        "scope_rule": ("A sequence is in scope when at least one track in it is "
                       "small. Every track in a kept sequence is annotated, "
                       "including large ones — an unlabelled object in an "
                       "annotated frame would score as a false positive."),
        "gt_format": "motchallenge_csv_9col",
        "gt_format_description": MOT_GT_DESCRIPTION,
        "coco_annotations": "mot/annotations/space_tracker_mot.json",
        "categories": {n: cat_id(MOT_CATEGORIES, n) for n in MOT_CATEGORIES},
        "groups": ("Sequences are foldered by the categories they contain: "
                   "car / airplane / ship / train hold single-category "
                   "sequences, mixed holds sequences with more than one."),
        "source_datasets": src["datasets"],
        "evaluation": src["evaluation"],
        "excluded_datasets": {"airmot": "no redistribution licence"},
        "boundary_cleanup": {
            "rule": ("Boxes are clipped to the image. A box left narrower than "
                     "1 px in either dimension is dropped, as is any track that "
                     "keeps no box. The sources continue tracks after the "
                     "object has left the frame."),
            "boxes_clipped": totals["boxes_clipped"],
            "boxes_dropped_offscreen": totals["boxes_dropped_offscreen"],
            "tracks_dropped_offscreen": totals["tracks_dropped_offscreen"],
            "sequences_regrouped": regrouped,
        },
        "n_sequences": len(seq_records),
        "sequences": seq_records,
    }
    (out / "mot" / "space_tracker_mot.json").write_text(json.dumps(manifest, indent=1))

    plan_dir.mkdir(parents=True, exist_ok=True)
    with open(plan_dir / "mot_images.tsv", "w", newline="") as f:
        csv.writer(f, delimiter="\t").writerows(plan_frames)
    (plan_dir / "mot_videos.json").write_text(json.dumps(plan_videos, indent=1))

    for w in warnings:
        print("  WARNING:", w)
    for r in regrouped:
        print("  regrouped:", r)
    print(f"MOT: {totals['sequences']} sequences, {totals['frames']} frames, "
          f"{totals['tracks']} tracks, {totals['boxes']} boxes")
    print(f"     boundary cleanup: {totals['boxes_clipped']} boxes clipped, "
          f"{totals['boxes_dropped_offscreen']} dropped off-screen, "
          f"{totals['tracks_dropped_offscreen']} tracks dropped")
    return {"kept": totals["sequences"], "frames": totals["frames"],
            "tracks": totals["tracks"], "boxes": totals["boxes"],
            "boxes_clipped": totals["boxes_clipped"],
            "boxes_dropped_offscreen": totals["boxes_dropped_offscreen"],
            "tracks_dropped_offscreen": totals["tracks_dropped_offscreen"],
            "regrouped": regrouped,
            "images_planned": len(plan_frames), "videos_planned": len(plan_videos),
            "warnings": warnings}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--plan-dir", type=Path, default=None,
                    help="where the image staging plan is written "
                         "(default: <out>.build); not part of the package")
    ap.add_argument("--task", choices=["sot", "mot", "both"], default="both")
    ap.add_argument("--size-metric", choices=["aabb", "obb"], default="aabb",
                    help="how a SOT target's size is measured for the <= 32 px "
                         "cut: 'aabb' (default) uses the axis-aligned box the "
                         "package ships; 'obb' uses OOTB's oriented polygon, "
                         "which measures OOTB on a different footprint from the "
                         "other two sources")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N sequences per task — smoke test only")
    args = ap.parse_args()

    out = args.out
    plan_dir = args.plan_dir or Path(str(out) + ".build")
    out.mkdir(parents=True, exist_ok=True)

    summary = {}
    if args.task in ("sot", "both"):
        summary["sot"] = build_sot(out, plan_dir, args.limit, args.size_metric)
    if args.task in ("mot", "both"):
        summary["mot"] = build_mot(out, plan_dir, args.limit)
    (plan_dir / "build_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
