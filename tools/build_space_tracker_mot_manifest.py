"""
Build the Space-tracker MOT manifest (``space_tracker_mot.json``).

Companion of ``space_tracker.json`` (SOT). This manifest catalogues every
sequence in AIRMOT / SAT-MTB / VISO (non-car) / SDM-Car / RsCarData along
with paths relative to each dataset's own root, a per-sequence ``gt_format``
tag, image / video resolution, track count, and the dataset's official
split assignment.

GT data is *not* embedded — readers download each dataset themselves under
its own license terms; the manifest only points at where the data lives.

VISO's car subset is **deliberately excluded** here: those sequences are
re-annotated and shipped as RsCarData (HiEUM, TPAMI 2024); loading both
would double-count. The other VISO categories (plane / ship / train) are
included.

Usage::

    python tools/build_space_tracker_mot_manifest.py \
        --out space_tracker/space_tracker_mot.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.airmot import AIRMOTDataset
from datasets.rscardata import RsCarDataset
from datasets.satmtb import SATMTBDataset, _MOT_CLASS_MAP
from datasets.sdmcar import SDMCarDataset
from datasets.viso import VISODataset


MANIFEST_VERSION = "1.0"

DATASET_ROOTS = {
    "airmot":    Path("/data/ESA_DLSTEM_2025/data/trafic/AIR-MOT-100"),
    "satmtb":    Path("/data/ESA_DLSTEM_2025/data/trafic/SAT-MTB"),
    "viso":      Path("/data/ESA_DLSTEM_2025/data/trafic/VISO"),
    "sdmcar":    Path("/data/ESA_DLSTEM_2025/data/trafic/SDM-Car"),
    "rscardata": Path("/data/ESA_DLSTEM_2025/data/trafic/RsCarData"),
}

# Public-facing dataset info block. Mirrors the SOT manifest's structure.
DATASET_INFO = {
    "airmot": {
        "name": "AIR-MOT-100",
        "paper": "He et al., IEEE GRSL 2022",
        "gt_format": "mot_csv_9col",
        "gt_format_description":
            "MOT CSV (comma-delimited, 9 cols): "
            "frame_id, track_id, x, y, w, h, conf, class, visibility. "
            "Coords are xywh top-left. class: 1=airplane, 2=ship. 1-indexed frames.",
        "image_format": "frames",
        "official_split": "stratified_80_10_10",
        "notes":
            "31/100 sequences ship with empty gt.txt and are skipped by the builder. "
            "Sequence 100 uses image filenames with a '_8' suffix (e.g. 000001_8.jpg).",
    },
    "satmtb": {
        "name": "SAT-MTB",
        "paper": "Li et al., IEEE TGRS 2023",
        "gt_format": "mot_csv_11col",
        "gt_format_description":
            "MOT CSV (comma-delimited, 11 cols): "
            "frame_id, obj_id, x, y, w, h, conf, cls_id, r1, r2, r3. "
            "Coords are xywh top-left. cls_id: 0=car, 1=airplane, 2=ship, 3=train. "
            "1-indexed frames. Only the coarse class id is carried by MOT; "
            "the 14 fine-grained labels (WA/NA/.../TN) live in HBB/OBB XML and "
            "seg JSON files and are NOT available for MOT.",
        "image_format": "frames",
        "official_split": "xlsx_train_test_then_carve_val_30pct",
        "notes":
            "Per-category annotation availability — MOT exists for: airplane, car, "
            "ship, train 1-7, train 11-16. train 8-10 do NOT have MOT. "
            "Car sequences ship MOT files with .txt extension; other categories ship "
            "them without an extension. Some sequences contain multiple coarse "
            "classes; we tag them category='mixed' and list each class in "
            "'categories_in_seq'.",
    },
    "viso": {
        "name": "VISO",
        "paper": "Yin et al., IEEE TGRS 2022",
        "gt_format": "viso_dual",
        "gt_format_description":
            "Two on-disk layouts in one dataset: car/train use "
            "comma-delimited xywh "
            "(frame,obj_id,x,y,w,h,...); plane/ship use space-delimited "
            "xyxy (frame obj_id x1 y1 x2 y2 ...). "
            "Auto-detect by inspecting the first line's delimiter.",
        "image_format": "frames",
        "official_split": "viso_official_frame_split_majority_vote",
        "notes":
            "Car subset (sequences 001-038) is EXCLUDED from this manifest because "
            "it has been re-annotated and shipped as RsCarData under the HiEUM "
            "protocol. To use the original VISO car labels, build VISODataset with "
            "categories=['car'] directly — but do not mix with RsCarData. Ship "
            "category has no official val split.",
        "skip_categories": ["car"],
    },
    "sdmcar": {
        "name": "SDM-Car",
        "paper": "Yin et al., 2023 (Luojia-3-01 satellite video)",
        "gt_format": "mot_csv_10col_0idx",
        "gt_format_description":
            "Per-sequence headerless CSV (comma-delimited, 10 cols): "
            "frame_id, target_id, x, y, w, h, -1, -1, -1, -1. "
            "Coords are xywh top-left. Single class (car). "
            "Frame ids are 0-INDEXED — unlike every other dataset in this manifest.",
        "image_format": "video",
        "official_split": "official_train_validation_test",
        "notes":
            "Each sequence ships as a single .avi video (no per-frame image files). "
            "Random-access decoding via OpenCV VideoCapture is slow; consider "
            "pre-extracting frames offline if you do many runs.",
    },
    "rscardata": {
        "name": "RsCarData",
        "paper": "Xiao et al., IEEE TPAMI 2024 (HiEUM)",
        "gt_format": "coco_mot_json",
        "gt_format_description":
            "COCO-MOT JSON ({images, annotations, videos, categories}) with "
            "video_id + video_frame_id (1-indexed) per image. bbox is xywh "
            "top-left. Single class (car).",
        "image_format": "frames",
        "official_split": "official_train_test_then_carve_val_10pct",
        "notes":
            "VISO car re-annotated by the HiEUM authors. Train+val sequences are "
            "512x512; test sequences are 1024x1024. The test split uses re-curated "
            "PASCAL-VOC XML labels under labeleddata20230227/ when present "
            "(per-frame gt_path_override, gt_format='pascal_voc_xml_per_frame'); "
            "this is the protocol HiEUM's paper evaluates against and carries ~66% "
            "more boxes than the COCO MOT JSON for those sequences. Train/val keep "
            "the COCO MOT JSON.",
    },
}

# Unified class taxonomy across the 5 datasets.
#
# native_ids: how each source dataset identifies this class natively.
#   - integer: the cls column value in the MOT CSV (AIRMOT, SAT-MTB)
#   - string : the directory or category name (VISO, RsCarData, SDM-Car)
CATEGORIES = {
    "car": {
        "datasets": ["rscardata", "sdmcar", "satmtb"],
        "native_ids": {"satmtb": 0, "sdmcar": "car", "rscardata": "car"},
        "notes": "VISO's car subset is excluded here — see DATASET_INFO['viso']['skip_categories'].",
    },
    "airplane": {
        "datasets": ["airmot", "satmtb", "viso"],
        "native_ids": {"airmot": 1, "satmtb": 1, "viso": "plane"},
        "notes": "VISO calls the directory 'plane'; unified label is 'airplane'.",
    },
    "ship": {
        "datasets": ["airmot", "satmtb", "viso"],
        "native_ids": {"airmot": 2, "satmtb": 2, "viso": "ship"},
    },
    "train": {
        "datasets": ["satmtb", "viso"],
        "native_ids": {"satmtb": 3, "viso": "train"},
        "notes": "SAT-MTB train sequences 8-10 have no MOT annotations (only HBB/OBB/seg).",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _viso_native_to_unified_category(cat: str) -> str:
    return "airplane" if cat == "plane" else cat


def _first_frame_size(ds, video) -> tuple[int, int]:
    """(width, height) — load the first frame via the dataset's own loader."""
    img = ds._load_frame(video, video.frame_ids[0])
    h, w = img.shape[:2]
    return int(w), int(h)


def _airmot_record(ds: AIRMOTDataset, video) -> dict:
    vid = video.video_id
    suffix = getattr(video, "_img_suffix", "")
    if suffix:
        pattern = f"{vid}/img/{{frame_id:06d}}_{suffix}.jpg"
    else:
        pattern = f"{vid}/img/{{frame_id:06d}}.jpg"

    ann_cache = ds._ann_cache.get(vid, {})
    tracks: set[int] = set()
    for objs in ann_cache.values():
        for o in objs:
            tracks.add(int(o["track_id"]))

    w, h = _first_frame_size(ds, video)
    return {
        "id":                   f"airmot/{vid}",
        "dataset":              "airmot",
        "video_id":             vid,
        "category":             video.category,
        "categories_in_seq":    [video.category],
        "n_frames":             int(video.num_frames),
        "n_tracks":             len(tracks),
        "img_width":            w,
        "img_height":           h,
        "image_format":         "frames",
        "image_path_pattern":   pattern,
        "video_path":           None,
        "gt_path":              f"{vid}/gt/gt.txt",
        "gt_path_override":     None,
        "gt_format":            "mot_csv_9col",
        "frame_index_base":     1,
        "split":                video.split,
    }


def _satmtb_record(ds: SATMTBDataset, video) -> dict:
    vid = video.video_id           # e.g. "ship/04"
    cat_dir, seq_num = vid.split("/")

    # MOT file may live as ``<seq_num>`` (no ext) or ``<seq_num>.txt``
    mot_dir = ds.root / "SAT-MTB_Dataset" / cat_dir / seq_num / "mot"
    mot_files = sorted(p for p in mot_dir.iterdir() if p.is_file())
    if not mot_files:
        raise FileNotFoundError(f"SAT-MTB MOT file missing: {mot_dir}")
    mot_file = mot_files[0]
    gt_rel = str(mot_file.relative_to(ds.root))

    ann_cache = ds._ann_cache.get(vid, {})
    tracks: set[int] = set()
    classes: Counter = Counter()
    for objs in ann_cache.values():
        for o in objs:
            tracks.add(int(o["track_id"]))
            classes[o["class"]] += 1

    classes_present = sorted(classes.keys())
    if len(classes_present) == 1:
        category = classes_present[0]
    else:
        category = "mixed"

    w, h = _first_frame_size(ds, video)
    return {
        "id":                   f"satmtb/{vid}",
        "dataset":              "satmtb",
        "video_id":             vid,
        "category":             category,
        "categories_in_seq":    classes_present,
        "n_frames":             int(video.num_frames),
        "n_tracks":             len(tracks),
        "img_width":            w,
        "img_height":           h,
        "image_format":         "frames",
        "image_path_pattern":   f"SAT-MTB_Dataset/{cat_dir}/{seq_num}/img/{{frame_id:06d}}.png",
        "video_path":           None,
        "gt_path":              gt_rel,
        "gt_path_override":     None,
        "gt_format":            "mot_csv_11col",
        "frame_index_base":     1,
        "split":                video.split,
    }


def _viso_record(ds: VISODataset, video) -> dict:
    vid = video.video_id           # e.g. "plane/039"
    native_cat, seq_num = vid.split("/")
    unified_cat = _viso_native_to_unified_category(native_cat)

    ann_cache = ds._ann_cache.get(vid, {})
    tracks: set[int] = set()
    for objs in ann_cache.values():
        for o in objs:
            tracks.add(int(o["track_id"]))

    w, h = _first_frame_size(ds, video)
    return {
        "id":                   f"viso/{vid}",
        "dataset":              "viso",
        "video_id":             vid,
        "category":             unified_cat,
        "categories_in_seq":    [unified_cat],
        "n_frames":             int(video.num_frames),
        "n_tracks":             len(tracks),
        "img_width":            w,
        "img_height":           h,
        "image_format":         "frames",
        "image_path_pattern":   f"mot/{native_cat}/{seq_num}/img/{{frame_id:06d}}.jpg",
        "video_path":           None,
        "gt_path":              f"mot/{native_cat}/{seq_num}/gt/gt.txt",
        "gt_path_override":     None,
        "gt_format":            "viso_dual",
        "frame_index_base":     1,
        "split":                video.split,
    }


def _sdmcar_record(ds: SDMCarDataset, video) -> dict:
    vid = video.video_id           # e.g. "train/1-1"
    split_canon, seq_name = vid.split("/", 1)
    split_dir = ds._SPLIT_DIRS[split_canon]

    ann_cache = ds._ann_cache.get(vid, {})
    tracks: set[int] = set()
    for objs in ann_cache.values():
        for (tid, *_rest) in objs:
            tracks.add(int(tid))

    # Read resolution from the video header instead of decoding a frame.
    avi_rel = f"{split_dir}/{seq_name}.avi"
    cap = cv2.VideoCapture(str(ds.root / avi_rel))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return {
        "id":                   f"sdmcar/{vid}",
        "dataset":              "sdmcar",
        "video_id":             vid,
        "category":             "car",
        "categories_in_seq":    ["car"],
        "n_frames":             int(video.num_frames),
        "n_tracks":             len(tracks),
        "img_width":            w,
        "img_height":           h,
        "image_format":         "video",
        "image_path_pattern":   None,
        "video_path":           avi_rel,
        "gt_path":              f"{split_dir}/{seq_name}-gt.csv",
        "gt_path_override":     None,
        "gt_format":            "mot_csv_10col_0idx",
        "frame_index_base":     0,
        "split":                split_canon,
    }


def _rscardata_record(ds: RsCarDataset, video) -> dict:
    vid = video.video_id           # e.g. "train/001" or "test1024/002"
    native_split, seq_name = vid.split("/", 1)
    img_dir_rel = ds._img_dir[vid]      # e.g. "images/train/001/img1"

    ann_cache = ds._ann_cache.get(vid, {})
    tracks: set[int] = set()
    for objs in ann_cache.values():
        for (tid, *_rest) in objs:
            tracks.add(int(tid))

    # Train/val: COCO MOT JSON; test: prefer PASCAL-VOC XML override if present.
    new_root = ds.root / RsCarDataset._NEW_TEST_LABELS_DIR
    if video.split == "test" and (new_root / seq_name / "img1").is_dir():
        gt_format = "pascal_voc_xml_per_frame"
        gt_path = f"{RsCarDataset._NEW_TEST_LABELS_DIR}/{seq_name}/img1"
        gt_override = None
        coco_json = "annotations/test1024_mot.json"
        gt_meta = {"coco_mot_json_fallback": coco_json}
    elif video.split == "test":
        gt_format = "coco_mot_json"
        gt_path = "annotations/test1024_mot.json"
        gt_override = None
        gt_meta = {}
    else:
        gt_format = "coco_mot_json"
        gt_path = "annotations/train_mot.json"
        gt_override = None
        gt_meta = {}

    w, h = _first_frame_size(ds, video)
    rec = {
        "id":                   f"rscardata/{vid}",
        "dataset":              "rscardata",
        "video_id":             vid,
        "category":             "car",
        "categories_in_seq":    ["car"],
        "n_frames":             int(video.num_frames),
        "n_tracks":             len(tracks),
        "img_width":            w,
        "img_height":           h,
        "image_format":         "frames",
        "image_path_pattern":   f"{img_dir_rel}/{{frame_id:06d}}.jpg",
        "video_path":           None,
        "gt_path":              gt_path,
        "gt_path_override":     gt_override,
        "gt_format":            gt_format,
        "frame_index_base":     1,
        "split":                video.split,
    }
    rec.update(gt_meta)
    return rec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="space_tracker/space_tracker_mot.json",
        help="Output manifest path.",
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("[1/2] loading datasets")
    airmot    = AIRMOTDataset(   root=DATASET_ROOTS["airmot"],    split="no_split", mode="detection")
    satmtb    = SATMTBDataset(   root=DATASET_ROOTS["satmtb"],    split="no_split", mode="detection", task="mot")
    viso      = VISODataset(     root=DATASET_ROOTS["viso"],      split="no_split", mode="detection",
                                 categories=("plane", "ship", "train"))  # car excluded
    sdmcar    = SDMCarDataset(   root=DATASET_ROOTS["sdmcar"],    split="no_split", mode="detection")
    rscardata = RsCarDataset(    root=DATASET_ROOTS["rscardata"], split="no_split", mode="detection")

    print("[2/2] building sequence records")
    sequences: list[dict] = []
    for ds_name, ds, builder in (
        ("airmot",    airmot,    _airmot_record),
        ("satmtb",    satmtb,    _satmtb_record),
        ("viso",      viso,      _viso_record),
        ("sdmcar",    sdmcar,    _sdmcar_record),
        ("rscardata", rscardata, _rscardata_record),
    ):
        before = len(sequences)
        for v in ds.videos:
            sequences.append(builder(ds, v))
        n = len(sequences) - before
        n_tracks = sum(s["n_tracks"] for s in sequences[-n:])
        n_frames = sum(s["n_frames"] for s in sequences[-n:])
        print(f"   {ds_name}: {n} sequences   ({n_tracks} tracks, {n_frames} frames)")

    manifest = {
        "version":     MANIFEST_VERSION,
        "name":        "space-tracker-mot",
        "task":        "mot",
        "description": (
            "Unified multi-object-tracking benchmark across AIRMOT, SAT-MTB, "
            "VISO (non-car), SDM-Car, and RsCarData. Manifest indexes sequences "
            "+ per-dataset native GT format tags; raw imagery and GT must be "
            "obtained from the original datasets per their license terms. "
            "VISO's car subset is excluded because it is re-annotated as "
            "RsCarData under the HiEUM protocol."
        ),
        "evaluation": {
            "aggregation":   "per_sequence",
            "metrics":       ["HOTA", "MOTA", "IDF1", "DetA", "AssA"],
            "iou_threshold": 0.5,
        },
        "categories":  CATEGORIES,
        "datasets":    DATASET_INFO,
        "n_sequences": len(sequences),
        "sequences":   sequences,
    }

    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    size_kb = out_path.stat().st_size / 1024
    print(f"\nWrote {out_path}  ({size_kb:.1f} KB, {len(sequences)} sequences)")

    # ---- sanity counts ------------------------------------------------
    print("\nPer-dataset / split breakdown:")
    for ds_name in DATASET_INFO:
        ds_seqs = [s for s in sequences if s["dataset"] == ds_name]
        by_split = Counter(s["split"] for s in ds_seqs)
        by_cat   = Counter(s["category"] for s in ds_seqs)
        print(f"  {ds_name:10s} n={len(ds_seqs):3d}   splits={dict(by_split)}   "
              f"cats={dict(by_cat)}")

    print("\nUnified-category sequence counts:")
    for cat in CATEGORIES:
        n = sum(1 for s in sequences if cat in s["categories_in_seq"])
        print(f"  {cat:10s} {n} sequences")


if __name__ == "__main__":
    main()
