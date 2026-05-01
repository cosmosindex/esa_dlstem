"""
Build the Space-tracker manifest (``space_tracker.json``).

The manifest is a single self-describing JSON file that catalogs every
sequence in SatSOT / SV248S / OOTB along with:

  - image folder + GT file (paths *relative* to the dataset's own root, so the
    manifest stays portable);
  - native sequence attributes from the source dataset;
  - unified attributes (BC, IV, ROT, OCC, SOB, DEF) under our taxonomy;
  - sequence-level scale stats (median sqrt-area in px) + the ``tiny`` flag;
  - the unified-attribute mapping table (so consumers do not need a
    separate spec file).

GT data is *not* embedded — readers are expected to download SatSOT /
SV248S / OOTB themselves (license terms differ across the three datasets);
the manifest only points to where the data lives. Same publication
convention as LaSOT / GOT-10k.

Usage::

    python tools/build_space_tracker_manifest.py \
        --out space_tracker/space_tracker.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.ootb import OOTBDataset
from datasets.satsot import SatSOTDataset
from datasets.sv248s import SV248SDataset
from tools.reaggregate_sot_per_sequence import (
    _ootb_seq_areas, _satsot_seq_areas, _sv248s_seq_areas,
    TINY_SQRT_AREA_THRESH,
)


MANIFEST_VERSION = "1.0"

DATASET_ROOTS = {
    "ootb":   Path("/data/ESA_DLSTEM_2025/data/trafic/OOTB"),
    "satsot": Path("/data/ESA_DLSTEM_2025/data/trafic/SatSOT"),
    "sv248s": Path("/data/ESA_DLSTEM_2025/data/trafic/SV248S"),
}

# Public download / paper hints for users who do NOT have the data yet.
DATASET_INFO = {
    "ootb": {
        "name": "OOTB",
        "paper": "Chen et al., ISPRS J. Photogramm. Remote Sens. 2024",
        "gt_format": "obb_8pt",
        "gt_format_description":
            "Per-frame line of 8 floats: x1,y1,x2,y2,x3,y3,x4,y4 — corners of an oriented bounding box.",
        "image_glob": "img/*.jpg",
    },
    "satsot": {
        "name": "SatSOT",
        "paper": "Zhao et al., IEEE TGRS 2022",
        "gt_format": "xywh_with_none",
        "gt_format_description":
            "Per-frame line of 4 floats x,y,w,h; lines containing 'none' mark frames where the target is absent.",
        "image_glob": "img/*",
    },
    "sv248s": {
        "name": "SV248S",
        "paper": "Li et al., IEEE TGRS 2022",
        "gt_format": "xywh_with_state",
        "gt_format_description":
            "Per-sequence: <ann_dir>/<seq_id>.rect (xywh per frame) and <ann_dir>/<seq_id>.state "
            "(per-frame flag: 0=visible, 1=invisible, 2=occluded). Sequences live at "
            "<video_dir>/sequences/<seq_id>/*.tiff with annotations at <video_dir>/annotations/<seq_id>.{rect,state,abs,poly,attr}.",
        "image_glob": "*.tiff",
    },
}

UNIFIED_ATTR_SPEC = {
    "BC": {
        "full_name":  "Background Clutter",
        "definition": "Background has similar appearance (texture/color) to the target.",
        "datasets":   {"satsot": ["BC"], "sv248s": [], "ootb": ["BC"]},
    },
    "IV": {
        "full_name":  "Illumination Variation",
        "definition": "Illumination around the target changes significantly.",
        "datasets":   {"satsot": ["IV"], "sv248s": ["IV"], "ootb": ["IV"]},
    },
    "ROT": {
        "full_name":  "Rotation (in-plane)",
        "definition": "Target rotates in the image plane (>= 30° for SV248S).",
        "datasets":   {"satsot": ["ROT"], "sv248s": ["IPR"], "ootb": ["IPR"]},
    },
    "OCC": {
        "full_name":  "Occlusion (any kind)",
        "definition": "Target is occluded by scene structures (any degree, any duration). Consolidates spatial (PO/FO/POC/FOC) and temporal (STO/LTO/CO) annotations across the three datasets.",
        "datasets":   {"satsot": ["POC", "FOC"], "sv248s": ["STO", "LTO", "CO"], "ootb": ["PO", "FO"]},
    },
    "SOB": {
        "full_name":  "Similar Object",
        "definition": "Nearby objects share shape / type / appearance with the target (≤ 2.5x object size for SV248S).",
        "datasets":   {"satsot": ["SOB"], "sv248s": ["DS"], "ootb": ["SA"]},
    },
    "DEF": {
        "full_name":  "Deformation",
        "definition": "Non-rigid object deformation.",
        "datasets":   {"satsot": ["DEF"], "sv248s": [], "ootb": ["DEF"]},
    },
}


def _polygon_area_from_corners(coords: np.ndarray) -> float:
    xs = coords[0::2]; ys = coords[1::2]
    return 0.5 * abs(
        xs[0] * (ys[1] - ys[3])
        + xs[1] * (ys[2] - ys[0])
        + xs[2] * (ys[3] - ys[1])
        + xs[3] * (ys[0] - ys[2])
    )


def _native_to_unified(dataset: str, native_attrs: list[str]) -> list[str]:
    out = []
    for unified, spec in UNIFIED_ATTR_SPEC.items():
        if set(native_attrs) & set(spec["datasets"].get(dataset, [])):
            out.append(unified)
    return out


def _seq_median_sqrt_area(dataset: str, root: Path, video_id: str) -> float | None:
    if dataset == "ootb":
        seq_dir = root / video_id
        areas = _ootb_seq_areas(seq_dir)
    elif dataset == "satsot":
        seq_dir = root / video_id
        areas = _satsot_seq_areas(seq_dir)
    elif dataset == "sv248s":
        # video_id format: "<video_dir>/<seq_id>"
        video_part, seq_id = video_id.split("/")
        ann_dir = root / video_part / "annotations"
        rect_path = ann_dir / f"{seq_id}.rect"
        state_path = ann_dir / f"{seq_id}.state"
        if not rect_path.exists():
            return None
        areas = _sv248s_seq_areas(rect_path, state_path)
    else:
        return None
    if not areas:
        return None
    return float(np.sqrt(np.median(areas)))


def build_sequence_record(
    dataset: str,
    root: Path,
    video,
    native_attrs: list[str],
) -> dict:
    """Assemble one ``sequences[]`` entry."""
    vid = video.video_id

    # Paths relative to the dataset root (consumer joins their own root).
    if dataset == "ootb":
        image_dir = f"{vid}/img"
        gt_path   = f"{vid}/groundtruth.txt"
    elif dataset == "satsot":
        image_dir = f"{vid}/img"
        gt_path   = f"{vid}/groundtruth.txt"
    else:  # sv248s
        video_part, seq_id = vid.split("/")
        image_dir = f"{video_part}/sequences/{seq_id}"
        gt_path   = f"{video_part}/annotations/{seq_id}.rect"

    median = _seq_median_sqrt_area(dataset, root, vid)

    return {
        "id":            f"{dataset}/{vid}",
        "dataset":       dataset,
        "video_id":      vid,
        "category":      video.category,
        "n_frames":      int(video.num_frames),
        "image_dir":     image_dir,
        "gt_path":       gt_path,
        "gt_format":     DATASET_INFO[dataset]["gt_format"],
        "native_attrs":  sorted(native_attrs),
        "unified_attrs": _native_to_unified(dataset, native_attrs),
        "median_sqrt_area_px": round(median, 3) if median is not None else None,
        "tiny":          bool(median is not None and median < TINY_SQRT_AREA_THRESH),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="space_tracker/space_tracker.json",
        help="Output manifest path.",
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("[1/2] loading datasets")
    ds_objs = {
        "ootb":   OOTBDataset(  root=DATASET_ROOTS["ootb"],   split="no_split", mode="detection"),
        "satsot": SatSOTDataset(root=DATASET_ROOTS["satsot"], split="no_split", mode="detection"),
        "sv248s": SV248SDataset(root=DATASET_ROOTS["sv248s"], split="no_split", mode="detection"),
    }

    print("[2/2] building sequence records")
    sequences: list[dict] = []
    for ds_name, ds in ds_objs.items():
        seq_attrs_map = ds.sequence_attributes()
        for v in ds.videos:
            native = seq_attrs_map.get(v.video_id, [])
            sequences.append(build_sequence_record(
                ds_name, DATASET_ROOTS[ds_name], v, native,
            ))
        print(f"   {ds_name}: {sum(1 for s in sequences if s['dataset'] == ds_name)} sequences "
              f"(tiny={sum(1 for s in sequences if s['dataset'] == ds_name and s['tiny'])})")

    manifest = {
        "version":       MANIFEST_VERSION,
        "name":          "space-tracker",
        "description":   "Unified single-object-tracking benchmark across SatSOT, SV248S, and OOTB. Manifest indexes sequences and unified-attribute groupings; raw imagery and GT must be obtained from the original datasets per their license terms.",
        "evaluation": {
            "aggregation":                  "per_sequence",
            "metrics":                      ["SR", "NPR", "PR", "P@5"],
            "pr_threshold_max_px":          50,
            "norm_pr_threshold_max":        0.5,
            "success_thresholds":           "linspace(0, 1, 21)  # IoU",
            "precision_thresholds_px":      "arange(0, 51, 1)   # CLE in px",
            "norm_precision_thresholds":    "linspace(0, 0.5, 21)  # normalised CLE",
            "tiny_threshold_sqrt_area_px":  TINY_SQRT_AREA_THRESH,
            "tiny_definition":              "A sequence is tiny if its median sqrt(GT area) across annotated frames is < 8 px.",
        },
        "datasets":            DATASET_INFO,
        "unified_attributes":  UNIFIED_ATTR_SPEC,
        "n_sequences":         len(sequences),
        "sequences":           sequences,
    }

    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {out_path}  ({out_path.stat().st_size / 1024:.1f} KB, "
          f"{len(sequences)} sequences)")


if __name__ == "__main__":
    main()
