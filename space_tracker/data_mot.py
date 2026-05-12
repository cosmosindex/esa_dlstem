"""Per-frame loader for the Space-tracker MOT manifest.

Six on-disk GT formats are supported, dispatched by ``seq.gt_format``:

* ``mot_csv_9col``               — AIRMOT
* ``mot_csv_11col``              — SAT-MTB
* ``viso_dual``                  — VISO (comma+xywh for car/train, space+xyxy for plane/ship)
* ``mot_csv_10col_0idx``         — SDM-Car (frame ids 0-indexed)
* ``coco_mot_json``              — RsCarData train/val (one JSON for all sequences in the split)
* ``pascal_voc_xml_per_frame``   — RsCarData test (HiEUM re-curated XML, one file per frame)

Two image-source modes:

* ``image_format='frames'`` — one JPEG/PNG per frame; iter globs the image
  directory and yields image paths.
* ``image_format='video'``  — one .avi per sequence (SDM-Car only); iter
  uses OpenCV ``VideoCapture`` to decode frames sequentially. Random
  access is slow; pre-extract frames offline if you do many runs.

The unified per-object ``category`` is one of ``car``, ``airplane``, ``ship``,
``train`` (VISO's ``plane`` is folded into ``airplane``).
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .manifest_mot import MOTSequenceRecord


# ---------------------------------------------------------------------------
# Per-object / per-frame records
# ---------------------------------------------------------------------------

@dataclass
class MOTObject:
    """One ground-truth object on one frame."""
    track_id: int
    category: str            # "car" | "airplane" | "ship" | "train"
    bbox_xyxy: np.ndarray    # (4,) float32  — (x1, y1, x2, y2) in absolute pixels


@dataclass
class MOTFrame:
    """One frame's worth of GT + (optionally) the decoded image."""
    frame_id: int                       # native frame id (matches GT's frame column)
    image: np.ndarray | None            # HxWxC uint8 RGB; None if decode_images=False
    image_path: Path | None             # path on disk (frames mode); None for video mode
    objects: list[MOTObject] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Class-id maps
# ---------------------------------------------------------------------------

# AIRMOT ``class`` column → unified category
_AIRMOT_CLS: dict[str, str] = {"1": "airplane", "2": "ship"}

# SAT-MTB ``cls_id`` column → unified category
_SATMTB_CLS: dict[int, str] = {0: "car", 1: "airplane", 2: "ship", 3: "train"}

# VISO directory name → unified category
_VISO_CLS: dict[str, str] = {"car": "car", "plane": "airplane", "ship": "ship", "train": "train"}


# ---------------------------------------------------------------------------
# Per-format parsers
# ---------------------------------------------------------------------------

def _parse_mot_csv_9col(gt_path: Path) -> dict[int, list[MOTObject]]:
    """AIRMOT: ``frame, track, x, y, w, h, conf, cls, vis`` (xywh top-left)."""
    out: dict[int, list[MOTObject]] = defaultdict(list)
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 9:
                continue
            fid       = int(parts[0])
            track_id  = int(parts[1])
            x, y, w, h = (float(p) for p in parts[2:6])
            cls_raw   = parts[7].strip()
            category  = _AIRMOT_CLS.get(cls_raw, f"class_{cls_raw}")
            out[fid].append(MOTObject(
                track_id=track_id,
                category=category,
                bbox_xyxy=np.array([x, y, x + w, y + h], dtype=np.float32),
            ))
    return dict(out)


def _parse_mot_csv_11col(gt_path: Path) -> dict[int, list[MOTObject]]:
    """SAT-MTB: ``frame, obj, x, y, w, h, conf, cls_id, r1, r2, r3`` (xywh top-left)."""
    out: dict[int, list[MOTObject]] = defaultdict(list)
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            fid       = int(parts[0])
            track_id  = int(parts[1])
            x, y, w, h = (float(p) for p in parts[2:6])
            cls_id    = int(parts[7])
            category  = _SATMTB_CLS.get(cls_id, f"class_{cls_id}")
            out[fid].append(MOTObject(
                track_id=track_id,
                category=category,
                bbox_xyxy=np.array([x, y, x + w, y + h], dtype=np.float32),
            ))
    return dict(out)


def _parse_viso_dual(gt_path: Path, native_category: str) -> dict[int, list[MOTObject]]:
    """VISO: auto-detect comma+xywh (car/train) vs. space+xyxy (plane/ship).

    All objects in a VISO sequence share the same category, taken from the
    parent directory (``native_category``).
    """
    unified_cat = _VISO_CLS.get(native_category, native_category)
    with open(gt_path) as f:
        first_line = f.readline()
    is_comma = "," in first_line

    out: dict[int, list[MOTObject]] = defaultdict(list)
    with open(gt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",") if is_comma else line.split()
            if len(parts) < 6:
                continue
            fid       = int(parts[0])
            track_id  = int(parts[1])
            if is_comma:
                x, y, w, h = (float(p) for p in parts[2:6])
                box = np.array([x, y, x + w, y + h], dtype=np.float32)
            else:
                x1, y1, x2, y2 = (float(p) for p in parts[2:6])
                box = np.array([x1, y1, x2, y2], dtype=np.float32)
            out[fid].append(MOTObject(
                track_id=track_id, category=unified_cat, bbox_xyxy=box,
            ))
    return dict(out)


def _parse_mot_csv_10col_0idx(gt_path: Path) -> dict[int, list[MOTObject]]:
    """SDM-Car: ``frame, track, x, y, w, h, -1, -1, -1, -1`` (xywh top-left, 0-indexed)."""
    out: dict[int, list[MOTObject]] = defaultdict(list)
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            fid       = int(parts[0])
            track_id  = int(parts[1])
            x, y, w, h = (float(p) for p in parts[2:6])
            out[fid].append(MOTObject(
                track_id=track_id, category="car",
                bbox_xyxy=np.array([x, y, x + w, y + h], dtype=np.float32),
            ))
    return dict(out)


@lru_cache(maxsize=4)
def _load_coco_mot_json(gt_path: Path) -> dict[str, dict[int, list[MOTObject]]]:
    """Load + index a COCO-MOT JSON once; return ``{video_name: {frame_id: [...]}}``.

    The JSON bundles every sequence in one file (one of train_mot.json /
    test1024_mot.json), so we cache by absolute path to avoid reparsing for
    each of the 70 (or 7) sequences.
    """
    with open(gt_path) as f:
        data = json.load(f)

    video_meta: dict[int, str] = {v["id"]: v["file_name"] for v in data["videos"]}
    img_to_video: dict[int, tuple[str, int]] = {}
    for img in data["images"]:
        vname = video_meta[img["video_id"]]
        fid   = int(img["video_frame_id"])     # 1-indexed within the sequence
        img_to_video[img["id"]] = (vname, fid)

    out: dict[str, dict[int, list[MOTObject]]] = defaultdict(lambda: defaultdict(list))
    for ann in data["annotations"]:
        vname, fid = img_to_video[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        out[vname][fid].append(MOTObject(
            track_id=int(ann["track_id"]),
            category="car",
            bbox_xyxy=np.array([x, y, x + w, y + h], dtype=np.float32),
        ))
    return {k: dict(v) for k, v in out.items()}


def _parse_coco_mot_json(gt_path: Path, video_id: str) -> dict[int, list[MOTObject]]:
    """RsCarData train/val. ``video_id`` looks like ``train/001``."""
    cache = _load_coco_mot_json(gt_path)
    seq_name = video_id.split("/", 1)[1]    # "001"
    return cache.get(seq_name, {})


def _parse_pascal_voc_xml_dir(xml_dir: Path) -> dict[int, list[MOTObject]]:
    """RsCarData test (HiEUM): one PASCAL-VOC XML per frame under ``xml_dir``."""
    out: dict[int, list[MOTObject]] = {}
    for xml_path in sorted(xml_dir.glob("*.xml")):
        try:
            fid = int(xml_path.stem)
        except ValueError:
            continue
        tree = ET.parse(xml_path)
        objs: list[MOTObject] = []
        for obj_el in tree.findall("object"):
            name_el = obj_el.find("name")
            if name_el is None or (name_el.text or "").lower() != "car":
                continue
            bb = obj_el.find("bndbox")
            if bb is None:
                continue
            x1 = float(bb.findtext("xmin", "0"))
            y1 = float(bb.findtext("ymin", "0"))
            x2 = float(bb.findtext("xmax", "0"))
            y2 = float(bb.findtext("ymax", "0"))
            tid_el = obj_el.find("id")
            tid = int(tid_el.text) if tid_el is not None and tid_el.text else -1
            objs.append(MOTObject(
                track_id=tid, category="car",
                bbox_xyxy=np.array([x1, y1, x2, y2], dtype=np.float32),
            ))
        if objs:
            out[fid] = objs
    return out


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def _parse_gt(seq: MOTSequenceRecord, root: Path) -> dict[int, list[MOTObject]]:
    """Parse the GT for one sequence into ``{frame_id: list[MOTObject]}``."""
    gt_full = root / seq.gt_path

    if seq.gt_format == "mot_csv_9col":
        return _parse_mot_csv_9col(gt_full)
    if seq.gt_format == "mot_csv_11col":
        return _parse_mot_csv_11col(gt_full)
    if seq.gt_format == "viso_dual":
        # native_category is the VISO directory name (car/plane/ship/train).
        # seq.video_id starts with the native directory name.
        native_cat = seq.video_id.split("/", 1)[0]
        return _parse_viso_dual(gt_full, native_cat)
    if seq.gt_format == "mot_csv_10col_0idx":
        return _parse_mot_csv_10col_0idx(gt_full)
    if seq.gt_format == "coco_mot_json":
        return _parse_coco_mot_json(gt_full, seq.video_id)
    if seq.gt_format == "pascal_voc_xml_per_frame":
        return _parse_pascal_voc_xml_dir(gt_full)
    raise ValueError(f"Unknown gt_format {seq.gt_format!r} for sequence {seq.id}")


_FRAME_STEM_RE = re.compile(r"^(\d+)")


def _enumerate_frame_paths(seq: MOTSequenceRecord, root: Path) -> list[tuple[int, Path]]:
    """For frames-mode sequences, return [(frame_id, image_path), ...] in capture order."""
    template = seq.image_path_pattern
    if template is None:
        raise ValueError(f"sequence {seq.id} has no image_path_pattern (image_format={seq.image_format})")

    template_path = Path(template)
    img_dir = root / template_path.parent
    ext = template_path.suffix
    paths = sorted(img_dir.glob(f"*{ext}"))

    out: list[tuple[int, Path]] = []
    for p in paths:
        m = _FRAME_STEM_RE.match(p.stem)
        if not m:
            continue
        out.append((int(m.group(1)), p))
    return out


def iter_mot_frames(
    seq: MOTSequenceRecord,
    dataset_roots: dict[str, str | Path],
    decode_images: bool = True,
) -> Iterator[MOTFrame]:
    """Yield one :class:`MOTFrame` per frame in ``seq``, in capture order.

    ``decode_images=False`` skips image decoding (handy when you only need GT
    for offline analysis); in video mode the call still advances the decoder
    via ``cap.grab`` so frames stay aligned.
    """
    if seq.dataset not in dataset_roots:
        raise KeyError(
            f"dataset_roots is missing an entry for '{seq.dataset}' "
            f"(needed by sequence {seq.id})"
        )
    root = Path(dataset_roots[seq.dataset])
    ann_cache = _parse_gt(seq, root)

    if seq.image_format == "video":
        if seq.video_path is None:
            raise ValueError(f"sequence {seq.id} has image_format=video but no video_path")
        cap = cv2.VideoCapture(str(root / seq.video_path))
        try:
            fid = seq.frame_index_base       # SDM-Car: 0
            while True:
                if decode_images:
                    ret, frame_bgr = cap.read()
                    if not ret:
                        break
                    image = frame_bgr[..., ::-1].copy()
                else:
                    if not cap.grab():
                        break
                    image = None
                yield MOTFrame(
                    frame_id=fid,
                    image=image,
                    image_path=None,
                    objects=ann_cache.get(fid, []),
                )
                fid += 1
        finally:
            cap.release()
        return

    # frames mode
    for fid, img_path in _enumerate_frame_paths(seq, root):
        if decode_images:
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                raise FileNotFoundError(f"failed to read {img_path}")
            image = bgr[..., ::-1].copy()
        else:
            image = None
        yield MOTFrame(
            frame_id=fid,
            image=image,
            image_path=img_path,
            objects=ann_cache.get(fid, []),
        )
