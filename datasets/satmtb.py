"""
SATMTBDataset
=============
Multi-task satellite video benchmark supporting:

  - **det_hbb** — Detection with Horizontal Bounding Boxes (XML)
  - **det_obb** — Detection with Oriented Bounding Boxes (XML → AABB)
  - **mot**     — Multi-Object Tracking (CSV, MOT format)
  - **seg**     — Instance Segmentation (COCO-like per-frame JSON)

Directory layout::

    <root>/SAT-MTB_Dataset/
        airplane/<seq>/img/000001.png …
                      /det/HBB/000001.xml …
                      /det/OBB/000001.xml …
                      /mot/<seq>            (no extension)
                      /seg/000001.json …
        car/<seq>/img/000001.png …
                 /mot/<seq>.txt
        ship/<seq>/…
        train/<seq>/…
        data_split.xlsx

Annotation availability by category:

    +-----------+-----+-----+-----+-----+
    | Category  | HBB | OBB | MOT | Seg |
    +-----------+-----+-----+-----+-----+
    | airplane  |  Y  |  Y  |  Y  |  Y  |
    | car       |  N  |  N  |  Y  |  N  |
    | ship      |  Y  |  Y  |  Y  |  Y  |
    | train 1-7 |  Y  |  Y  |  Y  |  Y  |
    | train 8-10|  Y  |  Y  |  N  |  Y  |
    | train11-16|  N  |  N  |  Y  |  N  |
    +-----------+-----+-----+-----+-----+

Official split: **train / test** (from ``data_split.xlsx``).
Val is carved from **30 %** of the official test set (stratified by
category, ``seed=42``); the remaining 70 % stays as test.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from .base import BaseVideoDataset, VideoInfo

_SPLIT_SEED = 42

# MOT class column → coarse-grained category name
_MOT_CLASS_MAP: dict[int, str] = {0: "car", 1: "airplane", 2: "ship", 3: "train"}

# Task → xlsx sheet name
_TASK_SHEET: dict[str, str] = {
    "det_hbb": "det_HBB",
    "det_obb": "det_OBB",
    "mot": "mot",
    "seg": "seg",
}

# Coarse-grained categories (level 1): 4 classes.
COARSE_CATEGORIES = ("airplane", "car", "ship", "train")

# Fine-grained categories (level 2): 14 classes, grouped by coarse parent.
#   airplane: WA, NA, RA, FA, CA
#   ship:     SB, YH, CS, FH, NV, OS
#   car:      LC, SC
#   train:    TN
FINE_CATEGORIES = (
    "WA", "NA", "RA", "FA", "CA",
    "SB", "YH", "CS", "FH", "NV", "OS",
    "LC", "SC",
    "TN",
)

# Fine-grained label string (as stored in XML/JSON annotations) → coarse name.
# Keys cover both the short codes (WA, NA, …) and common full-name variants
# seen in annotations, so the parser is tolerant of both conventions.
FINE_TO_COARSE: dict[str, str] = {
    # airplane
    "WA": "airplane", "NA": "airplane", "RA": "airplane",
    "FA": "airplane", "CA": "airplane",
    "wide-bodied aircraft": "airplane", "narrow-bodied aircraft": "airplane",
    "rear-engined aircraft": "airplane", "four-engine aircraft": "airplane",
    "corporate aircraft": "airplane",
    # ship
    "SB": "ship", "YH": "ship", "CS": "ship",
    "FH": "ship", "NV": "ship", "OS": "ship",
    "speed boat": "ship", "yacht": "ship", "cruise": "ship",
    "freighter": "ship", "naval vessel": "ship", "other ship": "ship",
    # car
    "LC": "car", "SC": "car",
    "large car": "car", "small car": "car",
    # train
    "TN": "train", "train": "train",
}

_CATEGORIES = COARSE_CATEGORIES  # back-compat alias for the private uses below

TaskType = Literal["det_hbb", "det_obb", "mot", "seg"]


def _derive_coarse(raw_name: str) -> str:
    """Return the coarse parent of an annotation's class string.

    Falls back to the raw string itself if it is already a coarse name
    (e.g. MOT annotations store only coarse-level labels), so callers don't
    have to know which granularity each task produces.
    """
    if raw_name in FINE_TO_COARSE:
        return FINE_TO_COARSE[raw_name]
    if raw_name in COARSE_CATEGORIES:
        return raw_name
    return raw_name  # unknown; caller's class_map will map it to -1


class SATMTBDataset(BaseVideoDataset):
    """
    SAT-MTB multi-task satellite video dataset.

    Args:
        root:   Path containing the ``SAT-MTB_Dataset/`` directory.
        split:  ``"train"``, ``"val"``, ``"test"``, or ``"no_split"``.
        task:   Annotation source — ``"det_hbb"``, ``"det_obb"``,
                ``"mot"``, or ``"seg"``.
        **kwargs:  Forwarded to :class:`BaseVideoDataset`.
    """

    TASKS = {"det_hbb", "det_obb", "mot", "seg"}

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        task: TaskType = "det_hbb",
        class_map_fine: dict[str, int] | None = None,
        **kwargs,
    ):
        if task not in self.TASKS:
            raise ValueError(f"task must be one of {self.TASKS}, got {task!r}")
        self.task = task
        # Optional second-level class map (14 fine-grained classes). When set,
        # ``_load_annotations`` returns a ``labels_fine`` array alongside the
        # coarse ``labels`` array. Annotation strings missing from this map
        # produce ``-1`` in the fine-label array — the coarse label is always
        # required for the sample to survive the coarse class_map filter.
        self.class_map_fine: dict[str, int] = class_map_fine or {}
        self._ann_cache: dict[str, dict[int, list[dict]]] = {}
        super().__init__(root=root, split=split, **kwargs)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        data_dir = self.root / "SAT-MTB_Dataset"
        xlsx_path = data_dir / "data_split.xlsx"

        # --- Step 1: parse official splits from xlsx ---
        split_map = self._parse_splits(xlsx_path)

        # --- Step 2: carve val from test (30 %, stratified by category) ---
        split_map = self._carve_val(split_map)

        # --- Step 3: discover sequences ---
        for cat in _CATEGORIES:
            cat_dir = data_dir / cat
            if not cat_dir.is_dir():
                continue

            for seq_dir in sorted(cat_dir.iterdir()):
                if not seq_dir.is_dir():
                    continue
                seq_num = seq_dir.name          # e.g. "02"
                video_id = f"{cat}/{seq_num}"

                # Skip if no annotations for this task
                if not self._has_annotations(seq_dir):
                    continue

                # Discover frames
                img_dir = seq_dir / "img"
                if not img_dir.is_dir():
                    continue
                frame_ids = sorted(int(p.stem) for p in img_dir.glob("*.png"))
                if not frame_ids:
                    continue

                # Parse & cache annotations
                self._cache_annotations(seq_dir, video_id, frame_ids)

                assigned_split = split_map.get(video_id, "no_split")
                video = VideoInfo(
                    video_id=video_id,
                    dataset="SAT-MTB",
                    category=cat,
                    split=assigned_split,
                    num_frames=len(frame_ids),
                    frame_ids=frame_ids,
                )

                if self.split == "no_split" or video.split == self.split:
                    self.videos.append(video)
                else:
                    self._ann_cache.pop(video_id, None)

    def _load_frame(self, video: VideoInfo, frame_id: int) -> np.ndarray:
        cat, seq_num = video.video_id.split("/")
        path = (
            self.root / "SAT-MTB_Dataset" / cat / seq_num
            / "img" / f"{frame_id:06d}.png"
        )
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Frame not found: {path}")
        return img[..., ::-1].copy()  # BGR → RGB

    def _load_annotations(
        self, video: VideoInfo, frame_id: int
    ) -> dict[str, np.ndarray]:
        objs = self._ann_cache.get(video.video_id, {}).get(frame_id, [])

        empty = {
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "labels": np.zeros(0, dtype=np.int64),
            "labels_fine": np.zeros(0, dtype=np.int64),
            "track_ids": np.zeros(0, dtype=np.int64),
        }
        if not objs:
            return empty

        boxes, labels, labels_fine, track_ids = [], [], [], []
        for obj in objs:
            lbl = self._map_label(obj["class"])
            if lbl < 0:
                continue
            fine_name = obj.get("class_fine")
            lbl_fine = (
                self.class_map_fine.get(fine_name, -1)
                if fine_name is not None else -1
            )
            boxes.append(obj["box"])
            labels.append(lbl)
            labels_fine.append(lbl_fine)
            track_ids.append(obj["track_id"])

        if not boxes:
            return empty

        return {
            "boxes": np.array(boxes, dtype=np.float32),
            "labels": np.array(labels, dtype=np.int64),
            "labels_fine": np.array(labels_fine, dtype=np.int64),
            "track_ids": np.array(track_ids, dtype=np.int64),
        }

    # ------------------------------------------------------------------
    # Segmentation support
    # ------------------------------------------------------------------

    def load_masks(
        self, video: VideoInfo, frame_id: int
    ) -> dict[str, np.ndarray]:
        """
        Load per-instance segmentation masks (task ``"seg"`` only).

        Returns:
            Dict with keys:
                masks:       ``(N, H, W)`` uint8 ``{0, 1}``
                labels:      ``(N,)`` int64 (coarse global class id)
                labels_fine: ``(N,)`` int64 (fine class id; -1 if unmapped)
                track_ids:   ``(N,)`` int64
        """
        if self.task != "seg":
            raise RuntimeError("load_masks() requires task='seg'")

        cat, seq_num = video.video_id.split("/")
        json_path = (
            self.root / "SAT-MTB_Dataset" / cat / seq_num
            / "seg" / f"{frame_id:06d}.json"
        )

        empty = {
            "masks": np.zeros((0, 0, 0), dtype=np.uint8),
            "labels": np.zeros(0, dtype=np.int64),
            "labels_fine": np.zeros(0, dtype=np.int64),
            "track_ids": np.zeros(0, dtype=np.int64),
        }
        if not json_path.exists():
            return empty

        with open(json_path) as f:
            data = json.load(f)

        img_info = data["images"][0]
        h, w = int(img_info["height"]), int(img_info["width"])

        # category_id → (coarse, fine-or-None) — mirrors _parse_seg.
        cat_id_map: dict[int, tuple[str, str | None]] = {}
        for c in data.get("categories", []):
            coarse = c.get("supercategory") or _derive_coarse(c.get("name", ""))
            fine = c.get("name") if c.get("supercategory") else None
            cat_id_map[c["id"]] = (coarse or "unknown", fine)

        masks_list, labels_list, labels_fine_list, tids_list = [], [], [], []
        for ann in data.get("annotations", []):
            coarse, fine = cat_id_map.get(ann["category_id"], ("unknown", None))
            lbl = self._map_label(coarse)
            if lbl < 0:
                continue
            lbl_fine = (
                self.class_map_fine.get(fine, -1) if fine is not None else -1
            )

            # Rasterise polygon(s) to binary mask
            mask = np.zeros((h, w), dtype=np.uint8)
            for poly in ann.get("segmentation", []):
                pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                cv2.fillPoly(mask, [pts.astype(np.int32)], 1)

            masks_list.append(mask)
            labels_list.append(lbl)
            labels_fine_list.append(lbl_fine)
            tids_list.append(ann.get("id", -1))

        if not masks_list:
            return {**empty, "masks": np.zeros((0, h, w), dtype=np.uint8)}

        return {
            "masks": np.stack(masks_list),
            "labels": np.array(labels_list, dtype=np.int64),
            "labels_fine": np.array(labels_fine_list, dtype=np.int64),
            "track_ids": np.array(tids_list, dtype=np.int64),
        }

    # ------------------------------------------------------------------
    # Split parsing
    # ------------------------------------------------------------------

    def _parse_splits(self, xlsx_path: Path) -> dict[str, str]:
        """Parse ``data_split.xlsx`` → ``{video_id: "train"/"test"}``."""
        import openpyxl

        sheet_name = _TASK_SHEET[self.task]
        wb = openpyxl.load_workbook(xlsx_path, read_only=True)
        ws = wb[sheet_name]
        rows = list(ws.iter_rows(values_only=True))
        wb.close()

        # Header row: (None, 'airplane', 'ship', 'train'[, 'car'])
        header = rows[0]
        categories = [str(c) for c in header[1:] if c is not None]

        split_map: dict[str, str] = {}
        current_split: str | None = None

        for row in rows[1:]:
            if row[0] is not None:
                current_split = str(row[0])
            if current_split is None:
                continue
            for ci, cat_name in enumerate(categories):
                val = row[ci + 1]
                if val is not None:
                    seq_num = f"{int(val):02d}"
                    video_id = f"{cat_name}/{seq_num}"
                    split_map[video_id] = current_split

        return split_map

    @staticmethod
    def _carve_val(split_map: dict[str, str]) -> dict[str, str]:
        """Split test into val (30 %) + test (70 %), stratified by category."""
        rng = np.random.RandomState(_SPLIT_SEED)

        by_cat: dict[str, list[str]] = defaultdict(list)
        for vid, split in split_map.items():
            if split == "test":
                cat = vid.split("/")[0]
                by_cat[cat].append(vid)

        new_map = dict(split_map)
        for cat in sorted(by_cat):
            seqs = sorted(by_cat[cat])     # sort for determinism before shuffle
            rng.shuffle(seqs)
            n_val = max(1, round(len(seqs) * 0.3))
            # If only 1 sequence → keep as test
            if len(seqs) <= 1:
                continue
            for vid in seqs[:n_val]:
                new_map[vid] = "val"

        return new_map

    # ------------------------------------------------------------------
    # Annotation helpers
    # ------------------------------------------------------------------

    def _has_annotations(self, seq_dir: Path) -> bool:
        """Check whether *seq_dir* has the annotations required by ``self.task``."""
        if self.task == "det_hbb":
            return (seq_dir / "det" / "HBB").is_dir()
        if self.task == "det_obb":
            return (seq_dir / "det" / "OBB").is_dir()
        if self.task == "mot":
            return (seq_dir / "mot").is_dir()
        if self.task == "seg":
            return (seq_dir / "seg").is_dir()
        return False

    def _cache_annotations(
        self, seq_dir: Path, video_id: str, frame_ids: list[int]
    ) -> None:
        if self.task == "det_hbb":
            self._ann_cache[video_id] = self._parse_det_hbb(
                seq_dir / "det" / "HBB", frame_ids
            )
        elif self.task == "det_obb":
            self._ann_cache[video_id] = self._parse_det_obb(
                seq_dir / "det" / "OBB", frame_ids
            )
        elif self.task == "mot":
            self._ann_cache[video_id] = self._parse_mot(seq_dir / "mot")
        elif self.task == "seg":
            self._ann_cache[video_id] = self._parse_seg(
                seq_dir / "seg", frame_ids
            )

    # --- det HBB ---

    @staticmethod
    def _parse_det_hbb(
        ann_dir: Path, frame_ids: list[int]
    ) -> dict[int, list[dict]]:
        cache: dict[int, list[dict]] = {}
        for fid in frame_ids:
            xml_path = ann_dir / f"{fid:06d}.xml"
            if not xml_path.exists():
                cache[fid] = []
                continue

            tree = ET.parse(xml_path)
            root = tree.getroot()
            objs: list[dict] = []
            for obj_el in root.findall("object"):
                # XML <name> is the fine-grained label (e.g. "WA"). Derive the
                # coarse parent for class_map filtering; keep the fine name for
                # class_map_fine lookup.
                name = obj_el.findtext("name", "unknown")
                obj_id = int(obj_el.findtext("objectID", "-1"))
                bb = obj_el.find("bndbox")
                xmin = float(bb.findtext("xmin", "0"))
                ymin = float(bb.findtext("ymin", "0"))
                xmax = float(bb.findtext("xmax", "0"))
                ymax = float(bb.findtext("ymax", "0"))
                objs.append({
                    "box": [xmin, ymin, xmax, ymax],
                    "class": _derive_coarse(name),
                    "class_fine": name,
                    "track_id": obj_id,
                })
            cache[fid] = objs
        return cache

    # --- det OBB ---

    @staticmethod
    def _parse_det_obb(
        ann_dir: Path, frame_ids: list[int]
    ) -> dict[int, list[dict]]:
        cache: dict[int, list[dict]] = {}
        for fid in frame_ids:
            xml_path = ann_dir / f"{fid:06d}.xml"
            if not xml_path.exists():
                cache[fid] = []
                continue

            tree = ET.parse(xml_path)
            root = tree.getroot()
            objs: list[dict] = []
            for obj_el in root.findall("object"):
                name = obj_el.findtext("name", "unknown")
                obj_id = int(obj_el.findtext("objectID", "-1"))
                rob = obj_el.find("robndbox")
                corners_x = [
                    float(rob.findtext(f"x{i}", "0")) for i in range(4)
                ]
                corners_y = [
                    float(rob.findtext(f"y{i}", "0")) for i in range(4)
                ]
                # OBB → AABB
                xmin, xmax = min(corners_x), max(corners_x)
                ymin, ymax = min(corners_y), max(corners_y)
                objs.append({
                    "box": [xmin, ymin, xmax, ymax],
                    "class": _derive_coarse(name),
                    "class_fine": name,
                    "track_id": obj_id,
                })
            cache[fid] = objs
        return cache

    # --- MOT ---

    @staticmethod
    def _parse_mot(mot_dir: Path) -> dict[int, list[dict]]:
        """Parse MOT CSV (one file per sequence)."""
        # File may be ``<num>`` (no ext) or ``<num>.txt``
        mot_files = sorted(p for p in mot_dir.iterdir() if p.is_file())
        if not mot_files:
            return {}
        mot_file = mot_files[0]

        cache: dict[int, list[dict]] = defaultdict(list)
        with open(mot_file) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 8:
                    continue
                fid = int(parts[0])
                obj_id = int(parts[1])
                x = float(parts[2])       # xmin (top-left)
                y = float(parts[3])       # ymin (top-left)
                w = float(parts[4])
                h = float(parts[5])
                cls_id = int(parts[7])
                cat_name = _MOT_CLASS_MAP.get(cls_id, "unknown")

                # MOT annotations only carry the coarse class id — no fine-
                # grained label is available, so ``class_fine`` is None and
                # the downstream labels_fine column will be -1 for MOT data.
                cache[fid].append({
                    "box": [x, y, x + w, y + h],
                    "class": cat_name,
                    "class_fine": None,
                    "track_id": obj_id,
                })
        return dict(cache)

    # --- Segmentation ---

    @staticmethod
    def _parse_seg(
        seg_dir: Path, frame_ids: list[int]
    ) -> dict[int, list[dict]]:
        """Parse COCO-like per-frame JSON annotations (boxes only for the cache)."""
        cache: dict[int, list[dict]] = {}
        for fid in frame_ids:
            json_path = seg_dir / f"{fid:06d}.json"
            if not json_path.exists():
                cache[fid] = []
                continue

            with open(json_path) as f:
                data = json.load(f)

            # category_id → (coarse supercategory, fine name) pair. COCO-like
            # schemas put the fine-grained label in ``name`` and the coarse
            # parent in ``supercategory``; fall back gracefully when only one
            # of the two is present.
            cat_id_map: dict[int, tuple[str, str | None]] = {}
            for c in data.get("categories", []):
                coarse = c.get("supercategory") or _derive_coarse(c.get("name", ""))
                fine = c.get("name") if c.get("supercategory") else None
                cat_id_map[c["id"]] = (coarse or "unknown", fine)

            objs: list[dict] = []
            for ann in data.get("annotations", []):
                coarse, fine = cat_id_map.get(
                    ann["category_id"], ("unknown", None)
                )
                bbox = ann["bbox"]      # [xmin, ymin, xmax, ymax]
                objs.append({
                    "box": [float(bbox[0]), float(bbox[1]),
                            float(bbox[2]), float(bbox[3])],
                    "class": coarse,
                    "class_fine": fine,
                    "track_id": ann.get("id", -1),
                })
            cache[fid] = objs
        return cache
