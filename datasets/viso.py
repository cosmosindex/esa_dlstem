"""
VISODataset
===========
Video satellite object detection & tracking dataset (VISO).

Supports **detection**, **SOT**, and **MOT** from the MOT annotation format.
All objects in a given sequence share the same category (determined by
the parent directory: ``car/``, ``plane/``, ``ship/``, ``train/``).

Directory layout::

    <root>/
        mot/
            car/001/img/000001.jpg …
                    gt/gt.txt
            car/002/…
            …
            plane/039/…
            ship/045/…
            train/046/…

47 sequences total:  car (38), plane (6), ship (2), train (1).

Annotation format (``gt.txt``) varies by category:

    Car / Train  — comma-delimited ``frame,obj_id,x,y,w,h,conf,cls,r1,r2``
    Plane / Ship — space-delimited ``frame obj_id x1 y1 x2 y2 r1 r2 r3 r4``

Split strategy: **Official** COCO/VOC frame-level split mapped to
sequence-level by majority vote.  Ship has no val; the *train*
category has no val/test (single sequence → always train).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .base import BaseVideoDataset, VideoInfo

_SPLIT_SEED = 42

_CATEGORIES = ("car", "plane", "ship", "train")

# ---------------------------------------------------------------------------
# Official split: COCO/VOC frame-level → sequence-level by majority vote.
#
# Car:   COCO train frames 1-9417  → seqs 001-024 (100 % train)
#         seq 025 is 29 % train / 71 % val → val
#        COCO val   frames 9418-10704 → seqs 025-028 (majority val)
#        COCO test  frames 10705-13421 → seqs 029-038 (majority test)
# Plane: seqs 039-042 train, 043 val, 044 test
# Ship:  045 train, 047 test  (no val in official split)
# Train: 046 train only (single sequence)
# ---------------------------------------------------------------------------
_OFFICIAL_SPLIT: dict[str, str] = {
    # car — 24 train, 4 val, 10 test
    **{f"car/{i:03d}": "train" for i in range(1, 25)},
    **{f"car/{i:03d}": "val" for i in range(25, 29)},
    **{f"car/{i:03d}": "test" for i in range(29, 39)},
    # plane — 4 train, 1 val, 1 test
    **{f"plane/{i:03d}": "train" for i in range(39, 43)},
    "plane/043": "val",
    "plane/044": "test",
    # ship — 1 train, 1 test (no val)
    "ship/045": "train",
    "ship/047": "test",
    # train — 1 sequence, all train
    "train/046": "train",
}


class VISODataset(BaseVideoDataset):
    """
    VISO satellite traffic dataset (MOT annotation format).

    Args:
        root:   Path to dataset root (must contain ``mot/`` sub-directory).
        split:  ``"train"``, ``"val"``, ``"test"``, or ``"no_split"``.
        **kwargs:  Forwarded to :class:`BaseVideoDataset`.
    """

    def __init__(self, root: str | Path, split: str = "train", **kwargs):
        self._ann_cache: dict[str, dict[int, list[dict]]] = {}
        super().__init__(root=root, split=split, **kwargs)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        mot_dir = self.root / "mot"

        for cat in _CATEGORIES:
            cat_dir = mot_dir / cat
            if not cat_dir.is_dir():
                continue

            for seq_dir in sorted(cat_dir.iterdir()):
                if not seq_dir.is_dir():
                    continue

                gt_path = seq_dir / "gt" / "gt.txt"
                img_dir = seq_dir / "img"
                if not gt_path.exists() or not img_dir.is_dir():
                    continue

                frame_ids = sorted(int(p.stem) for p in img_dir.glob("*.jpg"))
                if not frame_ids:
                    continue

                ann_cache = self._parse_gt(gt_path)
                if not ann_cache:
                    continue

                video_id = f"{cat}/{seq_dir.name}"
                assigned_split = _OFFICIAL_SPLIT.get(video_id, "no_split")

                video = VideoInfo(
                    video_id=video_id,
                    dataset="VISO",
                    category=cat,
                    split=assigned_split,
                    num_frames=len(frame_ids),
                    frame_ids=frame_ids,
                )

                if self.split == "no_split" or video.split == self.split:
                    self._ann_cache[video_id] = ann_cache
                    self.videos.append(video)

    def _load_frame(self, video: VideoInfo, frame_id: int) -> np.ndarray:
        cat, seq_num = video.video_id.split("/")
        path = self.root / "mot" / cat / seq_num / "img" / f"{frame_id:06d}.jpg"
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Frame not found: {path}")
        return img[..., ::-1].copy()  # BGR → RGB

    def _load_annotations(
        self, video: VideoInfo, frame_id: int
    ) -> dict[str, np.ndarray]:
        objs = self._ann_cache.get(video.video_id, {}).get(frame_id, [])

        if not objs:
            return {
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "labels": np.zeros(0, dtype=np.int64),
                "track_ids": np.zeros(0, dtype=np.int64),
            }

        boxes, labels, track_ids = [], [], []
        for obj in objs:
            lbl = self._map_label(video.category)
            if lbl < 0:
                continue
            boxes.append(obj["box"])
            labels.append(lbl)
            track_ids.append(obj["track_id"])

        if not boxes:
            return {
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "labels": np.zeros(0, dtype=np.int64),
                "track_ids": np.zeros(0, dtype=np.int64),
            }

        return {
            "boxes": np.array(boxes, dtype=np.float32),
            "labels": np.array(labels, dtype=np.int64),
            "track_ids": np.array(track_ids, dtype=np.int64),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_gt(gt_path: Path) -> dict[int, list[dict]]:
        """
        Parse MOT ``gt.txt`` → ``{frame_id: [obj dicts]}``.

        Auto-detects the annotation format:
            - Comma-delimited (car, train): ``frame,obj_id,x,y,w,h,...``  (xywh)
            - Space-delimited (plane, ship): ``frame obj_id x1 y1 x2 y2 ...``  (xyxy)
        """
        # Peek at first line to detect delimiter
        with open(gt_path) as f:
            first_line = f.readline().strip()
        is_comma = "," in first_line

        cache: dict[int, list[dict]] = defaultdict(list)
        with open(gt_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",") if is_comma else line.split()
                if len(parts) < 6:
                    continue

                fid = int(parts[0])
                obj_id = int(parts[1])

                if is_comma:
                    # xywh → xyxy
                    x, y = float(parts[2]), float(parts[3])
                    w, h = float(parts[4]), float(parts[5])
                    box = [x, y, x + w, y + h]
                else:
                    # Already xyxy
                    box = [float(parts[2]), float(parts[3]),
                           float(parts[4]), float(parts[5])]

                cache[fid].append({"box": box, "track_id": obj_id})

        return dict(cache)
