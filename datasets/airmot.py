"""
AIRMOTDataset
=============
Aerial / satellite Multi-Object Tracking dataset (AIR-MOT-100).

Directory layout::

    <root>/
        1/img/000001.jpg, 000002.jpg, …
         /gt/gt.txt
        2/…
        …
        100/img/000001_8.jpg, …    # special naming
           /gt/gt.txt

Annotation format (MOT CSV, 9 columns per line)::

    frame_id, track_id, x, y, w, h, conf, class, visibility

- ``x, y`` = top-left corner.
- ``class``: 1 = airplane, 2 = ship.
- 31 out of 100 sequences have empty annotations and are excluded.
- Some sequences have black padding bars (bottom and/or right); all
  annotations are within the valid content region.

No official split.  We create **80 / 10 / 10** stratified by dominant
class per sequence (``seed=42``).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from .base import BaseVideoDataset, VideoInfo

_SPLIT_SEED = 42

# Raw MOT class id → category name
_CLASS_MAP: dict[str, str] = {"1": "airplane", "2": "ship"}


class AIRMOTDataset(BaseVideoDataset):
    """
    AIR-MOT-100 multi-object tracking dataset.

    Args:
        root:   Path containing the numbered sequence directories (``1/`` … ``100/``).
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
        # --- Step 1: discover sequences with non-empty annotations ---
        seq_infos: list[dict] = []
        for seq_dir in sorted(self.root.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 0):
            if not seq_dir.is_dir() or not seq_dir.name.isdigit():
                continue
            gt_path = seq_dir / "gt" / "gt.txt"
            if not gt_path.exists() or gt_path.stat().st_size == 0:
                continue

            img_dir = seq_dir / "img"
            if not img_dir.is_dir():
                continue

            # Discover frames
            frame_ids = sorted(
                int(p.stem.split("_")[0])
                for p in img_dir.glob("*.jpg")
            )
            if not frame_ids:
                continue

            # Detect image naming pattern (seq 100 uses 000001_8.jpg)
            sample_img = next(img_dir.glob("*.jpg"))
            suffix = sample_img.stem.split("_", 1)[1] if "_" in sample_img.stem else ""

            # Parse annotations
            ann_cache, class_counts = self._parse_mot(gt_path)
            if not ann_cache:
                continue

            # Determine dominant class → category
            dominant_cls = class_counts.most_common(1)[0][0]
            category = _CLASS_MAP.get(dominant_cls, f"class_{dominant_cls}")

            seq_infos.append({
                "video_id": seq_dir.name,
                "category": category,
                "frame_ids": frame_ids,
                "ann_cache": ann_cache,
                "img_suffix": suffix,
            })

        # --- Step 2: stratified split 80/10/10 by category ---
        split_map = self._stratified_split(seq_infos)

        # --- Step 3: build VideoInfo entries ---
        for info in seq_infos:
            vid = info["video_id"]
            assigned_split = split_map.get(vid, "no_split")

            video = VideoInfo(
                video_id=vid,
                dataset="AIR-MOT",
                category=info["category"],
                split=assigned_split,
                num_frames=len(info["frame_ids"]),
                frame_ids=info["frame_ids"],
            )

            if self.split == "no_split" or video.split == self.split:
                self._ann_cache[vid] = info["ann_cache"]
                # Store the image suffix for _load_frame
                video._img_suffix = info["img_suffix"]  # type: ignore[attr-defined]
                self.videos.append(video)

    def _load_frame(self, video: VideoInfo, frame_id: int) -> np.ndarray:
        suffix = getattr(video, "_img_suffix", "")
        fname = f"{frame_id:06d}_{suffix}.jpg" if suffix else f"{frame_id:06d}.jpg"
        path = self.root / video.video_id / "img" / fname
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
            lbl = self._map_label(obj["class"])
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
    def _parse_mot(
        gt_path: Path,
    ) -> tuple[dict[int, list[dict]], Counter]:
        """Parse MOT CSV → ``{frame_id: [obj_dicts]}``, class counts."""
        cache: dict[int, list[dict]] = defaultdict(list)
        class_counts: Counter = Counter()

        with open(gt_path) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 9:
                    continue
                fid = int(parts[0])
                track_id = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                cls_raw = parts[7].strip()

                cat_name = _CLASS_MAP.get(cls_raw, f"class_{cls_raw}")
                class_counts[cls_raw] += 1

                cache[fid].append({
                    "box": [x, y, x + w, y + h],
                    "class": cat_name,
                    "track_id": track_id,
                })

        return dict(cache), class_counts

    @staticmethod
    def _stratified_split(
        seq_infos: list[dict],
    ) -> dict[str, str]:
        """80 / 10 / 10 stratified by category, ``seed=42``."""
        rng = np.random.RandomState(_SPLIT_SEED)

        by_cat: dict[str, list[str]] = defaultdict(list)
        for info in seq_infos:
            by_cat[info["category"]].append(info["video_id"])

        split_map: dict[str, str] = {}
        for cat in sorted(by_cat):
            vids = sorted(by_cat[cat])
            rng.shuffle(vids)
            n = len(vids)
            n_val = max(1, round(n * 0.1))
            n_test = max(1, round(n * 0.1))
            n_train = n - n_val - n_test
            for vid in vids[:n_train]:
                split_map[vid] = "train"
            for vid in vids[n_train : n_train + n_val]:
                split_map[vid] = "val"
            for vid in vids[n_train + n_val :]:
                split_map[vid] = "test"

        return split_map
