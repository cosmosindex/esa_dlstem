"""
SatSOTDataset
=============
Single-object tracking dataset from Jilin-1 satellite video.

Directory layout::

    <root>/
        <seq_name>/            e.g. car_01/, plane_03/
            img/
                0001.jpg
                0002.jpg
                ...
            groundtruth.txt    # one line per frame
                               # 4 floats: x, y, w, h  (top-left corner + size)
        SatSOT.json            # metadata with attributes (optional)

Categories: car, plane, ship, train.
"""

import json
import re
from collections import defaultdict

import cv2
import numpy as np
from pathlib import Path

from .base import BaseVideoDataset, VideoInfo

_SPLIT_SEED = 42

# Sequence attributes present in SatSOT.json (multi-label, per sequence).
# Order is nominal — attrs are stored as a list of strings in the metadata,
# so presence (not position) is what matters.
ATTR_NAMES = (
    "ARC", "BC", "BJT", "DEF", "FOC",
    "IV", "LQ", "POC", "ROT", "SOB", "TO",
)

_META_FILENAME = "SatSOT.json"


class SatSOTDataset(BaseVideoDataset):
    """
    SatSOT dataset loader.

    Args:
        root:      Path to dataset root (one sub-directory per sequence).
        split:     "train", "val", "test", or "no_split" (returns all videos).
        **kwargs:  Forwarded to BaseVideoDataset (mode, clip_len, transform, …).
    """

    def __init__(self, root: str | Path, split: str = "train", **kwargs):
        self._gt_cache: dict[str, np.ndarray] = {}
        self._attr_cache: dict[str, list[str]] = {}  # vid_id → list of attr names
        super().__init__(root=root, split=split, **kwargs)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        # Load per-sequence attribute metadata once (SatSOT.json) if present.
        meta_path = self.root / _META_FILENAME
        meta_attrs: dict[str, list[str]] = {}
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                meta_attrs = {
                    k: list(v.get("attr", [])) for k, v in meta.items()
                }
            except (json.JSONDecodeError, OSError):
                meta_attrs = {}

        all_videos: list[VideoInfo] = []
        for seq_dir in sorted(self.root.iterdir()):
            if not seq_dir.is_dir():
                continue
            gt_path = seq_dir / "groundtruth.txt"
            img_dir = seq_dir / "img"
            if not gt_path.exists() or not img_dir.exists():
                continue

            frames = sorted(img_dir.glob("*.jpg"))
            if not frames:
                continue

            gt = self._parse_gt(gt_path)
            n = min(len(gt), len(frames))
            if n == 0:
                continue

            category = re.sub(r"_\d+$", "", seq_dir.name)  # "car_01" → "car"
            all_videos.append(VideoInfo(
                video_id=seq_dir.name,
                dataset="SatSOT",
                category=category,
                split="",
                num_frames=n,
                frame_ids=list(range(n)),
            ))
            self._gt_cache[seq_dir.name] = gt[:n]
            if seq_dir.name in meta_attrs:
                self._attr_cache[seq_dir.name] = meta_attrs[seq_dir.name]

        # Stratified split 80/10/10 per category
        split_map = self._stratified_split(all_videos)

        for v in all_videos:
            v.split = split_map[v.video_id]
            if self.split == "no_split" or v.split == self.split:
                self.videos.append(v)
            else:
                self._gt_cache.pop(v.video_id, None)
                self._attr_cache.pop(v.video_id, None)

    @staticmethod
    def _stratified_split(
        videos: list[VideoInfo],
        ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    ) -> dict[str, str]:
        rng = np.random.RandomState(_SPLIT_SEED)

        by_cat: dict[str, list[str]] = defaultdict(list)
        for v in videos:
            by_cat[v.category].append(v.video_id)

        split_map: dict[str, str] = {}
        train_r, val_r, _ = ratios

        for cat in sorted(by_cat):
            ids = by_cat[cat]
            rng.shuffle(ids)
            n = len(ids)
            n_train = max(1, round(n * train_r))
            n_val = max(1, round(n * val_r))
            if n_train + n_val >= n:
                n_train = n - 2
                n_val = 1
            for vid in ids[:n_train]:
                split_map[vid] = "train"
            for vid in ids[n_train:n_train + n_val]:
                split_map[vid] = "val"
            for vid in ids[n_train + n_val:]:
                split_map[vid] = "test"

        return split_map

    def sequence_attributes(self) -> dict[str, list[str]]:
        """Return {video_id: [attr_name, ...]} for videos in this split.

        Attributes come from ``SatSOT.json`` (per-sequence ``attr`` list).
        Sequences missing from the metadata file yield an empty list.
        """
        return {v.video_id: list(self._attr_cache.get(v.video_id, [])) for v in self.videos}

    def _load_frame(self, video: VideoInfo, frame_id: int) -> np.ndarray:
        path = self.root / video.video_id / "img" / f"{frame_id + 1:04d}.jpg"
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Frame not found: {path}")
        return img[..., ::-1].copy()  # BGR → RGB

    def _load_annotations(self, video: VideoInfo, frame_id: int) -> dict:
        xywh = self._gt_cache[video.video_id][frame_id]  # (4,) float32
        # NaN means target absent/occluded in this frame
        if np.any(np.isnan(xywh)):
            return {
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "labels": np.zeros(0, dtype=np.int64),
                "track_ids": np.zeros(0, dtype=np.int64),
            }
        x, y, w, h = xywh
        box_xyxy = [x, y, x + w, y + h]
        return {
            "boxes": np.array([box_xyxy], dtype=np.float32),
            "labels": np.array([self._map_label(video.category)], dtype=np.int64),
            "track_ids": np.array([1], dtype=np.int64),  # SOT — single object
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_gt(path: Path) -> np.ndarray:
        """
        Parse groundtruth.txt → ndarray shape (N, 4) float32 (x, y, w, h).

        Lines containing "none" (target absent/occluded) are stored as NaN.
        """
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if "none" in line.lower():
                    rows.append([float("nan")] * 4)
                else:
                    values = re.split(r"[,\t ]+", line)
                    rows.append([float(v) for v in values[:4]])
        return np.array(rows, dtype=np.float32)
