"""
OOTBDataset
===========
Single-object tracking dataset with Oriented Bounding Box (OBB) ground truth.

Directory layout::

    <root>/
        <seq_name>/            e.g. car_1/, drone_3/
            img/
                0001.jpg
                0002.jpg
                ...
            groundtruth.txt    # one line per frame
                               # 8 floats: x1,y1,x2,y2,x3,y3,x4,y4  (OBB corners)
"""

import re
from collections import defaultdict

import cv2
import numpy as np
from pathlib import Path

from .base import BaseVideoDataset, VideoInfo

# Fixed seed for reproducible stratified splits
_SPLIT_SEED = 42


class OOTBDataset(BaseVideoDataset):
    """
    OOTB dataset loader.

    Args:
        root:      Path to dataset root (one sub-directory per sequence).
        split:     "train", "val", "test", or "no_split" (returns all videos).
        **kwargs:  Forwarded to BaseVideoDataset (mode, clip_len, transform, …).
    """

    def __init__(self, root: str | Path, split: str = "train", **kwargs):
        # Must be initialized before super().__init__() calls _build_index()
        self._gt_cache: dict[str, np.ndarray] = {}
        super().__init__(root=root, split=split, **kwargs)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        # --- Step 1: discover all sequences ---
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

            # Guard against annotation / image count mismatch
            n = min(len(gt), len(frames))
            if n == 0:
                continue

            category = re.sub(r"_\d+$", "", seq_dir.name)  # "car_1" → "car"
            all_videos.append(VideoInfo(
                video_id   = seq_dir.name,
                dataset    = "OOTB",
                category   = category,
                split      = "",  # assigned below
                num_frames = n,
                frame_ids  = list(range(n)),
            ))
            self._gt_cache[seq_dir.name] = gt[:n]

        # --- Step 2: stratified split (80/10/10) per category ---
        split_map = self._stratified_split(all_videos)

        # --- Step 3: keep only the requested split ---
        for v in all_videos:
            v.split = split_map[v.video_id]
            if self.split == "no_split" or v.split == self.split:
                self.videos.append(v)
            else:
                # Free gt cache for videos not in this split
                self._gt_cache.pop(v.video_id, None)

    @staticmethod
    def _stratified_split(
        videos: list[VideoInfo],
        ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    ) -> dict[str, str]:
        """
        Assign each video to train / val / test, stratified by category.

        Returns:
            Dict mapping video_id → split name.
        """
        rng = np.random.RandomState(_SPLIT_SEED)

        # Group video ids by category
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
            n_val   = max(1, round(n * val_r))
            # Ensure test gets at least 1
            if n_train + n_val >= n:
                n_train = n - 2
                n_val   = 1
            for vid in ids[:n_train]:
                split_map[vid] = "train"
            for vid in ids[n_train:n_train + n_val]:
                split_map[vid] = "val"
            for vid in ids[n_train + n_val:]:
                split_map[vid] = "test"

        return split_map

    def _load_frame(self, video: VideoInfo, frame_id: int) -> np.ndarray:
        path = self.root / video.video_id / "img" / f"{frame_id + 1:04d}.jpg"
        img  = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Frame not found: {path}")
        return img[..., ::-1].copy()  # BGR → RGB, make contiguous

    def _load_annotations(self, video: VideoInfo, frame_id: int) -> dict:
        coords   = self._gt_cache[video.video_id][frame_id]  # shape (8,) float32
        box_xyxy = self._obb_to_aabb(*coords)                # unpack 8 values
        return {
            "boxes":     np.array([box_xyxy], dtype=np.float32),
            "obb":       np.array([coords],   dtype=np.float32),  # (1, 8) raw OBB corners
            "labels":    np.array([self._map_label(video.category)], dtype=np.int64),
            "track_ids": np.array([1], dtype=np.int64),  # SOT — single object, ID=1
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_gt(path: Path) -> np.ndarray:
        """
        Parse groundtruth.txt → ndarray shape (N, 8) float32.

        Accepts comma, tab, or space delimiters; skips blank lines.
        """
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                values = re.split(r"[,\t ]+", line)
                rows.append([float(v) for v in values[:8]])
        return np.array(rows, dtype=np.float32)
