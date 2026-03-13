"""
BIRDSAIDataset
==============
BIRDSAI: Bird's-eye view drone-based dataset for wildlife tracking.

This loader uses the **tracking splits** (SOT sub-sequences extracted from
MOT videos), not the raw MOT CSV annotations directly.

Two tracking split variants are supported:
    "full"    — includes frames where the target is occluded / out-of-view
                (GT bbox = 0,0,0,0 for those frames).  More realistic.
    "perfect" — only frames where the target is fully visible.
                Every frame has a valid GT bbox.  More sequences, shorter.

Directory layout::

    <root>/
        TrainReal/
            images/<video_id>/
                <video_id>_<frame_idx>.jpg
            annotations/
                tracking/
                    data_split_full/<seq_id>/
                        groundtruth_rect.txt   # x,y,w,h per line
                        img_list.txt           # sequential→actual filename mapping
                    data_split_perfect/<seq_id>/
                        groundtruth_rect.txt
                        img_list.txt
        TestReal/
            images/<video_id>/...
            annotations/
                tracking/
                    data_split_full_test/<seq_id>/...
                    data_split_perfect_test/<seq_id>/...

Sequence directory naming: ``<video_id>_<track_id>_<start>-<end>``

Split strategy:
    TrainReal → train
    TestReal  → val + test  (stratified 30/70 by source video)
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from .base import BaseVideoDataset, VideoInfo

_SPLIT_SEED = 42

# MOT CSV class field: 0 = animal, 1 = human
_CLASS_NAMES = {0: "animal", 1: "human"}


class BIRDSAIDataset(BaseVideoDataset):
    """
    BIRDSAI SOT dataset loader.

    Args:
        root:          Path to BIRDSAI root (contains TrainReal/ and TestReal/).
        split:         "train", "val", "test", or "no_split".
        tracking_split: "full" or "perfect".
        **kwargs:      Forwarded to BaseVideoDataset.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        tracking_split: Literal["full", "perfect"] = "perfect",
        **kwargs,
    ):
        self.tracking_split = tracking_split

        # Caches populated in _build_index
        self._gt_cache: dict[str, np.ndarray] = {}       # seq_id → (T, 4) xywh
        self._img_list_cache: dict[str, list[str]] = {}   # seq_id → list of actual filenames
        self._video_dir_cache: dict[str, Path] = {}       # seq_id → path to image dir
        self._seq_category: dict[str, str] = {}            # seq_id → "animal" or "human"

        super().__init__(root=root, split=split, **kwargs)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        all_videos: list[VideoInfo] = []
        source_map: dict[str, str] = {}  # seq_id → "train" or "test" (original split)

        # Discover sequences from TrainReal and TestReal
        for orig_split, subdir in [("train", "TrainReal"), ("test", "TestReal")]:
            split_dir = self.root / subdir
            if not split_dir.exists():
                continue

            tracking_dir = self._get_tracking_dir(split_dir)
            if tracking_dir is None or not tracking_dir.exists():
                continue

            images_root = split_dir / "images"

            for seq_dir in sorted(tracking_dir.iterdir()):
                if not seq_dir.is_dir():
                    continue

                gt_path = seq_dir / "groundtruth_rect.txt"
                img_list_path = seq_dir / "img_list.txt"

                if not gt_path.exists() or not img_list_path.exists():
                    continue

                seq_id = seq_dir.name
                gt = self._parse_gt(gt_path)
                img_list = self._parse_img_list(img_list_path)

                n = min(len(gt), len(img_list))
                if n == 0:
                    continue

                # Determine valid frames (non-zero GT boxes)
                valid_mask = ~((gt[:n, 0] == 0) & (gt[:n, 1] == 0) &
                               (gt[:n, 2] == 0) & (gt[:n, 3] == 0))
                valid_frames = [i for i in range(n) if valid_mask[i]]

                if len(valid_frames) == 0:
                    continue

                # Resolve video image directory from img_list
                # Actual filename format: <video_id>_<frame>.jpg
                video_id = self._extract_video_id(seq_id)
                img_dir = images_root / video_id

                if not img_dir.exists():
                    continue

                self._gt_cache[seq_id] = gt[:n]
                self._img_list_cache[seq_id] = img_list[:n]
                self._video_dir_cache[seq_id] = img_dir

                # Determine category from MOT CSV
                track_id = self._extract_track_id(seq_id)
                category = self._lookup_track_class(
                    split_dir / "annotations", video_id, track_id
                )
                self._seq_category[seq_id] = category

                all_videos.append(VideoInfo(
                    video_id=seq_id,
                    dataset="BIRDSAI",
                    category=category,
                    split="",  # assigned below
                    num_frames=len(valid_frames),
                    frame_ids=valid_frames,
                ))
                source_map[seq_id] = orig_split

        # --- Assign splits ---
        # TrainReal sequences → "train"
        # TestReal sequences → stratified split into "val" + "test" by source video
        train_seqs = [v for v in all_videos if source_map[v.video_id] == "train"]
        test_seqs = [v for v in all_videos if source_map[v.video_id] == "test"]

        for v in train_seqs:
            v.split = "train"

        # Split TestReal into val/test (30/70, stratified by source video)
        val_test_map = self._stratified_val_test_split(test_seqs)

        for v in test_seqs:
            v.split = val_test_map[v.video_id]

        # Keep only requested split
        for v in all_videos:
            if self.split == "no_split" or v.split == self.split:
                self.videos.append(v)
            else:
                self._gt_cache.pop(v.video_id, None)
                self._img_list_cache.pop(v.video_id, None)
                self._video_dir_cache.pop(v.video_id, None)
                self._seq_category.pop(v.video_id, None)

    def _load_frame(self, video: VideoInfo, frame_id: int) -> np.ndarray:
        seq_id = video.video_id
        actual_filename = self._img_list_cache[seq_id][frame_id]
        img_dir = self._video_dir_cache[seq_id]
        path = img_dir / actual_filename

        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Frame not found: {path}")

        img = img[..., ::-1].copy()  # BGR → RGB

        # BIRDSAI images are grayscale — ensure 3 channels
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)

        return img

    def _load_annotations(self, video: VideoInfo, frame_id: int) -> dict:
        seq_id = video.video_id
        x, y, w, h = self._gt_cache[seq_id][frame_id]

        # Convert xywh → xyxy
        box_xyxy = [float(x), float(y), float(x + w), float(y + h)]

        category = self._seq_category[seq_id]
        return {
            "boxes": np.array([box_xyxy], dtype=np.float32),
            "labels": np.array([self._map_label(category)], dtype=np.int64),
            "track_ids": np.array([1], dtype=np.int64),  # SOT — single object
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_tracking_dir(self, split_dir: Path) -> Path | None:
        """Resolve the tracking sub-directory based on tracking_split and split."""
        ann_tracking = split_dir / "annotations" / "tracking"

        is_test = split_dir.name == "TestReal"

        if self.tracking_split == "full":
            dirname = "data_split_full_test" if is_test else "data_split_full"
        else:
            dirname = "data_split_perfect_test" if is_test else "data_split_perfect"

        path = ann_tracking / dirname
        return path if path.exists() else None

    @staticmethod
    def _extract_video_id(seq_id: str) -> str:
        """Extract the source video ID from a sequence directory name.

        Sequence naming: ``<vid_part1>_<vid_part2>_<track_id>_<start>-<end>``
        Video ID: ``<vid_part1>_<vid_part2>``
        """
        parts = seq_id.split("_")
        # Video ID is the first two parts (both are 10-digit zero-padded numbers)
        return f"{parts[0]}_{parts[1]}"

    @staticmethod
    def _extract_track_id(seq_id: str) -> int:
        """Extract the track ID from a sequence directory name.

        Sequence naming: ``<vid_part1>_<vid_part2>_<track_id>_<start>-<end>``
        """
        parts = seq_id.split("_")
        return int(parts[2])

    @staticmethod
    def _lookup_track_class(
        annotations_dir: Path, video_id: str, track_id: int
    ) -> str:
        """Look up the class of a track from the MOT CSV annotation.

        Returns "animal" or "human".  Falls back to "animal" if the CSV
        is missing or the track_id is not found.
        """
        csv_path = annotations_dir / f"{video_id}.csv"
        if not csv_path.exists():
            return "animal"
        with open(csv_path) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 7:
                    continue
                if int(parts[1]) == track_id:
                    cls = int(parts[6])
                    return _CLASS_NAMES.get(cls, "animal")
        return "animal"

    @staticmethod
    def _parse_gt(path: Path) -> np.ndarray:
        """Parse groundtruth_rect.txt → (N, 4) float32 array of x, y, w, h."""
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                values = re.split(r"[,\t ]+", line)
                rows.append([float(v) for v in values[:4]])
        return np.array(rows, dtype=np.float32)

    @staticmethod
    def _parse_img_list(path: Path) -> list[str]:
        """Parse img_list.txt → list of actual image filenames.

        Format per line: ``0001.jpg:<actual_filename>.jpg``
        """
        filenames = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Format: "0001.jpg:0000000010_0000000000_0000000002.jpg"
                parts = line.split(":")
                filenames.append(parts[1] if len(parts) > 1 else parts[0])
        return filenames

    @staticmethod
    def _stratified_val_test_split(
        sequences: list[VideoInfo],
        val_ratio: float = 0.3,
    ) -> dict[str, str]:
        """Split TestReal sequences into val/test, stratified by source video.

        Groups sequences by their source video (first two parts of video_id).
        Assigns entire source videos to either val or test to prevent
        data leakage between splits.

        Returns dict mapping seq_id → "val" or "test".
        """
        rng = np.random.RandomState(_SPLIT_SEED)

        # Group by source video
        by_source: dict[str, list[str]] = defaultdict(list)
        for v in sequences:
            source = BIRDSAIDataset._extract_video_id(v.video_id)
            by_source[source].append(v.video_id)

        source_ids = sorted(by_source.keys())
        rng.shuffle(source_ids)

        n_val = max(1, round(len(source_ids) * val_ratio))
        val_sources = set(source_ids[:n_val])

        split_map: dict[str, str] = {}
        for source, seq_ids in by_source.items():
            split_name = "val" if source in val_sources else "test"
            for sid in seq_ids:
                split_map[sid] = split_name

        return split_map
