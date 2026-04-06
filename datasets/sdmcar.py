"""
SDMCarDataset
=============
Multi-object tracking dataset for small and dim moving vehicles in satellite
video, captured by Luojia-3-01 satellite at 0.75 m resolution.

Directory layout::

    <root>/
        train/
            1-1.avi
            1-1-gt.csv
            ...
        validation/
            1-2.avi
            1-2-gt.csv
            ...
        test/
            10-2.avi
            10-2-gt.csv
            ...

Each ``-gt.csv`` is a headerless CSV with 10 columns::

    frame_id, target_id, bbox_x, bbox_y, bbox_w, bbox_h, -1, -1, -1, -1

Coordinates are **top-left xywh** absolute pixels, 0-indexed frames.
All objects are vehicles (single class ``"car"``).

Official split: **train (64) / validation (15) / test (20)**.
"""

from pathlib import Path

import cv2
import numpy as np

from .base import BaseVideoDataset, VideoInfo


class SDMCarDataset(BaseVideoDataset):
    """
    SDM-Car dataset loader.

    Args:
        root:      Path to dataset root (contains train/, validation/, test/).
        split:     "train", "val", "test", or "no_split".
        **kwargs:  Forwarded to BaseVideoDataset (mode, clip_len, transform, …).
    """

    # Map directory names to our canonical split names
    _SPLIT_DIRS = {
        "train": "train",
        "val": "validation",
        "test": "test",
    }

    def __init__(self, root: str | Path, split: str = "train", **kwargs):
        # Annotation cache: video_id → {frame_id → list of (track_id, x1, y1, x2, y2)}
        self._ann_cache: dict[str, dict[int, list[tuple[int, float, float, float, float]]]] = {}
        super().__init__(root=root, split=split, **kwargs)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        splits_to_load = (
            list(self._SPLIT_DIRS.keys())
            if self.split == "no_split"
            else [self.split]
        )

        for split_name in splits_to_load:
            split_dir = self.root / self._SPLIT_DIRS[split_name]
            if not split_dir.is_dir():
                continue

            for avi_path in sorted(split_dir.glob("*.avi")):
                seq_name = avi_path.stem  # e.g. "1-1"
                gt_path = split_dir / f"{seq_name}-gt.csv"
                if not gt_path.exists():
                    continue

                # Parse annotations
                ann_by_frame = self._parse_gt(gt_path)
                if not ann_by_frame:
                    continue

                # Get frame count from video
                cap = cv2.VideoCapture(str(avi_path))
                num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                if num_frames <= 0:
                    continue

                video_id = f"{split_name}/{seq_name}"
                self._ann_cache[video_id] = ann_by_frame

                self.videos.append(VideoInfo(
                    video_id=video_id,
                    dataset="SDM-Car",
                    category="car",
                    split=split_name,
                    num_frames=num_frames,
                    frame_ids=list(range(num_frames)),
                ))

    def _load_frame(self, video: VideoInfo, frame_id: int) -> np.ndarray:
        split_dir_name = self._SPLIT_DIRS[video.split]
        seq_name = video.video_id.split("/", 1)[1]
        avi_path = self.root / split_dir_name / f"{seq_name}.avi"

        cap = cv2.VideoCapture(str(avi_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            raise FileNotFoundError(
                f"Cannot read frame {frame_id} from {avi_path}"
            )
        return frame[..., ::-1].copy()  # BGR → RGB

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
        lbl = self._map_label("car")
        if lbl < 0:
            return {
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "labels": np.zeros(0, dtype=np.int64),
                "track_ids": np.zeros(0, dtype=np.int64),
            }

        for tid, x1, y1, x2, y2 in objs:
            boxes.append([x1, y1, x2, y2])
            labels.append(lbl)
            track_ids.append(tid)

        return {
            "boxes": np.array(boxes, dtype=np.float32),
            "labels": np.array(labels, dtype=np.int64),
            "track_ids": np.array(track_ids, dtype=np.int64),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_gt(
        path: Path,
    ) -> dict[int, list[tuple[int, float, float, float, float]]]:
        """
        Parse a gt.csv annotation file.

        Returns:
            Dict mapping frame_id → list of (track_id, x1, y1, x2, y2) in xyxy.
        """
        ann: dict[int, list[tuple[int, float, float, float, float]]] = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                frame_id = int(parts[0])
                track_id = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                # Convert xywh (top-left) → xyxy
                x1, y1, x2, y2 = x, y, x + w, y + h
                ann.setdefault(frame_id, []).append(
                    (track_id, x1, y1, x2, y2)
                )
        return ann
