"""
BIRDSAIMOTDataset
=================
BIRDSAI MOT (Multi-Object Tracking) dataset loader.

Uses the raw MOT CSV annotations (one CSV per video) rather than the
SOT tracking splits.  Each video can contain multiple simultaneously
tracked objects.

CSV format::

    <frame>, <object_id>, <x>, <y>, <w>, <h>, <class>, <species>, <occlusion>, <noise>

    class:     0 = animal, 1 = human
    species:   -1 unknown, 0 human, 1 elephant, 2 lion, 3 giraffe, 4 dog,
               5 crocodile, 6 hippo, 7 zebra, 8 rhino
    occlusion: 0 = none, 1 = occluded (IoU >= 0.3)
    noise:     0 = none, 1 = noise

Both animal and human annotations are loaded.

Directory layout::

    <root>/
        TrainReal/
            images/<video_id>/
                <video_id>_<frame_idx:010d>.jpg
            annotations/
                <video_id>.csv
        TestReal/
            images/<video_id>/
                <video_id>_<frame_idx:010d>.jpg
            annotations/
                <video_id>.csv

Split strategy:
    TrainReal  →  train
    TestReal   →  val (30%) + test (70%), stratified by video, seed=42
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .base import BaseVideoDataset, VideoInfo

_SPLIT_SEED = 42

# Map CSV class column to category name
_CLASS_NAMES = {0: "animal", 1: "human"}


class BIRDSAIMOTDataset(BaseVideoDataset):
    """
    BIRDSAI MOT dataset loader.

    Args:
        root:      Path to BIRDSAI root (contains TrainReal/ and TestReal/).
        split:     "train", "val", "test", or "no_split".
        **kwargs:  Forwarded to BaseVideoDataset.
    """

    def __init__(self, root: str | Path, split: str = "train", **kwargs):
        # Caches populated in _build_index
        # video_id → { frame_id → list of (track_id, class_name, x1, y1, x2, y2) }
        self._ann_cache: dict[str, dict[int, list[tuple[int, str, float, float, float, float]]]] = {}
        self._img_dir_cache: dict[str, Path] = {}

        super().__init__(root=root, split=split, **kwargs)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        all_videos: list[VideoInfo] = []
        source_map: dict[str, str] = {}  # video_id → "train" or "test"

        for orig_split, subdir in [("train", "TrainReal"), ("test", "TestReal")]:
            split_dir = self.root / subdir
            if not split_dir.exists():
                continue

            ann_dir = split_dir / "annotations"
            images_root = split_dir / "images"

            for csv_path in sorted(ann_dir.glob("*.csv")):
                if csv_path.name.startswith("._"):
                    continue

                video_id = csv_path.stem
                img_dir = images_root / video_id
                if not img_dir.exists():
                    continue

                # Parse CSV and build per-frame annotation dict
                frame_anns = self._parse_csv(csv_path)
                if not frame_anns:
                    continue

                # frame_ids = sorted annotated frames that have images
                frame_ids = [
                    fid for fid in sorted(frame_anns.keys())
                    if (img_dir / f"{video_id}_{fid:010d}.jpg").exists()
                ]
                if not frame_ids:
                    continue

                self._ann_cache[video_id] = frame_anns
                self._img_dir_cache[video_id] = img_dir

                all_videos.append(VideoInfo(
                    video_id=video_id,
                    dataset="BIRDSAI_MOT",
                    category="mixed",  # videos can contain both animal and human
                    split="",  # assigned below
                    num_frames=len(frame_ids),
                    frame_ids=frame_ids,
                ))
                source_map[video_id] = orig_split

        # --- Assign splits ---
        # TrainReal → train
        # TestReal  → val (30%) + test (70%), stratified by video
        train_vids = [v for v in all_videos if source_map[v.video_id] == "train"]
        test_vids = [v for v in all_videos if source_map[v.video_id] == "test"]

        for v in train_vids:
            v.split = "train"

        val_test_map = self._stratified_val_test_split(test_vids)
        for v in test_vids:
            v.split = val_test_map[v.video_id]

        # Keep only requested split
        for v in all_videos:
            if self.split == "no_split" or v.split == self.split:
                self.videos.append(v)
            else:
                self._ann_cache.pop(v.video_id, None)
                self._img_dir_cache.pop(v.video_id, None)

    def _load_frame(self, video: VideoInfo, frame_id: int) -> np.ndarray:
        img_dir = self._img_dir_cache[video.video_id]
        path = img_dir / f"{video.video_id}_{frame_id:010d}.jpg"

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
        anns = self._ann_cache[video.video_id].get(frame_id, [])

        if not anns:
            return {
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "labels": np.array([], dtype=np.int64),
                "track_ids": np.array([], dtype=np.int64),
            }

        boxes = []
        labels = []
        track_ids = []
        for track_id, class_name, x1, y1, x2, y2 in anns:
            label = self._map_label(class_name)
            boxes.append([x1, y1, x2, y2])
            labels.append(label)
            track_ids.append(track_id)

        return {
            "boxes": np.array(boxes, dtype=np.float32),
            "labels": np.array(labels, dtype=np.int64),
            "track_ids": np.array(track_ids, dtype=np.int64),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_csv(
        path: Path,
    ) -> dict[int, list[tuple[int, str, float, float, float, float]]]:
        """Parse a MOT CSV file.

        CSV columns: frame, object_id, x, y, w, h, class, species, occlusion, noise
        class: 0 = animal, 1 = human.

        Returns dict mapping frame_id → list of (track_id, class_name, x1, y1, x2, y2)
        in xyxy format.
        """
        frame_anns: dict[int, list[tuple[int, str, float, float, float, float]]] = defaultdict(list)

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 7:
                    continue

                frame_id = int(parts[0])
                track_id = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                cls = int(parts[6])

                class_name = _CLASS_NAMES.get(cls)
                if class_name is None:
                    continue

                # Convert xywh → xyxy
                frame_anns[frame_id].append((track_id, class_name, x, y, x + w, y + h))

        return dict(frame_anns)

    @staticmethod
    def _stratified_val_test_split(
        videos: list[VideoInfo],
        val_ratio: float = 0.3,
    ) -> dict[str, str]:
        """Split TestReal videos into val (30%) / test (70%)."""
        rng = np.random.RandomState(_SPLIT_SEED)

        video_ids = [v.video_id for v in videos]
        rng.shuffle(video_ids)

        n_val = max(1, round(len(video_ids) * val_ratio))
        val_set = set(video_ids[:n_val])

        return {vid: ("val" if vid in val_set else "test") for vid in video_ids}
