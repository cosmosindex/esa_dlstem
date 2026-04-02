"""
SV248SDataset
=============
Single-object tracking dataset (SV248S) from Jilin-1 satellite video.
248 targets across 6 videos, 4 classes: car, car-large, plane, ship.

Directory layout::

    <root>/
        01/
            sequences/
                000000/
                    000001.tiff
                    000002.tiff
                    ...
                000001/
                ...
            annotations/
                000000.abs    # JSON metadata (class, level, length, ...)
                000000.rect   # per-frame xywh (left_top_x, left_top_y, w, h)
                000000.poly   # per-frame tight polygon
                000000.state  # per-frame flag: 0=normal, 1=invisible, 2=occluded
                000000.attr   # sequence attributes (10 integers, csv)
                ...
        02/
        ...
        06/

Annotation notes:
- rect: comma-separated floats ``x,y,w,h`` (top-left corner + size), one line per frame.
- state: integer per line. 0=NOR (visible), 1=INV (invisible), 2=OCC (occluded).
  Frames with state != 0 still have rect annotations, but we return empty boxes for
  invisible frames (state=1) since the object has disappeared. Occluded frames (state=2)
  keep the annotation since the bbox is still meaningful.
- poly: comma-separated ``x1,y1,x2,y2,...`` polygon vertices per frame. Available for
  mask-level segmentation.
"""

import json
from collections import defaultdict

import cv2
import numpy as np
from pathlib import Path

from .base import BaseVideoDataset, VideoInfo

_SPLIT_SEED = 42


class SV248SDataset(BaseVideoDataset):
    """
    SV248S SOT dataset loader.

    Args:
        root:      Path to dataset root (contains video dirs 01–06).
        split:     "train", "val", "test", or "no_split".
        **kwargs:  Forwarded to BaseVideoDataset (mode, clip_len, transform, …).
    """

    def __init__(self, root: str | Path, split: str = "train", **kwargs):
        self._rect_cache: dict[str, np.ndarray] = {}
        self._state_cache: dict[str, np.ndarray] = {}
        self._poly_cache: dict[str, list[np.ndarray]] = {}
        super().__init__(root=root, split=split, **kwargs)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        all_videos: list[VideoInfo] = []

        for video_dir in sorted(self.root.iterdir()):
            if not video_dir.is_dir():
                continue
            seq_dir = video_dir / "sequences"
            ann_dir = video_dir / "annotations"
            if not seq_dir.exists() or not ann_dir.exists():
                continue

            for seq in sorted(seq_dir.iterdir()):
                if not seq.is_dir():
                    continue
                seq_id = seq.name
                abs_path = ann_dir / f"{seq_id}.abs"
                rect_path = ann_dir / f"{seq_id}.rect"
                state_path = ann_dir / f"{seq_id}.state"

                if not abs_path.exists() or not rect_path.exists():
                    continue

                # Parse metadata
                with open(abs_path) as f:
                    meta = json.load(f)
                category = meta["details"]["class_name"]

                # Parse rect annotations
                rects = self._parse_rect(rect_path)
                # Parse state flags
                states = self._parse_state(state_path) if state_path.exists() else np.zeros(len(rects), dtype=np.int32)

                # Count actual frames
                frames = sorted(seq.glob("*.tiff"))
                n = min(len(rects), len(states), len(frames))
                if n == 0:
                    continue

                vid_id = f"{video_dir.name}/{seq_id}"
                all_videos.append(VideoInfo(
                    video_id=vid_id,
                    dataset="SV248S",
                    category=category,
                    split="",
                    num_frames=n,
                    frame_ids=list(range(n)),
                ))
                self._rect_cache[vid_id] = rects[:n]
                self._state_cache[vid_id] = states[:n]

                # Parse polygon if available
                poly_path = ann_dir / f"{seq_id}.poly"
                if poly_path.exists():
                    self._poly_cache[vid_id] = self._parse_poly(poly_path, n)

        # Stratified split 80/10/10 per category
        split_map = self._stratified_split(all_videos)

        for v in all_videos:
            v.split = split_map[v.video_id]
            if self.split == "no_split" or v.split == self.split:
                self.videos.append(v)
            else:
                self._rect_cache.pop(v.video_id, None)
                self._state_cache.pop(v.video_id, None)
                self._poly_cache.pop(v.video_id, None)

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

    def _load_frame(self, video: VideoInfo, frame_id: int) -> np.ndarray:
        vid_dir, seq_id = video.video_id.split("/")
        path = self.root / vid_dir / "sequences" / seq_id / f"{frame_id + 1:06d}.tiff"
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Frame not found: {path}")
        # Handle grayscale or RGBA
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        # BGR → RGB
        return img[..., ::-1].copy()

    def _load_annotations(self, video: VideoInfo, frame_id: int) -> dict:
        xywh = self._rect_cache[video.video_id][frame_id]
        state = self._state_cache[video.video_id][frame_id]

        # State 1 = invisible (target disappeared) → return empty
        if state == 1:
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
            "track_ids": np.array([1], dtype=np.int64),
        }

    def load_mask(self, video: VideoInfo, frame_id: int) -> np.ndarray | None:
        """
        Load polygon mask for a frame.

        Returns binary mask (H, W) uint8 with 255 inside the polygon,
        or None if no polygon annotation is available.
        """
        polys = self._poly_cache.get(video.video_id)
        if polys is None or frame_id >= len(polys):
            return None

        poly = polys[frame_id]
        if len(poly) == 0:
            return None

        # Need image size to create mask
        img = self._load_frame(video, frame_id)
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = poly.reshape(-1, 1, 2).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
        return mask

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_rect(path: Path) -> np.ndarray:
        """Parse rect file → (N, 4) float32 array of (x, y, w, h)."""
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                values = line.split(",")
                rows.append([float(v) for v in values[:4]])
        return np.array(rows, dtype=np.float32)

    @staticmethod
    def _parse_state(path: Path) -> np.ndarray:
        """Parse state file → (N,) int32 array of frame flags."""
        states = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    states.append(int(line))
        return np.array(states, dtype=np.int32)

    @staticmethod
    def _parse_poly(path: Path, max_frames: int) -> list[np.ndarray]:
        """Parse poly file → list of (K, 2) float32 arrays of polygon vertices."""
        polys = []
        with open(path) as f:
            for i, line in enumerate(f):
                if i >= max_frames:
                    break
                line = line.strip()
                if not line:
                    polys.append(np.zeros((0, 2), dtype=np.float32))
                    continue
                values = [float(v) for v in line.split(",")]
                polys.append(np.array(values, dtype=np.float32).reshape(-1, 2))
        return polys
