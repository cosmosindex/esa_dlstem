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
import zlib
from collections import defaultdict

import cv2
import numpy as np
from pathlib import Path

from .base import BaseVideoDataset, VideoInfo

_SPLIT_SEED = 42
_RATIOS = (0.8, 0.1, 0.1)  # train, val, test

# Order of the 10 integers in `<seq_id>.attr` (SV248S paper, Table 5).
ATTR_NAMES = ("STO", "LTO", "DS", "IV", "BCH", "SM", "ND", "CO", "BCL", "IPR")

# Categories with ≤ this many sequences are pre-assigned round-robin to
# guarantee every split gets at least one instance.
_SMALL_CAT_THRESH = 10


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
        self._attr_cache: dict[str, np.ndarray] = {}  # vid_id → (10,) int array
        self._frame_paths: dict[str, list[Path]] = {}  # vid_id → sorted frame paths
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

                # Count actual frames (may not start from 000001.tiff)
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
                self._frame_paths[vid_id] = frames[:n]
                self._rect_cache[vid_id] = rects[:n]
                self._state_cache[vid_id] = states[:n]

                # Parse polygon if available
                poly_path = ann_dir / f"{seq_id}.poly"
                if poly_path.exists():
                    self._poly_cache[vid_id] = self._parse_poly(poly_path, n)

                # Parse sequence attributes (10 binary flags) if available
                attr_path = ann_dir / f"{seq_id}.attr"
                if attr_path.exists():
                    attrs = self._parse_attr(attr_path)
                    if attrs is not None:
                        self._attr_cache[vid_id] = attrs

        # Hybrid split: pre-assign tiny classes round-robin, then iterative
        # stratification on class ⊕ 10 attributes (keeps every attribute and
        # every class represented in train/val/test).
        split_map = self._hybrid_split(all_videos)

        for v in all_videos:
            v.split = split_map[v.video_id]
            if self.split == "no_split" or v.split == self.split:
                self.videos.append(v)
            else:
                self._frame_paths.pop(v.video_id, None)
                self._rect_cache.pop(v.video_id, None)
                self._state_cache.pop(v.video_id, None)
                self._poly_cache.pop(v.video_id, None)
                self._attr_cache.pop(v.video_id, None)

    def _hybrid_split(self, videos: list[VideoInfo]) -> dict[str, str]:
        """
        Hybrid 80/10/10 split, balanced on **both** category and sequence
        attributes (STO, LTO, DS, IV, BCH, SM, ND, CO, BCL, IPR).

        Step 1: tiny categories (n ≤ _SMALL_CAT_THRESH, i.e. plane & ship)
                are pre-assigned round-robin as test → val → train so that
                every split has at least one sequence of every class.
        Step 2: iterative stratification (Sechidis et al. 2011) assigns the
                remaining sequences using a label matrix that concatenates
                the class one-hot with the 10 binary attribute flags.

        Falls back to sequences whose `.attr` file is missing by using
        zeros for their attribute row — they behave as unlabelled w.r.t.
        attribute balance but still participate in class balance.
        """
        N = len(videos)
        cats = sorted({v.category for v in videos})
        cat_to_idx = {c: i for i, c in enumerate(cats)}

        # Build label matrix (N, |cats| + 10): one-hot class + 10 attrs.
        labels = np.zeros((N, len(cats) + 10), dtype=np.int32)
        for i, v in enumerate(videos):
            labels[i, cat_to_idx[v.category]] = 1
            attrs = self._attr_cache.get(v.video_id)
            if attrs is not None:
                labels[i, len(cats):] = attrs

        # Step 1 — tiny classes pre-assignment.
        preassigned: dict[int, int] = {}
        by_cat_idx: dict[str, list[int]] = defaultdict(list)
        for i, v in enumerate(videos):
            by_cat_idx[v.category].append(i)

        test_first_cycle = [2, 1, 0]  # test, val, train
        for cat, idxs in sorted(by_cat_idx.items()):
            if len(idxs) > _SMALL_CAT_THRESH:
                continue
            ids = idxs[:]
            rng2 = np.random.RandomState(_SPLIT_SEED + (zlib.crc32(cat.encode()) % 10_000))
            rng2.shuffle(ids)
            for k, idx in enumerate(ids):
                preassigned[idx] = test_first_cycle[k % 3]

        # Step 2 — iterative stratification on everyone else.
        split_idx = self._iterative_stratify(labels, preassigned)

        name = {0: "train", 1: "val", 2: "test"}
        return {videos[i].video_id: name[int(split_idx[i])] for i in range(N)}

    @staticmethod
    def _iterative_stratify(
        label_matrix: np.ndarray,
        preassigned: dict[int, int] | None = None,
    ) -> np.ndarray:
        """
        Multi-label iterative stratification (Sechidis et al. 2011).

        At each step, pick the label with the fewest remaining positives and
        place samples carrying it into the split whose deficit for that label
        is largest (tie-break: largest overall deficit, then random).
        Pre-assigned samples are frozen before the loop starts.
        """
        rng = np.random.RandomState(_SPLIT_SEED)
        N, _ = label_matrix.shape
        ratios = np.asarray(_RATIOS)
        targets = ratios * N
        per_label_targets = label_matrix.sum(axis=0)[:, None] * ratios

        split_idx = np.full(N, -1, dtype=np.int32)
        remaining = np.ones(N, dtype=bool)
        split_counts = np.zeros(3, dtype=np.float64)
        label_counts = np.zeros_like(per_label_targets)

        if preassigned:
            for i, s in preassigned.items():
                split_idx[i] = s
                split_counts[s] += 1
                for l in np.where(label_matrix[i] > 0)[0]:
                    label_counts[l, s] += 1
                remaining[i] = False

        remaining_label_counts = label_matrix[remaining].sum(axis=0).astype(np.float64)

        while remaining.any():
            active = remaining_label_counts > 0
            if not active.any():
                # Leftover samples with no active label: balance totals.
                deficit = targets - split_counts
                for idx in np.where(remaining)[0]:
                    s = int(np.argmax(deficit))
                    split_idx[idx] = s
                    split_counts[s] += 1
                    deficit = targets - split_counts
                    remaining[idx] = False
                break

            rare = np.where(active, remaining_label_counts, np.inf)
            label = int(np.argmin(rare))
            candidates = np.where(remaining & (label_matrix[:, label] > 0))[0]
            if candidates.size == 0:
                remaining_label_counts[label] = 0
                continue

            for idx in candidates:
                label_deficit = per_label_targets[label] - label_counts[label]
                best = np.where(label_deficit == label_deficit.max())[0]
                if best.size > 1:
                    total_deficit = targets - split_counts
                    td = total_deficit[best]
                    best = best[np.where(td == td.max())[0]]
                s = int(best[rng.randint(best.size)])

                split_idx[idx] = s
                split_counts[s] += 1
                for l in np.where(label_matrix[idx] > 0)[0]:
                    label_counts[l, s] += 1
                    remaining_label_counts[l] -= 1
                remaining[idx] = False

        return split_idx

    def sequence_attributes(self) -> dict[str, list[str]]:
        """Return {video_id: [attr_name, ...]} for videos in this split.

        Reads the already-loaded ``_attr_cache`` and decodes the 10 SV248S
        flags (``ATTR_NAMES``) into the list of attributes whose flag is 1.
        Sequences without a .attr file yield an empty list.
        """
        out: dict[str, list[str]] = {}
        for v in self.videos:
            flags = self._attr_cache.get(v.video_id)
            if flags is None:
                out[v.video_id] = []
            else:
                out[v.video_id] = [ATTR_NAMES[i] for i, f in enumerate(flags) if int(f) > 0]
        return out

    def _load_frame(self, video: VideoInfo, frame_id: int) -> np.ndarray:
        path = self._frame_paths[video.video_id][frame_id]
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
    def _parse_attr(path: Path) -> np.ndarray | None:
        """Parse `.attr` file → (10,) int32 array (STO, LTO, DS, IV, BCH, SM, ND, CO, BCL, IPR)."""
        text = path.read_text().strip()
        if not text:
            return None
        values = [int(v) for v in text.split(",") if v.strip() != ""]
        if len(values) != 10:
            return None
        return np.array(values, dtype=np.int32)

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
