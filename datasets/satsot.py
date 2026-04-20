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
import zlib
from collections import defaultdict

import cv2
import numpy as np
from pathlib import Path

from .base import BaseVideoDataset, VideoInfo

_SPLIT_SEED = 42
_RATIOS = (0.8, 0.1, 0.1)  # train, val, test

# Canonical 11-attribute ordering (SatSOT paper). Attributes come from
# ``SatSOT.json`` as per-sequence lists of strings; we encode them into a
# fixed-length binary vector so multi-label stratification can operate on them.
ATTR_NAMES = (
    "ARC", "BC", "BJT", "DEF", "FOC",
    "IV", "LQ", "POC", "ROT", "SOB", "TO",
)

_META_FILENAME = "SatSOT.json"

# Categories with ≤ this many sequences are pre-assigned round-robin so every
# split gets at least one instance. SatSOT has plane=9 and ship=5 — both need
# the safety net.
_SMALL_CAT_THRESH = 10

# Attributes with ≤ this many positive sequences are also pre-assigned round-
# robin (one positive per split) before iterative stratification runs. At
# 80/10/10, val/test only get ~10 % of samples, so anything with fewer than
# ~10 positives may end up with zero in val or test otherwise. SatSOT has
# IV (3 positives) and DEF (6 positives) that need this.
_RARE_ATTR_THRESH = 9


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
        self._attr_cache: dict[str, np.ndarray] = {}  # vid_id → (11,) int array
        super().__init__(root=root, split=split, **kwargs)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        # Load per-sequence attribute metadata once (SatSOT.json) if present.
        attr_to_idx = {a: i for i, a in enumerate(ATTR_NAMES)}
        meta_attrs: dict[str, np.ndarray] = {}
        meta_path = self.root / _META_FILENAME
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                for vid, entry in meta.items():
                    arr = np.zeros(len(ATTR_NAMES), dtype=np.int32)
                    for a in entry.get("attr", []):
                        if a in attr_to_idx:
                            arr[attr_to_idx[a]] = 1
                    meta_attrs[vid] = arr
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

        # Hybrid split balanced on class ⊕ 11 attributes.
        split_map = self._hybrid_split(all_videos)

        for v in all_videos:
            v.split = split_map[v.video_id]
            if self.split == "no_split" or v.split == self.split:
                self.videos.append(v)
            else:
                self._gt_cache.pop(v.video_id, None)
                self._attr_cache.pop(v.video_id, None)

    def _hybrid_split(self, videos: list[VideoInfo]) -> dict[str, str]:
        """
        Hybrid 80/10/10 split, balanced on **both** category and the 11 SatSOT
        sequence attributes (ARC, BC, BJT, DEF, FOC, IV, LQ, POC, ROT, SOB, TO).

        Step 1: tiny categories (n ≤ ``_SMALL_CAT_THRESH``) are pre-assigned
                round-robin as test → val → train. SatSOT's plane (9) and
                ship (5) land here so every split gets at least one of each.
        Step 2: rare attributes (positives ≤ ``_RARE_ATTR_THRESH``) are also
                pre-assigned — up to three positive-carrying sequences are
                forced into the splits still missing coverage. IV (3) and
                DEF (6) need this on SatSOT; without it, iterative strat's
                deficit rule dumps all their positives into train.
        Step 3: iterative stratification (Sechidis et al. 2011) places the
                remaining sequences using a label matrix that concatenates
                the class one-hot with the 11 binary attributes. Honours
                the preassignments from steps 1–2 as hard constraints.
        """
        N = len(videos)
        cats = sorted({v.category for v in videos})
        cat_to_idx = {c: i for i, c in enumerate(cats)}

        # Label matrix: one-hot class + 11 attrs (zero row if attrs missing).
        labels = np.zeros((N, len(cats) + len(ATTR_NAMES)), dtype=np.int32)
        for i, v in enumerate(videos):
            labels[i, cat_to_idx[v.category]] = 1
            attrs = self._attr_cache.get(v.video_id)
            if attrs is not None:
                labels[i, len(cats):] = attrs

        preassigned: dict[int, int] = {}
        test_first_cycle = [2, 1, 0]  # test, val, train

        # Step 1 — tiny classes pre-assignment.
        by_cat_idx: dict[str, list[int]] = defaultdict(list)
        for i, v in enumerate(videos):
            by_cat_idx[v.category].append(i)

        for cat, idxs in sorted(by_cat_idx.items()):
            if len(idxs) > _SMALL_CAT_THRESH:
                continue
            ids = idxs[:]
            rng2 = np.random.RandomState(_SPLIT_SEED + (zlib.crc32(cat.encode()) % 10_000))
            rng2.shuffle(ids)
            for k, idx in enumerate(ids):
                preassigned[idx] = test_first_cycle[k % 3]

        # Step 2 — rare attributes pre-assignment.
        attr_matrix = labels[:, len(cats):]
        for a_idx, a_name in enumerate(ATTR_NAMES):
            positives = np.where(attr_matrix[:, a_idx] > 0)[0].tolist()
            if not positives or len(positives) > _RARE_ATTR_THRESH:
                continue
            covered = {preassigned[i] for i in positives if i in preassigned}
            missing = [s for s in (2, 1, 0) if s not in covered]
            if not missing:
                continue
            unassigned = [i for i in positives if i not in preassigned]
            if not unassigned:
                continue
            rng3 = np.random.RandomState(_SPLIT_SEED + (zlib.crc32(a_name.encode()) % 10_000))
            rng3.shuffle(unassigned)
            for s in missing:
                if not unassigned:
                    break
                preassigned[unassigned.pop()] = s

        # Step 3 — iterative stratification on everyone else.
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

        Reads the already-loaded ``_attr_cache`` and decodes the 11 SatSOT
        flags (``ATTR_NAMES``) into the list of attributes whose flag is 1.
        Sequences missing from ``SatSOT.json`` yield an empty list.
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
