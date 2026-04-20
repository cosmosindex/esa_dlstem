"""
OOTBDataset
===========
Single-object tracking dataset with Oriented Bounding Box (OBB) ground truth.

Directory layout::

    <root>/
        <seq_name>/            e.g. car_1/, plane_3/
            img/
                0001.jpg
                0002.jpg
                ...
            groundtruth.txt    # one line per frame
                               # 8 floats: x1,y1,x2,y2,x3,y3,x4,y4  (OBB corners)
        anno/
            <seq_name>.txt     # 16 binary flags:
                               #   4 category one-hot (car, ship, train, plane)
                               # + 12 attributes (DEF, IPR, PO, FO, IV, MB, BC,
                               #   OON, SA, LT, IM, AM)
"""

import re
import zlib
from collections import defaultdict

import cv2
import numpy as np
from pathlib import Path

from .base import BaseVideoDataset, VideoInfo

_SPLIT_SEED = 42
_RATIOS = (0.8, 0.1, 0.1)  # train, val, test

# Order of the 16 integers in `anno/<seq_name>.txt` (OOTB paper, ISPRS 2024).
CAT_NAMES = ("car", "ship", "train", "plane")
ATTR_NAMES = (
    "DEF", "IPR", "PO", "FO", "IV", "MB",
    "BC", "OON", "SA", "LT", "IM", "AM",
)

# Categories with ≤ this many sequences are pre-assigned round-robin so
# every split gets at least one instance. Smallest OOTB class is `train`
# with 10 sequences (≥ 1 per split is safely achievable by iterative
# stratification alone), so no class needs the round-robin safety net.
_SMALL_CAT_THRESH = 5


class OOTBDataset(BaseVideoDataset):
    """
    OOTB dataset loader.

    Args:
        root:      Path to dataset root (one sub-directory per sequence plus anno/).
        split:     "train", "val", "test", or "no_split" (returns all videos).
        **kwargs:  Forwarded to BaseVideoDataset (mode, clip_len, transform, …).
    """

    def __init__(self, root: str | Path, split: str = "train", **kwargs):
        # Must be initialized before super().__init__() calls _build_index()
        self._gt_cache: dict[str, np.ndarray] = {}
        self._attr_cache: dict[str, np.ndarray] = {}  # vid_id → (12,) int array
        super().__init__(root=root, split=split, **kwargs)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        anno_dir = self.root / "anno"

        # --- Step 1: discover all sequences ---
        all_videos: list[VideoInfo] = []
        for seq_dir in sorted(self.root.iterdir()):
            if not seq_dir.is_dir() or seq_dir.name == "anno":
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

            # Parse the 16-flag anno file if available (skip macOS ._ files).
            attr_path = anno_dir / f"{seq_dir.name}.txt"
            if attr_path.exists() and not attr_path.name.startswith("."):
                attrs = self._parse_attr(attr_path)
                if attrs is not None:
                    self._attr_cache[seq_dir.name] = attrs

        # --- Step 2: hybrid split balanced on class ⊕ 12 attributes ---
        split_map = self._hybrid_split(all_videos)

        # --- Step 3: keep only the requested split ---
        for v in all_videos:
            v.split = split_map[v.video_id]
            if self.split == "no_split" or v.split == self.split:
                self.videos.append(v)
            else:
                self._gt_cache.pop(v.video_id, None)
                self._attr_cache.pop(v.video_id, None)

    def _hybrid_split(self, videos: list[VideoInfo]) -> dict[str, str]:
        """
        Hybrid 80/10/10 split, balanced on **both** category and the 12 OOTB
        sequence attributes (DEF, IPR, PO, FO, IV, MB, BC, OON, SA, LT, IM, AM).

        Step 1: tiny categories (n ≤ _SMALL_CAT_THRESH) are pre-assigned
                round-robin as test → val → train. With OOTB's class sizes
                (car=45, ship=30, plane=25, train=10) and _SMALL_CAT_THRESH=5,
                this step is a no-op — retained for parity with the SV248S
                splitter in case future datasets add rarer classes.
        Step 2: iterative stratification (Sechidis et al. 2011) assigns the
                remaining sequences using a label matrix that concatenates
                the class one-hot with the 12 binary attribute flags. This
                also keeps rare attributes (FO, IM, DEF) represented in test.

        Sequences whose anno file is missing get a zero attribute row —
        they still participate in class balance.
        """
        N = len(videos)
        cats = sorted({v.category for v in videos})
        cat_to_idx = {c: i for i, c in enumerate(cats)}

        # Build label matrix (N, |cats| + 12): one-hot class + 12 attrs.
        labels = np.zeros((N, len(cats) + len(ATTR_NAMES)), dtype=np.int32)
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

        Reads the already-loaded ``_attr_cache`` and decodes the 12 OOTB
        flags (``ATTR_NAMES``) into the list of attributes whose flag is 1.
        Sequences without an anno file yield an empty list.
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

    @staticmethod
    def _parse_attr(path: Path) -> np.ndarray | None:
        """
        Parse `anno/<seq>.txt` → (12,) int32 array of attribute flags.

        File contains 16 comma-separated binary ints: 4 category one-hot
        (car, ship, train, plane) followed by 12 attributes in the order
        defined in `ATTR_NAMES`. We keep only the attribute suffix.
        """
        try:
            text = path.read_text().strip()
        except UnicodeDecodeError:
            return None
        if not text:
            return None
        values = [int(v) for v in text.split(",") if v.strip() != ""]
        if len(values) != 4 + len(ATTR_NAMES):
            return None
        return np.array(values[4:], dtype=np.int32)
