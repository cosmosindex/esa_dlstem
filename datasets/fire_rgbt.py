"""
FireRGBTDataset
===============
Wildfire / smoke detection dataset (RGBT-3M, *Remote Sensing* 2025,
https://www.mdpi.com/2072-4292/17/15/2593).

**Detection only.** Frames are sampled from videos (``video1`` … ``video10``)
but the released annotations are per-frame YOLO bounding boxes with **no track
ids**, so ``track_ids`` is always ``-1`` and only ``mode="detection"`` is
meaningful here.  (The MEMORY note that this is "fire + person" is incomplete:
the labels contain **3** classes — ``smoke``, ``fire``, ``person``.)

Directory layout::

    <root>/                       # /data/ESA_DLSTEM_2025/data/fire/RGBT-3M
        images/
            train/video1_frame_00110.jpg …      # 7 854 frames
            test/ video1_frame_00110.jpg …      # 3 366 frames
        labels/
            train/video1_frame_00110.txt …      # one .txt per image
            test/ video1_frame_00110.txt …
            {train,test}/classes.txt            # "smoke\\nfire\\nperson"

Annotation format (``*.txt``, YOLO):

    ``<class_id> <cx> <cy> <w> <h>``   one object per line,
    all four geometry values **normalised to [0, 1]** w.r.t. the image
    (640 × 480 throughout).  class_id: 0=smoke, 1=fire, 2=person.

Split strategy
--------------
The dataset ships an **official frame-level train/test split** (separate
``images/{train,test}/`` dirs).  Note the *same* ``videoN`` appears in **both**
splits — the split is over frames, not videos.  We honour the official **test**
split unchanged (3 366 frames) so reported numbers stay comparable to the paper,
and carve a **val** split out of the official **train** side.

val is a **per-video proportional** frame sample: ``_TRAIN_TO_VAL_FRACTION`` of
*each* train video's frames, drawn with ``seed=42``.  Sampling the same fraction
from every video makes val an unbiased proportional miniature of train, so the
three splits stay **balanced per category and per object size** at once (the
design goal) — rather than skewing val toward one scene's class/size profile.
This frame-level carve mirrors the dataset's own protocol (its official
train/test is likewise a frame-level split of the same videos).

``split="no_split"`` returns every frame from both dirs (video_id is prefixed
with the on-disk dir, e.g. ``train/video1`` vs ``test/video1``, so the
identically-numbered frames never collide).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .base import BaseVideoDataset, VideoInfo

_SPLIT_SEED = 42
# Fraction of the official train-side videos held out as val (whole-video,
# leak-free). 8 train videos → max(1, round(0.1*8)) = 1 val video.
_TRAIN_TO_VAL_FRACTION = 0.1

# class_id (on disk) → raw category name, from labels/*/classes.txt
_CLASS_NAMES = ("smoke", "fire", "person")
# Default global class map (0-indexed, matches classes.txt order). Pass a
# custom ``class_map`` to e.g. drop smoke (``{"fire": 0, "person": 1}``) —
# objects whose name is absent are mapped to -1 and dropped by the base class.
_DEFAULT_CLASS_MAP = {"smoke": 0, "fire": 1, "person": 2}

# images/<dir>/<video>_frame_<fid>.jpg  →  (video, fid)
_STEM_RE = re.compile(r"^(video\d+)_frame_(\d+)$")


class FireRGBTDataset(BaseVideoDataset):
    """
    RGBT-3M wildfire/smoke/person detection dataset (YOLO-format labels).

    Args:
        root:   Dataset root (contains ``images/`` and ``labels/``).
        split:  ``"train"``, ``"val"``, ``"test"``, or ``"no_split"``.
        **kwargs:  Forwarded to :class:`BaseVideoDataset` (``mode``, ``transform``,
                   ``class_map`` …).  Defaults to the 3-class
                   smoke/fire/person map when ``class_map`` is None.
    """

    def __init__(self, root: str | Path, split: str = "train", **kwargs):
        kwargs.setdefault("class_map", dict(_DEFAULT_CLASS_MAP))
        # video_id -> on-disk split dir ("train" / "test")
        self._disk_dir: dict[str, str] = {}
        # video_id -> {frame_id: [(cls_id, cx, cy, w, h), ...]}  (normalised)
        self._ann_cache: dict[str, dict[int, list[tuple]]] = {}
        # (video_id, frame_id) -> (W, H), memoised lazily in _load_annotations
        self._size_cache: dict[tuple[str, int], tuple[int, int]] = {}
        super().__init__(root=root, split=split, **kwargs)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        # 1. Discover frames per (disk_dir, video) and parse YOLO labels once.
        #    raw[disk_dir][video] = {fid: [(cls, cx, cy, w, h), ...]}
        raw: dict[str, dict[str, dict[int, list[tuple]]]] = {
            d: defaultdict(dict) for d in ("train", "test")
        }
        for disk_dir in ("train", "test"):
            img_dir = self.root / "images" / disk_dir
            lbl_dir = self.root / "labels" / disk_dir
            if not img_dir.is_dir():
                continue
            for img_path in img_dir.glob("*.jpg"):
                m = _STEM_RE.match(img_path.stem)
                if m is None:
                    continue
                video, fid = m.group(1), int(m.group(2))
                raw[disk_dir][video][fid] = self._parse_label(
                    lbl_dir / f"{img_path.stem}.txt"
                )

        # 2. Frame-level val carve from the official train side, **per-video
        #    proportional**: val = ~_TRAIN_TO_VAL_FRACTION of EACH train video's
        #    frames, sampled with seed=42. Because every video contributes the
        #    same fraction, val becomes an unbiased proportional miniature of
        #    train and matches it on class mix, object-size mix, AND scene/video
        #    coverage simultaneously (the balance goal). Official test is left
        #    intact. The official train/test split is itself frame-level (the
        #    same videoN appears in both dirs), so carving val frames out of the
        #    train videos is consistent with the dataset's own protocol.
        frame_split: dict[tuple[str, str], dict[int, str]] = {}
        rng = np.random.RandomState(_SPLIT_SEED)
        for video in sorted(raw["train"].keys(), key=_video_key):
            fids = sorted(raw["train"][video].keys())
            perm = list(fids)
            rng.shuffle(perm)
            n_val = max(1, int(round(len(perm) * _TRAIN_TO_VAL_FRACTION)))
            val_fids = set(perm[:n_val])
            frame_split[("train", video)] = {
                f: ("val" if f in val_fids else "train") for f in fids
            }
        for video in raw["test"].keys():
            frame_split[("test", video)] = {f: "test" for f in raw["test"][video]}

        # 3. One VideoInfo per (dir, video), restricted to the frames assigned
        #    to self.split ("no_split" keeps every frame). A train-side video
        #    thus appears in the train split and the val split with disjoint
        #    frame subsets — never the same frame in two splits.
        for disk_dir in ("train", "test"):
            for video in sorted(raw[disk_dir].keys(), key=_video_key):
                frames = raw[disk_dir][video]
                fmap = frame_split[(disk_dir, video)]
                if self.split == "no_split":
                    sel_fids = sorted(frames.keys())
                    assigned = "no_split"
                else:
                    sel_fids = sorted(f for f, s in fmap.items() if s == self.split)
                    assigned = self.split
                if not sel_fids:
                    continue

                video_id = f"{disk_dir}/{video}"

                # category metadata over the frames selected for this split
                cls_count: Counter = Counter()
                for f in sel_fids:
                    for cls_id, *_ in frames[f]:
                        cls_count[cls_id] += 1
                present = tuple(
                    _CLASS_NAMES[c] for c in sorted(cls_count) if c < len(_CLASS_NAMES)
                )
                dominant = (
                    _CLASS_NAMES[cls_count.most_common(1)[0][0]]
                    if cls_count else "fire"
                )

                self._disk_dir[video_id] = disk_dir
                self._ann_cache[video_id] = frames
                self.videos.append(
                    VideoInfo(
                        video_id=video_id,
                        dataset="FireRGBT",
                        category=dominant,
                        split=assigned,
                        num_frames=len(sel_fids),
                        frame_ids=sel_fids,
                        categories_present=present,
                    )
                )

    def _load_frame(self, video: VideoInfo, frame_id: int) -> np.ndarray:
        path = self._frame_path(video.video_id, frame_id)
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Frame not found: {path}")
        return img[..., ::-1].copy()  # BGR → RGB

    def _load_annotations(
        self, video: VideoInfo, frame_id: int
    ) -> dict[str, np.ndarray]:
        objs = self._ann_cache.get(video.video_id, {}).get(frame_id, [])

        empty = {
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "labels": np.zeros(0, dtype=np.int64),
            "track_ids": np.zeros(0, dtype=np.int64),
        }
        if not objs:
            return empty

        w_img, h_img = self._image_size(video.video_id, frame_id)

        boxes, labels = [], []
        for cls_id, cx, cy, bw, bh in objs:
            if cls_id >= len(_CLASS_NAMES):
                continue
            lbl = self._map_label(_CLASS_NAMES[cls_id])
            if lbl < 0:  # class dropped by a custom class_map
                continue
            # normalised cxcywh → absolute xyxy
            x1 = (cx - bw / 2.0) * w_img
            y1 = (cy - bh / 2.0) * h_img
            x2 = (cx + bw / 2.0) * w_img
            y2 = (cy + bh / 2.0) * h_img
            # Drop degenerate / zero-area GT boxes (the dataset ships a couple
            # of w=0 annotations). albumentations rejects x_max <= x_min.
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(lbl)

        if not boxes:
            return empty

        return {
            "boxes": np.asarray(boxes, dtype=np.float32),
            "labels": np.asarray(labels, dtype=np.int64),
            # detection-only dataset: no tracking GT
            "track_ids": np.full(len(boxes), -1, dtype=np.int64),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _frame_path(self, video_id: str, frame_id: int) -> Path:
        disk_dir = self._disk_dir[video_id]
        video = video_id.split("/", 1)[1]
        return self.root / "images" / disk_dir / f"{video}_frame_{frame_id:05d}.jpg"

    def _image_size(self, video_id: str, frame_id: int) -> tuple[int, int]:
        """(W, H) for one frame, memoised via a header-only PIL read."""
        key = (video_id, frame_id)
        wh = self._size_cache.get(key)
        if wh is None:
            with Image.open(self._frame_path(video_id, frame_id)) as im:
                wh = im.size  # (W, H)
            self._size_cache[key] = wh
        return wh

    @staticmethod
    def _parse_label(path: Path) -> list[tuple]:
        """Parse a YOLO ``*.txt`` → ``[(cls_id, cx, cy, w, h), ...]`` (normalised)."""
        objs: list[tuple] = []
        if not path.exists():
            return objs
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls_id = int(float(parts[0]))
                cx, cy, w, h = (float(v) for v in parts[1:5])
                objs.append((cls_id, cx, cy, w, h))
        return objs


def _video_key(video: str) -> int:
    """Sort ``videoN`` numerically (video2 < video10), not lexicographically."""
    return int(video[len("video"):])
