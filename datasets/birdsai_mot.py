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

# Frame-level val carve from TrainReal: every _VAL_FRAME_STRIDE-th frame of each
# TrainReal video (offset to skip frame 0) becomes val (~14%); the rest is train.
_VAL_FRAME_STRIDE = 7
_VAL_FRAME_OFFSET = 3

# Coarse: CSV `class` column (col 6) → category name.
_CLASS_NAMES = {0: "animal", 1: "human"}

# Fine: CSV `species` column (col 7) → category name. Any species not listed here
# (dog=4 — test-only / never in train; crocodile/hippo/zebra/rhino=5..8 — absent)
# folds into "unknown", i.e. "an animal whose species the training vocab never saw".
# -1 (species genuinely unlabeled) is also "unknown" — the largest animal group.
_SPECIES_NAMES = {-1: "unknown", 0: "human", 1: "elephant", 2: "lion", 3: "giraffe"}


class BIRDSAIMOTDataset(BaseVideoDataset):
    """
    BIRDSAI MOT dataset loader.

    Args:
        root:      Path to BIRDSAI root (contains TrainReal/ and TestReal/).
        split:     "train", "val", "test", or "no_split".
        annotations_dirname: name of the per-split annotation subdir to read
                   CSVs from. Default "annotations" (original GT). Set to
                   "annotations_sam3" to use the SAM3 box-refined GT produced by
                   evaluation/relabel_birdsai_sam3.py (tighter thermal boxes;
                   same 10-column schema, same row counts).
        **kwargs:  Forwarded to BaseVideoDataset.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        granularity: str = "coarse",
        annotations_dirname: str = "annotations",
        **kwargs,
    ):
        self.annotations_dirname = annotations_dirname
        # granularity: "coarse" → {animal, human} (class col 6);
        #              "fine"   → species {human, elephant, giraffe, lion, unknown} (col 7).
        if granularity not in ("coarse", "fine"):
            raise ValueError(f"granularity must be 'coarse' or 'fine', got {granularity!r}")
        self.granularity = granularity

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

            ann_dir = split_dir / self.annotations_dirname
            if not ann_dir.exists():
                raise FileNotFoundError(
                    f"Annotation dir not found: {ann_dir} "
                    f"(annotations_dirname={self.annotations_dirname!r})"
                )
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
        # TrainReal → train + val ; TestReal → test (kept fully intact).
        #
        # Why frame-level val: BIRDSAI source videos are almost entirely single-class
        # (lion has only 1 video, giraffe 2), so a per-VIDEO val cannot be balanced by
        # class — a held-out video removes its class from train entirely. We instead
        # carve a per-video STRIDED frame subset (every _VAL_FRAME_STRIDE-th frame) out
        # of each TrainReal video: val then inherits every video's class *and* object-size
        # distribution, so all 5 classes and both size regimes are represented.
        #
        # This couples train/val temporally (a val frame is adjacent to train frames),
        # but val is used ONLY for checkpoint selection / early stopping — the reported
        # TEST set is all of TestReal, a fully disjoint source, so no leakage reaches the
        # test metrics. (The old code split TestReal into val/test by random video, which
        # gave a val with 0 elephants and 96% tiny boxes — useless for model selection.)
        kept = []
        for v in all_videos:
            if source_map[v.video_id] == "test":
                v.split = "test"
                if self.split in ("test", "no_split"):
                    kept.append(v)
                continue

            # TrainReal video
            if self.split == "no_split":
                v.split = "train"
                kept.append(v)
                continue

            val_fids = set(v.frame_ids[_VAL_FRAME_OFFSET::_VAL_FRAME_STRIDE])
            if self.split == "val":
                sel = [f for f in v.frame_ids if f in val_fids]
            elif self.split == "train":
                sel = [f for f in v.frame_ids if f not in val_fids]
            else:
                continue  # requested split is "test" → no TrainReal video
            if not sel:
                continue
            v.split = self.split
            v.frame_ids = sel
            v.num_frames = len(sel)
            kept.append(v)

        kept_ids = {v.video_id for v in kept}
        for v in all_videos:
            if v.video_id in kept_ids:
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

    def _parse_csv(
        self,
        path: Path,
    ) -> dict[int, list[tuple[int, str, float, float, float, float]]]:
        """Parse a MOT CSV file.

        CSV columns: frame, object_id, x, y, w, h, class, species, occlusion, noise
        class: 0 = animal, 1 = human.  species: -1 unknown, 0 human, 1 elephant,
        2 lion, 3 giraffe, 4 dog, 5 crocodile, 6 hippo, 7 zebra, 8 rhino.

        Category name depends on self.granularity ("coarse" uses the class col,
        "fine" uses the species col, folding unseen species → "unknown").

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

                if self.granularity == "fine":
                    if len(parts) < 8:
                        continue
                    # species col → fine name; unseen/unlabeled species → "unknown".
                    class_name = _SPECIES_NAMES.get(int(parts[7]), "unknown")
                else:
                    class_name = _CLASS_NAMES.get(cls)
                    if class_name is None:
                        continue

                # Convert xywh → xyxy
                frame_anns[frame_id].append((track_id, class_name, x, y, x + w, y + h))

        return dict(frame_anns)
