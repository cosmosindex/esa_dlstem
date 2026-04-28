"""
RsCarDataset
============
Single-class moving-vehicle MOT dataset, sourced from the
*RsCarData* re-annotation of the **car / vehicle subset of VISO** used by
the HiEUM paper (Xiao et al., TPAMI 2024). The original VISO dataset has
four categories (plane, car, ship, train); RsCarData keeps only the
vehicle sequences and ships re-curated MOT-format labels for them.

Directory layout::

    <root>/
        annotations/
            train_mot.json            # COCO-style + video metadata
            test1024_mot.json         # COCO-style + video metadata (1024x1024 test crop)
            instances_train2017.json  # plain COCO detection (no track ids) — unused
            instances_test2017_1024.json  # plain COCO detection — unused
        images/
            train/<seq>/img1/<frame>.jpg          # 70 sequences @ 512x512
            test1024/<seq>/img1/<frame>.jpg       # 7 sequences @ 1024x1024
        labeleddata20230227/<seq>/img1/<frame>.xml  # NEW (paper-protocol) test labels
        update_label_train.zip                    # NEW train labels (unused for test)

Test labels — old vs new
------------------------
HiEUM's official ``evaluation_final.py`` defaults to ``eval_new_mode='new'``,
which evaluates against the **re-curated** GT under
``labeleddata20230227/<seq>/img1/<frame>.xml`` rather than the COCO MOT
JSON shipped under ``annotations/test1024_mot.json``. The new labels carry
≈66 % more boxes (155 987 vs 93 491 across the 7 test sequences), and a
precision gap of >25 pp disappears once you switch.

This dataset class therefore prefers the new XML labels for the test
split when ``labeleddata20230227/<seq>/img1/`` is present and falls back
to the COCO MOT JSON otherwise. Train + val splits always use the COCO
JSON since the new labels for train ship as a separate zip
(``update_label_train.zip``) we currently do not consume.

JSON schema (the ``*_mot.json`` files we actually read)::

    {
      "images":     [{"id", "file_name", "video_id", "frame_id", ...}, ...],
      "annotations":[{"id", "category_id", "image_id", "track_id",
                      "bbox": [x, y, w, h],            # xywh, top-left
                      ...}, ...],
      "videos":     [{"id", "file_name"}, ...],
      "categories": [{"id": 1, "name": "car"}],
    }

All annotations belong to the single class ``"car"``.

Splits
------
The dataset ships **train (70 seqs @ 512x512) + test (7 seqs @ 1024x1024)**
only. We honour the official test split unchanged so that numbers reported
on RsCarData match HiEUM's paper protocol (``dataNum=[2,3,5,6,8,9,10]``,
the 7 test1024 sequences). Val is carved from the train side instead —
10 % of the 70 training sequences, shuffled with ``seed=42``:

    - train: 63 sequences  (~24 700 frames @ 512x512)
    - val:    7 sequences  (~ 2 700 frames @ 512x512)
    - test:   7 sequences  (  2 255 frames @ 1024x1024)

Use ``split="no_split"`` to load all 77 sequences regardless of split.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .base import BaseVideoDataset, VideoInfo


_SPLIT_SEED = 42

# Fraction of the official train split held out as val. The official 7
# test1024 sequences are kept untouched so reported numbers can be
# compared directly against HiEUM's paper protocol.
_TRAIN_TO_VAL_FRACTION = 0.10


class RsCarDataset(BaseVideoDataset):
    """
    RsCarData (HiEUM's VISO car subset) loader.

    Args:
        root:      Path to dataset root (contains ``annotations/`` and
                   ``images/{train,test1024}/``).
        split:     "train", "val", "test", or "no_split".
        **kwargs:  Forwarded to BaseVideoDataset (mode, clip_len, transform, …).
    """

    # Original split name on disk → our split tag (val carved from test below)
    _OFFICIAL_SPLITS = {
        "train":     "annotations/train_mot.json",
        "test1024":  "annotations/test1024_mot.json",
    }

    def __init__(self, root: str | Path, split: str = "train", **kwargs):
        # Cache: video_id (str) → {frame_id (int) → list of (track_id, x1, y1, x2, y2)}
        self._ann_cache: dict[str, dict[int, list[tuple[int, float, float, float, float]]]] = {}
        # video_id → relative path to img1/ dir, e.g. "images/train/001/img1"
        self._img_dir: dict[str, str] = {}
        super().__init__(root=root, split=split, **kwargs)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    # Path to the re-curated test-split labels HiEUM's paper evaluates on
    # (``eval_new_mode='new'`` in ``evaluation_final.py``). When this
    # directory is present we prefer it for the test split.
    _NEW_TEST_LABELS_DIR = "labeleddata20230227"

    def _build_index(self) -> None:
        # Parse both COCO MOT JSONs once. Each provides one official split.
        train_videos = self._parse_mot_json(
            self.root / self._OFFICIAL_SPLITS["train"], official_split="train",
        )
        test_videos = self._parse_mot_json(
            self.root / self._OFFICIAL_SPLITS["test1024"], official_split="test1024",
        )

        # Test split: prefer the paper's re-curated XML labels when present.
        new_root = self.root / self._NEW_TEST_LABELS_DIR
        if new_root.is_dir():
            self._override_test_anns_from_xml(test_videos, new_root)

        # Carve val out of the official **train** split — keeps the 7
        # test1024 sequences intact so evaluations on this dataset can be
        # compared one-to-one with HiEUM's paper.
        rng = np.random.RandomState(_SPLIT_SEED)
        train_ids = sorted(v.video_id for v in train_videos)
        rng.shuffle(train_ids)
        n_val = max(1, int(round(len(train_ids) * _TRAIN_TO_VAL_FRACTION)))
        val_set = set(train_ids[:n_val])

        for v in train_videos:
            v.split = "val" if v.video_id in val_set else "train"

        all_videos = train_videos + test_videos

        if self.split == "no_split":
            self.videos = all_videos
        else:
            self.videos = [v for v in all_videos if v.split == self.split]

    def _load_frame(self, video: VideoInfo, frame_id: int) -> np.ndarray:
        path = self.root / self._img_dir[video.video_id] / f"{frame_id:06d}.jpg"
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Frame not found: {path}")
        return img[..., ::-1].copy()                    # BGR → RGB

    def _load_annotations(
        self, video: VideoInfo, frame_id: int,
    ) -> dict[str, np.ndarray]:
        objs = self._ann_cache.get(video.video_id, {}).get(frame_id, [])
        lbl = self._map_label("car")
        if not objs or lbl < 0:
            return {
                "boxes":     np.zeros((0, 4), dtype=np.float32),
                "labels":    np.zeros(0,      dtype=np.int64),
                "track_ids": np.zeros(0,      dtype=np.int64),
            }

        boxes, labels, track_ids = [], [], []
        for tid, x1, y1, x2, y2 in objs:
            boxes.append([x1, y1, x2, y2])
            labels.append(lbl)
            track_ids.append(tid)
        return {
            "boxes":     np.array(boxes,     dtype=np.float32),
            "labels":    np.array(labels,    dtype=np.int64),
            "track_ids": np.array(track_ids, dtype=np.int64),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_mot_json(self, path: Path, official_split: str) -> list[VideoInfo]:
        """Parse one COCO MOT JSON into a list of VideoInfo + populate caches.

        ``official_split`` is the on-disk folder name ("train" or "test1024");
        we re-tag test1024 → "test" for our canonical naming. Val is
        decided by the caller.
        """
        with open(path) as f:
            data = json.load(f)

        # video_id → (file_name, [frame_ids sorted])
        video_meta: dict[int, dict] = {v["id"]: v for v in data["videos"]}
        video_frames: dict[int, list[int]] = defaultdict(list)
        # image_id (COCO) → (video_id, frame_id_in_video)
        img_to_video: dict[int, tuple[int, int]] = {}

        for img in data["images"]:
            vid = img["video_id"]
            # ``video_frame_id`` is 1-indexed within the sequence — matches the
            # frame filename ``%06d.jpg``.
            fid = int(img["video_frame_id"])
            video_frames[vid].append(fid)
            img_to_video[img["id"]] = (vid, fid)

        # Group annotations by (video_id, frame_id)
        ann_by_video: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
        for ann in data["annotations"]:
            vid, fid = img_to_video[ann["image_id"]]
            seq_name = video_meta[vid]["file_name"]   # "001"
            video_id = f"{official_split}/{seq_name}"
            x, y, w, h = ann["bbox"]
            ann_by_video[video_id][fid].append(
                (int(ann["track_id"]), float(x), float(y), float(x + w), float(y + h))
            )

        canonical_split = "test" if official_split == "test1024" else official_split

        videos: list[VideoInfo] = []
        for vid, frame_ids in video_frames.items():
            seq_name = video_meta[vid]["file_name"]   # "001"
            video_id = f"{official_split}/{seq_name}"
            frame_ids = sorted(set(frame_ids))

            self._ann_cache[video_id] = dict(ann_by_video.get(video_id, {}))
            self._img_dir[video_id] = f"images/{official_split}/{seq_name}/img1"

            videos.append(VideoInfo(
                video_id=video_id,
                dataset="RsCarData",
                category="car",
                split=canonical_split,        # may be re-tagged to "val" by caller
                num_frames=len(frame_ids),
                frame_ids=frame_ids,
            ))
        videos.sort(key=lambda v: v.video_id)
        return videos

    def _override_test_anns_from_xml(
        self, test_videos: list[VideoInfo], new_root: Path,
    ) -> None:
        """Replace ``self._ann_cache[video_id]`` entries for the test split
        with boxes parsed from the paper's re-curated PASCAL-VOC XML labels.

        Only entries for sequences that actually have an XML directory are
        overwritten; everything else is left on the COCO-MOT JSON path.
        Frame coverage is identical (verified upstream against the JSON).
        """
        replaced = 0
        for v in test_videos:
            seq_name = v.video_id.split("/", 1)[1]
            xml_dir = new_root / seq_name / "img1"
            if not xml_dir.is_dir():
                continue
            ann_by_frame: dict[int, list[tuple[int, float, float, float, float]]] = {}
            for xml_path in xml_dir.glob("*.xml"):
                fid = int(xml_path.stem)
                tree = ET.parse(xml_path)
                rows: list[tuple[int, float, float, float, float]] = []
                for obj in tree.findall("object"):
                    name_el = obj.find("name")
                    if name_el is None or (name_el.text or "").lower() != "car":
                        continue
                    bb = obj.find("bndbox")
                    if bb is None:
                        continue
                    x1 = float(bb.findtext("xmin", "0"))
                    y1 = float(bb.findtext("ymin", "0"))
                    x2 = float(bb.findtext("xmax", "0"))
                    y2 = float(bb.findtext("ymax", "0"))
                    tid_el = obj.find("id")
                    tid = int(tid_el.text) if tid_el is not None and tid_el.text else -1
                    rows.append((tid, x1, y1, x2, y2))
                if rows:
                    ann_by_frame[fid] = rows
            self._ann_cache[v.video_id] = ann_by_frame
            replaced += 1
