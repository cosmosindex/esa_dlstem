"""
LMODDataset
===========
Multi-class moving object detection dataset for satellite videos.

Directory layout::

    <root>/
        Seq1/
            Seq1/
                JPEGImages/
                    000001.jpg
                    ...
                gt/
                    000001.xml   # Pascal VOC format
                    ...
        Seq2/
            Seq2/
                ...
        ...

Categories: car, plane, ship, train.

Split strategy: 80/10/10 by frame within each sequence (temporal order).
Each sequence becomes one VideoInfo entry; frame_ids are filtered to the
requested split.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from .base import BaseVideoDataset, VideoInfo


class LMODDataset(BaseVideoDataset):
    """
    LMOD dataset loader.

    Args:
        root:      Path to dataset root (contains Seq1/ … Seq8/).
        split:     "train", "val", "test", or "no_split".
        **kwargs:  Forwarded to BaseVideoDataset (mode, clip_len, transform, …).
    """

    def __init__(self, root: str | Path, split: str = "train", **kwargs):
        self._gt_cache: dict[str, dict[int, list[dict]]] = {}
        super().__init__(root=root, split=split, **kwargs)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        seq_dirs = sorted(
            d for d in self.root.iterdir()
            if d.is_dir() and d.name.startswith("Seq")
        )

        for seq_dir in seq_dirs:
            inner = seq_dir / seq_dir.name
            gt_dir = inner / "gt"
            img_dir = inner / "JPEGImages"
            if not gt_dir.exists() or not img_dir.exists():
                continue

            xml_files = sorted(gt_dir.glob("*.xml"))
            if not xml_files:
                continue

            # Parse all annotations for this sequence
            frame_ids: list[int] = []
            gt_data: dict[int, list[dict]] = {}
            categories_in_seq: set[str] = set()

            for xml_path in xml_files:
                fid = int(xml_path.stem)
                objects = self._parse_xml(xml_path)
                gt_data[fid] = objects
                frame_ids.append(fid)
                for obj in objects:
                    categories_in_seq.add(obj["name"])

            # Determine the dominant non-car category for this sequence
            # (used as the sequence-level category label)
            category = sorted(categories_in_seq)[0]  # fallback
            for c in sorted(categories_in_seq):
                if c != "car":
                    category = c
                    break

            self._gt_cache[seq_dir.name] = gt_data

            # Split frames 80/10/10 by temporal order
            n = len(frame_ids)
            n_train = int(n * 0.8)
            n_val = int(n * 0.1)

            split_ranges = {
                "train": frame_ids[:n_train],
                "val":   frame_ids[n_train:n_train + n_val],
                "test":  frame_ids[n_train + n_val:],
            }

            for split_name, split_fids in split_ranges.items():
                if not split_fids:
                    continue
                if self.split != "no_split" and self.split != split_name:
                    continue
                self.videos.append(VideoInfo(
                    video_id=f"{seq_dir.name}_{split_name}",
                    dataset="LMOD",
                    category=category,
                    categories_present=tuple(sorted(categories_in_seq)),
                    split=split_name,
                    num_frames=len(split_fids),
                    frame_ids=split_fids,
                ))

    def _load_frame(self, video: VideoInfo, frame_id: int) -> np.ndarray:
        seq_name = video.video_id.rsplit("_", 1)[0]  # "Seq1_train" → "Seq1"
        path = self.root / seq_name / seq_name / "JPEGImages" / f"{frame_id:06d}.jpg"
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Frame not found: {path}")
        return img[..., ::-1].copy()  # BGR → RGB

    def _load_annotations(
        self, video: VideoInfo, frame_id: int
    ) -> dict[str, np.ndarray]:
        seq_name = video.video_id.rsplit("_", 1)[0]
        objects = self._gt_cache[seq_name][frame_id]

        boxes, labels, track_ids = [], [], []
        for obj in objects:
            label = self._map_label(obj["name"])
            boxes.append([obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"]])
            labels.append(label)
            track_ids.append(-1)

        if boxes:
            return {
                "boxes": np.array(boxes, dtype=np.float32),
                "labels": np.array(labels, dtype=np.int64),
                "track_ids": np.array(track_ids, dtype=np.int64),
            }
        return {
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "labels": np.zeros((0,), dtype=np.int64),
            "track_ids": np.zeros((0,), dtype=np.int64),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_xml(xml_path: Path) -> list[dict]:
        """Parse a single Pascal VOC XML annotation file."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        objects = []
        for obj in root.findall("object"):
            name = obj.find("name").text
            bb = obj.find("bndbox")
            objects.append({
                "name": name,
                "xmin": int(bb.find("xmin").text),
                "ymin": int(bb.find("ymin").text),
                "xmax": int(bb.find("xmax").text),
                "ymax": int(bb.find("ymax").text),
            })
        return objects
