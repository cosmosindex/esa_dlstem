"""
IRSatVideoDataset
=================
Multi-object satellite video dataset with bounding box and segmentation mask
annotations.  Supports Det / SOT / MOT / Seg tasks.

Directory layout::

    <root>/
        IRSatVideo-LEO/
            images/<seq_name>/0000.png, 0001.png, ...
            masks/<seq_name>/0000.png, 0001.png, ...     # binary uint8 (0/255)
            img_idx/<seq_name>.txt                        # frame indices (zero-padded)
            video_idx/
                train_IRSatVideo-LEO.txt                  # 160 sequences
                test_IRSatVideo-LEO.txt                   # 40 sequences
                test_IRSatVideo-LEO-easy.txt
                test_IRSatVideo-LEO-middle.txt
                test_IRSatVideo-LEO-hard.txt
        voc/<seq_name>/0000.xml, 0001.xml, ...            # Pascal VOC annotations

Objects are named ``target0``, ``target1``, etc. in the XML — the numeric
suffix is the **track ID**.  All objects belong to a single category
(``"target"``).

Official split: **train (160) / test (40)**.
Val is carved from 30% of the official test set (stratified by region,
``seed=42``); the remaining 70% stays as test.
"""

import re
import xml.etree.ElementTree as ET
from collections import defaultdict

import cv2
import numpy as np
from pathlib import Path

from .base import BaseVideoDataset, VideoInfo

_SPLIT_SEED = 42

# Regex to extract region name from sequence id, e.g.
#   "NorthAmericaEast12_51" → "NorthAmericaEast"
#   "EastAfrica-0_12"       → "EastAfrica"
_REGION_RE = re.compile(r"^([A-Za-z]+)")


class IRSatVideoDataset(BaseVideoDataset):
    """
    IRSatVideo-LEO dataset loader.

    Args:
        root:      Path to the dataset root (contains ``IRSatVideo-LEO/`` and
                   ``voc/`` sub-directories).
        split:     "train", "val", "test", or "no_split".
        **kwargs:  Forwarded to BaseVideoDataset (mode, clip_len, transform, …).
    """

    def __init__(self, root: str | Path, split: str = "train", **kwargs):
        self._ann_cache: dict[str, dict[int, list[dict]]] = {}
        super().__init__(root=root, split=split, **kwargs)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        data_dir = self.root / "IRSatVideo-LEO"
        voc_dir = self.root / "voc"
        video_idx_dir = data_dir / "video_idx"
        img_idx_dir = data_dir / "img_idx"

        # --- Step 1: read official split lists ---
        train_seqs = self._read_split_file(
            video_idx_dir / "train_IRSatVideo-LEO.txt"
        )
        test_seqs = self._read_split_file(
            video_idx_dir / "test_IRSatVideo-LEO.txt"
        )

        # --- Step 2: build val from 30% of test (stratified by region) ---
        val_seqs, remaining_test_seqs = self._split_test_to_val(
            test_seqs, val_ratio=0.3
        )

        split_map: dict[str, str] = {}
        for s in train_seqs:
            split_map[s] = "train"
        for s in val_seqs:
            split_map[s] = "val"
        for s in remaining_test_seqs:
            split_map[s] = "test"

        # --- Step 3: discover all sequences and build VideoInfo ---
        all_seqs = set(train_seqs) | set(test_seqs)
        for seq_name in sorted(all_seqs):
            img_dir = data_dir / "images" / seq_name
            ann_dir = voc_dir / seq_name
            if not img_dir.is_dir():
                continue

            # Read frame index
            idx_file = img_idx_dir / f"{seq_name}.txt"
            if idx_file.exists():
                frame_ids = self._parse_img_idx(idx_file)
            else:
                frame_ids = sorted(
                    int(p.stem) for p in img_dir.glob("*.png")
                )

            if not frame_ids:
                continue

            # Parse annotations and cache
            self._ann_cache[seq_name] = self._parse_voc_sequence(
                ann_dir, frame_ids
            )

            region = self._extract_region(seq_name)
            assigned_split = split_map.get(seq_name, "no_split")

            video = VideoInfo(
                video_id=seq_name,
                dataset="IRSatVideo-LEO",
                category=region,
                split=assigned_split,
                num_frames=len(frame_ids),
                frame_ids=frame_ids,
            )

            if self.split == "no_split" or video.split == self.split:
                self.videos.append(video)
            else:
                self._ann_cache.pop(seq_name, None)

    def _load_frame(self, video: VideoInfo, frame_id: int) -> np.ndarray:
        path = (
            self.root
            / "IRSatVideo-LEO"
            / "images"
            / video.video_id
            / f"{frame_id:04d}.png"
        )
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Frame not found: {path}")
        return img[..., ::-1].copy()  # BGR → RGB

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
        for obj in objs:
            lbl = self._map_label("target")
            if lbl < 0:
                continue
            boxes.append(obj["box"])
            labels.append(lbl)
            track_ids.append(obj["track_id"])

        if not boxes:
            return {
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "labels": np.zeros(0, dtype=np.int64),
                "track_ids": np.zeros(0, dtype=np.int64),
            }

        return {
            "boxes": np.array(boxes, dtype=np.float32),
            "labels": np.array(labels, dtype=np.int64),
            "track_ids": np.array(track_ids, dtype=np.int64),
        }

    # ------------------------------------------------------------------
    # Segmentation support
    # ------------------------------------------------------------------

    def load_mask(self, video: VideoInfo, frame_id: int) -> np.ndarray:
        """
        Load binary segmentation mask for a frame.

        Returns:
            np.ndarray of shape (H, W), dtype uint8, values in {0, 1}.
        """
        path = (
            self.root
            / "IRSatVideo-LEO"
            / "masks"
            / video.video_id
            / f"{frame_id:04d}.png"
        )
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {path}")
        return (mask > 0).astype(np.uint8)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_split_file(path: Path) -> list[str]:
        """Read a video_idx split file — one sequence name per line."""
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]

    @staticmethod
    def _split_test_to_val(
        test_seqs: list[str], val_ratio: float = 0.3
    ) -> tuple[list[str], list[str]]:
        """
        Stratified split of official test set into val + test by region.
        """
        rng = np.random.RandomState(_SPLIT_SEED)

        by_region: dict[str, list[str]] = defaultdict(list)
        for s in test_seqs:
            region = IRSatVideoDataset._extract_region(s)
            by_region[region].append(s)

        val, test = [], []
        for region in sorted(by_region):
            seqs = by_region[region]
            rng.shuffle(seqs)
            n_val = max(1, round(len(seqs) * val_ratio))
            # If only 1 sequence, assign to test (prioritise test coverage)
            if len(seqs) == 1:
                test.extend(seqs)
            else:
                val.extend(seqs[:n_val])
                test.extend(seqs[n_val:])

        return val, test

    @staticmethod
    def _extract_region(seq_name: str) -> str:
        """Extract geographic region prefix from a sequence name."""
        m = _REGION_RE.match(seq_name)
        return m.group(1) if m else "unknown"

    @staticmethod
    def _parse_img_idx(path: Path) -> list[int]:
        """Parse an img_idx file — 4-digit zero-padded frame indices, no separator."""
        text = path.read_text().strip()
        if not text:
            return []
        return [int(text[i : i + 4]) for i in range(0, len(text), 4)]

    @staticmethod
    def _parse_voc_sequence(
        ann_dir: Path, frame_ids: list[int]
    ) -> dict[int, list[dict]]:
        """
        Parse all VOC XML annotations for a sequence.

        Returns:
            Dict mapping frame_id → list of object dicts with keys:
                box:      [xmin, ymin, xmax, ymax] (floats)
                track_id: int (extracted from object name, e.g. target2 → 2)
        """
        cache: dict[int, list[dict]] = {}
        for fid in frame_ids:
            xml_path = ann_dir / f"{fid:04d}.xml"
            if not xml_path.exists():
                cache[fid] = []
                continue

            tree = ET.parse(xml_path)
            root = tree.getroot()
            objs = []
            for obj_elem in root.findall("object"):
                name = obj_elem.findtext("name", "target0")
                # Extract track id from name: "target2" → 2
                tid_match = re.search(r"(\d+)$", name)
                track_id = int(tid_match.group(1)) if tid_match else 0

                bndbox = obj_elem.find("bndbox")
                xmin = float(bndbox.findtext("xmin", "0"))
                ymin = float(bndbox.findtext("ymin", "0"))
                xmax = float(bndbox.findtext("xmax", "0"))
                ymax = float(bndbox.findtext("ymax", "0"))

                objs.append({
                    "box": [xmin, ymin, xmax, ymax],
                    "track_id": track_id,
                })
            cache[fid] = objs
        return cache
