"""
Space-Tracker SOT release, exposed as a single-object tracking dataset.

Reads the built release directly -- ``sot/<class>/<seq>/{img/,groundtruth.txt}``
plus the manifest beside them -- so the whole of Space-Tracker-SOT is one
dataset here, exactly as the paper scores it, rather than three source datasets
whose numbers are pooled afterwards.

Layout::

    <root>/
      splits.json                       train / val / test per sequence
      sot/
        <class>/<seq>/img/000001.jpg    1-indexed frames
        <class>/<seq>/groundtruth.txt   15 columns, one row per frame
        space_tracker_sot.json          manifest: attributes, provenance, sizes

**Ground truth.** One row per frame:
``frame_id, x, y, w, h, visible, state, px1, py1, …, px4, py4``. A row with
``visible = 0`` (all-``-1`` geometry) means the target is absent from that
frame, and is returned as zero objects -- the same contract
:class:`~datasets.satsot.SatSOTDataset` gives for its literal ``none``.
``px1..py4`` are oriented corners where the source annotates them and ``-1``
elsewhere; a sequence that has them exposes ``obb`` alongside ``boxes``, so an
evaluator can score either geometry without knowing which source a sequence
came from.

**Split.** Read from ``splits.json`` at the release root, which is scene-
disjoint: sequences cut from the same parent satellite scene are assigned
together, so a tracker cannot meet a test scene's rooftops during training.
``split="no_split"`` returns all 395.

**Attributes.** :meth:`sequence_attributes` returns each sequence's full
taxonomy list -- every one of the 18 attributes it carries, plus any occlusion
sub-type -- taken from the manifest rather than recomputed, so it agrees with
the released package by construction. :meth:`pooled_attributes` names the five
that more than one source annotates and may therefore be scored across the
benchmark; the rest are reported on their one annotating source.

Relation to the per-source configs
----------------------------------
``configs/SOT/<tracker>_{ootb,satsot,sv248s}.yaml`` evaluate the same sequences
through the source datasets in their original layouts, and the published
numbers were produced that way, then aggregated over the released set by
``tools/make_wacv_sot_table.py``. This class is the entry point for evaluating
on the release itself; ``tools/check_sot_release_equivalence.py`` verifies that
the two paths yield identical sequences and boxes.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .base import BaseVideoDataset, VideoInfo

#: Columns of ``groundtruth.txt``.
_GT_COLS = 15
#: Where the oriented-corner block starts.
_POLY_AT = 7
#: The tiny-object threshold, strict: a sequence is tiny when its median
#: sqrt(area) is *below* this. 238 of the 395 released sequences are.
TINY_PX = 8.0


class SpaceTrackerSOTDataset(BaseVideoDataset):
    """
    Space-Tracker-SOT loader.

    Args:
        root:       Release root -- the directory holding ``sot/`` and ``mot/``.
                    Defaults to ``$SPACE_TRACKER_ROOT``.
        split:      "train", "val", "test", or "no_split" (all 395 sequences).
        categories: Keep only sequences of these classes, e.g. ``["ship"]``.
        attributes: Keep only sequences carrying any of these taxonomy
                    attributes, e.g. ``["OCC", "SOB"]``.
        max_size:   Keep only sequences whose median sqrt(area) is at most
                    this many pixels. **Inclusive**; for the paper's tiny
                    subset use ``tiny=True``, which is strict.
        tiny:       Keep only the tiny regime, ``s < 8`` px -- 238 sequences.
                    Not the same as ``max_size=8``, which keeps 240: two
                    sequences sit at exactly 8.0 px.
        **kwargs:   Forwarded to BaseVideoDataset (mode, clip_len, transform, …).
    """

    def __init__(
        self,
        root: str | Path | None = None,
        split: str = "test",
        *,
        attributes: list[str] | None = None,
        max_size: float | None = None,
        tiny: bool = False,
        **kwargs,
    ):
        import os
        if root is None:
            root = os.environ.get("SPACE_TRACKER_ROOT", "")
            if not root:
                raise ValueError(
                    "no release root: pass root=... or set $SPACE_TRACKER_ROOT "
                    "to the directory holding sot/ and mot/.")
        # Must exist before super().__init__() calls _build_index().
        self._want_attrs = set(attributes) if attributes else None
        self._max_size = max_size
        self._tiny = tiny
        self._gt_cache: dict[str, np.ndarray] = {}
        self._attr_cache: dict[str, list[str]] = {}
        self._size_cache: dict[str, float] = {}
        self._rel_dir: dict[str, str] = {}
        self._has_poly: dict[str, bool] = {}
        self._ext: dict[str, str] = {}
        super().__init__(root=root, split=split, **kwargs)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        manifest_path = self.root / "sot" / "space_tracker_sot.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"{manifest_path} not found. `root` must be the release root -- "
                f"the directory holding sot/ and mot/.")
        manifest = json.loads(manifest_path.read_text())
        split_map = self._load_split_map()

        self._taxonomy = manifest["attribute_taxonomy"]["attributes"]
        self._pooled = [k for k, v in self._taxonomy.items() if v.get("pooled")]

        videos: list[VideoInfo] = []
        for rec in manifest["sequences"]:
            name = rec["name"]
            attrs = list(rec.get("taxonomy_attrs", []))
            if self._want_attrs is not None and not self._want_attrs & set(attrs):
                continue
            size = float(rec.get("median_sqrt_area_px", 0.0))
            if self._max_size is not None and size > self._max_size:
                continue
            # The tiny regime is strict: s < 8, as the paper defines it. Two
            # sequences sit at exactly 8.0 px, so `max_size=8` is not the same
            # subset and would report the tiny column over 240 rather than 238.
            if self._tiny and size >= TINY_PX:
                continue

            gt = self._parse_gt(self.root / rec["gt_path"])
            n = min(len(gt), int(rec["n_frames"]))
            if n == 0:
                continue

            videos.append(VideoInfo(
                video_id   = name,
                dataset    = "SpaceTrackerSOT",
                category   = rec["category"],
                split      = split_map.get(name, "no_split"),
                num_frames = n,
                frame_ids  = list(range(n)),
            ))
            self._gt_cache[name] = gt[:n]
            self._attr_cache[name] = attrs
            self._size_cache[name] = size
            self._rel_dir[name] = rec["path"]
            self._ext[name] = rec.get("img_ext", ".jpg")
            # -1 everywhere in the corner block means the source annotates no
            # orientation; checked once per sequence rather than per frame.
            self._has_poly[name] = bool((gt[:n, _POLY_AT:] >= 0).any())

        if self.split != "no_split":
            videos = [v for v in videos if v.split == self.split]
        self.videos = videos

    def _load_frame(self, video: VideoInfo, frame_id: int) -> np.ndarray:
        # Frames are 1-indexed on disk; frame_id is 0-based within the sequence.
        path = (self.root / self._rel_dir[video.video_id] / "img"
                / f"{frame_id + 1:06d}{self._ext[video.video_id]}")
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Frame not found: {path}")
        return img[..., ::-1].copy()  # BGR → RGB, contiguous

    def _load_annotations(self, video: VideoInfo, frame_id: int) -> dict:
        row = self._gt_cache[video.video_id][frame_id]
        if row[5] <= 0:  # visible == 0: the target is not in this frame
            return {
                "boxes":     np.zeros((0, 4), dtype=np.float32),
                "labels":    np.zeros(0, dtype=np.int64),
                "track_ids": np.zeros(0, dtype=np.int64),
            }
        x, y, w, h = row[1:5]
        out = {
            "boxes":     np.array([[x, y, x + w, y + h]], dtype=np.float32),
            "labels":    np.array([self._map_label(video.category)], dtype=np.int64),
            "track_ids": np.array([1], dtype=np.int64),  # SOT — single object
        }
        if self._has_poly[video.video_id]:
            out["obb"] = np.array([row[_POLY_AT:]], dtype=np.float32)
        return out

    # ------------------------------------------------------------------
    # Released metadata
    # ------------------------------------------------------------------

    def sequence_attributes(self) -> dict[str, list[str]]:
        """``{video_id: [attribute, ...]}`` for the videos in this split.

        The complete list the release carries: all 18 taxonomy attributes the
        sequence holds, plus any occlusion sub-type. Pooled and single-source
        alike -- which of them may be *scored across sources* is
        :meth:`pooled_attributes`, and is a property of the row, not of the
        sequence.
        """
        return {v.video_id: list(self._attr_cache[v.video_id]) for v in self.videos}

    def pooled_attributes(self) -> list[str]:
        """The five attributes more than one source annotates, so a score over
        them spans the benchmark rather than the one dataset that defined it."""
        return list(self._pooled)

    def sequence_sizes(self) -> dict[str, float]:
        """``{video_id: median sqrt(area) in px}``, as released."""
        return {v.video_id: self._size_cache[v.video_id] for v in self.videos}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_split_map(self) -> dict[str, str]:
        """Released sequence name → split, from the package's own splits.json."""
        for path in (self.root / "splits.json",
                     Path(__file__).resolve().parents[1] / "space_tracker" / "splits.json"):
            if path.exists():
                manifest = json.loads(path.read_text())
                # Keys are "sot/<name>"; the MOT half is not ours.
                return {k.split("/", 1)[1]: v
                        for k, v in manifest["splits"]["sot"].items()}
        return {}

    @staticmethod
    def _parse_gt(path: Path) -> np.ndarray:
        """Parse ``groundtruth.txt`` → ndarray ``(N, 15)`` float32."""
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                values = [float(v) for v in line.replace("\t", ",").split(",")]
                if len(values) != _GT_COLS:
                    raise ValueError(
                        f"{path}: expected {_GT_COLS} columns, got {len(values)}")
                rows.append(values)
        return np.array(rows, dtype=np.float32)
