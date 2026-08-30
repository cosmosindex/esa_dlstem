"""
Space-Tracker MOT release, exposed as a detection / video dataset.

Reads the built release directly (MOTChallenge layout plus the COCO-VID
`space_tracker_mot.json` index), so training and evaluation consume exactly the
artifact we publish rather than the internal source datasets.

Two filters matter and neither is cosmetic:

**Split.** The release ships scene-disjoint train/val/test splits (283/40/80
sequences) that are not yet stamped into the annotation JSON, so they are read
from the split manifest (`docs/space_tracker/splits.csv` by default). Sequences
sharing a parent scene are kept together, which is the whole point of that file.

**Annotation completeness.** Outside SAT-MTB, the MOT ground truth labels only
*moving* objects, and even inside SAT-MTB only the 139 non-car sequences that
ship detection XML were completed with static targets (+842 tracks / +200,821
boxes). Training a single-frame detector on a movers-only sequence is actively
harmful: a parked aircraft is an unlabelled positive, so the model is taught
that the very thing it must find is background. `complete_only=True` (the
default for detector training) keeps just the completed sequences. Note that
`car` is movers-only *by design* across the whole benchmark and can never be
used to train a single-frame detector -- use a moving-object detector for the
car half instead. See `docs/static_annotation/README.md`.
"""

from __future__ import annotations

import configparser
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Literal, Optional

import cv2
import numpy as np

from .base import BaseVideoDataset, VideoInfo

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SPLITS = _REPO_ROOT / "docs" / "space_tracker" / "splits.csv"
_DEFAULT_MERGE_RECORD = _REPO_ROOT / "docs" / "static_annotation" / "merge_det_to_mot.csv"


class SpaceTrackerMOTDataset(BaseVideoDataset):
    """
    Args:
        root:          Release root (the directory holding `mot/` and `sot/`).
        split:         "train" / "val" / "test", or "no_split" for everything.
        complete_only: Keep only sequences whose annotation was completed with
                       static objects. Required for detector training.
        splits_csv:    Split manifest; defaults to the in-repo one.
        merge_csv:     Record of which sequences had detection XML merged in.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        mode: Literal["detection", "video"] = "detection",
        clip_len: int = 8,
        clip_stride: int = 1,
        clip_overlap: float = 0.0,
        transform: Optional[Callable] = None,
        class_map: Optional[dict[str, int]] = None,
        categories: Optional[list[str]] = None,
        complete_only: bool = True,
        splits_csv: str | Path | None = None,
        merge_csv: str | Path | None = None,
        frame_stride: int = 1,
    ):
        self.complete_only = complete_only
        self.frame_stride = max(1, int(frame_stride))
        self.splits_csv = Path(splits_csv) if splits_csv else _DEFAULT_SPLITS
        self.merge_csv = Path(merge_csv) if merge_csv else _DEFAULT_MERGE_RECORD
        # frame-level annotation cache, filled by _build_index
        self._ann: dict[str, dict[int, list[tuple[list[float], int, int]]]] = {}
        self._seq_dir: dict[str, Path] = {}
        self._img_ext: dict[str, str] = {}
        super().__init__(
            root=root, split=split, mode=mode, clip_len=clip_len,
            clip_stride=clip_stride, clip_overlap=clip_overlap,
            transform=transform, class_map=class_map, categories=categories,
        )
        # Subsampling applies to TRAINING only. val/test must stay complete or
        # the score changes meaning (and the best-checkpoint choice gets noisy),
        # so the same per_dataset_kwargs can be passed to every split safely.
        if self.split != "train":
            self.frame_stride = 1
        if self.frame_stride > 1 and mode == "detection":
            self.set_epoch(0)

    # ------------------------------------------------------------------
    # Frame subsampling
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """
        Keep one frame in every `frame_stride`, rotating the phase each epoch.

        Consecutive frames of a satellite sequence are near-identical -- a parked
        aircraft is literally the same pixels every frame -- so training on all
        of them costs a full epoch of time for almost no new signal, and makes it
        easy for the model to memorise the handful of scenes instead of
        generalising. Sampling one frame in N cuts the epoch by N while the
        rotating phase means every frame is still reached across the run.

        The per-video count is held FIXED (n // stride, at least 1) so the
        dataset length never changes between epochs; otherwise the already-built
        DataLoader sampler and the dataset would disagree mid-fit.
        """
        if self.frame_stride <= 1 or self.mode != "detection" or self.split != "train":
            return
        off = epoch % self.frame_stride
        index: list[tuple[int, int]] = []
        for vi, v in enumerate(self.videos):
            fids = v.frame_ids
            n = len(fids)
            if n == 0:
                continue
            keep = max(1, n // self.frame_stride)
            for j in range(keep):
                index.append((vi, fids[(j * self.frame_stride + off) % n]))
        self._frame_index = index

    # ------------------------------------------------------------------

    def _load_split_map(self) -> dict[str, str]:
        """release sequence name → split."""
        out: dict[str, str] = {}
        if not self.splits_csv.exists():
            return out
        with open(self.splits_csv) as f:
            for row in csv.DictReader(f):
                if row.get("half") != "mot":
                    continue
                # `sequence` is "mot/<name>"
                out[row["sequence"].split("/", 1)[1]] = row["split"]
        return out

    def _load_completed_source_ids(self) -> set[str]:
        """source_sequence_id values whose GT was completed with static objects."""
        out: set[str] = set()
        if not self.merge_csv.exists():
            return out
        with open(self.merge_csv) as f:
            for row in csv.DictReader(f):
                if row.get("has_det_xml") == "True" and row.get("category") != "car":
                    out.add(row["seq_id"])
        return out

    def _build_index(self) -> None:
        ann_path = self.root / "mot" / "annotations" / "space_tracker_mot.json"
        data = json.loads(ann_path.read_text())

        cats = {c["id"]: c["name"] for c in data["categories"]}
        split_map = self._load_split_map()
        completed = self._load_completed_source_ids() if self.complete_only else None

        # video id → record, applying split + completeness filters up front
        keep_vids: dict[int, dict] = {}
        for v in data["videos"]:
            if completed is not None and v["source_sequence_id"] not in completed:
                continue
            seq_split = split_map.get(v["name"], "no_split")
            if self.split != "no_split" and seq_split != self.split:
                continue
            keep_vids[v["id"]] = v

        # images: video id → {frame index → file name}
        frames: dict[int, dict[int, str]] = defaultdict(dict)
        img_meta: dict[int, tuple[int, int]] = {}
        for im in data["images"]:
            if im["video_id"] not in keep_vids:
                continue
            frames[im["video_id"]][int(im["frame_id"])] = im.get("file_name", "")
            img_meta[im["id"]] = (im["video_id"], int(im["frame_id"]))

        # annotations, dropping any class outside class_map (e.g. car)
        per_video: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
        for a in data["annotations"]:
            loc = img_meta.get(a["image_id"])
            if loc is None:
                continue
            vid_id, frame_id = loc
            name = cats.get(a["category_id"], "")
            gid = self._map_label(name)
            if gid < 0:
                continue
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            vname = keep_vids[vid_id]["name"]
            per_video[vname][frame_id].append(
                ([x, y, x + w, y + h], gid, int(a.get("track_id", -1)))
            )

        for vid_id, v in keep_vids.items():
            vname = v["name"]
            seq_dir = self.root / "mot" / v["category"] / vname
            if not seq_dir.is_dir():
                continue
            ext = ".png"
            ini = seq_dir / "seqinfo.ini"
            if ini.exists():
                cp = configparser.ConfigParser()
                cp.read(ini)
                ext = cp.get("Sequence", "imExt", fallback=".png")
            frame_ids = sorted(frames.get(vid_id, {}))
            if not frame_ids:
                continue
            self._seq_dir[vname] = seq_dir
            self._img_ext[vname] = ext
            self._ann[vname] = per_video.get(vname, {})
            self.videos.append(VideoInfo(
                video_id=vname,
                dataset="Space-Tracker-MOT",
                category=v["category"],
                split=split_map.get(vname, "no_split"),
                num_frames=len(frame_ids),
                frame_ids=frame_ids,
                categories_present=tuple(v.get("categories_in_sequence", []) or ()),
            ))

    # ------------------------------------------------------------------

    def _load_frame(self, video: VideoInfo, frame_id: int) -> np.ndarray:
        seq_dir = self._seq_dir[video.video_id]
        path = seq_dir / "img1" / f"{frame_id:06d}{self._img_ext[video.video_id]}"
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"frame not readable: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _load_annotations(self, video: VideoInfo, frame_id: int) -> dict:
        recs = self._ann.get(video.video_id, {}).get(frame_id, [])
        if recs:
            boxes = np.array([r[0] for r in recs], dtype=np.float32)
            labels = np.array([r[1] for r in recs], dtype=np.int64)
            track_ids = np.array([r[2] for r in recs], dtype=np.int64)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
            track_ids = np.zeros((0,), dtype=np.int64)
        return {"boxes": boxes, "labels": labels, "track_ids": track_ids}
