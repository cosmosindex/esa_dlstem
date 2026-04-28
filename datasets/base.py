"""
BaseVideoDataset
================
Abstract base class for all video detection / tracking datasets.

Two modes
---------
mode='detection'
    Returns one frame at a time as a (image, target) pair.
    Used for training Faster RCNN, DINOv2+head, and YOLO.
    ByteTrack is applied at inference time on top of per-frame detections,
    so training does NOT require temporal ordering or track_ids.

mode='video'
    Returns a clip of T consecutive frames.
    Used for SAM2 (first-frame prompt → propagation) and for
    tracking evaluation where frame ordering matters.

Canonical annotation format (output of _load_annotations)
----------------------------------------------------------
All subclasses must convert their native format and return:
    boxes:     np.ndarray  [N, 4]  float32   xyxy, absolute pixel coords
    labels:    np.ndarray  [N]     int64     global class id (via _map_label)
    track_ids: np.ndarray  [N]     int64     per-object identity; -1 if unknown

collate functions
-----------------
detection_collate_fn  →  (list[Tensor], list[dict])
    Compatible with ObjectDetectionModule.training_step / validation_step.

video_collate_fn      →  list[VideoClipSample]
    Returns a plain Python list; each element is one clip.
    Stacking is left to the model since clip lengths may differ.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class VideoInfo:
    """Metadata for a single video / sequence."""
    video_id:   str
    dataset:    str              # e.g. "OOTB", "BIRDSAI"
    category:   str              # dominant category name (single class), e.g. "car"
    split:      str              # "train" / "val" / "test" / "no_split"
    num_frames: int
    frame_ids:  list[int]        # ordered list of usable frame indices
    # Optional: every distinct category present *anywhere* in the sequence,
    # for datasets where a single sequence can mix multiple classes (e.g.
    # LMOD's car+plane+ship+train seqs). Used by ``BaseVideoDataset``'s
    # ``categories`` filter so single-class evaluators can opt into
    # "sequence contains class X" semantics rather than "dominant == X".
    categories_present: tuple[str, ...] = ()


@dataclass
class DetectionSample:
    """Single-frame sample for detection training / evaluation."""
    image:     torch.Tensor      # [C, H, W]  float32  values in [0, 1]
    boxes:     torch.Tensor      # [N, 4]     float32  xyxy absolute pixels
    labels:    torch.Tensor      # [N]        int64    global class id
    track_ids: torch.Tensor      # [N]        int64    -1 when unavailable
    # ---- meta (not fed into model loss) ----
    video_id:  str
    frame_id:  int
    orig_size: tuple[int, int]   # (H, W) before any resize transform
    dataset:   str


@dataclass
class VideoClipSample:
    """Multi-frame clip sample for SAM2 and tracking evaluation."""
    frames:    torch.Tensor           # [T, C, H, W]  float32  [0, 1]
    boxes:     list[torch.Tensor]     # T × [N_t, 4]  xyxy per frame
    labels:    list[torch.Tensor]     # T × [N_t]     global class id
    track_ids: list[torch.Tensor]     # T × [N_t]     -1 when unavailable
    frame_ids: list[int]
    video_id:  str
    orig_size: tuple[int, int]
    dataset:   str
    category:  str = ""           # dominant/raw category name of the source video
    obb:       list[torch.Tensor] | None = None  # T × [N_t, 8] OBB corners (optional)


# ---------------------------------------------------------------------------
# Base dataset
# ---------------------------------------------------------------------------

class BaseVideoDataset(ABC, Dataset):
    """
    Abstract base for all video datasets.

    Subclasses must implement three methods:
        _build_index()          populate self.videos: list[VideoInfo]
        _load_frame()           return one frame as HxWxC uint8 numpy array
        _load_annotations()     return dict with boxes / labels / track_ids

    Args:
        root:           Path to the dataset root directory.
        split:          Which split to load ("train", "val", "test", "no_split").
        mode:           "detection" (single frame) or "video" (clip).
        clip_len:       Number of frames per clip (video mode only).
        clip_stride:    Frame step inside a clip, e.g. stride=2 skips every
                        other frame.  Total frames spanned = clip_len * stride.
        clip_overlap:   Fraction of overlap between consecutive clips [0, 1).
                        0.0 = non-overlapping clips.
        transform:      Callable(image: np.ndarray, ann: dict) → (image, ann).
                        Applies augmentation / resize in place; must keep the
                        annotation dict keys intact.
        class_map:      Dict mapping raw category name → global int class id.
                        Categories absent from the map are mapped to -1 and
                        filtered out in __getitem__.
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
    ):
        self.root         = Path(root)
        self.split        = split
        self.mode         = mode
        self.clip_len     = clip_len
        self.clip_stride  = clip_stride
        self.clip_overlap = clip_overlap
        self.transform    = transform
        self.class_map    = class_map or {}
        # If set, drop any video whose VideoInfo.category isn't in this list.
        # Useful when a single-class detector (e.g. HiEUM) should only see
        # sequences whose dominant class matches its training distribution.
        self.categories   = list(categories) if categories else None

        # Populated by _build_index()
        self.videos: list[VideoInfo] = []
        self._build_index()
        if self.categories is not None:
            wanted = set(self.categories)
            self.videos = [
                v for v in self.videos
                if (v.categories_present
                    and any(c in wanted for c in v.categories_present))
                or v.category in wanted
            ]

        if not self.videos:
            raise RuntimeError(
                f"{self.__class__.__name__}: no videos found for split='{split}' "
                f"at root='{root}'."
            )

        # Build flat index depending on mode
        if mode == "detection":
            # One entry per frame: (video_index, frame_id)
            self._frame_index: list[tuple[int, int]] = [
                (vi, fid)
                for vi, v in enumerate(self.videos)
                for fid in v.frame_ids
            ]
        elif mode == "video":
            self._clip_index: list[tuple[int, int]] = []
            self._build_clip_index()
        else:
            raise ValueError(f"mode must be 'detection' or 'video', got '{mode}'")

    # -----------------------------------------------------------------------
    # Abstract interface — subclasses must implement these three methods
    # -----------------------------------------------------------------------

    @abstractmethod
    def _build_index(self) -> None:
        """
        Parse the dataset directory and fill self.videos with VideoInfo entries.

        Called once in __init__ before any other method.
        """

    @abstractmethod
    def _load_frame(self, video: VideoInfo, frame_id: int) -> np.ndarray:
        """
        Load a single frame.

        Returns:
            np.ndarray of shape (H, W, 3), dtype uint8, RGB channel order.
        """

    @abstractmethod
    def _load_annotations(
        self, video: VideoInfo, frame_id: int
    ) -> dict[str, np.ndarray]:
        """
        Load annotations for one frame and convert to the canonical format.

        Returns a dict with exactly these keys:
            "boxes":     np.ndarray [N, 4]  float32  xyxy absolute pixels
            "labels":    np.ndarray [N]     int64    global class id
            "track_ids": np.ndarray [N]     int64    -1 if no tracking GT

        Use self._map_label(raw_name) to convert raw category names to global ids.
        Boxes for objects whose label maps to -1 should be excluded.
        """

    # -----------------------------------------------------------------------
    # Utility helpers available to subclasses
    # -----------------------------------------------------------------------

    def _map_label(self, raw_name: str) -> int:
        """Map a raw category string to a global class id; -1 if not in map."""
        return self.class_map.get(raw_name, -1)

    @staticmethod
    def _to_tensor(img: np.ndarray) -> torch.Tensor:
        """HxWxC uint8 → CxHxW float32 in [0, 1]."""
        return torch.from_numpy(
            np.ascontiguousarray(img.transpose(2, 0, 1))
        ).float() / 255.0

    @staticmethod
    def _obb_to_aabb(
        x1: float, y1: float,
        x2: float, y2: float,
        x3: float, y3: float,
        x4: float, y4: float,
    ) -> tuple[float, float, float, float]:
        """
        Convert an oriented bounding box (4 corner points) to an axis-aligned
        bounding box (xyxy).  Used by datasets with OBB annotations (e.g. OOTB).
        """
        xs = (x1, x2, x3, x4)
        ys = (y1, y2, y3, y4)
        return min(xs), min(ys), max(xs), max(ys)

    # -----------------------------------------------------------------------
    # Internal: clip index construction
    # -----------------------------------------------------------------------

    def _build_clip_index(self) -> None:
        """
        Build self._clip_index as (video_index, start_position_in_frame_ids).

        start_position is an index into video.frame_ids, not an absolute frame id.
        """
        # How far to advance between clip starts
        total_span = self.clip_len * self.clip_stride
        step = max(1, int(total_span * (1.0 - self.clip_overlap)))

        for vi, v in enumerate(self.videos):
            n = len(v.frame_ids)
            if n < total_span:
                # Video too short for a full clip — emit one clip starting at
                # frame 0; _get_clip_sample truncates to the frames that exist
                # (selected_fids filters `p < len(fids)`).
                if n > 0:
                    self._clip_index.append((vi, 0))
                continue
            for start in range(0, n - total_span + 1, step):
                self._clip_index.append((vi, start))

    # -----------------------------------------------------------------------
    # Dataset interface
    # -----------------------------------------------------------------------

    def __len__(self) -> int:
        if self.mode == "detection":
            return len(self._frame_index)
        return len(self._clip_index)

    def __getitem__(self, idx: int) -> DetectionSample | VideoClipSample:
        if self.mode == "detection":
            return self._get_detection_sample(idx)
        return self._get_clip_sample(idx)

    # -----------------------------------------------------------------------
    # Sample builders
    # -----------------------------------------------------------------------

    def _get_detection_sample(self, idx: int) -> DetectionSample:
        vi, fid = self._frame_index[idx]
        video   = self.videos[vi]

        img = self._load_frame(video, fid)         # (H, W, 3) uint8
        ann = self._load_annotations(video, fid)   # dict of ndarray
        orig_size = img.shape[:2]                  # (H, W)

        # Clip boxes to image bounds (OBB→AABB can slightly exceed)
        h, w = orig_size
        if "boxes" in ann and len(ann["boxes"]) > 0:
            ann["boxes"][:, [0, 2]] = np.clip(ann["boxes"][:, [0, 2]], 0, w)
            ann["boxes"][:, [1, 3]] = np.clip(ann["boxes"][:, [1, 3]], 0, h)

        if self.transform is not None:
            img, ann = self.transform(img, ann)

        # Drop objects with unknown class (-1)
        keep = ann["labels"] >= 0
        return DetectionSample(
            image     = self._to_tensor(img),
            boxes     = torch.as_tensor(ann["boxes"][keep],     dtype=torch.float32),
            labels    = torch.as_tensor(ann["labels"][keep],    dtype=torch.int64),
            track_ids = torch.as_tensor(ann["track_ids"][keep], dtype=torch.int64),
            video_id  = video.video_id,
            frame_id  = fid,
            orig_size = orig_size,
            dataset   = video.dataset,
        )

    def _get_clip_sample(self, idx: int) -> VideoClipSample:
        vi, start = self._clip_index[idx]
        video     = self.videos[vi]
        fids      = video.frame_ids

        # Select frames: start, start+stride, start+2*stride, ..., clip_len frames
        selected_positions = range(start, start + self.clip_len * self.clip_stride,
                                   self.clip_stride)
        selected_fids = [fids[p] for p in selected_positions if p < len(fids)]

        frames, boxes_list, labels_list, tids_list = [], [], [], []
        obb_list: list[torch.Tensor] = []
        has_obb = False
        orig_size = None

        for fid in selected_fids:
            img = self._load_frame(video, fid)
            ann = self._load_annotations(video, fid)

            if orig_size is None:
                orig_size = img.shape[:2]

            # Clip boxes to image bounds (OBB→AABB can slightly exceed)
            h, w = img.shape[:2]
            if "boxes" in ann and len(ann["boxes"]) > 0:
                ann["boxes"][:, [0, 2]] = np.clip(ann["boxes"][:, [0, 2]], 0, w)
                ann["boxes"][:, [1, 3]] = np.clip(ann["boxes"][:, [1, 3]], 0, h)

            if self.transform is not None:
                img, ann = self.transform(img, ann)

            keep = ann["labels"] >= 0
            frames.append(self._to_tensor(img))
            boxes_list.append(torch.as_tensor(ann["boxes"][keep],     dtype=torch.float32))
            labels_list.append(torch.as_tensor(ann["labels"][keep],   dtype=torch.int64))
            tids_list.append(torch.as_tensor(ann["track_ids"][keep],  dtype=torch.int64))

            if "obb" in ann:
                has_obb = True
                obb_list.append(torch.as_tensor(ann["obb"][keep], dtype=torch.float32))

        return VideoClipSample(
            frames    = torch.stack(frames),        # [T, C, H, W]
            boxes     = boxes_list,
            labels    = labels_list,
            track_ids = tids_list,
            frame_ids = selected_fids,
            video_id  = video.video_id,
            orig_size = orig_size,
            dataset   = video.dataset,
            category  = video.category,
            obb       = obb_list if has_obb else None,
        )

    # -----------------------------------------------------------------------
    # Repr
    # -----------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"split={self.split!r}, "
            f"mode={self.mode!r}, "
            f"videos={len(self.videos)}, "
            f"samples={len(self)})"
        )


# ---------------------------------------------------------------------------
# collate functions
# ---------------------------------------------------------------------------

def detection_collate_fn(
    batch: list[DetectionSample],
) -> tuple[list[torch.Tensor], list[dict]]:
    """
    Collate a list of DetectionSamples into the format expected by
    ObjectDetectionModule: (images, targets).

        images:  list[Tensor]  each [C, H, W]  — kept as a list because
                 images may have different spatial sizes before padding.
        targets: list[dict]    each dict has:
                     'boxes'     Tensor [N, 4]  xyxy float32
                     'labels'    Tensor [N]     int64
                     'track_ids' Tensor [N]     int64
                     'video_id'  str
                     'frame_id'  int
                     'orig_size' tuple[int, int]
                     'dataset'   str
    """
    images  = [s.image for s in batch]
    targets = [
        {
            "boxes":     s.boxes,
            "labels":    s.labels,
            "track_ids": s.track_ids,
            "video_id":  s.video_id,
            "frame_id":  s.frame_id,
            "orig_size": s.orig_size,
            "dataset":   s.dataset,
        }
        for s in batch
    ]
    return images, targets


def video_collate_fn(
    batch: list[VideoClipSample],
) -> list[VideoClipSample]:
    """
    Collate for video mode.

    Clips in a batch may have different numbers of objects per frame, so
    stacking into a single tensor is not straightforward.  This collate
    simply returns the list as-is and lets the model handle batching.

    If all clips have the same spatial size, the model can call
        torch.stack([s.frames for s in batch])   → [B, T, C, H, W]
    after receiving the list.
    """
    return batch
