"""
Shared image + annotation transforms for detection datasets.

Wraps albumentations so that transforms conform to the dataset interface:
    transform(image: np.ndarray, ann: dict) → (image, ann)

Boxes are in xyxy absolute pixel coordinates (pascal_voc format in albumentations).
"""

from __future__ import annotations

import albumentations as A
import numpy as np


class AlbumentationsWrapper:
    """
    Bridge between our dataset's (image, ann) interface and albumentations.

    Args:
        transforms: An albumentations Compose pipeline with
                    ``bbox_params=A.BboxParams(format='pascal_voc', ...)``.
    """

    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

    def __call__(
        self, image: np.ndarray, ann: dict
    ) -> tuple[np.ndarray, dict]:
        boxes = ann["boxes"]       # (N, 4) float32 xyxy
        labels = ann["labels"]     # (N,)   int64
        track_ids = ann["track_ids"]  # (N,) int64

        if len(boxes) == 0:
            result = self.transforms(image=image, bboxes=[], labels=[])
            return result["image"], ann

        result = self.transforms(
            image=image,
            bboxes=boxes.tolist(),
            labels=labels.tolist(),
        )

        out_boxes = np.array(result["bboxes"], dtype=np.float32).reshape(-1, 4)
        out_labels = np.array(result["labels"], dtype=np.int64)

        # albumentations may drop boxes that go out of bounds after augmentation;
        # keep only the track_ids that survived
        n_out = len(out_labels)
        out_track_ids = track_ids[:n_out] if n_out <= len(track_ids) else track_ids

        ann = {
            "boxes": out_boxes,
            "labels": out_labels,
            "track_ids": out_track_ids,
        }
        return result["image"], ann


def build_train_transform(img_size: tuple[int, int] = (640, 640)) -> AlbumentationsWrapper:
    """Training: resize + horizontal flip."""
    return AlbumentationsWrapper(
        A.Compose(
            [
                A.Resize(height=img_size[0], width=img_size[1]),
                A.HorizontalFlip(p=0.5),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["labels"],
                min_visibility=0.3,
            ),
        )
    )


def build_eval_transform(img_size: tuple[int, int] = (640, 640)) -> AlbumentationsWrapper:
    """Validation / test: resize only."""
    return AlbumentationsWrapper(
        A.Compose(
            [
                A.Resize(height=img_size[0], width=img_size[1]),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["labels"],
                min_visibility=0.3,
            ),
        )
    )


def build_satmot_train_transform(crop_size: int = 1024) -> AlbumentationsWrapper:
    """Cross-dataset satellite-MOT training transform.

    Designed for the LMOD/SAT-MTB/VISO/SDM-Car/AIRMOT mix: most objects are
    very small (LMOD avg-area ≈ 51 px² with 98.8% < 32×32), so we **crop**
    instead of resizing to preserve pixel area. PadIfNeeded handles frames
    smaller than ``crop_size``. Boxes whose post-augmentation visibility
    drops below ``min_visibility=0.1`` are dropped — a slightly looser
    threshold than the default 0.3 since small-object boxes are easy to
    clip.
    """
    return AlbumentationsWrapper(
        A.Compose(
            [
                A.PadIfNeeded(
                    min_height=crop_size, min_width=crop_size,
                    border_mode=0, fill=0,
                ),
                A.RandomCrop(height=crop_size, width=crop_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomBrightnessContrast(
                    brightness_limit=0.15, contrast_limit=0.15, p=0.3,
                ),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["labels"],
                min_visibility=0.1,
            ),
        )
    )


def build_satmot_eval_transform() -> AlbumentationsWrapper:
    """Cross-dataset satellite-MOT eval transform.

    No spatial transform — the FasterRCNN model's internal GeneralizedRCNN
    transform handles resizing to ``min_size``/``max_size`` while preserving
    aspect ratio. Boxes pass through unchanged.
    """
    return AlbumentationsWrapper(
        A.Compose(
            [],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["labels"],
                min_visibility=0.0,
            ),
        )
    )
