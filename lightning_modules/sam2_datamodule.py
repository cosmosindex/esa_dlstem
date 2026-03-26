"""
SAM2DataModule
==============
Lightning DataModule for SAM2 video object segmentation / tracking.

SAM2 operates on video clips: frame 0 provides the prompt (GT bounding boxes),
and the remaining frames are used for mask propagation and evaluation.

Uses ``mode='video'`` from BaseVideoDataset, which returns VideoClipSample
with T consecutive frames per clip.

Usage::

    cfg = SAM2DataModuleConfig(
        datasets={"OOTB": "/data/OOTB"},
        class_map={"car": 0, "plane": 1, "ship": 2, "train": 3},
        clip_len=8,
    )
    dm = SAM2DataModule(cfg)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import lightning as L
from torch.utils.data import ConcatDataset, DataLoader

from datasets import video_collate_fn
from datasets.ootb import OOTBDataset
from datasets.birdsai import BIRDSAIDataset
from datasets.birdsai_mot import BIRDSAIMOTDataset
from datasets.lmod import LMODDataset

# ---------------------------------------------------------------------------
# Dataset registry — shared with DetectionDataModule; add new classes here
# ---------------------------------------------------------------------------

_DATASET_REGISTRY: dict[str, type] = {
    "OOTB": OOTBDataset,
    "BIRDSAI": BIRDSAIDataset,
    "BIRDSAI_MOT": BIRDSAIMOTDataset,
    "LMOD": LMODDataset,
}

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SAM2DataModuleConfig:
    """Configuration for SAM2DataModule."""

    # Mapping of dataset name → root path
    datasets: dict[str, str] = field(default_factory=dict)

    # Category → global class id
    class_map: dict[str, int] = field(default_factory=dict)

    # Clip parameters
    clip_len: int = 8
    clip_stride: int = 1
    clip_overlap: float = 0.0

    batch_size: int = 2
    num_workers: int = 4
    pin_memory: bool = True

    # Extra kwargs forwarded to every dataset constructor
    dataset_kwargs: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------


class SAM2DataModule(L.LightningDataModule):
    """
    Lightning DataModule for SAM2 video-level training and evaluation.

    Each sample is a VideoClipSample with T frames. Frame 0 carries the
    prompt boxes (passed to SAM2Tracker.add_prompts); frames 1..T-1 are
    used for propagation and metric computation.

    Args:
        cfg:             A SAM2DataModuleConfig instance.
        train_transform: Callable for training augmentation.
        eval_transform:  Callable for val/test preprocessing.
    """

    def __init__(
        self,
        cfg: SAM2DataModuleConfig,
        train_transform: Optional[Callable] = None,
        eval_transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.train_transform = train_transform
        self.eval_transform = eval_transform

        self.train_dataset: ConcatDataset | None = None
        self.val_dataset: ConcatDataset | None = None
        self.test_dataset: ConcatDataset | None = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, stage: str | None = None) -> None:
        shared = dict(
            class_map=self.cfg.class_map,
            mode="video",
            clip_len=self.cfg.clip_len,
            clip_stride=self.cfg.clip_stride,
            clip_overlap=self.cfg.clip_overlap,
            **self.cfg.dataset_kwargs,
        )

        if stage in ("fit", None):
            self.train_dataset = self._build_concat("train", transform=self.train_transform, **shared)
            self.val_dataset = self._build_concat("val", transform=self.eval_transform, **shared)

        if stage in ("test", None):
            self.test_dataset = self._build_concat("test", transform=self.eval_transform, **shared)

    def _build_concat(self, split: str, **kwargs) -> ConcatDataset:
        """Instantiate every configured dataset for the given split and concatenate."""
        parts = []
        for name, root in self.cfg.datasets.items():
            cls = _DATASET_REGISTRY.get(name)
            if cls is None:
                raise ValueError(
                    f"Unknown dataset '{name}'. "
                    f"Registered: {list(_DATASET_REGISTRY.keys())}"
                )
            parts.append(cls(root=root, split=split, **kwargs))
        return ConcatDataset(parts)

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            collate_fn=video_collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            collate_fn=video_collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            collate_fn=video_collate_fn,
        )
