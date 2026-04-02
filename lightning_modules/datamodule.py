"""
DetectionDataModule
====================
A general-purpose Lightning DataModule for object detection.

Supports any combination of datasets that subclass BaseVideoDataset.
New datasets only need to be registered in ``_DATASET_REGISTRY``.

Usage::

    cfg = DataModuleConfig(
        datasets={"OOTB": "/data/OOTB", "SatSOT": "/data/SatSOT"},
        class_map={"car": 0, "plane": 1, "ship": 2, "train": 3},
        batch_size=8,
    )
    dm = DetectionDataModule(cfg)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import lightning as L
from torch.utils.data import ConcatDataset, DataLoader

from datasets import detection_collate_fn
from datasets.ootb import OOTBDataset
from datasets.birdsai import BIRDSAIDataset
from datasets.birdsai_mot import BIRDSAIMOTDataset
from datasets.lmod import LMODDataset
from datasets.irsatvideo import IRSatVideoDataset
from datasets.satsot import SatSOTDataset
from datasets.satmtb import SATMTBDataset
from datasets.airmot import AIRMOTDataset
from datasets.viso import VISODataset
from datasets.sv248s import SV248SDataset

# ---------------------------------------------------------------------------
# Dataset registry — add new dataset classes here
# ---------------------------------------------------------------------------

_DATASET_REGISTRY: dict[str, type] = {
    "OOTB": OOTBDataset,
    "BIRDSAI": BIRDSAIDataset,
    "BIRDSAI_MOT": BIRDSAIMOTDataset,
    "LMOD": LMODDataset,
    "IRSatVideo-LEO": IRSatVideoDataset,
    "SatSOT": SatSOTDataset,
    "SAT-MTB": SATMTBDataset,
    "AIR-MOT": AIRMOTDataset,
    "VISO": VISODataset,
    "SV248S": SV248SDataset,
}

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DataModuleConfig:
    """Configuration for DetectionDataModule."""

    # Mapping of dataset name → root path, e.g. {"OOTB": "/data/OOTB"}
    datasets: dict[str, str] = field(default_factory=dict)

    # Category → global class id
    class_map: dict[str, int] = field(default_factory=dict)

    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True

    # Image size for transforms (height, width)
    img_size: tuple[int, int] = (640, 640)

    # Extra kwargs forwarded to every dataset constructor
    dataset_kwargs: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------


class DetectionDataModule(L.LightningDataModule):
    """
    Lightning DataModule that builds train / val / test DataLoaders
    from one or more detection datasets.

    Args:
        cfg:       A DataModuleConfig instance.
        transform: Optional callable ``(image, ann) → (image, ann)``.
                   If None, no augmentation is applied.
    """

    def __init__(
        self,
        cfg: DataModuleConfig,
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
            mode="detection",
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
            collate_fn=detection_collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            collate_fn=detection_collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            collate_fn=detection_collate_fn,
        )
