from .base import (
    BaseVideoDataset,
    VideoInfo,
    DetectionSample,
    VideoClipSample,
    detection_collate_fn,
    video_collate_fn,
)
from .lmod import LMODDataset
from .irsatvideo import IRSatVideoDataset
from .satsot import SatSOTDataset
from .satmtb import SATMTBDataset
from .airmot import AIRMOTDataset
from .viso import VISODataset
from .sv248s import SV248SDataset
from .sdmcar import SDMCarDataset
from .rscardata import RsCarDataset
# Use-case datasets (thermal wildlife, wildfire) are not part of this
# benchmark release; import them only when their loaders are present.
try:
    from .birdsai import BIRDSAIDataset
    from .birdsai_mot import BIRDSAIMOTDataset
    from .fire_rgbt import FireRGBTDataset
except ImportError:  # loaders kept out of the released tree
    BIRDSAIDataset = BIRDSAIMOTDataset = FireRGBTDataset = None
from .space_tracker_mot import SpaceTrackerMOTDataset

__all__ = [
    "BaseVideoDataset",
    "VideoInfo",
    "DetectionSample",
    "VideoClipSample",
    "detection_collate_fn",
    "video_collate_fn",
    "LMODDataset",
    "IRSatVideoDataset",
    "SatSOTDataset",
    "SATMTBDataset",
    "AIRMOTDataset",
    "VISODataset",
    "SV248SDataset",
    "SDMCarDataset",
    "RsCarDataset",
    "SpaceTrackerMOTDataset",
]
