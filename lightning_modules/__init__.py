from .module import ObjectDetectionModule
from .datamodule import DetectionDataModule, DataModuleConfig
from .sam2_datamodule import SAM2DataModule, SAM2DataModuleConfig
from .video_tracker_module import VideoTrackerEvaluationModule
from .visualization import DetectionVisualizationCallback, SAM2VisualizationCallback
from .sot_callback import SOTEvalCallback, SAM2SOTEvalCallback
from .mot_format_dump import MOTFormatDumpCallback
from .prediction_dump import VideoPredictionDumpCallback

# Backwards-compatible alias (legacy name used by eval_sam2*.py scripts)
SAM2EvaluationModule = VideoTrackerEvaluationModule

__all__ = [
    "ObjectDetectionModule",
    "DetectionDataModule",
    "DataModuleConfig",
    "SAM2DataModule",
    "SAM2DataModuleConfig",
    "VideoTrackerEvaluationModule",
    "SAM2EvaluationModule",  # legacy alias
    "DetectionVisualizationCallback",
    "SAM2VisualizationCallback",
    "SOTEvalCallback",
    "SAM2SOTEvalCallback",
    "MOTFormatDumpCallback",
    "VideoPredictionDumpCallback",
]
