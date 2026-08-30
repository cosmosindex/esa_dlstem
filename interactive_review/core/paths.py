"""Dataset roots and manifest lookup, shared by every module in the tool.

Kept in one place so the review tool can be pointed at a relocated dataset by
setting ``SPACE_TRACKER_DATA`` instead of editing code.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from space_tracker.manifest_mot import MOTManifest, MOTSequenceRecord

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("SPACE_TRACKER_DATA", "/data/ESA_DLSTEM_2025/data/trafic"))

MOT_ROOTS = {
    "airmot":    DATA / "AIR-MOT-100",
    "satmtb":    DATA / "SAT-MTB",
    "viso":      DATA / "VISO",
    "sdmcar":    DATA / "SDM-Car",
    "rscardata": DATA / "RsCarData",
}

MOT_MANIFEST = REPO / "space_tracker" / "space_tracker_mot.json"
DET_VS_MOT_CSV = REPO / "space_tracker" / "data" / "satmtb_det_vs_mot.csv"
SIZE_SPLIT = REPO / "space_tracker" / "data" / "size_split.json"

#: Where human decisions are written. One file, git-friendly, never touches raw
#: data. Keyed by sequence — see :mod:`.vdecisions`.
DEFAULT_OVERRIDES = REPO / "docs" / "annotation_review" / "review.json"

CATEGORIES = ["car", "airplane", "ship", "train"]


def satmtb_det_dir(video_id: str) -> Path:
    """Per-frame detection XML directory for a SAT-MTB sequence (``"airplane/28"``)."""
    cat_dir, num = video_id.split("/", 1)
    return MOT_ROOTS["satmtb"] / "SAT-MTB_Dataset" / cat_dir / num / "det" / "HBB"


@lru_cache(maxsize=1)
def manifest() -> MOTManifest:
    return MOTManifest.load(MOT_MANIFEST)


@lru_cache(maxsize=1)
def _by_id() -> dict[str, MOTSequenceRecord]:
    return {s.id: s for s in manifest().sequences}


def sequence_by_id(seq_id: str) -> MOTSequenceRecord:
    return _by_id()[seq_id]


def frame_path(seq: MOTSequenceRecord, frame_id: int) -> Path:
    """Absolute path to one frame image."""
    if seq.image_path_pattern is None:
        raise ValueError(f"sequence {seq.id} is video-backed, not frames")
    return MOT_ROOTS[seq.dataset] / seq.image_path_pattern.format(frame_id=frame_id)
