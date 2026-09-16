"""Dataset roots and manifest lookup, shared by every module in the tool.

The sequences come from the **released package**, read at start-up from
``$SPACE_TRACKER_ROOT``. Nothing here depends on the source-side index the build
pipeline used to emit (``space_tracker/space_tracker_mot.json``): that file
catalogued 491 sequences in their original, mutually incompatible on-disk
formats, it is not part of the release, and keeping it as the tool's input meant
the tool could not be run by anyone who had only downloaded the benchmark.

Reading the release instead also settles the formats: one MOTChallenge CSV per
sequence, frames on disk, 1-indexed throughout, so the per-source parsers are no
longer reached. The only thing still read from a source dataset is SAT-MTB's
per-frame detection XML (:func:`satmtb_det_dir`), which no release field can
replace because it is a *second annotation* of the same video rather than a
different encoding of the released one.

Sequence ids stay the source-side ``"<dataset>/<video_id>"``, which the release
carries as ``source_sequence_id``, so decisions recorded against an earlier run
still resolve.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from space_tracker.manifest_mot import MOTManifest, MOTSequenceRecord
from project_paths import DATA_ROOT

REPO = Path(__file__).resolve().parents[2]

#: The released benchmark: the directory holding ``sot/`` and ``mot/``.
RELEASE = Path(os.environ.get("SPACE_TRACKER_ROOT",
                              f"{DATA_ROOT}/release/space_tracker"))

#: Source datasets in their original layout. Needed only by
#: :func:`satmtb_det_dir`; every other read goes to :data:`RELEASE`.
DATA = Path(os.environ.get("SPACE_TRACKER_DATA", f"{DATA_ROOT}/data/trafic"))
SOURCE_ROOTS = {
    "satmtb":    DATA / "SAT-MTB",
    "viso":      DATA / "VISO",
    "sdmcar":    DATA / "SDM-Car",
    "rscardata": DATA / "RsCarData",
}

DET_VS_MOT_CSV = REPO / "space_tracker" / "data" / "satmtb_det_vs_mot.csv"
SIZE_SPLIT = REPO / "space_tracker" / "data" / "size_split.json"

#: Where human decisions are written. One file, git-friendly, never touches raw
#: data. Keyed by sequence — see :mod:`.vdecisions`. Set ``SPACE_TRACKER_REVIEW``
#: to keep it somewhere else; it is local state, not a release artifact.
DEFAULT_OVERRIDES = Path(os.environ.get(
    "SPACE_TRACKER_REVIEW", REPO / "docs" / "annotation_review" / "review.json"))

CATEGORIES = ["car", "airplane", "ship", "train"]

#: Every sequence in the release is written in this one format.
RELEASE_GT_FORMAT = "mot_csv_9col"

#: The released MOT manifest. The tool's single source of sequences.
MOT_MANIFEST = RELEASE / "mot" / "space_tracker_mot.json"


def satmtb_det_dir(video_id: str) -> Path:
    """Per-frame detection XML directory for a SAT-MTB sequence (``"airplane/28"``)."""
    cat_dir, num = video_id.split("/", 1)
    return SOURCE_ROOTS["satmtb"] / "SAT-MTB_Dataset" / cat_dir / num / "det" / "HBB"


def _record(r: dict, splits: dict[str, str]) -> MOTSequenceRecord:
    """One released sequence, described the way the tool's readers expect."""
    ds = r["dataset"]
    sid = r["source_sequence_id"]
    return MOTSequenceRecord(
        id=sid,
        dataset=ds,
        video_id=sid[len(ds) + 1:] if sid.startswith(ds + "/") else sid,
        category=r["category"],
        categories_in_seq=list(r.get("categories_in_sequence", [])),
        n_frames=int(r["n_frames"]),
        n_tracks=int(r.get("n_tracks", 0)),
        img_width=int(r.get("img_width", 0)),
        img_height=int(r.get("img_height", 0)),
        image_format="frames",
        image_path_pattern=r["image_path_pattern"],
        video_path=None,
        gt_path=r["gt_path"],
        gt_path_override=None,
        gt_format=RELEASE_GT_FORMAT,
        frame_index_base=1,
        split=splits.get(f"mot/{r['name']}", "no_split"),
    )


@lru_cache(maxsize=1)
def manifest() -> MOTManifest:
    """The MOT half of the release, as a :class:`MOTManifest`."""
    path = MOT_MANIFEST
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Point $SPACE_TRACKER_ROOT at the unpacked "
            f"Space-Tracker release (the directory holding sot/ and mot/).")
    doc = json.loads(path.read_text())

    splits_path = RELEASE / "splits.json"
    if not splits_path.exists():
        splits_path = REPO / "space_tracker" / "splits.json"
    splits = json.loads(splits_path.read_text())["splits"]["mot"] \
        if splits_path.exists() else {}
    splits = {f"mot/{k.split('/', 1)[-1]}": v for k, v in splits.items()}

    return MOTManifest(
        version=doc.get("version", ""),
        task="mot",
        description=doc.get("description", ""),
        evaluation=doc.get("evaluation", {}),
        categories=doc.get("categories", {}),
        datasets=doc.get("source_datasets", {}),
        sequences=[_record(r, splits) for r in doc["sequences"]],
    )


@lru_cache(maxsize=1)
def _by_id() -> dict[str, MOTSequenceRecord]:
    return {s.id: s for s in manifest().sequences}


def sequence_by_id(seq_id: str) -> MOTSequenceRecord:
    return _by_id()[seq_id]


def sequence_root(seq: MOTSequenceRecord) -> Path:
    """Root the sequence's ``gt_path`` and ``image_path_pattern`` resolve against.

    One root for every dataset, because the release normalised them into one
    tree. Kept as a function so a caller reads intent rather than a constant.
    """
    return RELEASE


def frame_path(seq: MOTSequenceRecord, frame_id: int) -> Path:
    """Absolute path to one frame image."""
    if seq.image_path_pattern is None:
        raise ValueError(f"sequence {seq.id} has no frames on disk")
    return sequence_root(seq) / seq.image_path_pattern.format(frame_id=frame_id)
