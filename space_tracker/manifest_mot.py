"""MOT manifest loader + filtering API.

The manifest (``space_tracker_mot.json``, ~280 KB) catalogues every sequence
in AIRMOT / SAT-MTB / VISO(non-car) / SDM-Car / RsCarData along with each
dataset's native GT format tag, image-source mode, resolution, track count,
and official split assignment. Read once, query in memory.

For the single-object-tracking companion, see :mod:`space_tracker.manifest`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class MOTSequenceRecord:
    """One row from ``manifest['sequences']`` with light typing."""
    id: str                           # "<dataset>/<video_id>" — globally unique
    dataset: str                      # "airmot" | "satmtb" | "viso" | "sdmcar" | "rscardata"
    video_id: str
    category: str                     # "car" | "airplane" | "ship" | "train" | "mixed"
    categories_in_seq: list[str]
    n_frames: int
    n_tracks: int
    img_width: int
    img_height: int
    image_format: str                 # "frames" | "video"
    image_path_pattern: str | None    # e.g. "5/img/{frame_id:06d}.jpg" — None for video mode
    video_path: str | None            # e.g. "train/1-1.avi" — None for frames mode
    gt_path: str
    gt_path_override: str | None
    gt_format: str                    # see data_mot._parse_gt
    frame_index_base: int             # 0 (SDM-Car) or 1 (everyone else)
    split: str                        # "train" | "val" | "test" | "no_split"

    @classmethod
    def from_dict(cls, d: dict) -> "MOTSequenceRecord":
        return cls(
            id=d["id"],
            dataset=d["dataset"],
            video_id=d["video_id"],
            category=d["category"],
            categories_in_seq=list(d.get("categories_in_seq", [])),
            n_frames=int(d["n_frames"]),
            n_tracks=int(d.get("n_tracks", 0)),
            img_width=int(d.get("img_width", 0)),
            img_height=int(d.get("img_height", 0)),
            image_format=d["image_format"],
            image_path_pattern=d.get("image_path_pattern"),
            video_path=d.get("video_path"),
            gt_path=d["gt_path"],
            gt_path_override=d.get("gt_path_override"),
            gt_format=d["gt_format"],
            frame_index_base=int(d.get("frame_index_base", 1)),
            split=d.get("split", "no_split"),
        )


@dataclass
class MOTManifest:
    """In-memory representation of ``space_tracker_mot.json``."""
    version: str
    task: str
    description: str
    evaluation: dict
    categories: dict
    datasets: dict
    sequences: list[MOTSequenceRecord] = field(default_factory=list)

    # ---------- I/O ----------

    @classmethod
    def load(cls, path: str | Path) -> "MOTManifest":
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        return cls(
            version=data["version"],
            task=data.get("task", "mot"),
            description=data.get("description", ""),
            evaluation=data["evaluation"],
            categories=data["categories"],
            datasets=data["datasets"],
            sequences=[MOTSequenceRecord.from_dict(s) for s in data["sequences"]],
        )

    # ---------- queries ----------

    def filter(
        self,
        datasets: Iterable[str] | None = None,
        categories: Iterable[str] | None = None,
        splits: Iterable[str] | None = None,
        match: str = "any",
    ) -> list[MOTSequenceRecord]:
        """Return sequences matching all provided filters.

        ``categories`` matches against ``categories_in_seq`` (the actual set of
        classes present in the sequence), not the dominant ``category`` tag —
        so passing ``categories=['car']`` will include SAT-MTB sequences whose
        primary class is airplane but which also contain cars, when ``match``
        is ``"any"``. Use ``match='all'`` to require every requested category.
        """
        if match not in ("any", "all"):
            raise ValueError(f"match must be 'any' or 'all', got {match!r}")
        ds_set    = set(datasets)   if datasets   else None
        cat_set   = set(categories) if categories else None
        split_set = set(splits)     if splits     else None

        out: list[MOTSequenceRecord] = []
        for s in self.sequences:
            if ds_set is not None and s.dataset not in ds_set:
                continue
            if split_set is not None and s.split not in split_set:
                continue
            if cat_set is not None:
                got = set(s.categories_in_seq)
                if (match == "any" and not (cat_set & got)) or \
                   (match == "all" and not cat_set.issubset(got)):
                    continue
            out.append(s)
        return out

    def by_id(self) -> dict[str, MOTSequenceRecord]:
        return {s.id: s for s in self.sequences}

    def datasets_with(self, category: str) -> list[str]:
        """Datasets that annotate ``category`` natively."""
        if category not in self.categories:
            raise KeyError(category)
        return list(self.categories[category]["datasets"])
