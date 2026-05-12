"""SOT manifest loader + filtering API.

The manifest (``space_tracker.json``, ~260 KB) catalogues every sequence in
SatSOT / SV248S / OOTB along with native and unified attribute labels and
sequence-level scale stats. Read once, query in memory.

For the multi-object-tracking companion, see :mod:`space_tracker.manifest_mot`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class SequenceRecord:
    """One row from ``manifest['sequences']`` with light typing."""
    id: str                          # "<dataset>/<video_id>" — globally unique
    dataset: str                     # "ootb" | "satsot" | "sv248s"
    video_id: str
    category: str
    n_frames: int
    image_dir: str                   # relative to the dataset root
    gt_path: str                     # relative to the dataset root
    gt_format: str                   # "obb_8pt" | "xywh_with_none" | "xywh_with_state"
    native_attrs: list[str]
    unified_attrs: list[str]         # 6-row shared taxonomy (BC/IV/ROT/OCC/SOB/DEF)
    taxonomy_attrs: list[str]        # full paper taxonomy — includes the 6
                                     # unified rows + aspect-ratio + dataset-unique
                                     # + occlusion sub-types (POC/FOC/STO/LTO/CO)
    median_sqrt_area_px: float | None
    tiny: bool

    @classmethod
    def from_dict(cls, d: dict) -> "SequenceRecord":
        return cls(
            id=d["id"],
            dataset=d["dataset"],
            video_id=d["video_id"],
            category=d["category"],
            n_frames=int(d["n_frames"]),
            image_dir=d["image_dir"],
            gt_path=d["gt_path"],
            gt_format=d["gt_format"],
            native_attrs=list(d.get("native_attrs", [])),
            unified_attrs=list(d.get("unified_attrs", [])),
            taxonomy_attrs=list(d.get("taxonomy_attrs", [])),
            median_sqrt_area_px=d.get("median_sqrt_area_px"),
            tiny=bool(d.get("tiny", False)),
        )


@dataclass
class Manifest:
    """In-memory representation of ``space_tracker.json``."""
    version: str
    description: str
    evaluation: dict
    datasets: dict
    unified_attributes: dict
    attribute_taxonomy: dict          # added in v1.1; full paper taxonomy
                                      # (groups + per-attribute specs)
    sequences: list[SequenceRecord] = field(default_factory=list)

    # ---------- I/O ----------

    @classmethod
    def load(cls, path: str | Path) -> "Manifest":
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        return cls(
            version=data["version"],
            description=data.get("description", ""),
            evaluation=data["evaluation"],
            datasets=data["datasets"],
            unified_attributes=data["unified_attributes"],
            attribute_taxonomy=data.get("attribute_taxonomy", {}),
            sequences=[SequenceRecord.from_dict(s) for s in data["sequences"]],
        )

    # ---------- queries ----------

    def filter(
        self,
        datasets: Iterable[str] | None = None,
        unified_attrs: Iterable[str] | None = None,
        native_attrs: Iterable[str] | None = None,
        taxonomy_attrs: Iterable[str] | None = None,
        tiny: bool | None = None,
        category: Iterable[str] | None = None,
        match: str = "any",
    ) -> list[SequenceRecord]:
        """Return sequences matching all provided filters.

        ``unified_attrs`` / ``native_attrs`` / ``taxonomy_attrs`` semantics are
        controlled by ``match``: ``"any"`` (default) → sequence matches if it
        carries any of the requested attributes; ``"all"`` → must carry every
        requested one.

        ``taxonomy_attrs`` accepts any name from the full paper taxonomy —
        the 6 unified rows (BC/IV/ROT/OCC/SOB/DEF), the aspect-ratio
        attributes (ARC/OON), the dataset-unique-other attributes
        (LQ/BJT/BCH/ND/IBG/SM/LT/MB/IM/AM), or the occlusion sub-types
        (POC/FOC/STO/LTO/CO). Drill down from a unified row (e.g. OCC)
        to its sub-types (e.g. POC, FOC) by switching this argument.
        """
        if match not in ("any", "all"):
            raise ValueError(f"match must be 'any' or 'all', got {match!r}")
        ds_set = set(datasets) if datasets else None
        u_set = set(unified_attrs) if unified_attrs else None
        n_set = set(native_attrs) if native_attrs else None
        t_set = set(taxonomy_attrs) if taxonomy_attrs else None
        c_set = set(category) if category else None

        out = []
        for s in self.sequences:
            if ds_set is not None and s.dataset not in ds_set:
                continue
            if c_set is not None and s.category not in c_set:
                continue
            if tiny is not None and bool(s.tiny) != bool(tiny):
                continue
            if u_set is not None:
                got = set(s.unified_attrs)
                if (match == "any" and not (u_set & got)) or \
                   (match == "all" and not u_set.issubset(got)):
                    continue
            if n_set is not None:
                got = set(s.native_attrs)
                if (match == "any" and not (n_set & got)) or \
                   (match == "all" and not n_set.issubset(got)):
                    continue
            if t_set is not None:
                got = set(s.taxonomy_attrs)
                if (match == "any" and not (t_set & got)) or \
                   (match == "all" and not t_set.issubset(got)):
                    continue
            out.append(s)
        return out

    def by_id(self) -> dict[str, SequenceRecord]:
        return {s.id: s for s in self.sequences}

    def datasets_annotating(self, unified_attr: str) -> list[str]:
        """Datasets whose native labels map into ``unified_attr``."""
        if unified_attr not in self.unified_attributes:
            raise KeyError(unified_attr)
        spec = self.unified_attributes[unified_attr]
        return [d for d, labels in spec["datasets"].items() if labels]

    # ---------- full-taxonomy helpers ----------

    def taxonomy_spec(self) -> dict[str, dict]:
        """Per-attribute spec dict from ``attribute_taxonomy['attributes']``."""
        return self.attribute_taxonomy.get("attributes", {})

    def taxonomy_groups(self) -> dict[str, dict]:
        """Group definitions from ``attribute_taxonomy['groups']``."""
        return self.attribute_taxonomy.get("groups", {})

    def datasets_annotating_taxonomy(self, attr: str) -> list[str]:
        """Datasets whose native labels map into ``attr`` under the paper taxonomy.

        ``attr`` may be any name from the full taxonomy (shared, aspect-ratio,
        dataset-unique-other, or occlusion sub-type). Useful when iterating
        sub-types of a unified row (e.g. ``OCC`` → ``[POC, FOC, STO, LTO, CO]``
        via ``taxonomy_spec()[\"OCC\"][\"subtypes\"]``).
        """
        spec = self.taxonomy_spec().get(attr)
        if spec is None:
            raise KeyError(attr)
        return [d for d, labels in spec.get("datasets", {}).items() if labels]

    def occlusion_subtypes(self) -> list[str]:
        """Convenience: the 5 occlusion sub-types (POC/FOC/STO/LTO/CO)."""
        return list(
            self.taxonomy_spec().get("OCC", {}).get(
                "subtypes",
                ["POC", "FOC", "STO", "LTO", "CO"],
            )
        )
