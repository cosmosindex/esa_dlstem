"""Access to geometry produced by ``tools/refine_satmtb_sam3.py``.

That script rewrites every box against the imagery and writes one JSON per
sequence. Both the review UI and the exporter read it through here, so they
cannot disagree about precedence: geometry decided by hand in the UI outranks
the batch pass, which outranks the raw source annotation.

The directory is not baked in — pass ``--refined-dir`` or set
``SPACE_TRACKER_REFINED``. With neither, nothing is loaded and every view falls
back to the original boxes.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

#: ``{track key id: {frame id: xyxy}}`` for one sequence.
Geometry = dict[str, dict[int, list[float]]]


def default_dir() -> Path | None:
    d = os.environ.get("SPACE_TRACKER_REFINED")
    return Path(d) if d else None


@lru_cache(maxsize=8)
def load_refined(refined_dir: str | None, seq_id: str) -> Geometry:
    """Batch-refined boxes for one sequence, empty when none exist.

    Keyed by string rather than Path so the cache key stays hashable and stable.
    """
    if not refined_dir:
        return {}
    path = Path(refined_dir) / (seq_id.replace("/", "_") + ".json")
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return {k: {int(f): [float(v) for v in b] for f, b in fb.items()}
            for k, fb in doc.get("refined", {}).items()}


def stats_for(refined_dir: str | None, seq_id: str) -> dict:
    """Guard counters and area ratio for one sequence, summed over sources.

    Counts add across sources; the area ratio is averaged, since a median of
    medians is not a median and pretending otherwise would misreport it.
    """
    if not refined_dir:
        return {}
    path = Path(refined_dir) / (seq_id.replace("/", "_") + ".json")
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text()).get("stats", {})
    except (OSError, ValueError):
        return {}
    if not raw:
        return {}
    # Older files store one flat dict; newer ones key it by source.
    per_source = [raw] if "boxes" in raw else list(raw.values())

    out: dict = {}
    ratios = []
    for s in per_source:
        for k, v in s.items():
            if k == "median_area_ratio":
                if v:
                    ratios.append(v)
            elif isinstance(v, (int, float)):
                out[k] = out.get(k, 0) + v
    out["median_area_ratio"] = sum(ratios) / len(ratios) if ratios else 0.0
    out["sources"] = len(per_source)
    return out
