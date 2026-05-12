"""Space-tracker: unified satellite-video tracking benchmark.

Two parallel evaluation surfaces:

* **SOT** — single-object tracking across SatSOT / SV248S / OOTB. Manifest:
  ``space_tracker/space_tracker.json`` (the original NeurIPS 2026 submission
  artifact; filename kept stable for citation / Croissant URL integrity).
  Use :class:`Benchmark`.
* **MOT** — multi-object tracking across AIRMOT / SAT-MTB / VISO(non-car) /
  SDM-Car / RsCarData. Manifest: ``space_tracker/space_tracker_mot.json``.
  Use :class:`MOTBenchmark`.

Quick start (SOT)::

    from space_tracker import Benchmark

    bench = Benchmark.load(
        manifest="space_tracker/space_tracker.json",
        dataset_roots={
            "ootb":   "/path/to/OOTB",
            "satsot": "/path/to/SatSOT",
            "sv248s": "/path/to/SV248S",
        },
    )
    for seq in bench.filter(unified_attrs=["OCC"], datasets=["sv248s"]):
        for frame in bench.frames(seq):
            ...                      # frame.image_path / frame.gt_box_xyxy

Quick start (MOT)::

    from space_tracker import MOTBenchmark

    bench = MOTBenchmark.load(
        manifest="space_tracker/space_tracker_mot.json",
        dataset_roots={
            "airmot":    "/path/to/AIR-MOT-100",
            "satmtb":    "/path/to/SAT-MTB",
            "viso":      "/path/to/VISO",
            "sdmcar":    "/path/to/SDM-Car",
            "rscardata": "/path/to/RsCarData",
        },
    )
    for seq in bench.filter(categories=["car"], splits=["test"]):
        for frame in bench.frames(seq):
            ...                      # frame.image / frame.objects (list[MOTObject])
"""

from .manifest import Manifest, SequenceRecord
from .benchmark import Benchmark, EvalResult
from .metrics import per_sequence_metrics, aggregate
from .manifest_mot import MOTManifest, MOTSequenceRecord
from .data_mot import MOTFrame, MOTObject, iter_mot_frames
from .benchmark_mot import MOTBenchmark

__all__ = [
    # SOT
    "Benchmark",
    "EvalResult",
    "Manifest",
    "SequenceRecord",
    "per_sequence_metrics",
    "aggregate",
    # MOT
    "MOTBenchmark",
    "MOTManifest",
    "MOTSequenceRecord",
    "MOTFrame",
    "MOTObject",
    "iter_mot_frames",
]
