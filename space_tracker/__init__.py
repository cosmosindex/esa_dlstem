"""Space-tracker: unified SOT benchmark across SatSOT, SV248S, OOTB.

Quick start::

    from space_tracker import Benchmark

    bench = Benchmark.load(
        manifest="space_tracker/space_tracker.json",
        dataset_roots={
            "ootb":   "/path/to/OOTB",
            "satsot": "/path/to/SatSOT",
            "sv248s": "/path/to/SV248S",
        },
    )

    # Pick a subset, e.g. all sequences carrying our unified OCC attribute,
    # restricted to the SAT 248S dataset:
    for seq in bench.filter(unified_attrs=["OCC"], datasets=["sv248s"]):
        for frame in bench.frames(seq):
            ...  # frame is a dict with image_path, gt_box, gt_obb (if any), ...

    # Run a tracker and score it:
    result = bench.evaluate(my_tracker_callable, unified_attrs=["OCC"])
    print(result.summary())   # SR / NPR / PR / P@5 per dataset and overall
"""

from .manifest import Manifest, SequenceRecord
from .benchmark import Benchmark, EvalResult
from .metrics import per_sequence_metrics, aggregate

__all__ = [
    "Benchmark",
    "EvalResult",
    "Manifest",
    "SequenceRecord",
    "per_sequence_metrics",
    "aggregate",
]
