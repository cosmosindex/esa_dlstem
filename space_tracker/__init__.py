"""Space-Tracker — a tracking benchmark for small objects in satellite video.

Two halves, single-object (SOT) and multi-object (MOT), assembled from seven
source datasets and rewritten into one annotation format per task. This package
reads the released benchmark; see ``README.md`` for where to download it.

    from space_tracker import SpaceTracker

    st = SpaceTracker("/path/to/space_tracker")   # or set $SPACE_TRACKER_ROOT
    print(st.summary())

    for seq in st.sot.filter(split="test", attribute="OCC"):
        for f in seq.frames(visible_only=True):
            f.box            # (x, y, w, h) in absolute pixels
            f.image_path     # pathlib.Path

    for seq in st.mot.filter(split="test", contains="ship"):
        for frame_id, objects in seq.frames():
            for o in objects:
                o.track_id, o.category, o.box

Everything above is standard library only. Reading a frame needs Pillow or
OpenCV; ``load_coco()`` needs ``pycocotools``.

The other modules here (``manifest``, ``data``, ``manifest_mot``, ``data_mot``,
``benchmark``, ``benchmark_mot``, ``metrics``) are the internal layer that
indexes the *source* datasets in their original formats and builds the release
from them. They are kept because the annotation tool and the build scripts in
this repository import them, and they are imported lazily so that reading the
release costs nothing extra. A reader of the benchmark does not need them.
"""

from .release import (
    MOTHalf,
    MOTObject,
    MOTSequence,
    MOTTrack,
    SOTFrame,
    SOTHalf,
    SOTSequence,
    SpaceTracker,
)

__all__ = [
    # the released benchmark
    "SpaceTracker",
    "SOTHalf",
    "MOTHalf",
    "SOTSequence",
    "MOTSequence",
    "SOTFrame",
    "MOTObject",
    "MOTTrack",
    # internal source-dataset layer, resolved on first use
    "Benchmark",
    "EvalResult",
    "Manifest",
    "SequenceRecord",
    "per_sequence_metrics",
    "aggregate",
    "MOTBenchmark",
    "MOTManifest",
    "MOTSequenceRecord",
    "MOTFrame",
    "iter_mot_frames",
]

_LAZY = {
    "Manifest": "manifest",
    "SequenceRecord": "manifest",
    "Benchmark": "benchmark",
    "EvalResult": "benchmark",
    "per_sequence_metrics": "metrics",
    "aggregate": "metrics",
    "MOTManifest": "manifest_mot",
    "MOTSequenceRecord": "manifest_mot",
    "MOTFrame": "data_mot",
    "iter_mot_frames": "data_mot",
    "MOTBenchmark": "benchmark_mot",
}


def __getattr__(name):
    """Resolve the internal source-dataset layer on first use (PEP 562)."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f".{module}", __name__), name)


def __dir__():
    return sorted(__all__)
