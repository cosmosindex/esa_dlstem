"""High-level MOT Benchmark API.

External users typically interact with just this class::

    from space_tracker import MOTBenchmark

    bench = MOTBenchmark.load(
        manifest="space_tracker/space_tracker_mot.json",
        dataset_roots={
            "airmot":    "/data/AIR-MOT-100",
            "satmtb":    "/data/SAT-MTB",
            "viso":      "/data/VISO",
            "sdmcar":    "/data/SDM-Car",
            "rscardata": "/data/RsCarData",
        },
    )

    for seq in bench.filter(categories=["car"], splits=["test"]):
        for frame in bench.frames(seq):
            # frame.image      — HxWxC uint8 RGB
            # frame.frame_id   — native frame id
            # frame.objects    — list[MOTObject(track_id, category, bbox_xyxy)]
            ...

    # Run a tracker and collect its predictions.
    preds = bench.run(my_tracker_fn, categories=["car"], splits=["test"])
    # preds[seq_id] = list[ list[ (track_id, category, bbox_xyxy) ] ]  per frame.

Scoring (HOTA / MOTA / IDF1) is intentionally **not** bundled — use the
official `TrackEval <https://github.com/JonathonLuiten/TrackEval>`__ or
``py-motmetrics`` on the predictions you collect here. The manifest's
``evaluation`` block records the metric and IoU threshold to report.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

from .data_mot import MOTFrame, iter_mot_frames
from .manifest_mot import MOTManifest, MOTSequenceRecord


# A tracker is any callable taking (frames_iter, image_size) and yielding one
# list of (track_id, category, bbox_xyxy) tuples per frame in order. The
# iterator is finite — yield exactly ``seq.n_frames`` entries (yield ``[]``
# for empty frames). ``image_size`` is ``(width, height)``.
TrackPrediction = tuple[int, str, "object"]    # (track_id, category, bbox_xyxy: np.ndarray (4,))
TrackerFn = Callable[
    [Iterator[MOTFrame], tuple[int, int]],
    Iterable[list[TrackPrediction]],
]


@dataclass
class MOTBenchmark:
    manifest: MOTManifest
    dataset_roots: dict[str, Path]

    # ---------- construction ----------

    @classmethod
    def load(
        cls,
        manifest: str | Path,
        dataset_roots: dict[str, str | Path],
    ) -> "MOTBenchmark":
        m = MOTManifest.load(manifest)
        roots = {k: Path(v) for k, v in dataset_roots.items()}
        missing = set(roots) - {"airmot", "satmtb", "viso", "sdmcar", "rscardata"}
        if missing:
            # Just a friendly warning — extras don't break anything.
            print(f"[MOTBenchmark] note: dataset_roots has unrecognised keys: {sorted(missing)}")
        return cls(manifest=m, dataset_roots=roots)

    # ---------- queries ----------

    def filter(self, **kwargs) -> list[MOTSequenceRecord]:
        """Forward to :meth:`MOTManifest.filter`."""
        return self.manifest.filter(**kwargs)

    def frames(
        self,
        seq: MOTSequenceRecord,
        decode_images: bool = True,
    ) -> Iterator[MOTFrame]:
        """Iterate frames + GT for ``seq`` in capture order."""
        return iter_mot_frames(seq, self.dataset_roots, decode_images=decode_images)

    # ---------- inference helper ----------

    def run(
        self,
        tracker_fn: TrackerFn,
        seqs: list[MOTSequenceRecord] | None = None,
        decode_images: bool = True,
        verbose: bool = True,
        **filter_kwargs,
    ) -> dict[str, list[list[TrackPrediction]]]:
        """Call ``tracker_fn`` over every selected sequence and collect outputs.

        Predictions are returned as ``{seq.id: [[ (track_id, category, xyxy), ... ], ...]}``
        — one inner list per frame in capture order. No metrics are computed
        here; feed the dict to TrackEval / py-motmetrics or your own scorer.
        """
        if seqs is None:
            seqs = self.manifest.filter(**filter_kwargs)
        if verbose:
            print(f"[MOTBenchmark.run] {len(seqs)} sequences selected")

        out: dict[str, list[list[TrackPrediction]]] = {}
        for i, seq in enumerate(seqs, 1):
            img_size = (seq.img_width, seq.img_height)
            frames_iter = self.frames(seq, decode_images=decode_images)
            preds = list(tracker_fn(frames_iter, img_size))
            out[seq.id] = preds
            if verbose and (i % 25 == 0 or i == len(seqs)):
                print(f"  [{i}/{len(seqs)}] {seq.id}  n_frames={seq.n_frames}  "
                      f"got={len(preds)}")
        return out
