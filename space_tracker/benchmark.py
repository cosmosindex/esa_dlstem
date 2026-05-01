"""High-level Benchmark API.

External users typically interact with just this class:

    from space_tracker import Benchmark

    bench = Benchmark.load(
        manifest="space_tracker/space_tracker.json",
        dataset_roots={
            "ootb":   "/data/OOTB",
            "satsot": "/data/SatSOT",
            "sv248s": "/data/SV248S",
        },
    )

    def my_tracker(frames, init_box):
        # frames: iterator of Frame; init_box: xyxy of frame 0's GT.
        # yield one np.ndarray (xyxy) per frame, or None for "no detection".
        ...

    result = bench.evaluate(my_tracker, unified_attrs=["OCC", "TO"])
    print(result.summary())   # SR / NPR / PR / P@5 by dataset and overall
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np

from .data import Frame, iter_frames
from .manifest import Manifest, SequenceRecord
from .metrics import aggregate, per_frame_records, per_sequence_metrics


# A tracker is any callable taking (frames_iter, init_box_xyxy) and yielding
# one prediction per *visible-GT* frame in order. The first yielded prediction
# is for the frame whose GT was used as init_box.
TrackerFn = Callable[[Iterator[Frame], np.ndarray], Iterable[np.ndarray | None]]


@dataclass
class EvalResult:
    """Outcome of ``Benchmark.evaluate``."""
    overall: dict
    per_dataset: dict[str, dict]
    per_unified_attribute: dict[str, dict]
    per_sequence: dict[str, dict] = field(default_factory=dict)

    def summary(self) -> str:
        lines = ["=== Overall ==="]
        ov = self.overall
        lines.append(f"  n_seq={ov.get('n_sequences', 0)}  "
                     f"n_frames={ov.get('n_frames', 0)}")
        for k in ("SR", "NPR", "PR", "P@5"):
            lines.append(f"  {k:5s} = {ov.get(k, float('nan')):.4f}")
        lines.append("")
        lines.append("=== Per dataset ===")
        for ds, r in self.per_dataset.items():
            lines.append(f"  {ds:7s}  n_seq={r.get('n_sequences', 0):3d}  "
                         f"SR={r.get('SR', 0):.3f}  NPR={r.get('NPR', 0):.3f}  "
                         f"PR={r.get('PR', 0):.3f}  P@5={r.get('P@5', 0):.3f}")
        if self.per_unified_attribute:
            lines.append("")
            lines.append("=== Per unified attribute (averaged across annotating datasets) ===")
            for attr, r in self.per_unified_attribute.items():
                lines.append(f"  {attr:4s} n_ds={r.get('n_datasets', 0)}  "
                             f"SR={r.get('SR', 0):.3f}  NPR={r.get('NPR', 0):.3f}  "
                             f"PR={r.get('PR', 0):.3f}  P@5={r.get('P@5', 0):.3f}")
        return "\n".join(lines)


@dataclass
class Benchmark:
    manifest: Manifest
    dataset_roots: dict[str, Path]

    # ---------- construction ----------

    @classmethod
    def load(
        cls,
        manifest: str | Path,
        dataset_roots: dict[str, str | Path],
    ) -> "Benchmark":
        m = Manifest.load(manifest)
        roots = {k: Path(v) for k, v in dataset_roots.items()}
        return cls(manifest=m, dataset_roots=roots)

    # ---------- queries ----------

    def filter(self, **kwargs) -> list[SequenceRecord]:
        """Forward to ``Manifest.filter``."""
        return self.manifest.filter(**kwargs)

    def frames(self, seq: SequenceRecord) -> Iterator[Frame]:
        """Iterate frames + GT for ``seq``."""
        return iter_frames(seq, self.dataset_roots)

    # ---------- evaluation ----------

    def score_predictions(
        self,
        seq_predictions: dict[str, list[np.ndarray | None]],
        seqs: list[SequenceRecord] | None = None,
        unified_attrs: Iterable[str] | None = None,
    ) -> EvalResult:
        """Score precomputed predictions.

        ``seq_predictions`` maps ``seq.id`` → list of predicted xyxy boxes
        aligned 1-to-1 with the **visible-GT** frames yielded by ``frames()``.
        Use ``None`` for "no prediction this frame".
        """
        if seqs is None:
            seqs = list(self.manifest.sequences)
        # GT loading + per-sequence summary.
        per_seq_summary: dict[str, dict] = {}
        per_seq_record: dict[str, SequenceRecord] = {}
        for seq in seqs:
            preds = seq_predictions.get(seq.id)
            if preds is None:
                continue
            gt_boxes: list[np.ndarray | None] = []
            for fr in self.frames(seq):
                if fr.visible:
                    gt_boxes.append(fr.gt_box_xyxy)
            n = min(len(gt_boxes), len(preds))
            recs = per_frame_records(gt_boxes[:n], list(preds[:n]))
            per_seq_summary[seq.id] = per_sequence_metrics(recs)
            per_seq_record[seq.id] = seq

        # Aggregate.
        overall = aggregate(per_seq_summary.values())
        per_dataset = {}
        for ds in sorted({s.dataset for s in per_seq_record.values()}):
            per_dataset[ds] = aggregate(
                per_seq_summary[sid] for sid, s in per_seq_record.items()
                if s.dataset == ds
            )
        per_unified: dict[str, dict] = {}
        if unified_attrs is not None:
            unified_attrs = list(unified_attrs)
        else:
            unified_attrs = list(self.manifest.unified_attributes.keys())
        for attr in unified_attrs:
            ds_means = []
            for ds in self.manifest.datasets_annotating(attr):
                ds_seqs = [
                    sid for sid, s in per_seq_record.items()
                    if s.dataset == ds and attr in s.unified_attrs
                ]
                if not ds_seqs:
                    continue
                ds_means.append(aggregate(per_seq_summary[sid] for sid in ds_seqs))
            if not ds_means:
                continue
            per_unified[attr] = {
                "n_datasets": len(ds_means),
                "SR":  float(np.mean([m["SR"]  for m in ds_means])),
                "NPR": float(np.mean([m["NPR"] for m in ds_means])),
                "PR":  float(np.mean([m["PR"]  for m in ds_means])),
                "P@5": float(np.mean([m["P@5"] for m in ds_means])),
            }

        return EvalResult(
            overall=overall,
            per_dataset=per_dataset,
            per_unified_attribute=per_unified,
            per_sequence=per_seq_summary,
        )

    def evaluate(
        self,
        tracker_fn: TrackerFn,
        datasets: Iterable[str] | None = None,
        unified_attrs: Iterable[str] | None = None,
        tiny: bool | None = None,
        match: str = "any",
        verbose: bool = True,
    ) -> EvalResult:
        """Run ``tracker_fn`` over the filtered subset and score it.

        ``tracker_fn(frames_iter, init_box)`` is called once per sequence;
        ``init_box`` is the xyxy of the *first visible-GT frame*. The callable
        must yield one prediction per visible-GT frame, in order. Yield
        ``None`` for "no prediction".
        """
        seqs = self.manifest.filter(
            datasets=datasets, unified_attrs=unified_attrs,
            tiny=tiny, match=match,
        )
        if verbose:
            print(f"[evaluate] {len(seqs)} sequences selected")

        seq_preds: dict[str, list[np.ndarray | None]] = {}
        for i, seq in enumerate(seqs, 1):
            visible_frames = [f for f in self.frames(seq) if f.visible]
            if not visible_frames:
                continue
            init_box = visible_frames[0].gt_box_xyxy
            preds = list(tracker_fn(iter(visible_frames), init_box))
            seq_preds[seq.id] = preds
            if verbose and (i % 25 == 0 or i == len(seqs)):
                print(f"  [{i}/{len(seqs)}] {seq.id}")

        return self.score_predictions(
            seq_preds, seqs=seqs, unified_attrs=unified_attrs,
        )
