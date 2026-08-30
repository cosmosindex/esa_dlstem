"""Ship the reviewed ground truth as something the benchmark can just load.

Everything the review produces — SAT-MTB's recovered static tracks, SAM 3
geometry, hand corrections, hand-drawn tracks, deletions — is layered on top of
five datasets that store ground truth in six different formats, one of which
(``coco_mot_json``) keeps many sequences in a single file. Writing each layer
back into its native format would be six writers, and would leave the result
scattered across five dataset roots that must never be modified.

So the export normalises: **one 11-column MOT CSV per sequence**, unified class
ids (0 car, 1 airplane, 2 ship, 3 train), plus a complete manifest pointing at
them. Point ``MOTManifest.load()`` at that manifest and every downstream loader
works unchanged. Frame ids stay native — 1-indexed everywhere except SDM-Car,
which is 0-indexed — and each record's ``frame_index_base`` says which.

Review status travels *in* the manifest rather than in a side document, because
the failure this guards against is shipping unreviewed annotation as reviewed —
which is exactly what happens when the pointer and the provenance live in
different files and only one of them gets copied.

Usage::

    python -m interactive_review.export --out /work/<user>/space_tracker_mot_reviewed
    python -m interactive_review.export --out ... --accepted-only
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from .core.gtsource import (MERGED_ROOT, frame_objects, load_frames,
                           merged_available, visible_objects)
from .core.paths import DEFAULT_OVERRIDES, MOT_MANIFEST, manifest
from .core.idsplit import split_reused_ids
from .core.sizefix import invalid_frames
from .core.vdecisions import SequenceDecisions
from .core.vqueue import mode_for

CLS_ID = {"car": 0, "airplane": 1, "ship": 2, "train": 3}

log = logging.getLogger(__name__)


def sequence_lines(seq_id: str, decisions: SequenceDecisions) -> tuple[list[str], Counter]:
    """The 11-column rows for one sequence, with every layer applied.

    Boxes that cannot be annotation on their own terms — zero area, or many
    times their own track's size — are dropped here rather than written. They
    are defects in the sources (RsCarData ships 122 zero-area boxes in eight
    sequences alone, before any review saw them) and not decisions anyone made,
    so the review document stays a record of human judgement and the dataset
    files stay untouched. See :mod:`.core.sizefix` for why dropping beats
    repairing. The counts travel in the manifest.
    """
    deleted = decisions.deleted(seq_id)
    counts: Counter = Counter()
    rows = []
    frames = set(load_frames(seq_id))
    frames.update(f for fb in decisions.drawn(seq_id).values() for f in fb)

    # Gathered per track first: "many times its own size" needs the whole track,
    # and an id carrying two objects can only be seen with every box in hand.
    gathered: dict[tuple, dict[int, list[list[float]]]] = {}
    provenance: dict[tuple, dict[int, str]] = {}
    for fid in sorted(frames):
        for o in frame_objects(seq_id, fid, decisions):
            if o.key in deleted:
                counts["boxes_deleted"] += 1
                continue
            ident = (o.category, o.track_id)
            box = [float(v) for v in o.box]
            here = gathered.setdefault(ident, {}).setdefault(fid, [])
            # SAT-MTB writes some rows twice: `satmtb/airplane/27` carries every
            # box of 457 car tracks in duplicate, 34,489 rows, in its own MOT
            # file and in the merged one alike. Byte-identical, so the second is
            # redundant -- but one id on one frame must still produce one row,
            # or every consumer reads two objects.
            if any(all(abs(a - b) < 1e-6 for a, b in zip(prev, box))
                   for prev in here):
                counts["boxes_duplicate"] += 1
                continue
            here.append(box)
            provenance.setdefault(ident, {})[fid] = o.provenance

    # What is left on a frame after de-duplication is two *different* boxes, so
    # two objects under one id. Splitting keeps both; dropping one would delete
    # a real object's annotation. New ids are appended, so the report has to be
    # read to know an id was ever shared.
    tracks, split = split_reused_ids(gathered)
    if split["ids_split"]:
        counts["ids_split"] += split["ids_split"]
        counts["tracks_created_by_split"] += split["tracks_created"]
        counts["boxes_recovered_by_split"] += split["frames_recovered"]
        for line in split["detail"]:
            log.info("idsplit: %s | %s", seq_id, line)

    for ident, track in tracks.items():
        bad = invalid_frames(track)
        for fid, box in track.items():
            reason = bad.get(fid)
            if reason:
                counts[f"boxes_dropped_{reason}"] += 1
                continue
            # A split track's frames keep the provenance recorded under the id
            # they arrived on; a new id has no entry of its own.
            counts[f"boxes_{provenance.get(ident, {}).get(fid, 'original')}"] += 1
            x1, y1, x2, y2 = box
            rows.append((fid, ident[1], x1, y1, x2 - x1, y2 - y1, CLS_ID[ident[0]]))

    rows.sort(key=lambda r: (r[0], r[1]))
    lines = [f"{fid},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,{cls},-1,-1,-1"
             for fid, tid, x, y, w, h, cls in rows]
    counts["boxes"] = len(rows)
    return lines, counts


#: Median sqrt(area) at or below which a track counts as a small object. The
#: benchmark's own boundary, and the same number the sequence-level split uses —
#: applied here to each track instead of to the sequence as a whole.
SMALL_OBJECT_PX = 32.0


def _has_small_object(lines: list[str]) -> bool:
    """Whether any track in these rows is a small object.

    Asked of the rows about to be written rather than of the source, so a
    sequence is judged on the annotation it actually ships with.
    """
    import numpy as np

    per: dict[tuple, list[float]] = {}
    for l in lines:
        f = l.split(",")
        per.setdefault((f[1], f[7]), []).append(
            float(np.sqrt(float(f[4]) * float(f[5]))))
    return any(float(np.median(v)) <= SMALL_OBJECT_PX for v in per.values())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--overrides", type=Path, default=DEFAULT_OVERRIDES)
    ap.add_argument("--exclude-datasets", nargs="*", default=(),
                    help="datasets to leave out entirely. A release is a "
                         "redistribution, and AIR-MOT was never licensed for "
                         "one — so a shippable export is `--exclude-datasets "
                         "airmot`, and a complete one is the default")
    ap.add_argument("--require-small-object", action="store_true",
                    help="keep only sequences carrying at least one track whose "
                         "median sqrt(area) is <= 32 px. This is the benchmark's "
                         "scope and it is asked per *object*: a sequence whose "
                         "median object is large can still carry small ones, and "
                         "37 of the 55 held out by the sequence-level split do. "
                         "Independent of space_tracker/data/size_split.json, which "
                         "stays frozen so the size-split experiments do not move")
    ap.add_argument("--queue-only", action="store_true",
                    help="export exactly the sequences the review covered: "
                         "licensed for redistribution and in the small-object "
                         "half, read from space_tracker/data/size_split.json so "
                         "this and the size-split experiments cannot drift "
                         "apart. This is the benchmark's own scope — a large "
                         "object is out of it whatever its annotation says")
    ap.add_argument("--accepted-only", action="store_true",
                    help="export only sequences signed off in the review; the "
                         "default exports everything and records the status")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    decisions = SequenceDecisions(args.overrides)
    src = json.loads(MOT_MANIFEST.read_text())
    gt_dir = args.out / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)

    kept: list[dict] = []
    dropped: list[dict] = []
    totals: Counter = Counter()
    excluded_ds = {d.lower() for d in (args.exclude_datasets or ())}
    in_queue = None
    if args.queue_only:
        from .core.vqueue import UNLICENSED_DATASETS, build_queue
        in_queue = {it.seq_id for it in build_queue(
            small_only=True, exclude_datasets=UNLICENSED_DATASETS)}
    for record in src["sequences"]:
        seq_id = record["id"]
        if seq_id.split("/", 1)[0].lower() in excluded_ds:
            totals["excluded_by_licence"] += 1
            continue
        if in_queue is not None and seq_id not in in_queue:
            totals["excluded_out_of_scope"] += 1
            continue
        status = decisions.status_of(seq_id)
        if status == "excluded":
            # Never silently: a sequence that disappears without a recorded
            # reason is indistinguishable from one that was lost.
            dropped.append({"id": seq_id,
                            "reason": decisions.get(seq_id).get("note", ""),
                            "tags": decisions.tags(seq_id),
                            "reviewer": decisions.get(seq_id).get("reviewer", ""),
                            "updated": decisions.get(seq_id).get("updated")})
            totals["excluded"] += 1
            continue
        if args.accepted_only and status != "accepted":
            totals["skipped_unaccepted"] += 1
            continue

        lines, counts = sequence_lines(seq_id, decisions)
        if args.require_small_object and not _has_small_object(lines):
            totals["excluded_no_small_object"] += 1
            continue
        rel = Path("gt") / (seq_id.replace("/", "_") + ".txt")
        (args.out / rel).write_text("\n".join(lines) + "\n")
        totals.update(counts)
        totals["sequences"] += 1
        if status:
            totals[f"status_{status}"] += 1

        out_record = dict(record)
        out_record.update({
            "gt_path": str(rel),
            "gt_path_override": None,
            "gt_format": "mot_csv_11col",
            # frame_index_base is deliberately NOT overridden: the rows above
            # carry each source's native frame ids (SDM-Car's are 0-based), so
            # declaring 1 here would shift every SDM-Car sequence by one frame.
            # visible_objects, not frame_objects: the rows above drop deleted
            # tracks, and a manifest that counted them would promise more tracks
            # than the file it describes contains.
            # Counted off the rows actually written, not off the objects that
            # went in: a track whose every box was dropped as invalid is not in
            # the file, and a manifest that still counted it would promise more
            # than the file contains.
            "n_tracks": len({(int(l.split(",")[1]), int(l.split(",")[7]))
                             for l in lines}),
            "review": {
                "status": status,
                "mode": mode_for(next(s for s in manifest().sequences
                                      if s.id == seq_id)),
                "merged_from_detection_xml": merged_available(seq_id),
                "edits": decisions.edit_counts(seq_id),
                "note": decisions.get(seq_id).get("note", ""),
            },
            # Top level, not under "review": these describe the *video*, not
            # anyone's opinion of it, and they are useful to a user who never
            # reads the review provenance at all.
            "tags": decisions.tags(seq_id),
        })
        kept.append(out_record)

    out_manifest = dict(src)
    out_manifest["sequences"] = kept
    out_manifest["n_sequences"] = len(kept)
    out_manifest["name"] = src.get("name", "") + " (reviewed)"
    out_manifest["description"] = (
        src.get("description", "") + " "
        "Ground truth normalised to 11-column MOT CSV by "
        "interactive_review.export: SAT-MTB static objects restored from "
        "detection XML, geometry from the SAM 3 batch pass, plus corrections "
        "made in the video review. Per-sequence provenance is in each record's "
        "'review' field.")
    out_manifest["source_manifest"] = str(MOT_MANIFEST.name)
    out_manifest["merged_root"] = str(MERGED_ROOT)
    out_manifest["tags"] = dict(sorted(decisions.tag_counts().items()))
    (args.out / "space_tracker_mot_reviewed.json").write_text(
        json.dumps(out_manifest, indent=1))
    if dropped:
        (args.out / "excluded.json").write_text(json.dumps(
            {"n": len(dropped), "sequences": dropped}, indent=1))

    print(f"{totals['sequences']} sequences, {totals['boxes']} boxes")
    if dropped:
        print(f"  {len(dropped)} excluded by the review -> {args.out}/excluded.json")
    if decisions.tag_counts():
        print("  tags: " + ", ".join(f"{t} {n}" for t, n
                                     in decisions.tag_counts().most_common()))
    for k in sorted(totals):
        if k.startswith(("boxes_", "status_", "skipped_")):
            print(f"  {k:<26} {totals[k]}")
    signed = totals.get("status_accepted", 0)
    if signed < totals["sequences"]:
        print(f"\n[note] {totals['sequences'] - signed} of {totals['sequences']} "
              f"exported sequences are NOT signed off — each record's "
              f"review.status says so.")
    print(f"\n-> {args.out}/space_tracker_mot_reviewed.json")


if __name__ == "__main__":
    main()
