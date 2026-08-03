"""Ship the reviewed ground truth as something the benchmark can just load.

Everything the review produces — SAT-MTB's recovered static tracks, SAM 3
geometry, hand corrections, hand-drawn tracks, deletions — is layered on top of
five datasets that store ground truth in six different formats, one of which
(``coco_mot_json``) keeps many sequences in a single file. Writing each layer
back into its native format would be six writers, and would leave the result
scattered across five dataset roots that must never be modified.

So the export normalises: **one 11-column MOT CSV per sequence**, 1-indexed
frames, unified class ids (0 car, 1 airplane, 2 ship, 3 train), plus a complete
manifest pointing at them. Point ``MOTManifest.load()`` at that manifest and
every downstream loader works unchanged.

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
from collections import Counter
from pathlib import Path

from .core.gtsource import MERGED_ROOT, frame_objects, load_frames, merged_available
from .core.paths import DEFAULT_OVERRIDES, MOT_MANIFEST, manifest
from .core.vdecisions import SequenceDecisions
from .core.vqueue import mode_for

CLS_ID = {"car": 0, "airplane": 1, "ship": 2, "train": 3}


def sequence_lines(seq_id: str, decisions: SequenceDecisions) -> tuple[list[str], Counter]:
    """The 11-column rows for one sequence, with every layer applied."""
    deleted = decisions.deleted(seq_id)
    counts: Counter = Counter()
    rows = []
    frames = set(load_frames(seq_id))
    frames.update(f for fb in decisions.drawn(seq_id).values() for f in fb)
    for fid in sorted(frames):
        for o in frame_objects(seq_id, fid, decisions):
            if o.key in deleted:
                counts["boxes_deleted"] += 1
                continue
            counts[f"boxes_{o.provenance}"] += 1
            x1, y1, x2, y2 = (float(v) for v in o.box)
            rows.append((fid, o.track_id, x1, y1, x2 - x1, y2 - y1, CLS_ID[o.category]))

    rows.sort(key=lambda r: (r[0], r[1]))
    lines = [f"{fid},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,{cls},-1,-1,-1"
             for fid, tid, x, y, w, h, cls in rows]
    counts["boxes"] = len(rows)
    return lines, counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--overrides", type=Path, default=DEFAULT_OVERRIDES)
    ap.add_argument("--accepted-only", action="store_true",
                    help="export only sequences signed off in the review; the "
                         "default exports everything and records the status")
    args = ap.parse_args()

    decisions = SequenceDecisions(args.overrides)
    src = json.loads(MOT_MANIFEST.read_text())
    gt_dir = args.out / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)

    kept: list[dict] = []
    dropped: list[dict] = []
    totals: Counter = Counter()
    for record in src["sequences"]:
        seq_id = record["id"]
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
            "frame_index_base": 1,
            "n_tracks": len({(o.category, o.track_id)
                             for fid in load_frames(seq_id)
                             for o in frame_objects(seq_id, fid, decisions)}),
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
