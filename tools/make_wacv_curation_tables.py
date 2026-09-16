#!/usr/bin/env python
"""Emit the two tables \\cref{sec:st_mot} promises the supplementary will carry.

The MOT section states an audit total and then says the per-dataset breakdown
and the form of curation each source received are in the supplementary.  Nothing
there carried them.  This writes both, from the artifacts of the curation itself
rather than from a hand-kept tally:

``tables/curation_matrix.tex``
    Which curation step each of the seven sources received, both parts.  It
    exists to make the difference between the two parts legible: the SOT sources
    needed harmonization and the MOT sources needed re-annotation, and neither
    is "the other half was left alone".

``tables/mot_audit.tex``
    The per-dataset counts behind the audit total: tracks and boxes added, by
    which of the two mechanisms, and tracks removed, relabelled or re-fitted.

Sources of truth
----------------
``<merged>/provenance/<seq>.json``
    One file per SAT-MTB sequence, written by the detection-XML merge. Carries
    ``tracks_added`` / ``boxes_added`` / ``boxes_hole_filled`` /
    ``boxes_regeometried``.
``<release>/space_tracker_mot_reviewed.json``
    The review manifest. ``review.edits`` per sequence carries the manual pass:
    ``tracks_drawn`` / ``boxes_drawn`` / ``tracks_deleted`` /
    ``tracks_relabelled``.

Both are restricted to the 403 released sequences, which is why the totals here
are smaller than a tally taken over everything ever reviewed: AIR-MOT (69
sequences) is not redistributable and 18 SAT-MTB sequences hold no object under
the size criterion, so neither is in the benchmark and neither may be counted
towards its curation.

Usage
-----
    python tools/make_wacv_curation_tables.py
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

import os as _bootstrap_os, sys as _bootstrap_sys
_bootstrap_sys.path.insert(0, _bootstrap_os.path.dirname(
    _bootstrap_os.path.dirname(_bootstrap_os.path.abspath(__file__))))
from project_paths import DATA_ROOT

#: The released package. ``SPACE_TRACKER_ROOT`` is the name the README and the
#: loading code use; ``SPACE_TRACKER_RELEASE`` is accepted as the older spelling.
RELEASE = Path(os.environ.get("SPACE_TRACKER_ROOT")
               or os.environ.get("SPACE_TRACKER_RELEASE")
               or f"{DATA_ROOT}/release/space_tracker")
REVIEWED = Path(os.environ.get("SPACE_TRACKER_MOT_REVIEWED",
                               "/data/anon/mot_release/space_tracker_mot_reviewed.json"))
PROVENANCE = Path(os.environ.get("SPACE_TRACKER_MOT_PROVENANCE",
                                 "/data/anon/mot_merged/provenance"))

SRC_NAME = {"satsot": "SatSOT", "sv248s": "SV248S", "ootb": "OOTB",
            "satmtb": "SAT-MTB", "sdmcar": "SDM-Car",
            "rscardata": "RsCarData", "viso": "VISO"}
MOT_ORDER = ["satmtb", "sdmcar", "rscardata", "viso"]
SOT_ORDER = ["sv248s", "satsot", "ootb"]

YES, NO = "\\checkmark", "--"


def num(x: int) -> str:
    return f"{x:,}".replace(",", "{,}")


def collect_mot(released_ids: dict[str, str]) -> dict[str, collections.Counter]:
    """Per source: the merge counts and the manual-review counts, released only."""
    agg: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    manifest = json.loads(REVIEWED.read_text())
    for s in manifest["sequences"]:
        if s["id"] not in released_ids:
            continue
        a = agg[s["dataset"]]
        a["seq"] += 1
        e = (s.get("review") or {}).get("edits") or {}
        a["m_tracks"] += e.get("tracks_drawn", 0)
        a["m_boxes"] += e.get("boxes_drawn", 0)
        a["removed"] += e.get("tracks_deleted", 0)
        a["relabelled"] += e.get("tracks_relabelled", 0)

        f = PROVENANCE / (s["id"].replace("/", "_") + ".json")
        if f.exists():
            p = json.loads(f.read_text())["summary"]
            a["x_tracks"] += p["tracks_added"]
            a["x_boxes"] += p["boxes_added"]
            a["hole"] += p["boxes_hole_filled"]
            a["regeom"] += p["boxes_regeometried"]
    return agg


def write_matrix(out: Path, sot: dict, mot: dict, agg: dict) -> None:
    """Which curation step each source received."""
    def r(key, part, n_seq, harm, attr, geom, split, motion, xml, manual):
        return (f"{SRC_NAME[key]} & {n_seq} & {harm} & {attr} & {geom} & "
                f"{split} & {motion} & {xml} & {manual} \\\\")

    rows = ["\\multicolumn{9}{@{}l}{\\emph{SOT sources}} \\\\"]
    for k in SOT_ORDER:
        # Oriented geometry: native in OOTB, imported from SV248S's own contour
        # files, and absent from SatSOT, which publishes horizontal boxes only.
        rows.append(r(k, "SOT", sot[k]["seq"], YES, YES,
                      YES if sot[k]["oriented"] else NO, YES, NO, NO, NO))
    rows.append("\\midrule")
    rows.append("\\multicolumn{9}{@{}l}{\\emph{MOT sources}} \\\\")
    for k in MOT_ORDER:
        rows.append(r(k, "MOT", agg[k]["seq"], YES, NO, NO, YES, YES,
                      YES if agg[k]["x_tracks"] or agg[k]["regeom"] else NO, YES))

    lines = [
        "% Generated by tools/make_wacv_curation_tables.py -- do not edit by hand.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{6pt}",
        "\\renewcommand{\\arraystretch}{1.1}",
        "\\caption{Which form of curation each source received. The two parts "
        "needed different work, and the point of the table is that neither was "
        "passed through untouched. \\emph{Schema}: decoded to frames and "
        "rewritten into the unified row format and COCO-VID JSON. "
        "\\emph{Attr.}: native attribute vocabulary aligned into the 18-attribute "
        "taxonomy. \\emph{Geom.}: oriented geometry present---native in OOTB, "
        "imported from SV248S's own contour files, and unavailable for SatSOT, "
        "which publishes horizontal boxes only. \\emph{Split}: assigned by parent "
        "scene group. \\emph{Motion}: per-track motion state. \\emph{Det.\\,XML}: "
        "static and off-class instances recovered from the source's own detection "
        "annotation and re-fitted with SAM\\,3; only SAT-MTB ships one. "
        "\\emph{Manual}: every sequence opened in the review tool and its tracks "
        "accepted, drawn, deleted or relabelled by hand. Counts behind the last "
        "two columns are in \\cref{tab:mot_audit}.}",
        "\\label{tab:curation_matrix}",
        "\\begin{tabular}{@{}l r cccc c cc@{}}",
        "\\toprule",
        "\\multirow{2}{*}{Source} & \\multirow{2}{*}{\\#Seq} & "
        "\\multicolumn{4}{c}{Harmonization} & \\multicolumn{3}{c}{Re-annotation} \\\\",
        "\\cmidrule(lr){3-6}\\cmidrule(l){7-9}",
        " & & Schema & Attr. & Geom. & Split & Motion & Det.\\,XML & Manual \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
    ]
    out.write_text("\n".join(lines) + "\n")
    print(f"written: {out}")


def write_audit(out: Path, agg: dict) -> None:
    """The per-dataset counts behind the MOT audit total."""
    tot = collections.Counter()
    rows = []
    for k in MOT_ORDER:
        a = agg[k]
        tot.update(a)
        rows.append(
            f"{SRC_NAME[k]} & {a['seq']} & {num(a['x_tracks'])} & "
            f"{num(a['m_tracks'])} & {num(a['x_boxes'])} & {num(a['m_boxes'])} & "
            f"{num(a['removed'])} & {num(a['relabelled'])} & {num(a['regeom'])} \\\\")
    rows.append("\\midrule")
    rows.append(
        f"\\textbf{{Total}} & {tot['seq']} & {num(tot['x_tracks'])} & "
        f"{num(tot['m_tracks'])} & {num(tot['x_boxes'])} & {num(tot['m_boxes'])} & "
        f"{num(tot['removed'])} & {num(tot['relabelled'])} & {num(tot['regeom'])} \\\\")

    lines = [
        "% Generated by tools/make_wacv_curation_tables.py -- do not edit by hand.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{6pt}",
        "\\renewcommand{\\arraystretch}{1.1}",
        "\\caption{The MOT audit, per source dataset. Tracks and boxes were "
        "added by two mechanisms, reported separately because they carry "
        "different evidence. \\emph{Det.\\,XML} recovers instances that the "
        "source itself annotates for detection but omits from its MOT ground "
        "truth---static aircraft and vessels, and classes the sequence was not "
        "labelled for---and re-fits their geometry with SAM\\,3; only SAT-MTB "
        "publishes such an annotation. \\emph{Manual} is the review pass, in "
        "which every one of the 403 sequences was opened and its tracks "
        "accepted, drawn, deleted or relabelled by hand. "
        "\\emph{Removed} and \\emph{Relab.} are tracks deleted as spurious and "
        "tracks whose class was corrected; \\emph{Re-fit} counts boxes whose "
        "geometry was replaced by a SAM\\,3 mask fit, which changes no identity "
        "and adds no instance. All counts are over the 403 released sequences "
        "only: work spent on sequences the licence or the size criterion later "
        "excluded is not claimed here.}",
        "\\label{tab:mot_audit}",
        "\\begin{tabular}{@{}l r rr rr rr r@{}}",
        "\\toprule",
        "\\multirow{2}{*}{Source} & \\multirow{2}{*}{\\#Seq} & "
        "\\multicolumn{2}{c}{$+$Tracks} & \\multicolumn{2}{c}{$+$Boxes} & "
        "\\multicolumn{2}{c}{Corrected} & \\multirow{2}{*}{Re-fit} \\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}",
        " & & Det.\\,XML & Manual & Det.\\,XML & Manual & Removed & Relab. & \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
    ]
    out.write_text("\n".join(lines) + "\n")
    print(f"written: {out}")
    print(f"  totals: +{tot['x_tracks'] + tot['m_tracks']} tracks, "
          f"+{tot['x_boxes'] + tot['m_boxes']} boxes, "
          f"{tot['removed']} removed, {tot['relabelled']} relabelled, "
          f"{tot['regeom']} re-fitted, {tot['hole']} holes filled")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("wacv-2027-author-kit-template/tables"))
    args = ap.parse_args()

    sot_json = json.loads((RELEASE / "sot" / "annotations"
                           / "space_tracker_sot.json").read_text())
    sot: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for v in sot_json["videos"]:
        sot[v["source_dataset"]]["seq"] += 1
    vid_src = {v["id"]: v["source_dataset"] for v in sot_json["videos"]}
    for a in sot_json["annotations"]:
        if a.get("segmentation"):
            sot[vid_src[a["video_id"]]]["oriented"] += 1

    mot_json = json.loads((RELEASE / "mot" / "annotations"
                           / "space_tracker_mot.json").read_text())
    released = {v["source_sequence_id"]: v["name"] for v in mot_json["videos"]}

    agg = collect_mot(released)
    write_matrix(args.out_dir / "curation_matrix.tex", sot, mot_json, agg)
    write_audit(args.out_dir / "mot_audit.tex", agg)


if __name__ == "__main__":
    main()
