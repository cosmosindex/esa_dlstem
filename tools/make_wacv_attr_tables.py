"""
Build the WACV per-attribute SOT tables (whole set and tiny subset).

Differences from the NeurIPS tables this replaces:

1. **The taxonomy changed.** The release exposes 23 labels
   (`taxonomy_attributes`): 5 pooled attributes annotated by more than one
   source, 13 single-source attributes, and 5 occlusion sub-types. The NeurIPS
   tables were built on the older 6-attribute unified set, so every number here
   is recomputed rather than carried over.

2. **Per-sequence aggregation, not macro-of-macro.** NeurIPS averaged per
   (dataset, attribute) and then across datasets, which keeps the source
   boundaries alive inside an attribute score. Space-Tracker is one dataset
   here, so every released sequence carrying the attribute counts once.

3. **Sample size is a column.** Attributes range from 236 sequences (SOB) down
   to 1 (ARC). A reader cannot judge an attribute row without knowing which of
   those it is, and the reviewers explicitly asked for uncertainty on the small
   groups.

4. **SR only.** Nine trackers x three metrics is 28 columns; at that width the
   table is unreadable. SR is the primary ranking metric and the global NPR/PR
   already appear in the main table.

OOTB is rescored to HBB exactly as in the main table, so attribute rows are
comparable to the headline numbers. Scores come from the tau=0.5 runs, which is
the paper's protocol.

Usage:
    python tools/make_wacv_attr_tables.py \
        --runs /path/to/SOT_whole_dataset_<date> \
        --release /path/to/release/space_tracker \
        --out-dir wacv-2027-author-kit-template/tables
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lightning_modules.sot_metrics import _compute_from_sequences
from tools.aggregate_sot_release import records_by_sequence
from tools.make_wacv_sot_table import find_run

TINY_PX = 8.0
DATASETS = ["ootb", "satsot", "sv248s"]

TRACKERS = [
    ("siamrpn",    "SiamRPN++"),
    ("siamfc",     "SiamFC"),
    ("ostrack",    "OSTrack-384"),
    ("odtrack",    "ODTrack"),
    ("lorat",      "LoRAT-g378"),
    ("smalltrack", "SmallTrack"),
    ("sam2",       "SAM 2"),
    ("samurai",    "SAMURAI"),
    ("sam3",       "SAM 3"),
]

#: Heading for each taxonomy group, in the order the tables print them.
#: The membership itself is read from the release manifest -- hardcoding it
#: here once let the tables and the package disagree about which rows pool.
GROUP_TITLES = [
    ("pooled", "Pooled --- annotated by more than one source"),
    ("single_source", "Single-source --- reported on the annotating source only"),
    ("occlusion_subtypes",
     r"Occlusion sub-types --- not interchangeable across sources"),
]


def taxonomy_groups(release_root: Path):
    """``[(heading, [attribute, ...]), ...]``, straight out of the manifest."""
    manifest = json.loads(
        (release_root / "sot" / "space_tracker_sot.json").read_text())
    groups = manifest["attribute_taxonomy"]["groups"]
    return [(title, list(groups[key]["members"])) for key, title in GROUP_TITLES]


def sequence_attributes(release_root: Path):
    """(dataset, source video id) -> (set of attributes, median sqrt(area))."""
    ann = release_root / "sot" / "annotations" / "space_tracker_sot.json"
    data = json.loads(ann.read_text())
    out = {}
    for v in data["videos"]:
        ds = v["source_dataset"]
        sid = v["source_sequence_id"]
        vid = sid[len(ds) + 1:] if sid.startswith(ds + "/") else sid
        out[(ds, vid)] = (set(v.get("taxonomy_attributes") or []),
                          float(v["median_sqrt_area_px"]))
    return out


def collect(runs_root: Path, tracker: str, attrs: dict):
    """key 'ds/vid' -> (records, tiny flag, attribute set) for one tracker."""
    per_seq = {}
    for ds in DATASETS:
        pim = find_run(runs_root, tracker, ds)
        if pim is None:
            raise FileNotFoundError(f"{tracker}/{ds}")
        keep = {vid for (d, vid) in attrs if d == ds}
        seqs, _ = records_by_sequence(pim, keep, rescore_aabb=(ds == "ootb"))
        for vid, recs in seqs.items():
            a, size = attrs[(ds, vid)]
            per_seq[f"{ds}/{vid}"] = (recs, size < TINY_PX, a)
    return per_seq


def fmt_row(vals):
    """Bold best, underline second best; '--' where an attribute has no data."""
    live = [(i, v) for i, v in enumerate(vals) if v is not None]
    out = ["--"] * len(vals)
    if not live:
        return out
    order = sorted(live, key=lambda t: -t[1])
    best = order[0][0]
    second = order[1][0] if len(order) > 1 else None
    for i, v in live:
        s = f"{v:.3f}"
        out[i] = (rf"\textbf{{{s}}}" if i == best
                  else rf"\underline{{{s}}}" if i == second else s)
    return out


def build(per_tracker, groups, tiny_only: bool):
    rows = []
    for title, attrs in groups:
        block = []
        for a in attrs:
            n_seq = None
            vals = []
            for key, _ in TRACKERS:
                seqs = per_tracker[key]
                sel = [recs for (recs, tiny, aset) in seqs.values()
                       if a in aset and (tiny or not tiny_only)]
                if tiny_only:
                    sel = [recs for (recs, tiny, aset) in seqs.values()
                           if a in aset and tiny]
                if n_seq is None:
                    n_seq = len(sel)
                vals.append(_compute_from_sequences(sel).get("success_auc")
                            if sel else None)
            block.append((a, n_seq, vals))
        rows.append((title, block))
    return rows


def emit(rows, tiny: bool, out_path: Path, n_total: int):
    L = []
    L.append("% Generated by tools/make_wacv_attr_tables.py -- do not edit by hand.")
    L.append("% Requires: \\usepackage{booktabs, multirow, array, graphicx, adjustbox}"
             " and the \\rothead slanted-header macro from preamble.tex")
    L.append(r"\begin{table*}[t]")
    L.append(r"\centering")
    scope = (r"the \textbf{tiny} subset (median $s=\sqrt{\mathrm{area}}<8$\,px)"
             if tiny else r"\textbf{all} released sequences")
    L.append(
        r"\caption{Per-attribute Success Rate on Space-Tracker-SOT, over "
        + scope + r". Rows follow the taxonomy of \cref{tab:attr_taxonomy}; "
        r"\textbf{\#Seq} is the number of sequences the row is computed on, and "
        r"is given because the attributes span three orders of magnitude in "
        r"support -- a row backed by a handful of sequences should not be read "
        r"as a ranking. Scores use per-sequence aggregation over the released "
        r"set, with OOTB reduced to horizontal boxes as in the main table, at "
        r"the paper's operating threshold $\tau=0.5$. Single-source rows are "
        r"computed on the annotating source alone and occlusion sub-types are "
        r"not interchangeable across sources, so neither group is pooled. "
        r"``--'' marks an attribute with no sequence in this scope. "
        r"\textbf{Bold} = best, \underline{underline} = second best, per row.}")
    L.append(r"\label{tab:sot_attr_" + ("tiny" if tiny else "whole") + "}")
    # No \resizebox. The tabular's natural width is well under \textwidth, so
    # scaling to fill it MAGNIFIES the type -- the font ends up larger than the
    # body text rather than smaller. Fixed p{3.1em} columns were the other half
    # of the problem: "OSTrack-384" does not fit in 3.1em and wrapped into two
    # crushed lines. Natural c columns with the method names set vertically fit
    # nine methods in ~350pt and keep every name on one line. The names are set at
    # 45 rather than 90 degrees (\rothead, defined in preamble.tex): a fully
    # vertical name costs its own length in header-row height and is read with a
    # tilted head, while the 45-degree slant cuts that height by ~1/sqrt(2) and
    # stays legible. \rothead laps the overhang instead of claiming column width,
    # so the slant does not widen the table.
    L.append(r"\footnotesize")
    L.append(r"\setlength{\tabcolsep}{3.5pt}")
    L.append(r"\renewcommand{\arraystretch}{1.05}")
    L.append(r"\begin{tabular}{l r *{9}{c}}")
    L.append(r"\toprule")
    L.append(r"\textbf{Attr.} & \textbf{\#Seq} & "
             + " & ".join(rf"\rothead{{\textbf{{{n}}}}}" for _, n in TRACKERS)
             + r" \\")
    L.append(r"\midrule")
    ncol = 2 + len(TRACKERS)
    for title, block in rows:
        L.append(rf"\multicolumn{{{ncol}}}{{l}}{{\emph{{{title}}}}} \\")
        for a, n_seq, vals in block:
            cells = fmt_row(vals)
            L.append(f"{a} & {n_seq} & " + " & ".join(cells) + r" \\")
        if title != rows[-1][0]:
            L.append(r"\midrule")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table*}")
    out_path.write_text("\n".join(L) + "\n")
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--release", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    attrs = sequence_attributes(Path(args.release))
    groups = taxonomy_groups(Path(args.release))
    print(f"released sequences: {len(attrs)}")
    print("  taxonomy: " + ", ".join(f"{len(a)} {t.split(' ---')[0].lower()}"
                                     for t, a in groups))

    per_tracker = {}
    for key, name in TRACKERS:
        per_tracker[key] = collect(Path(args.runs), key, attrs)
        print(f"  {name:12s} {len(per_tracker[key])} sequences")

    out_dir = Path(args.out_dir)
    for tiny in (False, True):
        rows = build(per_tracker, groups, tiny_only=tiny)
        emit(rows, tiny, out_dir / f"sot_attr_{'tiny' if tiny else 'whole'}.tex",
             len(attrs))


if __name__ == "__main__":
    main()
