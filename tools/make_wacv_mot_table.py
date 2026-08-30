"""
Build the non-car (airplane + ship) MOT main table for the WACV paper.

**Why the headline numbers are class-agnostic.** Two of the paradigms in this
table cannot produce a class label at all: MOTRv2 and MOTIP are trained
class-agnostically and emit -1. Scoring the SORT family per class and the
query-based methods pooled would compare different quantities in the same
column, so every method here is scored with all ground truth and all predictions
merged into one foreground class. The per-category split is reported separately
(\\cref{tab:mot_nocar_perclass}) for the methods that do predict a class.

All rows consume the SAME Faster R-CNN detections (\\cref{tab:mot_detection})
and run at one unified operating point, so differences are association
behaviour, not detector tuning. MOTRv2's proposals come from that same
detection file -- its checkpoint is evaluated through `MOTRv2/eval_motrv2.py
--proposal-tag frcnn`, NOT the GT-box oracle used in the association study.

Metrics follow \\cref{sec:eval_metrics}: HOTA (primary), DetA, AssA, LocA,
MOTA, MOTP, IDF1, IDsw, all under the MOTChallenge protocol at IoU >= 0.5 via
TrackEval.

Usage:
    python tools/make_wacv_mot_table.py \
        --agnostic docs/space_tracker/hota_nocar_class_agnostic.csv \
        --perclass docs/space_tracker/hota_nocar_perclass.csv \
        --out-dir wacv-2027-author-kit-template/tables
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

# (csv key, display name, citation, paradigm group). A key of None marks a row
# whose run has not finished; it is emitted as a placeholder rather than
# silently dropped, so the table shows what the benchmark still owes.
ROWS = [
    ("--- Tracking-by-detection (SORT family)", None, None, None),
    ("sort",         "SORT",           r"\citep{bewley2016sort}",   None),
    ("bytetrack",    "ByteTrack",      r"\citep{zhang2022bytetrack}", None),
    ("ocsort",       "OC-SORT",        r"\citep{cao2023ocsort}",    None),
    ("botsort",      "BoT-SORT",       r"\citep{aharon2022botsort}", None),
    ("botsort_reid", "BoT-SORT-ReID",  r"\citep{aharon2022botsort}", None),
    ("--- Learned association", None, None, None),
    ("tracktrack",   "TrackTrack",     r"\citep{shim2025tracktrack}", None),
    ("masa",         "MASA",           r"\citep{li2024masa}",       None),
    ("--- Joint detection and tracking", None, None, None),
    ("fairmot_all",  "FairMOT",        r"\citep{zhang2021fairmot}", None),
    ("tgram_all",    "TGraM",          r"\citep{He2022TGraM}",      None),
    ("--- Query-based end-to-end", None, None, None),
    ("motrv2",       "MOTRv2",         r"\citep{zhang2023motrv2}",  None),
    ("motip",        "MOTIP",          r"\citep{gao2024motip}",     None),
]

# (csv column, header, higher-is-better, decimals)
METRICS = [
    ("HOTA", r"\textbf{HOTA}", True,  3),
    ("DetA", "DetA",           True,  3),
    ("AssA", "AssA",           True,  3),
    ("LocA", "LocA",           True,  3),
    ("MOTA", "MOTA",           True,  3),
    ("MOTP", "MOTP",           True,  3),
    ("IDF1", "IDF1",           True,  3),
    ("IDsw", "IDsw",           False, 0),
]
# Tiny block, mirroring the SOT main table. The SOT version restricts to whole
# sequences whose median object is under 8 px; that cannot be done here -- only
# ONE of the 22 non-car test sequences qualifies, because MOT sequences mix
# scales internally (a `mixed_*` sequence holds aircraft and ships in the same
# frame). The restriction is therefore per ground-truth BOX, using the exact
# decomposition of tools/compute_mot_size_curve.py. Precision cannot come along:
# a false positive has no ground-truth object and so no ground-truth scale, so
# DetA/MOTA/IDsw have no sub-8-px counterpart and the block reports the two
# axes that do, plus their geometric mean as the analogue of HOTA.
TINY_METRICS = [
    ("HOTA_re", r"HOTA$_\mathrm{re}$", True, 3),
    ("DetRe",   "DetRe",                True, 3),
    ("AssA",    "AssA",                 True, 3),
]
PLACEHOLDER = r"\tbd"


def read(path: Path, cls: str) -> dict[str, dict]:
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("class") != cls:
                continue
            out[r["tracker"]] = r
    return out


def fmt_column(vals: list, higher: bool, nd: int) -> list[str]:
    """Bold best, underline second; placeholders pass through untouched."""
    live = [(i, v) for i, v in enumerate(vals) if v is not None]
    cells = [PLACEHOLDER] * len(vals)
    order = sorted(live, key=lambda t: -t[1] if higher else t[1])
    best = order[0][0] if order else None
    second = order[1][0] if len(order) > 1 else None
    for i, v in live:
        s = f"{v:,.0f}" if nd == 0 else f"{v:.{nd}f}"
        # A leading ASCII hyphen is a legal break point, so a negative value in
        # a narrow p{} column splits across two lines ("-" above, digits below).
        # $-$ is a math minus and carries no break opportunity.
        s = s.replace("-", "$-$")
        cells[i] = (rf"\textbf{{{s}}}" if i == best
                    else rf"\underline{{{s}}}" if i == second else s)
    return cells


def read_tiny(path: Path) -> dict[str, dict]:
    return {r["tracker"]: r for r in csv.DictReader(open(path))}


def build(data: dict, out: Path, n_seq: int, n_det: int, tiny: dict | None = None,
          n_tiny_trk: int = 0):
    live_rows = [r for r in ROWS if not r[0].startswith("---")] if False else \
                [r for r in ROWS if r[1] is not None]
    n_tiny_box = int(next(iter(tiny.values()))["n_gt"]) if tiny else 0
    cols = {}
    for key, _, higher, nd in METRICS:
        vals = []
        for csvkey, _, _, _ in live_rows:
            rec = data.get(csvkey) if csvkey else None
            vals.append(float(rec[key]) if rec else None)
        cols[key] = fmt_column(vals, higher, nd)
    if tiny:
        for key, _, higher, nd in TINY_METRICS:
            vals = []
            for csvkey, _, _, _ in live_rows:
                rec = tiny.get(csvkey) if csvkey else None
                vals.append(float(rec[key]) if rec else None)
            cols["tiny_" + key] = fmt_column(vals, higher, nd)

    L = []
    L.append("% Generated by tools/make_wacv_mot_table.py -- do not edit by hand.")
    L.append(r"% \tbd is the not-yet-finished placeholder; define it in the preamble, e.g.")
    L.append(r"%   \newcommand{\tbd}{\textcolor{gray}{--}}")
    L.append(r"\begin{table*}[t]" if tiny else r"\begin{table}[t]")
    L.append(r"\centering")
    L.append(
        r"\caption{Multi-object tracking on the \textbf{non-car} "
        r"(\emph{airplane}, \emph{ship}) Space-Tracker-MOT test split, "
        f"{n_seq} sequences. "
        r"All methods consume the \emph{same} Faster R-CNN detections "
        r"(\cref{tab:mot_detection}) at one unified operating point, so the "
        r"spread is association behaviour rather than detector tuning; MOTRv2 "
        r"takes those detections as its proposal channel. Scoring is "
        r"\textbf{class-agnostic} -- all ground truth and all predictions are "
        r"pooled into a single foreground class -- because the query-based "
        r"methods are trained class-agnostically and emit no category, so a "
        r"per-class column would not mean the same thing in every row; "
        r"\cref{tab:mot_nocar_perclass} splits the class-predicting methods by "
        r"category. Metrics follow \cref{sec:eval_metrics} at "
        r"$\mathrm{IoU}\geq0.5$. \textbf{Bold} is best, \underline{underline} "
        r"second best; lower is better for IDsw only. "
        r"FairMOT and TGraM are the non-car retrains: their original union "
        r"models were trained on a split overlapping this test set and are not "
        r"used here." + (
            r" \blue{The \textbf{tiny} block restricts the metrics to ground-truth "
            rf"boxes below $8$\,px ({n_tiny_box:,} boxes, {n_tiny_trk} tracks). "
            r"Unlike \cref{tab:sot_results}, which selects whole sequences by "
            r"their median object scale, the restriction here is per box: only "
            r"one of the "
            rf"{n_seq} sequences has a median under $8$\,px, because a MOT "
            r"sequence mixes scales internally. Precision does not survive that "
            r"restriction -- a false positive has no ground-truth object and "
            r"therefore no ground-truth scale -- so the block reports the two "
            r"axes that do, and "
            r"$\mathrm{HOTA}_\mathrm{re}=\sqrt{\mathrm{DetRe}\cdot\mathrm{AssA}}$ "
            r"as their combination. It is the analogue of HOTA, not HOTA.}"
            if tiny else "") + r"}")
    L.append(r"\label{tab:mot_nocar}")
    L.append(r"\setlength{\tabcolsep}{3.5pt}")
    L.append(r"\renewcommand{\arraystretch}{1.12}")
    if tiny:
        L.append(r"\resizebox{\textwidth}{!}{%")
        L.append(r"\begin{tabular}{l *{7}{>{\centering\arraybackslash}p{3.0em}} r c "
                 r"*{3}{>{\centering\arraybackslash}p{3.0em}}}")
        L.append(r"\toprule")
        L.append(rf"& \multicolumn{{{len(METRICS)}}}{{c}}{{\textbf{{All}}}} & & "
                 rf"\multicolumn{{{len(TINY_METRICS)}}}{{c}}"
                 r"{\textbf{Tiny} \, ($<8$\,px)} \\")
        L.append(rf"\cmidrule(lr){{2-{1+len(METRICS)}}} "
                 rf"\cmidrule(lr){{{3+len(METRICS)}-{2+len(METRICS)+len(TINY_METRICS)}}}")
        L.append(r"\textbf{Method} & " + " & ".join(h for _, h, _, _ in METRICS)
                 + " & & " + " & ".join(h for _, h, _, _ in TINY_METRICS) + r" \\")
        L.append(r"\midrule")
    else:
        L.append(r"\resizebox{\columnwidth}{!}{%")
        L.append(r"\begin{tabular}{l *{7}{>{\centering\arraybackslash}p{3.0em}} r}")
        L.append(r"\toprule")
        L.append(r"\textbf{Method} & " + " & ".join(h for _, h, _, _ in METRICS) + r" \\")
        L.append(r"\midrule")

    ncol = 1 + len(METRICS) + (1 + len(TINY_METRICS) if tiny else 0)
    i = 0
    for entry in ROWS:
        if entry[1] is None:
            title = entry[0][4:]
            if i:
                L.append(r"\midrule")
            L.append(rf"\multicolumn{{{ncol}}}{{l}}{{\emph{{{title}}}}} \\")
            continue
        _, name, cite, _ = entry
        cells = [cols[k][i] for k, _, _, _ in METRICS]
        if tiny:
            cells += [""] + [cols["tiny_" + k][i] for k, _, _, _ in TINY_METRICS]
        L.append(f"{name}{cite or ''} & " + " & ".join(cells) + r" \\")
        i += 1

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"}")
    L.append(r"\end{table*}" if tiny else r"\end{table}")
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}")


def build_tiny(tiny: dict, ag: dict, out: Path, n_seq: int, n_trk: int):
    """Standalone sub-8-px table for the supplementary.

    Deliberately NOT a block inside the main table: these are per-ground-truth-box
    restrictions of two HOTA components, not the same quantities as the DetA /
    MOTA / IDsw columns there, and putting them side by side invites a comparison
    that is not defined. The `All` HOTA column is repeated here only so the
    re-ranking can be read without turning back.
    """
    live = [r for r in ROWS if r[1] is not None]
    keys = [r[0] for r in live]
    cols = {"HOTA": fmt_column([float(ag[k]["HOTA"]) if k in ag else None
                                for k in keys], True, 3)}
    for key, _, higher, nd in TINY_METRICS:
        cols[key] = fmt_column([float(tiny[k][key]) if k in tiny else None
                                for k in keys], higher, nd)
    n_box = int(next(iter(tiny.values()))["n_gt"])

    L = ["% Generated by tools/make_wacv_mot_table.py -- do not edit by hand.",
         r"\begin{table}[t]", r"\centering",
         r"\caption{\textbf{Non-car MOT restricted to sub-$8$\,px ground truth} "
         rf"({n_box:,} boxes, {n_trk} tracks, over the same {n_seq} test "
         r"sequences as \cref{tab:mot_nocar}). The restriction is per "
         r"ground-truth \emph{box}, not per sequence as in the tiny part of "
         rf"\cref{{tab:sot_results}}: only one of the {n_seq} sequences has a "
         r"median object scale under $8$\,px, because a MOT sequence mixes "
         r"scales internally. Precision does not survive the restriction --- a "
         r"false positive has no ground-truth object and therefore no "
         r"ground-truth scale --- so only the two HOTA components that are "
         r"functions of ground-truth scale are defined here, together with "
         r"$\mathrm{HOTA}_\mathrm{re}=\sqrt{\mathrm{DetRe}\cdot\mathrm{AssA}}$ "
         r"as their combination; it is the analogue of HOTA, not HOTA, and is "
         r"not comparable to the \textbf{All} column beside it. That column is "
         r"repeated from \cref{tab:mot_nocar} only to show the re-ranking: "
         r"MOTIP is second overall and second-to-last here. \textbf{Bold} is "
         r"best, \underline{underline} second best, within each column.}",
         r"\label{tab:mot_nocar_tiny}",
         r"\setlength{\tabcolsep}{5pt}",
         r"\renewcommand{\arraystretch}{1.1}",
         r"\begin{tabular}{l c c ccc}", r"\toprule",
         r"\textbf{Method} & All & & \multicolumn{3}{c}{Tiny \, ($<8$\,px)} \\",
         r"\cmidrule(lr){2-2} \cmidrule(lr){4-6}",
         r"& HOTA & & " + " & ".join(h for _, h, _, _ in TINY_METRICS) + r" \\",
         r"\midrule"]
    i = 0
    for entry in ROWS:
        if entry[1] is None:
            if i:
                L.append(r"\midrule")
            L.append(rf"\multicolumn{{6}}{{l}}{{\emph{{{entry[0][4:]}}}}} \\")
            continue
        _, name, cite, _ = entry
        cells = [cols["HOTA"][i], ""] + [cols[k][i] for k, _, _, _ in TINY_METRICS]
        L.append(f"{name}{cite or ''} & " + " & ".join(cells) + r" \\")
        i += 1
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}")


def build_perclass(pc: dict[str, dict[str, dict]], out: Path):
    """airplane / ship side by side, for the methods that predict a class."""
    keys = [r[0] for r in ROWS if r[1] is not None and r[0]]
    keys = [k for k in keys if any(k in pc[c] for c in ("airplane", "ship"))]
    names = {r[0]: r[1] for r in ROWS if r[1] is not None and r[0]}
    sub = [("HOTA", r"\textbf{HOTA}", 3), ("DetA", "DetA", 3),
           ("AssA", "AssA", 3), ("IDF1", "IDF1", 3)]

    cols = {}
    for cat in ("airplane", "ship"):
        for m, _, nd in sub:
            vals = [float(pc[cat][k][m]) if k in pc[cat] else None for k in keys]
            cols[(cat, m)] = fmt_column(vals, True, nd)

    L = []
    L.append("% Generated by tools/make_wacv_mot_table.py -- do not edit by hand.")
    L.append(r"\begin{table}[t]")
    L.append(r"\centering")
    L.append(
        r"\caption{Per-category MOT on the non-car test split, for the methods "
        r"that predict a category. The query-based rows of "
        r"\cref{tab:mot_nocar} are absent by construction: they are trained "
        r"class-agnostically and emit no label, which is exactly why the main "
        r"table pools classes. The two categories are far from equally hard: "
        r"every method loses $0.26$--$0.30$ HOTA on \emph{ship} relative to "
        r"\emph{airplane} (mean $0.277$ over the seven methods), and the loss "
        r"is split almost evenly between the two error sources (mean "
        r"$\Delta$DetA $0.262$, mean $\Delta$AssA $0.263$). Ship targets are "
        r"therefore not simply harder to \emph{find}; once found they are also "
        r"harder to keep, so a better detector alone would close only about "
        r"half of the gap. The spread across methods is also asymmetric: on "
        r"\emph{airplane} all seven fall within $0.010$ HOTA, whereas on "
        r"\emph{ship} they span $0.044$ and the ordering changes---association "
        r"design stops being irrelevant exactly where detection gets hard. "
        r"\textbf{Bold} is best within a column.}")
    L.append(r"\label{tab:mot_nocar_perclass}")
    L.append(r"\setlength{\tabcolsep}{4pt}")
    L.append(r"\renewcommand{\arraystretch}{1.12}")
    L.append(r"\resizebox{\columnwidth}{!}{%")
    L.append(r"\begin{tabular}{l *{8}{>{\centering\arraybackslash}p{2.5em}}}")
    L.append(r"\toprule")
    L.append(r"\multirow{2}{*}{\textbf{Method}} & "
             r"\multicolumn{4}{c}{\textbf{airplane}} & "
             r"\multicolumn{4}{c}{\textbf{ship}} \\")
    L.append(r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}")
    L.append(" & " + " & ".join(h for _, h, _ in sub) + " & "
             + " & ".join(h for _, h, _ in sub) + r" \\")
    L.append(r"\midrule")
    for i, k in enumerate(keys):
        cells = [cols[("airplane", m)][i] for m, _, _ in sub] + \
                [cols[("ship", m)][i] for m, _, _ in sub]
        L.append(f"{names[k]} & " + " & ".join(cells) + r" \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"}")
    L.append(r"\end{table}")
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agnostic", required=True)
    ap.add_argument("--perclass", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tiny", help="per-box <8px CSV from tools/compute_mot_size_curve.py "
                                   "run with --edges 0,8; omit to keep the single-column table")
    ap.add_argument("--tiny-tracks", type=int, default=0,
                    help="distinct GT tracks under 8px, for the caption")
    args = ap.parse_args()

    ag = read(Path(args.agnostic), "all")
    pc = {c: read(Path(args.perclass), c) for c in ("airplane", "ship")}
    n_seq = int(next(iter(ag.values()))["n_seqs"])

    missing = [r[1] for r in ROWS if r[1] is not None and r[0] and r[0] not in ag]
    if missing:
        raise SystemExit(f"expected in the agnostic CSV but absent: {missing}")
    pend = [r[1] for r in ROWS if r[1] is not None and r[0] is None]
    print(f"class-agnostic rows: {len(ag)}   placeholders: {pend}")
    for c in ("airplane", "ship"):
        print(f"  per-class {c}: {sorted(pc[c])}")

    out = Path(args.out_dir)
    tiny = read_tiny(Path(args.tiny)) if args.tiny else None
    if tiny:
        missing = [r[1] for r in ROWS if r[1] is not None and r[0] and r[0] not in tiny]
        if missing:
            raise SystemExit(f"absent from the tiny CSV: {missing}")
        n_box = int(next(iter(tiny.values()))["n_gt"])
        print(f"tiny rows: {len(tiny)}   {n_box} GT boxes under 8px")
        for k, name, _, _ in [r for r in ROWS if r[1] is not None]:
            print(f"  {name:14s} HOTA_re={float(tiny[k]['HOTA_re']):.3f} "
                  f"DetRe={float(tiny[k]['DetRe']):.3f} AssA={float(tiny[k]['AssA']):.3f}")
    # The main table stays single-column: the tiny columns are a different kind
    # of quantity and are given their own supplementary table instead.
    build(ag, out / "mot_nocar_main.tex", n_seq, 0)
    if tiny:
        build_tiny(tiny, ag, out / "mot_nocar_tiny.tex", n_seq, args.tiny_tracks)
    build_perclass(pc, out / "mot_nocar_perclass.tex")


if __name__ == "__main__":
    main()
