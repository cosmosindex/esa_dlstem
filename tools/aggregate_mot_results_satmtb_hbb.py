"""Aggregate the per-(dataset, tracker, class) HOTA rows produced by
``compute_hota_multiclass.py`` into a single Space-Tracker-MOT
(airplane/ship/train) table, matching the layout of
``aggregate_mot_results.py`` for the car case.

Outputs:
  * mot_satmtb_hbb_overall_table.tex   — main table (rows = trackers,
    macro-mean across (dataset × class))
  * mot_satmtb_hbb_per_dataset_table.tex — one block per dataset (rows =
    trackers, columns = HOTA / DetA / AssA / MOTA / IDF1 / IDsw)
  * mot_satmtb_hbb_per_class_table.tex — one block per class (rows =
    trackers, columns aggregated across the datasets that annotate the
    class)
  * mot_satmtb_hbb_aggregated.csv — raw aggregated values
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

DEFAULT_HOTA_CSV = Path(
    "/data/ESA_DLSTEM_2025/experiments/MOT/tracker_satmtb_hbb_LATEST/hota_summary.csv"
)
DEFAULT_OUT_DIR = Path(
    "/home/ziwen/code/esa_dlstem/Formatting Instructions For NeurIPS 2026/tables/MOT"
)

TRACKER_META = [
    ("sort",      "SORT",      "bewley2016sort",     "ICIP 2016"),
    ("bytetrack", "ByteTrack", "zhang2022bytetrack", "ECCV 2022"),
    ("ocsort",    "OC-SORT",   "cao2023ocsort",      "CVPR 2023"),
    ("botsort",   "BoT-SORT",  "aharon2022botsort",  "arXiv 2022"),
]

DATASET_DISPLAY = {
    "satmtb_nocar": "SAT-MTB",
    "viso_nocar":   "VISO",
    "airmot":       "AIR-MOT-100",
}
DATASET_ORDER = ["satmtb_nocar", "viso_nocar", "airmot"]
CLASS_ORDER = ["airplane", "ship", "train", "plane"]   # plane = VISO alias for airplane

RATE_METRICS = ["HOTA", "DetA", "AssA", "MOTA", "IDF1"]
COUNT_METRICS = ["IDsw", "MT", "ML"]


def read_hota_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for k in RATE_METRICS + ["LocA", "MOTP"]:
                if row.get(k) not in (None, ""):
                    row[k] = float(row[k])
            for k in COUNT_METRICS + ["n_dets", "n_seqs"]:
                if row.get(k) not in (None, ""):
                    row[k] = int(row[k])
            rows.append(row)
    return rows


def macro_mean(rows: list[dict], rate_metrics, count_metrics) -> dict:
    """Mean for rate metrics, sum for count metrics."""
    out = {}
    for m in rate_metrics:
        vals = [r[m] for r in rows if r.get(m) not in (None, "")]
        out[m] = sum(vals) / len(vals) if vals else 0.0
    for m in count_metrics:
        vals = [r[m] for r in rows if r.get(m) not in (None, "")]
        out[m] = sum(vals) if vals else 0
    return out


def fmt_rate(v: float) -> str:
    return f"{v:.3f}"


def fmt_int(v: int) -> str:
    return f"{int(v):,}"


def rank_marks(values: list[float], higher_is_better: bool) -> list[str]:
    n = len(values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: values[i], reverse=higher_is_better)
    marks = [""] * n
    if n >= 1: marks[order[0]] = r"\textbf"
    if n >= 2: marks[order[1]] = r"\underline"
    return marks


def write_overall_table(per_tracker_agg: dict[str, dict], out_path: Path) -> None:
    rate_cols = RATE_METRICS
    count_cols = COUNT_METRICS

    rate_values = {m: [per_tracker_agg[t[0]][m] for t in TRACKER_META] for m in rate_cols}
    count_values = {m: [per_tracker_agg[t[0]][m] for t in TRACKER_META] for m in count_cols}

    rate_marks  = {m: rank_marks(rate_values[m], True)  for m in rate_cols}
    count_marks = {
        "IDsw": rank_marks(count_values["IDsw"], False),
        "MT":   rank_marks(count_values["MT"],   True),
        "ML":   rank_marks(count_values["ML"],   False),
    }

    lines = []
    lines.append(r"% Requires: \usepackage{booktabs, multirow, array}")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Multi-Object Tracking results on Space-Tracker-MOT~"
                 r"(airplane / ship / train), pooled across SAT-MTB (test split), "
                 r"VISO and AIR-MOT-100. Detections come from a Faster~R-CNN trained "
                 r"on SAT-MTB det\_hbb (3-class HBB; held-out test split for SAT-MTB; "
                 r"VISO and AIR-MOT-100 never seen at training). Trackers are run "
                 r"per-class so identities cannot leak across categories. Rate metrics "
                 r"are macro-averaged across (dataset $\times$ class), counts are "
                 r"summed. \textbf{Bold} = best, \underline{underline} = second.}")
    lines.append(r"\label{tab:mot_results_satmtb_hbb}")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\begin{tabular}{l l ccccc ccc}")
    lines.append(r"\toprule")
    lines.append(r"\multirow{2}{*}{\textbf{Method}} & \multirow{2}{*}{\textbf{Venue}} "
                 r"& \multicolumn{5}{c}{\textbf{Rate metrics ($\uparrow$)}} "
                 r"& \multicolumn{3}{c}{\textbf{Count metrics}} \\")
    lines.append(r"\cmidrule(lr){3-7} \cmidrule(lr){8-10}")
    lines.append(r"& & HOTA & DetA & AssA & MOTA & IDF1 "
                 r"& IDsw $\downarrow$ & MT $\uparrow$ & ML $\downarrow$ \\")
    lines.append(r"\midrule")

    for i, (key, name, cite, venue) in enumerate(TRACKER_META):
        cells = [f"{name}~\\cite{{{cite}}}", venue]
        for m in rate_cols:
            mk = rate_marks[m][i]
            v = fmt_rate(rate_values[m][i])
            cells.append(f"{mk}{{{v}}}" if mk else v)
        for m in count_cols:
            mk = count_marks[m][i]
            v = fmt_int(count_values[m][i])
            cells.append(f"{mk}{{{v}}}" if mk else v)
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    out_path.write_text("\n".join(lines) + "\n")


def write_per_dataset_table(rows: list[dict], out_path: Path) -> None:
    """Per-dataset breakdown: pool classes within each dataset, one cell block
    per dataset. Columns: HOTA / DetA / AssA / MOTA / IDF1 / IDsw."""
    metrics_seq = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDsw"]

    by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_pair[(r["dataset"], r["tracker"])].append(r)
    pair_agg: dict[tuple[str, str], dict] = {
        k: macro_mean(v, RATE_METRICS, COUNT_METRICS) for k, v in by_pair.items()
    }

    lines = []
    lines.append(r"% Requires: \usepackage{booktabs, multirow, array}")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Per-dataset breakdown of the multi-class MOT results "
                 r"underlying Tab.~\ref{tab:mot_results_satmtb_hbb}. Within each "
                 r"dataset we macro-average across classes that have GT in that "
                 r"dataset. \textbf{Bold} / \underline{underline} mark best / "
                 r"second-best within each (dataset, metric) cell.}")
    lines.append(r"\label{tab:mot_results_satmtb_hbb_per_dataset}")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.10}")
    lines.append(r"\resizebox{\linewidth}{!}{")
    lines.append(r"\begin{tabular}{l " + " ".join(["cccccc"] * len(DATASET_ORDER)) + r"}")
    lines.append(r"\toprule")
    headers = [r"\multirow{2}{*}{\textbf{Method}}"]
    for ds in DATASET_ORDER:
        headers.append(r"\multicolumn{6}{c}{\textbf{" + DATASET_DISPLAY[ds] + r"}}")
    lines.append(" & ".join(headers) + r" \\")
    cmid = " ".join(rf"\cmidrule(lr){{{2 + 6 * i}-{7 + 6 * i}}}" for i in range(len(DATASET_ORDER)))
    lines.append(cmid)
    sub = [""]
    for _ in DATASET_ORDER:
        sub += ["HOTA", "DetA", "AssA", "MOTA", "IDF1", r"IDsw $\downarrow$"]
    lines.append(" & ".join(sub) + r" \\")
    lines.append(r"\midrule")

    # marks per (dataset, metric)
    marks_lookup: dict[tuple[str, str], list[str]] = {}
    for ds in DATASET_ORDER:
        for m in metrics_seq:
            vals = [pair_agg[(ds, t[0])][m] for t in TRACKER_META if (ds, t[0]) in pair_agg]
            higher = (m != "IDsw")
            marks = rank_marks(vals, higher)
            for i, t in enumerate(TRACKER_META):
                if (ds, t[0]) in pair_agg:
                    marks_lookup.setdefault((ds, m), []).append(marks[i])

    for ti, (key, name, cite, venue) in enumerate(TRACKER_META):
        cells = [f"{name}~\\cite{{{cite}}}"]
        for ds in DATASET_ORDER:
            for mi, m in enumerate(metrics_seq):
                if (ds, key) not in pair_agg:
                    cells.append("--")
                    continue
                idx_in_ds = sum(1 for t in TRACKER_META[:ti] if (ds, t[0]) in pair_agg)
                mk = marks_lookup.get((ds, m), [""] * len(TRACKER_META))[idx_in_ds] \
                     if idx_in_ds < len(marks_lookup.get((ds, m), [])) else ""
                v = fmt_int(int(pair_agg[(ds, key)][m])) if m == "IDsw" \
                    else fmt_rate(pair_agg[(ds, key)][m])
                cells.append(f"{mk}{{{v}}}" if mk else v)
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")
    out_path.write_text("\n".join(lines) + "\n")


def write_aggregated_csv(per_tracker_agg: dict[str, dict], out_path: Path) -> None:
    fields = ["tracker"] + RATE_METRICS + COUNT_METRICS
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for key, name, _, _ in TRACKER_META:
            row = [name] + [round(per_tracker_agg[key][m], 4) for m in RATE_METRICS] + \
                  [int(per_tracker_agg[key][m]) for m in COUNT_METRICS]
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hota-csv", default=str(DEFAULT_HOTA_CSV))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args()

    rows = read_hota_csv(Path(args.hota_csv))
    if not rows:
        print(f"no rows in {args.hota_csv}")
        return

    by_tracker: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_tracker[r["tracker"]].append(r)
    per_tracker_agg = {tr: macro_mean(by_tracker[tr], RATE_METRICS, COUNT_METRICS)
                       for tr in by_tracker}

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    write_overall_table(per_tracker_agg, out_dir / "mot_satmtb_hbb_overall_table.tex")
    write_per_dataset_table(rows, out_dir / "mot_satmtb_hbb_per_dataset_table.tex")
    write_aggregated_csv(per_tracker_agg, out_dir / "mot_satmtb_hbb_aggregated.csv")
    print(f"[ok] wrote tables under {out_dir}")
    print()
    print(f"{'Tracker':16s}  " + "  ".join(f"{m:>6s}" for m in RATE_METRICS) +
          "  " + "  ".join(f"{m:>9s}" for m in COUNT_METRICS))
    for key, name, _, _ in TRACKER_META:
        if key not in per_tracker_agg:
            continue
        a = per_tracker_agg[key]
        rates = "  ".join(f"{a[m]:6.3f}" for m in RATE_METRICS)
        cnts = "  ".join(f"{int(a[m]):9,d}" for m in COUNT_METRICS)
        print(f"{name:16s}  {rates}  {cnts}")


if __name__ == "__main__":
    main()
