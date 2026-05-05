"""Aggregate MOT results across (RsCarData, SAT-MTB-car, SDM-Car) into a single
"Space-Tracker-MOT car" table. Reads hota_summary.csv produced by
tools/compute_hota.py, averages rate metrics across the three datasets and sums
count metrics, then emits a NeurIPS-style LaTeX table plus a per-dataset
breakdown for the supplementary material.

Excludes SAM3 / SAM3.1 (handled separately).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

DEFAULT_HOTA_CSV = Path(
    "/data/ESA_DLSTEM_2025/experiments/MOT/tracker_20260427/hota_summary.csv"
)
DEFAULT_OUT_DIR = Path(
    "/home/ziwen/code/esa_dlstem/Formatting Instructions For NeurIPS 2026/tables/MOT"
)

# (display_name, citation_key, venue) — preserves the publication order.
TRACKER_META = [
    ("sort",          "SORT",           "bewley2016sort",   "ICIP 2016"),
    ("bytetrack",     "ByteTrack",      "zhang2022bytetrack", "ECCV 2022"),
    ("ocsort",        "OC-SORT",        "cao2023ocsort",    "CVPR 2023"),
    ("botsort",       "BoT-SORT",       "aharon2022botsort", "arXiv 2022"),
    ("botsort_reid",  "BoT-SORT-ReID",  "aharon2022botsort", "arXiv 2022"),
    ("tracktrack",    "TrackTrack",     "kim2025tracktrack", "CVPR 2025"),
]

DATASETS = ["rscardata", "satmtb", "sdmcar"]
DATASET_DISPLAY = {
    "rscardata": "RsCarData",
    "satmtb":    "SAT-MTB",
    "sdmcar":    "SDM-Car",
}

# Higher-is-better rate metrics — averaged across datasets.
RATE_METRICS = ["HOTA", "DetA", "AssA", "MOTA", "IDF1"]
# Count metrics — summed across datasets.
COUNT_METRICS = ["IDsw", "MT", "ML"]


def read_hota_csv(path: Path) -> dict[tuple[str, str], dict]:
    rows: dict[tuple[str, str], dict] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["dataset"], row["tracker"])
            for k, v in list(row.items()):
                if k in ("dataset", "tracker"):
                    continue
                row[k] = float(v)
            rows[key] = row
    return rows


def aggregate(rows: dict[tuple[str, str], dict]) -> dict[str, dict]:
    """For each tracker, compute mean of rates and sum of counts across datasets."""
    out: dict[str, dict] = {}
    for tracker_key, _, _, _ in TRACKER_META:
        per_ds = [rows[(d, tracker_key)] for d in DATASETS if (d, tracker_key) in rows]
        if len(per_ds) != len(DATASETS):
            print(f"[warn] {tracker_key}: only {len(per_ds)}/{len(DATASETS)} datasets present")
        agg = {}
        for m in RATE_METRICS:
            agg[m] = sum(r[m] for r in per_ds) / len(per_ds)
        for m in COUNT_METRICS:
            agg[m] = sum(int(r[m]) for r in per_ds)
        out[tracker_key] = agg
    return out


def fmt_rate(v: float) -> str:
    return f"{v:.3f}"


def fmt_int(v: int) -> str:
    return f"{int(v):,}"


def rank_marks(values: list[float], higher_is_better: bool) -> list[str]:
    """Return list of '\\textbf', '\\underline', or '' for each value."""
    n = len(values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: values[i], reverse=higher_is_better)
    marks = [""] * n
    if n >= 1:
        marks[order[0]] = r"\textbf"
    if n >= 2:
        marks[order[1]] = r"\underline"
    return marks


def write_overall_table(agg: dict[str, dict], out_path: Path) -> None:
    rate_cols = RATE_METRICS  # HOTA, DetA, AssA, MOTA, IDF1
    count_cols = COUNT_METRICS  # IDsw, MT, ML

    rate_values = {m: [agg[t[0]][m] for t in TRACKER_META] for m in rate_cols}
    count_values = {m: [agg[t[0]][m] for t in TRACKER_META] for m in count_cols}

    # ranking: rates higher-better; counts: IDsw/ML lower-better, MT higher-better
    rate_marks = {m: rank_marks(rate_values[m], higher_is_better=True) for m in rate_cols}
    count_marks = {
        "IDsw": rank_marks(count_values["IDsw"], higher_is_better=False),
        "MT":   rank_marks(count_values["MT"],   higher_is_better=True),
        "ML":   rank_marks(count_values["ML"],   higher_is_better=False),
    }

    lines = []
    lines.append(r"% Requires: \usepackage{booktabs, multirow, array}")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Multi-Object Tracking (MOT) results on Space-Tracker-MOT~(car), "
                 r"averaged across the three constituent car-only MOT splits "
                 r"(RsCarData, SAT-MTB-car, SDM-Car). Detections are produced once by "
                 r"HiEUM~\cite{TODO_hieum} and shared across trackers; differences in "
                 r"the table therefore reflect association quality only. Rate metrics "
                 r"(HOTA, DetA, AssA, MOTA, IDF1) are macro-averaged across datasets; "
                 r"count metrics (IDsw, MT, ML) are summed. $\uparrow$ higher is better, "
                 r"$\downarrow$ lower is better. \textbf{Bold} marks the best score and "
                 r"\underline{underline} the second best in each column.}")
    lines.append(r"\label{tab:mot_results}")
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
            mark = rate_marks[m][i]
            v = fmt_rate(rate_values[m][i])
            cells.append(f"{mark}{{{v}}}" if mark else v)
        for m in count_cols:
            mark = count_marks[m][i]
            v = fmt_int(count_values[m][i])
            cells.append(f"{mark}{{{v}}}" if mark else v)
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    out_path.write_text("\n".join(lines) + "\n")


def write_per_dataset_table(rows: dict[tuple[str, str], dict], out_path: Path) -> None:
    """Per-dataset breakdown, supplementary."""
    lines = []
    lines.append(r"% Requires: \usepackage{booktabs, multirow, array}")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Per-dataset breakdown of MOT results underlying "
                 r"Tab.~\ref{tab:mot_results}. All trackers use the same HiEUM "
                 r"detections. \textbf{Bold} marks the best score, \underline{underline} "
                 r"the second best, within each (dataset, metric) cell.}")
    lines.append(r"\label{tab:mot_results_per_dataset}")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.10}")
    # 5 rates + IDsw per dataset = 6 columns × 3 datasets = 18 numeric columns
    lines.append(r"\resizebox{\linewidth}{!}{")
    lines.append(r"\begin{tabular}{l " + " ".join(["cccccc"] * len(DATASETS)) + "}")
    lines.append(r"\toprule")
    headers = [r"\multirow{2}{*}{\textbf{Method}}"]
    for ds in DATASETS:
        headers.append(r"\multicolumn{6}{c}{\textbf{" + DATASET_DISPLAY[ds] + r"}}")
    lines.append(" & ".join(headers) + r" \\")
    cmid = " ".join(rf"\cmidrule(lr){{{2 + 6 * i}-{7 + 6 * i}}}" for i in range(len(DATASETS)))
    lines.append(cmid)
    sub_headers = [""]
    for _ in DATASETS:
        sub_headers += ["HOTA", "DetA", "AssA", "MOTA", "IDF1", r"IDsw $\downarrow$"]
    lines.append(" & ".join(sub_headers) + r" \\")
    lines.append(r"\midrule")

    # ranking marks per (dataset, metric)
    metrics_seq = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDsw"]
    for ds in DATASETS:
        for m in metrics_seq:
            vals = [rows[(ds, t[0])][m] for t in TRACKER_META]
            higher = (m != "IDsw")
            rows_marks = rank_marks(vals, higher_is_better=higher)
            for i, t in enumerate(TRACKER_META):
                rows[(ds, t[0])].setdefault("_mark", {})[(ds, m)] = rows_marks[i]

    for i, (key, name, cite, venue) in enumerate(TRACKER_META):
        cells = [f"{name}~\\cite{{{cite}}}"]
        for ds in DATASETS:
            row = rows[(ds, key)]
            for m in metrics_seq:
                mark = row["_mark"][(ds, m)]
                if m == "IDsw":
                    v = fmt_int(int(row[m]))
                else:
                    v = fmt_rate(row[m])
                cells.append(f"{mark}{{{v}}}" if mark else v)
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")
    out_path.write_text("\n".join(lines) + "\n")


def write_aggregated_csv(agg: dict[str, dict], out_path: Path) -> None:
    fields = ["tracker"] + RATE_METRICS + COUNT_METRICS
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for key, name, _, _ in TRACKER_META:
            row = [name] + [round(agg[key][m], 4) for m in RATE_METRICS] + \
                  [int(agg[key][m]) for m in COUNT_METRICS]
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hota-csv", default=str(DEFAULT_HOTA_CSV))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args()

    rows = read_hota_csv(Path(args.hota_csv))
    agg = aggregate(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_overall_table(agg, out_dir / "mot_overall_table.tex")
    write_per_dataset_table(rows, out_dir / "mot_per_dataset_table.tex")
    write_aggregated_csv(agg, out_dir / "mot_aggregated.csv")
    print(f"[ok] wrote tables under {out_dir}")
    # Also dump the aggregate to stdout for a quick eyeball.
    print()
    print("Tracker          " + "  ".join(f"{m:>6s}" for m in RATE_METRICS) +
          "  " + "  ".join(f"{m:>9s}" for m in COUNT_METRICS))
    for key, name, _, _ in TRACKER_META:
        rate_str = "  ".join(f"{agg[key][m]:6.3f}" for m in RATE_METRICS)
        cnt_str = "  ".join(f"{int(agg[key][m]):9,d}" for m in COUNT_METRICS)
        print(f"{name:16s} {rate_str}  {cnt_str}")


if __name__ == "__main__":
    main()
