"""Build the combined Space-Tracker-MOT overall table that puts the six
tracking-by-detection (TbD) trackers and the three end-to-end trackers
(FairMOT, TGraM, MOTRv2) side by side.

The two families cannot be scored under one protocol, so the table has two
panels and every number in it comes from an already-completed run:

  Panel (a) -- full pipeline, each method with the detections it actually
    consumes. The six TbD trackers share a cached detector (HiEUM on the car
    half, Faster R-CNN on the airplane/ship/train half); FairMOT and TGraM
    detect with their own one-shot heads. MOTRv2 has no detector at all (its
    detection port is a proposal channel) so its row is empty here.

  Panel (b) -- the GT-box association oracle (Exp2). Every method receives the
    same ground-truth boxes, so all nine are directly comparable; only
    association-side metrics are reported (see docs/mot_exp2_assa_vs_size_results.md
    for why HOTA/MOTA are excluded under the oracle).

Sources (read-only; nothing is re-run here)::

  tracker_20260427/hota_summary.csv              TbD, car half
  <out-dir>/_nocar_testsplit_hota.csv            airplane/ship/train half, BOTH
                                                 families, test split only
  allclass_20260608/hota_summary_allclass.csv    JDT, rscardata + sdmcar
  <out-dir>/_jdt_satmtb_caronly.csv              JDT, SAT-MTB re-scored car-only
  exp2_oracle_20260608/assa_vs_size_le32.csv     GT-box oracle, all 9 methods

Outputs ``mot_combined_overall_jdt_table.tex`` + ``mot_combined_overall_jdt.csv``.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

MOT_ROOT = Path("/data/ESA_DLSTEM_2025/experiments/MOT")
DEFAULT_TBD_CAR_CSV = MOT_ROOT / "tracker_20260427" / "hota_summary.csv"
DEFAULT_NOCAR_CSV = None   # -> <out-dir>/_nocar_testsplit_hota.csv
DEFAULT_JDT_CSV = MOT_ROOT / "allclass_20260608" / "hota_summary_allclass.csv"
DEFAULT_ORACLE_CSV = (MOT_ROOT / "exp2_oracle_20260608"
                      / "assa_vs_size_le32.csv")
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "NeurIPS 2026" / "tables" / "MOT"

# key_car / key_nocar / key_oracle are the tracker slugs each source CSV uses.
# ``det`` is the detection source shown in the table.
METHODS = [
    # (display, cite, venue, det, family, key_car, key_nocar, key_oracle)
    ("SORT",          "bewley2016sort",     "ICIP 2016",  "shared",
     "tbd", "sort",         "sort",         "sort"),
    ("ByteTrack",     "zhang2022bytetrack", "ECCV 2022",  "shared",
     "tbd", "bytetrack",    "bytetrack",    "bytetrack"),
    ("OC-SORT",       "cao2023ocsort",      "CVPR 2023",  "shared",
     "tbd", "ocsort",       "ocsort",       "ocsort"),
    ("BoT-SORT",      "aharon2022botsort",  "arXiv 2022", "shared",
     "tbd", "botsort",      "botsort",      "botsort"),
    ("BoT-SORT-ReID", "aharon2022botsort",  "arXiv 2022", "shared",
     "tbd", "botsort_reid", "botsort_reid", "botsort_reid"),
    ("TrackTrack",    "kim2025tracktrack",  "CVPR 2025",  "shared",
     "tbd", "tracktrack",   "tracktrack",   "tracktrack"),
    ("FairMOT",       "zhang2021fairmot",   "IJCV 2021",  "own",
     "jdt", "fairmot_all",  "fairmot_all",  "fairmot"),
    ("TGraM",         "He2022TGraM",        "TGRS 2022",  "own",
     "jdt", "tgram_all",    "tgram_all",    "tgram"),
    ("MOTRv2",        "zhang2023motrv2",    "CVPR 2023",  "GT prop.",
     "query", None,         None,           "motrv2"),
]

# Panel (a) annotates each family with the detections it consumes; panel (b)
# feeds everyone the same GT boxes, so the annotation would be wrong there.
FAMILY_HEADER = {
    "tbd":   r"\textit{Tracking-by-detection} (shared cached detections)",
    "jdt":   r"\textit{Joint detection \& tracking} (own one-shot detector)",
    "query": r"\textit{Query-based end-to-end} (proposal port)",
}
FAMILY_HEADER_PLAIN = {
    "tbd":   r"\textit{Tracking-by-detection}",
    "jdt":   r"\textit{Joint detection \& tracking}",
    "query": r"\textit{Query-based end-to-end}",
}
FAMILY_ORDER = ["tbd", "jdt", "query"]

CAR_DATASETS = ["rscardata", "satmtb", "sdmcar"]
ORACLE_DATASETS = ["rscardata", "satmtb", "sdmcar", "airmot", "viso_no_car"]
SMALL_BINS = ["<5", "5-8"]          # pooled into the "AssA @ <=8 px" column

RATE_METRICS = ["HOTA", "DetA", "AssA", "MOTA", "IDF1"]
COUNT_METRICS = ["IDsw", "MT", "ML"]


# ----------------------------------------------------------------------
# readers
# ----------------------------------------------------------------------

def _read_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _agg(per_cell: list[dict]) -> dict:
    """Macro-mean the rate metrics, sum the count metrics."""
    out = {m: sum(float(r[m]) for r in per_cell) / len(per_cell)
           for m in RATE_METRICS}
    out.update({m: sum(int(float(r[m])) for r in per_cell)
                for m in COUNT_METRICS})
    return out


def load_car_half(tbd_csv: Path, jdt_csv: Path, jdt_satmtb_csv: Path) -> dict[str, dict]:
    """Macro-average over RsCarData / SAT-MTB (car-only) / SDM-Car."""
    by_key: dict[tuple[str, str], dict] = {}
    for r in _read_rows(tbd_csv):
        by_key[(r["dataset"], r["tracker"])] = r
    for r in _read_rows(jdt_csv):
        by_key[(r["dataset"], r["tracker"])] = r
    # The all-class JDT eval pools every class on SAT-MTB; the TbD rows are
    # car-only. Override those two cells with the car-only re-score.
    for r in _read_rows(jdt_satmtb_csv):
        by_key[(r["dataset"], r["tracker"])] = r

    out: dict[str, dict] = {}
    for disp, _c, _v, _d, _f, key_car, _kn, _ko in METHODS:
        if key_car is None:
            continue
        cells = [by_key[(d, key_car)] for d in CAR_DATASETS
                 if (d, key_car) in by_key]
        if len(cells) != len(CAR_DATASETS):
            raise SystemExit(
                f"{disp}: car half has {len(cells)}/{len(CAR_DATASETS)} datasets")
        out[disp] = _agg(cells)
    return out


def load_nocar_half(path: Path) -> dict[str, dict]:
    """Macro-average over every (dataset, class) cell.

    Test split only for every dataset — see tools/hota_nocar_testsplit.py: the
    published non-car table scores VISO / AIR-MOT on all sequences, which the
    JDT models trained on. Cells with no GT (VISO train) are absent for every
    method, so the macro-average is over the same 7 cells throughout.
    """
    cells: dict[str, list[dict]] = defaultdict(list)
    for r in _read_rows(path):
        cells[r["tracker"]].append(r)
    out: dict[str, dict] = {}
    for disp, _c, _v, _d, _f, _kc, key_nocar, _ko in METHODS:
        if key_nocar is None:
            continue
        out[disp] = _agg(cells[key_nocar])
    return out


def load_oracle(path: Path) -> dict[str, dict]:
    """Pool the 5 oracle datasets, weighting rates by GT-track count."""
    rows = _read_rows(path)
    acc: dict[str, dict] = defaultdict(
        lambda: {"w": 0.0, "AssA": 0.0, "IDF1": 0.0, "DetA": 0.0,
                 "IDsw": 0, "w_small": 0.0, "AssA_small": 0.0})
    for r in rows:
        method = r["method"]
        if r["dataset"] not in ORACLE_DATASETS:
            continue
        n = float(r["n_gt_tracks"])
        if n <= 0:
            continue
        a = acc[method]
        if r["size_bin"] == "all":
            a["w"] += n
            a["AssA"] += n * float(r["AssA"])
            a["IDF1"] += n * float(r["IDF1"])
            a["DetA"] += n * float(r["DetA"])
            a["IDsw"] += int(float(r["IDsw"]))
        elif r["size_bin"] in SMALL_BINS:
            a["w_small"] += n
            a["AssA_small"] += n * float(r["AssA"])

    out: dict[str, dict] = {}
    for disp, _c, _v, _d, _f, _kc, _kn, key in METHODS:
        a = acc[key]
        out[disp] = {
            "AssA": a["AssA"] / a["w"],
            "IDF1": a["IDF1"] / a["w"],
            "DetA": a["DetA"] / a["w"],
            "IDsw": a["IDsw"],
            "AssA_small": a["AssA_small"] / a["w_small"],
        }
    return out


# ----------------------------------------------------------------------
# formatting
# ----------------------------------------------------------------------

def fmt_rate(v: float) -> str:
    return f"{v:.3f}"


def fmt_int(v: float) -> str:
    return f"{int(v):,}"


def rank_marks(values: list[float | None], higher_is_better: bool) -> list[str]:
    """'\\textbf' for the best value, '\\underline' for the second best."""
    idx = [i for i, v in enumerate(values) if v is not None]
    marks = [""] * len(values)
    order = sorted(idx, key=lambda i: values[i], reverse=higher_is_better)
    if order:
        marks[order[0]] = r"\textbf"
    if len(order) > 1:
        marks[order[1]] = r"\underline"
    return marks


def _cell(mark: str, text: str) -> str:
    return f"{mark}{{{text}}}" if mark else text


def _marked_column(values: list[float | None], higher: bool, fmt) -> list[str]:
    marks = rank_marks(values, higher)
    return [("--" if v is None else _cell(marks[i], fmt(v)))
            for i, v in enumerate(values)]


def build_panel_a(car: dict, nocar: dict) -> list[str]:
    names = [m[0] for m in METHODS]
    cols: dict[str, list[str]] = {}
    for half, src in (("car", car), ("nocar", nocar)):
        for m in RATE_METRICS:
            vals = [src.get(n, {}).get(m) for n in names]
            cols[f"{half}_{m}"] = _marked_column(vals, True, fmt_rate)
        for m, higher in (("IDsw", False), ("MT", True), ("ML", False)):
            vals = [src.get(n, {}).get(m) for n in names]
            cols[f"{half}_{m}"] = _marked_column(vals, higher, fmt_int)

    order = RATE_METRICS[:3] + ["MOTA", "IDF1"] + COUNT_METRICS  # HOTA DetA AssA MOTA IDF1 IDsw MT ML
    lines = []
    for fam in FAMILY_ORDER:
        lines.append(r"\multicolumn{19}{l}{" + FAMILY_HEADER[fam] + r"} \\")
        for i, (disp, cite, venue, det, family, *_rest) in enumerate(METHODS):
            if family != fam:
                continue
            cells = [f"{disp}~\\citep{{{cite}}}", venue, det]
            for half in ("car", "nocar"):
                cells += [cols[f"{half}_{m}"][i] for m in order]
            lines.append("  " + " & ".join(cells) + r" \\")
        if fam != FAMILY_ORDER[-1]:
            lines.append(r"\midrule")
    return lines


def build_panel_b(oracle: dict) -> list[str]:
    names = [m[0] for m in METHODS]
    cols = {
        "AssA":       _marked_column([oracle[n]["AssA"] for n in names], True, fmt_rate),
        "AssA_small": _marked_column([oracle[n]["AssA_small"] for n in names], True, fmt_rate),
        "IDF1":       _marked_column([oracle[n]["IDF1"] for n in names], True, fmt_rate),
        "IDsw":       _marked_column([oracle[n]["IDsw"] for n in names], False, fmt_int),
        "DetA":       [fmt_rate(oracle[n]["DetA"]) for n in names],  # context, not ranked
    }
    lines = []
    for fam in FAMILY_ORDER:
        lines.append(r"\multicolumn{6}{l}{" + FAMILY_HEADER_PLAIN[fam] + r"} \\")
        for i, (disp, cite, _venue, _det, family, *_rest) in enumerate(METHODS):
            if family != fam:
                continue
            cells = [f"{disp}~\\citep{{{cite}}}"]
            cells += [cols[k][i] for k in
                      ("AssA", "AssA_small", "IDF1", "IDsw", "DetA")]
            lines.append("  " + " & ".join(cells) + r" \\")
        if fam != FAMILY_ORDER[-1]:
            lines.append(r"\midrule")
    return lines


def write_tex(car: dict, nocar: dict, oracle: dict, out_path: Path) -> None:
    L: list[str] = []
    add = L.append
    add(r"% Requires: \usepackage{booktabs, multirow, array}")
    add(r"% Generated by tools/aggregate_mot_overall_with_jdt.py -- do not hand-edit.")
    add(r"% Panel (a): full pipeline, each method with the detections it consumes.")
    add(r"% Panel (b): GT-box association oracle -- all nine methods comparable.")
    add("")
    add(r"\textbf{(a) Full pipeline.} Tracking-by-detection methods share one cached"
        r" detector per half (HiEUM on \emph{Car}, Faster~R-CNN on"
        r" \emph{Airplane/Ship/Train}); FairMOT and TGraM detect with their own"
        r" one-shot heads. MOTRv2 has no detector and is therefore absent here."
        r" Each half is a macro-average over its constituent splits: the"
        r" \emph{Car} half over RsCarData / SAT-MTB\,(car) / SDM-Car, the"
        r" right half over the seven annotated (dataset, class) cells of the"
        r" \emph{test} split.")
    add("")
    add(r"\setlength{\tabcolsep}{2.5pt}")
    add(r"\renewcommand{\arraystretch}{1.10}")
    add(r"\resizebox{\linewidth}{!}{%")
    add(r"\begin{tabular}{l l c cccccccc cccccccc}")
    add(r"\toprule")
    add(r"\multirow{2}{*}{\textbf{Method}} & \multirow{2}{*}{\textbf{Venue}}"
        r" & \multirow{2}{*}{\textbf{Det.}}")
    add(r"  & \multicolumn{8}{c}{\textbf{Car} --- RsCarData / SAT-MTB / SDM-Car}")
    add(r"  & \multicolumn{8}{c}{\textbf{Airplane / Ship / Train} --- SAT-MTB / VISO / AIR-MOT-100 (test split)} \\")
    add(r"\cmidrule(lr){4-11} \cmidrule(lr){12-19}")
    add(r"& & & HOTA $\uparrow$ & DetA $\uparrow$ & AssA $\uparrow$ & MOTA $\uparrow$ & IDF1 $\uparrow$ & IDsw $\downarrow$ & MT $\uparrow$ & ML $\downarrow$")
    add(r"  & HOTA $\uparrow$ & DetA $\uparrow$ & AssA $\uparrow$ & MOTA $\uparrow$ & IDF1 $\uparrow$ & IDsw $\downarrow$ & MT $\uparrow$ & ML $\downarrow$ \\")
    add(r"\midrule")
    L.extend(build_panel_a(car, nocar))
    add(r"\bottomrule")
    add(r"\end{tabular}%")
    add(r"}")
    add("")
    add(r"\vspace{0.9em}")
    add("")
    add(r"\textbf{(b) Association oracle.} The same ground-truth boxes are fed to"
        r" every method's unchanged association stage and pooled over all five MOT"
        r" splits, so detection quality is removed and all nine are directly"
        r" comparable. DetA is reported for context only: under the oracle it"
        r" measures how much of the perfect input a method chooses to emit"
        r" (track warm-up / confirmation), not box quality.")
    add("")
    add(r"\setlength{\tabcolsep}{5pt}")
    add(r"\begin{tabular}{l ccccc}")
    add(r"\toprule")
    add(r"\textbf{Method} & AssA $\uparrow$ & AssA$_{\le 8\,\text{px}}$ $\uparrow$"
        r" & IDF1 $\uparrow$ & IDsw $\downarrow$ & DetA$^{\dagger}$ \\")
    add(r"\midrule")
    L.extend(build_panel_b(oracle))
    add(r"\bottomrule")
    add(r"\end{tabular}")
    out_path.write_text("\n".join(L) + "\n")


def write_csv(car: dict, nocar: dict, oracle: dict, out_path: Path) -> None:
    fields = (["method", "venue", "detections", "family"]
              + [f"car_{m}" for m in RATE_METRICS + COUNT_METRICS]
              + [f"nocar_{m}" for m in RATE_METRICS + COUNT_METRICS]
              + ["oracle_AssA", "oracle_AssA_le8", "oracle_IDF1",
                 "oracle_IDsw", "oracle_DetA"])
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for disp, _cite, venue, det, family, *_rest in METHODS:
            row = {"method": disp, "venue": venue, "detections": det,
                   "family": family}
            for half, src in (("car", car), ("nocar", nocar)):
                for m in RATE_METRICS + COUNT_METRICS:
                    v = src.get(disp, {}).get(m)
                    row[f"{half}_{m}"] = (
                        "" if v is None else
                        (round(v, 4) if m in RATE_METRICS else int(v)))
            o = oracle[disp]
            row.update({
                "oracle_AssA":     round(o["AssA"], 4),
                "oracle_AssA_le8": round(o["AssA_small"], 4),
                "oracle_IDF1":     round(o["IDF1"], 4),
                "oracle_IDsw":     int(o["IDsw"]),
                "oracle_DetA":     round(o["DetA"], 4),
            })
            w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tbd-car-csv", default=str(DEFAULT_TBD_CAR_CSV))
    ap.add_argument("--nocar-csv", default=None,
                    help="Per-(dataset, class) non-car HOTA for both families "
                         "(default: <out-dir>/_nocar_testsplit_hota.csv).")
    ap.add_argument("--jdt-csv", default=str(DEFAULT_JDT_CSV))
    ap.add_argument("--jdt-satmtb-csv", default=None,
                    help="Car-only re-score of the JDT SAT-MTB tracks "
                         "(default: <out-dir>/_jdt_satmtb_caronly.csv).")
    ap.add_argument("--oracle-csv", default=str(DEFAULT_ORACLE_CSV))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jdt_satmtb = Path(args.jdt_satmtb_csv or (out_dir / "_jdt_satmtb_caronly.csv"))

    car = load_car_half(Path(args.tbd_car_csv), Path(args.jdt_csv), jdt_satmtb)
    nocar = load_nocar_half(
        Path(args.nocar_csv or (out_dir / "_nocar_testsplit_hota.csv")))
    oracle = load_oracle(Path(args.oracle_csv))

    write_tex(car, nocar, oracle, out_dir / "mot_combined_overall_jdt_table.tex")
    write_csv(car, nocar, oracle, out_dir / "mot_combined_overall_jdt.csv")
    print(f"[ok] wrote tables under {out_dir}")

    hdr = f"{'Method':16s} {'car HOTA':>9s} {'car AssA':>9s} " \
          f"{'nocar HOTA':>11s} {'orc AssA':>9s} {'orc<=8':>8s} {'orc DetA':>9s}"
    print("\n" + hdr)
    for disp, *_ in METHODS:
        c = car.get(disp, {})
        n = nocar.get(disp, {})
        o = oracle[disp]
        f = lambda d, k: f"{d[k]:.3f}" if k in d else "  --"
        print(f"{disp:16s} {f(c,'HOTA'):>9s} {f(c,'AssA'):>9s} "
              f"{f(n,'HOTA'):>11s} {o['AssA']:9.3f} {o['AssA_small']:8.3f} "
              f"{o['DetA']:9.3f}")


if __name__ == "__main__":
    main()
